import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import os
import re

from .generators import CaptchaFactory
from .store import CaptchaStore


_STORE = CaptchaStore()
_FACTORY = CaptchaFactory()
_TTL_SECONDS = 120
_DEBUG = False


def _json_bytes(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


def _ensure_image_size(info: dict) -> tuple[int, int]:
    """Try best-effort to obtain (image_width, image_height).

    Priority:
      1) explicit image_width/image_height in info
      2) parse from data_uri (embedded SVG)
      3) fallback to meta.width/meta.height
    """
    try:
        w = int(float(info.get("image_width") or 0))
        h = int(float(info.get("image_height") or 0))
    except Exception:
        w = h = 0
    if w and h:
        return w, h
    # Parse from data_uri (SVG)
    try:
        data_uri = info.get("data_uri") or ""
        if isinstance(data_uri, str) and data_uri.startswith("data:image/svg+xml"):
            import base64, urllib.parse
            payload = data_uri.split(",", 1)[1] if "," in data_uri else ""
            # handle base64 or urlencoded
            if ";base64" in data_uri:
                svg = base64.b64decode(payload).decode("utf-8", errors="ignore")
            else:
                svg = urllib.parse.unquote(payload)
            # viewBox first
            m = re.search(r"viewBox\s*=\s*['\"]\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*['\"]", svg, re.I)
            if m:
                w = w or int(float(m.group(1)))
                h = h or int(float(m.group(2)))
            else:
                m2 = re.search(r"width=['\"]([0-9.]+)['\"][^>]*height=['\"]([0-9.]+)['\"]", svg, re.I)
                if m2:
                    w = w or int(float(m2.group(1)))
                    h = h or int(float(m2.group(2)))
            if w and h:
                return w, h
    except Exception:
        pass
    # Fallback to meta
    try:
        meta = info.get("meta") or {}
        w = w or int(float(meta.get("width") or 0))
        h = h or int(float(meta.get("height") or 0))
    except Exception:
        pass
    return int(w or 0), int(h or 0)


def _iter_strings(obj):
    """Yield string fragments from nested dict/list structures."""
    try:
        if obj is None:
            return
        if isinstance(obj, str):
            yield obj
        elif isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(k, str):
                    yield k
                yield from _iter_strings(v)
        elif isinstance(obj, (list, tuple)):
            for it in obj:
                yield from _iter_strings(it)
    except Exception:
        return


def _try_extract_click_xy(reco: dict, w: int, h: int) -> tuple[int, int] | None:
    """Best-effort extract a bottom-left-origin click (x,y) from reco JSON.

    Accepts patterns:
      - fields: {"x":123, "y":45}
      - point dict: {"point": {"x":123, "y":45}} 或 {"calc_center": {...}}
      - 自定义标记: {"标记坐标": [{"x":..,"y":..}, ...]}
      - text like: "point 123,45" / "坐标: 123,45" / "x=123, y=45"
    Returns clamped integers within [0..w],[0..h] or None if not found.
    """
    # 标记坐标数组: 取第一个点
    try:
        marks = reco.get("标记坐标") if isinstance(reco, dict) else None
        if isinstance(marks, list) and marks:
            pt = marks[0]
            if isinstance(pt, dict) and isinstance(pt.get("x"), (int, float)) and isinstance(pt.get("y"), (int, float)):
                x = int(round(float(pt.get("x"))))
                y = int(round(float(pt.get("y"))))
                if w and h and (x < 0 or y < 0 or x > int(w) or y > int(h)):
                    return None
                return x, y
    except Exception:
        pass
    # Direct fields
    try:
        if isinstance(reco, dict) and isinstance(reco.get("x"), (int, float)) and isinstance(reco.get("y"), (int, float)):
            x = int(round(float(reco.get("x"))))
            y = int(round(float(reco.get("y"))))
            if w and h:
                # 若越界，说明使用了不同坐标系，直接放弃直提，交给下游 LLM 处理
                if x < 0 or y < 0 or x > int(w) or y > int(h):
                    return None
            return x, y
        # nested point / calc_center
        for key in ("point", "calc_center"):
            pt = reco.get(key) if isinstance(reco, dict) else None
            if (
                isinstance(pt, dict)
                and isinstance(pt.get("x"), (int, float))
                and isinstance(pt.get("y"), (int, float))
            ):
                x = int(round(float(pt.get("x"))))
                y = int(round(float(pt.get("y"))))
                if w and h:
                    if x < 0 or y < 0 or x > int(w) or y > int(h):
                        return None
                return x, y
    except Exception:
        pass
    # Scan text
    try:
        for s in _iter_strings(reco):
            if not isinstance(s, str) or len(s) > 4000:
                continue
            m = re.search(r"(?:point|坐标)\s*[:：]?\s*([0-9]+)\s*,\s*([0-9]+)", s, re.I)
            if not m:
                m = re.search(r"x\s*[:=]\s*([0-9]+)\s*,?\s*y\s*[:=]\s*([0-9]+)", s, re.I)
            if m:
                x = int(m.group(1))
                y = int(m.group(2))
                if w and h:
                    if x < 0 or y < 0 or x > int(w) or y > int(h):
                        return None
                return x, y
    except Exception:
        pass
    return None


def _try_extract_input_value(info: dict, reco: dict) -> str | None:
    """Try to deterministically extract an input value from recognition output.

    Heuristics:
      1) reco.value / reco.answer if present
      2) From steps: prefer label in {正确答案, 答案, 最终答案, 输入值}
         - pick the longest alnum/arrow/compare token (e.g. jjgqr, 123, >, YES)
      3) Fallback: scan all strings in reco and pick a concise candidate

    Apply norm rules: lower/int/exact.
    """
    norm = (info.get("norm") or "exact").lower()

    def norm_val(s: str) -> str:
        if not isinstance(s, str):
            s = str(s)
        if norm == "lower":
            return s.lower()
        if norm == "int":
            try:
                return str(int(float(s.strip())))
            except Exception:
                return None  # invalid
        # exact or others
        return s

    # 1) direct fields
    for k in ("value", "answer"):
        v = reco.get(k)
        if isinstance(v, str) and v.strip():
            nv = norm_val(v.strip())
            if nv is not None:
                return nv

    import re as _re
    preferred = {"正确答案", "答案", "最终答案", "输入值"}

    def extract_token(text: str) -> str | None:
        # allow letters/digits/arrow symbols/compare signs
        m = _re.findall(r"[A-Za-z0-9><=]+", text)
        if not m:
            return None
        # 如果全部是单字符且有多个，按出现顺序拼接，适用于“3 4 p B W”这类 case 题
        if all(len(t) == 1 for t in m) and len(m) > 1:
            joined = "".join(m)
            nv = norm_val(joined)
            return nv
        # otherwise prefer the longest token
        cand = max(m, key=len)
        nv = norm_val(cand)
        return nv

    # 2) math-aware pass: if norm=int，尝试从描述中识别算式并计算/取等号后数值
    def _math_value(text: str) -> str | None:
        try:
            # 优先取等号后的数字，如 "13 - 12 = 1"
            m_eq = _re.search(r"=\s*([0-9]+)", text)
            if m_eq:
                return str(int(m_eq.group(1)))
            m = _re.search(r"([0-9]+)\s*([+\-×x*/÷])\s*([0-9]+)", text)
            if not m:
                return None
            a = int(m.group(1)); b = int(m.group(3)); op = m.group(2)
            if op in {"+", "＋"}:
                return str(a + b)
            if op in {"-", "−"}:
                return str(a - b)
            if op in {"×", "x", "X", "*"}:
                return str(a * b)
            if op in {"÷", "/"}:
                if b != 0 and a % b == 0:
                    return str(a // b)
                if b != 0:
                    return str(a / b)
            return None
        except Exception:
            return None

    if norm == "int":
        steps = reco.get("steps")
        if isinstance(steps, list):
            for it in steps:
                try:
                    det = str((it or {}).get("detail") or "")
                    mv = _math_value(det)
                    if mv is not None:
                        return mv
                except Exception:
                    pass

    # 3) scan steps with preferred labels first
    steps = reco.get("steps")
    if isinstance(steps, list):
        # preferred labels
        for it in steps:
            try:
                lbl = str((it or {}).get("label") or "")
                if lbl in preferred:
                    det = str((it or {}).get("detail") or "")
                    tok = extract_token(det)
                    if tok:
                        return tok
            except Exception:
                pass
        # any detail
        for it in steps:
            try:
                det = str((it or {}).get("detail") or "")
                tok = extract_token(det)
                if tok:
                    return tok
            except Exception:
                pass

    # 4) scan all strings
    for s in _iter_strings(reco):
        if not isinstance(s, str) or len(s) > 4000:
            continue
        # math 优先
        if norm == "int":
            mv = _math_value(s)
            if mv is not None:
                return mv
        tok = extract_token(s)
        if tok:
            return tok
    return None


def _try_extract_grid_indices(info: dict, reco: dict) -> list[int] | None:
    """Attempt to extract full grid indices from recognition output without LLM.

    Sources considered:
      - reco.indices: [ints]
      - reco.points: [{x,y}, ...] (bottom-left origin, pixels) → map to indices
      - reco["标记坐标"]: [{x,y}, ...] (同上)
      - steps[].detail or any string containing a JSON-like indices array
    """
    try:
        if not isinstance(reco, dict):
            return None
        # 标记坐标 → indices
        marks = reco.get("标记坐标")
        if isinstance(marks, list) and marks:
            w, h = _ensure_image_size(info)
            rows = int(((info.get("meta") or {}).get("rows") or 3))
            cols = int(((info.get("meta") or {}).get("cols") or 3))
            def p2i(px: float, py: float) -> int:
                try:
                    col = max(0, min(cols-1, int((float(px) / float(w)) * cols))) if w else 0
                    row = max(0, min(rows-1, int(((float(h) - float(py)) / float(h)) * rows))) if h else 0
                    return row * cols + col
                except Exception:
                    return 0
            out = []
            for p in marks:
                if isinstance(p, dict) and isinstance(p.get("x"), (int, float)) and isinstance(p.get("y"), (int, float)):
                    out.append(p2i(p.get("x"), p.get("y")))
            if out:
                return sorted(set(out))
        # 1) direct indices
        inds = reco.get("indices")
        if isinstance(inds, list):
            out = []
            for v in inds:
                try:
                    out.append(int(v))
                except Exception:
                    pass
            if out:
                # unique + sorted
                return sorted(set(out))
        # 2) points → indices
        pts = reco.get("points")
        if isinstance(pts, list) and pts:
            w, h = _ensure_image_size(info)
            rows = int(((info.get("meta") or {}).get("rows") or 3))
            cols = int(((info.get("meta") or {}).get("cols") or 3))
            def p2i(px: float, py: float) -> int:
                try:
                    col = max(0, min(cols-1, int((float(px) / float(w)) * cols))) if w else 0
                    row = max(0, min(rows-1, int(((float(h) - float(py)) / float(h)) * rows))) if h else 0
                    return row * cols + col
                except Exception:
                    return 0
            out = []
            for p in pts:
                if not isinstance(p, dict):
                    continue
                if not (isinstance(p.get("x"),(int,float)) and isinstance(p.get("y"),(int,float))):
                    continue
                out.append(p2i(p.get("x"), p.get("y")))
            if out:
                return sorted(set(out))
        # 3) scan JSON-like indices from strings
        import json as _json
        for s in _iter_strings(reco):
            try:
                if not isinstance(s, str) or len(s) > 4000:
                    continue
                # simple detection of indices array
                if "indices" in s and "[" in s and "]" in s:
                    # try to parse the first {...} that contains "indices"
                    m = re.search(r"\{[\s\S]*?\}", s)
                    if m:
                        obj = _json.loads(m.group(0))
                        inds2 = obj.get("indices")
                        if isinstance(inds2, list) and inds2:
                            out = []
                            for v in inds2:
                                try:
                                    out.append(int(v))
                                except Exception:
                                    pass
                            if out:
                                return sorted(set(out))
            except Exception:
                continue
    except Exception:
        return None
    return None


def _try_extract_seq_indices(info: dict, reco: dict) -> list[int] | None:
    """Extract ordered indices for sequence tasks without LLM.

    Accepts:
      - reco.indices: keep order, drop non-integers
      - reco.points: keep order, map BL-origin pixels to indices
      - reco["标记坐标"]: list of points → ordered indices
    """
    try:
        if not isinstance(reco, dict):
            return None
        marks = reco.get("标记坐标")
        if isinstance(marks, list) and marks:
            w, h = _ensure_image_size(info)
            rows = int(((info.get("meta") or {}).get("rows") or 3))
            cols = int(((info.get("meta") or {}).get("cols") or 3))
            def p2i(px: float, py: float) -> int:
                try:
                    col = max(0, min(cols-1, int((float(px) / float(w)) * cols))) if w else 0
                    row = max(0, min(rows-1, int(((float(h) - float(py)) / float(h)) * rows))) if h else 0
                    return row * cols + col
                except Exception:
                    return 0
            ordered = []
            seen = set()
            for p in marks:
                if isinstance(p, dict) and isinstance(p.get("x"), (int, float)) and isinstance(p.get("y"), (int, float)):
                    idx = p2i(p.get("x"), p.get("y"))
                    if idx not in seen:
                        seen.add(idx)
                        ordered.append(idx)
            if ordered:
                return ordered
        inds = reco.get("indices")
        if isinstance(inds, list) and inds:
            out = []
            for v in inds:
                try:
                    out.append(int(v))
                except Exception:
                    continue
            # preserve order but drop duplicates (first occurrence wins)
            seen = set()
            ordered = []
            for v in out:
                if v not in seen:
                    seen.add(v)
                    ordered.append(v)
            return ordered if ordered else None
        pts = reco.get("points")
        if isinstance(pts, list) and pts:
            w, h = _ensure_image_size(info)
            rows = int(((info.get("meta") or {}).get("rows") or 3))
            cols = int(((info.get("meta") or {}).get("cols") or 3))
            def p2i(px: float, py: float) -> int:
                try:
                    col = max(0, min(cols-1, int((float(px) / float(w)) * cols))) if w else 0
                    row = max(0, min(rows-1, int(((float(h) - float(py)) / float(h)) * rows))) if h else 0
                    return row * cols + col
                except Exception:
                    return 0
            ordered = []
            seen = set()
            for p in pts:
                if not isinstance(p, dict):
                    continue
                if not (isinstance(p.get("x"),(int,float)) and isinstance(p.get("y"),(int,float))):
                    continue
                idx = p2i(p.get("x"), p.get("y"))
                if idx not in seen:
                    seen.add(idx)
                    ordered.append(idx)
            return ordered if ordered else None
        # 3) scan textual coordinates in order (e.g., "point 200,40 → 140,20 → …")
        import re as _re
        w, h = _ensure_image_size(info)
        rows = int(((info.get("meta") or {}).get("rows") or 3))
        cols = int(((info.get("meta") or {}).get("cols") or 3))
        def p2i(px: float, py: float) -> int:
            try:
                col = max(0, min(cols-1, int((float(px) / float(w)) * cols))) if w else 0
                row = max(0, min(rows-1, int(((float(h) - float(py)) / float(h)) * rows))) if h else 0
                return row * cols + col
            except Exception:
                return 0
        for s in _iter_strings(reco):
            if not isinstance(s, str) or len(s) > 4000:
                continue
            # only consider lines that likely describe a sequence
            if not ("point" in s.lower() or "→" in s or "坐标" in s):
                continue
            pairs = _re.findall(r"([0-9]{1,4})\s*,\s*([0-9]{1,4})", s)
            if len(pairs) >= 2:
                ordered = []
                seen = set()
                for x_str, y_str in pairs:
                    try:
                        idx = p2i(float(x_str), float(y_str))
                        if idx not in seen:
                            seen.add(idx)
                            ordered.append(idx)
                    except Exception:
                        continue
                if ordered:
                    return ordered
    except Exception:
        return None
    return None


## 本地规则已移除；改由 LLM 方案生成


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        pass
    # 优先从 ```json ... ``` 代码块中提取
    try:
        m = re.search(r"```\s*json\s*\n([\s\S]*?)\n```", text, re.I)
        if m:
            return json.loads(m.group(1))
    except Exception:
        pass
    # 尝试从代码块或正文中提取第一个 JSON 对象
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        frag = m.group(0)
        try:
            return json.loads(frag)
        except Exception:
            pass
    return {"summary": text.strip()[:500], "steps": []}


def _agent_guidance_llm(info: dict) -> dict:
    """使用 OpenAI 官方 GPT‑4o 模型进行推理，只输出 JSON 结构。

    说明（坐标系约定）：
    - 对于点选类题目（type=click/odd），请输出点击坐标 point x,y（以像素为单位）。
    - 坐标系以图像“左下角”为原点 (0,0)，x 向右、y 向上。
    - x ∈ [0, image_width]，y ∈ [0, image_height]。当无法精确时，也给出一个“合理中心点”。
    - 服务端会将该坐标换算为前端点击所需的“左上角为原点”的坐标进行验证。
    """
    from openai import OpenAI  # 延迟导入

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing for LLM agent")

    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.qingyuntop.top/v1")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o")
    client = OpenAI(base_url=base_url, api_key=api_key)

    kind = (info.get("kind") or info.get("type") or "").lower()
    prompt = info.get("prompt") or "请完成验证"
    norm = info.get("norm") or "exact"
    meta = info.get("meta") or {}
    data_uri = info.get("data_uri")
    image_url = info.get("image_url")
    image_png = info.get("image_png")

    sys = (
        "你是“识别Agent”，负责验证码的讲解与执行指导。务必输出‘可直接执行的具体操作方案’，包括观察要点与明确结论。\n"
        "重要：每次你给出的坐标会被标记到图像上再返给你复审，必须通过观察标记后的图判断落点是否在目标内，禁止仅靠公式/经验计算；只有确认在目标内部，才会交给执行Agent。\n"
        "在任何复审场景下，你不仅要判断坐标是否落在某个图形内部，还必须重新对照题目要求（颜色、形状、数量、位置等）检查该坐标所指图形是否满足题意：如果坐标在错误图形内，即便几何位置正确也必须视为 in_target=false 并给出新的合法坐标。\n"
        "只输出 JSON，结构固定为：\n"
        "{\n"
        "  \"summary\": \"一句话概述类型与策略（包含 type 与 norm 关键点）\",\n"
        "  \"point\": {\"x\": <像素>, \"y\": <像素>},  // 点击题(click/odd)必填，左下角为原点\n"
        "  \"in_target\": true | false,              // 明确说明该坐标是否在目标内部，若为 false 必须给出新的合法坐标\n"
        "  \"标记坐标\": [{\"x\":<像素>,\"y\":<像素>}, ...], // 对点击/网格/顺序题，列出所有需要点击的点（左下角为原点）；至少包含一个点\n"
        "  \"steps\": [\n"
        "    {\"label\":\"题目类型\", \"detail\":\"如：type=text/grid/seq 等；norm=lower/exact/int/set/list/point\"},\n"
        "    {\"label\":\"题目要求\", \"detail\":\"简述用户需要做什么（例如：点击与众不同的图形 / 依次点击 1→N / 输入算式结果）\"},\n"
        "    {\"label\":\"观察要点\", \"detail\":\"列出能从图面读到的关键信息（数量、颜色、形状差异、网格大小等）\"},\n"
        "    {\"label\":\"正确答案\", \"detail\":\"给出明确结论（例如：绿色五角星；或‘第 r,c 格’）。点击题必须直接用图像坐标轴/网格读数给出目标图形‘中心点’的像素坐标 point x,y（左下角为原点，坐标需落在图形内部而非边缘/背景）；给出前请自检该点是否在目标内，不在则调整后再输出。顺序/多选题优先给出 points 列表，按点击顺序。禁止输出仅基于公式推算的点，也不要返回边界框/推导过程，只给最终坐标。\"},\n"
        "    {\"label\":\"操作步骤\", \"detail\":\"逐条、可执行、含 UI 指令（先‘开启坐标工具(20×20 网格)’→按坐标点击… / 网格类可给出按顺序的 points 或索引）。点击题需先圈定边界，再直接在图上用坐标轴/网格读数确认点是否落在目标内部；坐标必须来自实际图像读数，禁止凭经验/模板推算或靠公式估算；只输出最终 point，勿输出计算步骤。请明确写出你观察与检查的动作，例如‘标记坐标后检查是否落在红色三角形内部’。\"},\n"
        "    {\"label\":\"答案格式示例\", \"detail\":\"提供‘格式模板’：输入类 <c1><c2>…；set 0,3,5；list 1,2,3,4,5；point x,y（像素）\"},\n"
        "    {\"label\":\"判定与注意\", \"detail\":\"相似字符区分(O/0, I/l/1)、干扰线、点击可重选、TTL 等实操提示\"}\n"
        "  ]\n"
        "}\n"
        "类型提示：输入类(text/distort/case/subset/anagram/math/compare/sumdigits/digitcount/vowelcount/hex2dec/count)，"
        "点选类(click/odd)，多选类(grid/gridcolor/gridshape/gridvowel)，顺序点击(seq/charseq/arrowseq/alphaseq/numseq)。\n"
        "坐标系：若为点击题，请使用‘左下角为原点’坐标，像素单位；若提供百分比也请换算为像素。必须利用图上的坐标轴/网格读取坐标，禁止估算或通过公式计算；在输出前确认点落在目标内部。必须直接观看图像并读取坐标轴刻度，先标点再复核是否在目标里。\n"
        "尺寸优先级：若同时提供 image_width/image_height 与 meta.width/meta.height，必须以 image_width/image_height 为唯一依据（不要使用 meta 尺寸）。\n"
        "注意：必须严格匹配提示词中的‘颜色 + 形状’，例如‘五角星’≠‘三角形’；若有同色不同形状，请只选择指定形状。允许给出‘正确答案’与点击位置；但避免输出账号/隐私等无关内容。\n"
        "请仔细分辨图形本身以及图形的颜色，尤其要区分相近颜色（如蓝色/紫色）和不同形状组合，只选择与题目提示完全一致的目标。\n"
    )

    # 针对顺序类任务，追加严格规则
    if kind in {"seq", "charseq", "arrowseq", "alphaseq", "numseq"}:
        sys += (
            "顺序题特别要求：必须根据图中标注的‘数字/字符/箭头’本身确定顺序，严禁按阅读顺序或空间布局推断。\n"
            "输出应包含 points（左下角像素，按点击顺序）并尽量包含 indices（0基索引，基于 rows×cols 与 image_width/height 计算）。\n"
        )
    # 针对网格多选题，强化“颜色+形状”约束
    if kind in {"grid", "gridcolor", "gridshape"}:
        sys += (
            "网格多选题要求特别注意：\n"
            "- 若提示词同时包含颜色和形状（如“蓝色方形”、“紫色圆形”），只能选择同时满足这两点的格子：既是该颜色，又是该形状；颜色对但形状错、形状对但颜色错都必须排除。\n"
            "- 若提示词只给形状（gridshape），忽略颜色，只按形状筛选；若只给颜色（gridcolor），忽略形状，只按颜色筛选。\n"
            "- 请先从提示词中拆解出“目标颜色 + 目标形状”，在 steps.观察要点/正确答案中明确写出筛选规则，再根据图像逐格检查，只输出符合规则的目标格子的中心点坐标。\n"
        )
    # 针对点击类，强调坐标必须位于目标中心
    if kind in {"click", "odd"}:
        sys += (
            "点击类特别要求：返回目标图形的中心点坐标（圆心/方形中心/三角形几何中心），坐标必须落在目标内部，不得落在边缘或旁边背景；若不确定，也请给出你判定的中心点。\n"
            "请先确定目标图形的边界框或半径，再用公式 x_center=(xmin+xmax)/2, y_center=(ymin+ymax)/2 计算中心；图上网格/坐标轴可用于精确读取坐标。\n"
            "在输出前，请模拟在图上标记该点，检查是否落在目标着色区域内；若发现落在外部/边缘，必须重新计算并只输出调整后的坐标，并在 steps 中标注你的检查/修正。允许在 steps 中说明“初始点→不在目标→微调为 (x,y)”。\n"
            "如收到 revise.prev_point，请先检查该坐标是否位于目标内：若在，直接确认输出；若不在，计算正确中心点并输出修正坐标，并在 steps 说明修正理由。\n"
            "务必在输出 JSON 中补充辅助信息（如有）：calc_center:{\"x\":...,\"y\":...}, bbox:{\"xmin\":...,\"xmax\":...,\"ymin\":...,\"ymax\":...}。\n"
            "颜色与形状的匹配也同样重要：请先在图中找到至少一个显然的目标样例（如真正的橙色三角形），再与其它图形对比颜色和轮廓；三角形只有三个明显的尖角，圆形没有角，紫色整体偏蓝而橙色偏红、偏暖，严禁把紫色圆形误判为橙色三角形等。\n"
        )
    # 针对元音/数字计数题，限定规则，避免误判
    if kind in {"vowelcount"}:
        sys += (
            "计数规则：只统计英文字母 AEIOU（大小写等价），不包含 Y、V 等辅音；忽略所有非字母字符（数字/符号）。\n"
        )

    # 组织文本信息（包含尺寸，先拼好再构建 content）
    # 如果是复审（带有 revise.prev_point），先提示模型验证/修正该点
    prev_point = None
    if isinstance(info.get("revise"), dict):
        pp = info["revise"].get("prev_point")
        if isinstance(pp, dict) and "x" in pp and "y" in pp:
            try:
                px = float(pp["x"]); py = float(pp["y"])
                prev_point = (px, py)
            except Exception:
                prev_point = None

    user_text_lines = [
        "请根据题面信息给出解题方法（只输出 JSON）:",
        f"type: {kind}",
        f"norm: {norm}",
        f"prompt: {prompt}",
        f"meta: {json.dumps(meta, ensure_ascii=False)}",
        "对于点击题，答案坐标必须是目标图形的中心点（圆心/方形中心/三角形几何中心），坐标需落在图形内部；不要返回大致位置或落在边缘/背景。",
        "坐标必须基于图像的坐标轴/网格读数得出，不得脱离图片凭经验或仅用公式计算；请先看图再给数，只给最终坐标，不要推导过程、边界框或计算公式。",
        "建议步骤：先识别并圈定目标的大致范围 → 用坐标轴/网格直接读出中心点坐标并给出，并在图上想象标记校验；只需给最终坐标，不要描述计算过程。",
        "输出前，请在脑中将点标到图上检查是否在目标内，并确认该点所指图形在“颜色 + 形状 + 位置”等方面完全符合题目要求；如发现不在目标内或图形不满足题意，请说明并给出你调整后的最终点。"
    ]
    if prev_point:
        user_text_lines.append(
            f"复审：此前的点为 point {int(prev_point[0])},{int(prev_point[1])}（已标在图上），必须直接查看标记后的图片判断是否在目标内；若不在，请基于坐标轴读数给出新的点，并在 steps 中说明修正过程。不得重复输出同一坐标；若确认在目标内，可重申并说明理由。若你再次输出相同坐标，将被视为失败。禁止仅靠公式估算，必须以标记后的图形为准，且不要输出边界框/推导步骤，只输出最终坐标。"
        )
        user_text_lines.append("注意：橙色十字标记使用的是左下角为原点的坐标，请直接用 BL 坐标判断是否在目标内，不要翻转 y。")
    # 提供图像尺寸，便于模型严格给出像素坐标（必须在 content 构造前加入）
    # 兜底：若客户端未提供尺寸，服务端尝试从 SVG data_uri 推断
    w, h = _ensure_image_size(info)
    if w and h:
        user_text_lines.append(f"image_width: {w}")
        user_text_lines.append(f"image_height: {h}")

    # 构造 user content（先文本后图片，图片优先使用 PNG）
    user_content = [
        {"type": "text", "text": "\n".join(user_text_lines)}
    ]
    # 优先使用 PNG dataURL 以规避代理对 SVG/WEBP 的兼容问题
    if isinstance(image_png, str) and image_png.startswith("data:image/png"):
        user_content.append({"type": "image_url", "image_url": {"url": image_png}})
    elif isinstance(data_uri, str) and data_uri.startswith("data:image"):
        user_content.append({"type": "image_url", "image_url": {"url": data_uri}})
    elif image_url:
        user_content.append({"type": "image_url", "image_url": {"url": image_url}})

    # 若有 revise.prev_point，显式告知模型需要检查该点并必要时修改
    if prev_point:
        user_content.append({"type": "text", "text": "已在图中用橙色十字标记了上一次的点击点，请结合网格核对是否在目标内；如不在，给出新的坐标（不要重复原坐标）。"})

    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": user_content},
    ]
    resp = client.chat.completions.create(model=model, messages=messages)
    text = resp.choices[0].message.content or ""
    return _extract_json(text)


def _agent_guidance_fallback(info: dict, err: str | None = None) -> dict:
    kind = (info.get("kind") or info.get("type") or "").lower()
    norm = (info.get("norm") or "exact").lower()
    prompt = info.get("prompt") or "请完成验证"
    meta = info.get("meta") or {}
    w, h = _ensure_image_size(info)
    steps = []
    steps.append({"label": "题目类型", "detail": f"type={kind}；norm={norm}"})
    steps.append({"label": "题目要求", "detail": prompt})
    if w and h:
        steps.append({"label": "观察要点", "detail": f"图像尺寸 {w}×{h}"})
    # 通用指导（不暴露真值）
    steps.append({"label": "操作步骤", "detail": "请根据题面完成操作；如需坐标，请开启坐标工具并点击目标"})
    if err:
        steps.append({"label": "回退说明", "detail": f"LLM 不可用/超时：{err}"})
    return {"summary": f"本地回退：{kind} 指南（不含坐标/索引/真值）", "steps": steps}


def _agent_guidance(info: dict) -> dict:
    # 默认使用 LLM；仅在 LLM 不可用时 fallback/取真值
    try:
        return _agent_guidance_llm(info)
    except Exception as e:
        return _agent_guidance_fallback(info, str(e))


def _agent_exec_input_llm(info: dict, reco: dict) -> dict:
    """执行Agent（输入类）：严格依据识别Agent输出，给出可直接输入的 value。

    约束：不得自行“推理答案”，只允许从识别Agent提供的 JSON/文字中抽取；若缺失则返回错误说明。
    输出 JSON:
      { "action": "input", "value": "..." }
    """
    def _normalize_value(val: str) -> str:
        norm = (info.get("norm") or "exact").lower()
        if not isinstance(val, str):
            val = str(val)
        s = val.strip()
        if norm == "lower":
            return s.lower()
        if norm == "int":
            try:
                return str(int(float(s)))
            except Exception:
                return s
        # 对 exact 类型，去除所有空白（包括零宽字符），避免 LLM 输出带空格/换行导致误判
        try:
            s = "".join(ch for ch in s if not ch.isspace())
            # 去零宽空格
            s = s.replace("\u200b", "").replace("\ufeff", "")
        except Exception:
            pass
        return s

    # 优先尝试无模型直提，避免网络波动导致的超时
    try:
        v = _try_extract_input_value(info, reco)
        if isinstance(v, str) and v.strip():
            nv = _normalize_value(v)
            return {"action": "input", "value": nv, "summary": reco.get("summary", "直接抽取答案"), "steps": reco.get("steps") or reco.get("plan") or []}
    except Exception:
        pass

    from openai import OpenAI  # 延迟导入

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing for LLM exec agent")

    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.qingyuntop.top/v1")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o")
    client = OpenAI(base_url=base_url, api_key=api_key)

    sys = (
        "你是‘执行Agent(输入类)’。只允许根据‘识别Agent’提供的结果来执行输入，"
        "不得自行推理或计算答案。\n"
        "只输出 JSON: {\"action\":\"input\", \"value\":\"<待输入字符串>\", \"summary\":\"本次执行概述\", \"steps\":[{\"step\":\"…\"}]}。\n"
        "若识别Agent未明确提供可输入的最终字符串，请输出 {\"error\":\"missing_value\"}。\n"
    )

    user = [
        {"type": "text", "text": "以下是题面信息与识别Agent输出，请给出输入值（只输出JSON）。"},
        {"type": "text", "text": "题面信息:"},
        {"type": "text", "text": json.dumps(info, ensure_ascii=False)},
        {"type": "text", "text": "识别Agent输出:"},
        {"type": "text", "text": json.dumps(reco, ensure_ascii=False)},
    ]
    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ]
    resp = client.chat.completions.create(model=model, messages=messages)
    text = resp.choices[0].message.content or ""
    out = _extract_json(text)
    try:
        if isinstance(out, dict) and isinstance(out.get("value"), str):
            out["value"] = _normalize_value(out["value"])
    except Exception:
        pass
    return out


def _agent_exec_click_llm(info: dict, reco: dict) -> dict:
    """执行Agent（点选类+网格/顺序）：严格依据识别Agent输出，给出执行指令。

    只能‘复述/结构化’识别Agent的结论，不得自行推理图片内容。
    允许的输出 JSON：
      - 单点点击（click/odd）：{"action":"click","x":<像素>,"y":<像素>}  坐标以左下角为原点。
      - 网格多选（grid/gridcolor/gridshape/gridvowel）：{"action":"grid","indices":[...]}  0基索引。
      - 顺序点击（seq/charseq/arrowseq/alphaseq/numseq）：{"action":"seq","indices":[...]}  0基索引、有序。

    若识别结果仅提供了‘点列表’而非索引，也可输出 {"action":"grid","points":[{"x":..,"y":..},...]}
    或 {"action":"seq","points":[...]}，坐标同样以左下角为原点，像素单位。

    若缺少必要信息，请输出 {"error":"missing_data"}。
    """
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing for LLM exec agent")

    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.qingyuntop.top/v1")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o")
    client = OpenAI(base_url=base_url, api_key=api_key)

    # 兜底：若客户端未提供尺寸，服务端尝试从 SVG data_uri 推断
    w, h = _ensure_image_size(info)

    kind = (info.get("kind") or info.get("type") or "").lower()
    is_grid = kind in {"grid", "gridcolor", "gridshape", "gridvowel"}
    is_seq = kind in {"seq", "charseq", "arrowseq", "alphaseq", "numseq"}
    is_clickish = kind in {"click", "odd"}

    # 优先尝试从识别Agent的输出中直接提取——仅限于匹配的题型
    if isinstance(reco, dict):
        if is_clickish:
            xy = _try_extract_click_xy(reco, w, h)
            if xy is not None:
                x, y = xy
                return {"action": "click", "x": int(x), "y": int(y), "summary": reco.get("summary", "直接提取坐标"), "steps": reco.get("steps") or reco.get("plan") or []}
        if is_grid:
            idxs = _try_extract_grid_indices(info, reco)
            if idxs:
                return {"action": "grid", "indices": idxs, "summary": reco.get("summary", "直接提取索引"), "steps": reco.get("steps") or reco.get("plan") or []}
        if is_seq:
            sidx = _try_extract_seq_indices(info, reco)
            if sidx:
                return {"action": "seq", "indices": sidx, "summary": reco.get("summary", "直接提取顺序索引"), "steps": reco.get("steps") or reco.get("plan") or []}
    sys = (
        "你是‘执行Agent(点击/网格/顺序)’。严格根据‘识别Agent’的输出，生成执行指令，不得自行推理图片。\n"
        "输出三种之一：\n"
        '1) {"action":"click", "x":<像素>, "y":<像素>}（左下角为原点）\n'
        '2) {"action":"grid", "indices":[0,3,5]} 或使用 points 数组（同为左下角原点像素）\n'
        '3) {"action":"seq",  "indices":[1,4,2]} 或使用 points 数组（同为左下角原点像素）\n'
        f"图像尺寸: width={w}, height={h}; 网格: rows={info.get('meta',{}).get('rows')}, cols={info.get('meta',{}).get('cols')}（如有）。\n"
        "若识别输出给的是‘集合/序列索引’，直接填入 indices；若给的是 ‘point x,y’，按需归类为 click/grid/seq。\n"
        '输出中可包含 \"summary\" 与 \"steps\" 字段，说明你的执行思路与步骤（简要、非推理）。\n'
        "尺寸优先级：务必以上述 width/height 为唯一坐标系尺寸，不能使用 meta.width/height。\n"
        "严格要求：\n- 对于多选题(grid/gridcolor/gridshape/gridvowel)，必须输出覆盖所有目标格子的完整 indices 列表，去重且升序；不得仅输出单个点或单个索引。\n- 对于顺序题(seq/charseq/arrowseq/alphaseq/numseq)，必须给出完整的 indices 列表，并保持点击顺序（按 1→k）。若识别结果给了多个‘point x,y’，按出现顺序转换为 indices。\n"
        '若缺少信息，输出 {"error":"missing_data"}。\n'
    )

    content = [
        {"type": "text", "text": "题面信息(仅作上下文，不可推理):"},
        {"type": "text", "text": json.dumps({k:info.get(k) for k in ("type","kind","prompt","meta","norm","image_width","image_height")}, ensure_ascii=False)},
        {"type": "text", "text": "识别Agent输出(仅据此生成执行指令):"},
        {"type": "text", "text": json.dumps(reco, ensure_ascii=False)},
    ]

    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": content},
    ]
    resp = client.chat.completions.create(model=model, messages=messages)
    text = resp.choices[0].message.content or ""
    return _extract_json(text)


INDEX_HTML = r"""
<!doctype html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>验证码演示</title>
  <style>
    :root {
      --bg: #f5f7fb;
      --card: #ffffff;
      --text: #1f2937;
      --muted: #6b7280;
      --primary: #268bd2;
      --primary-600: #1f78b5;
      --border: #e5e7eb;
      --shadow: 0 8px 20px rgba(0,0,0,0.06);
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #0e1116; --card:#151922; --text:#e5e7eb; --muted:#9aa4b2; --border:#263041; --primary:#3aa0ff; --primary-600:#2d84d1;
      }
    }
    html, body { background: var(--bg); color: var(--text); }
    body { font-family: -apple-system, BlinkMacSystemFont, Arial, sans-serif; margin: 0; }
    .container { max-width: 1100px; margin: 0 auto; padding: 2rem; }
    .row { display: flex; gap: 1.5rem; align-items: flex-start; flex-wrap: wrap; }
    .card { background: var(--card); border: 1px solid var(--border); padding: 1rem; border-radius: 12px; margin-top: 1rem; box-shadow: var(--shadow); }
    .stage { position: relative; display: block; width: 100%; max-width: 560px; border-radius: 10px; overflow: hidden; }
    #cap-img { width: 100%; height: auto; min-height: 180px; border: 1px solid var(--border); border-radius: 10px; background: #fff; cursor: crosshair; display: block; box-shadow: inset 0 0 0 1px rgba(0,0,0,0.02); }
    button { padding: 0.55rem 0.9rem; border: 1px solid var(--border); border-radius: 8px; background: linear-gradient(180deg,#fff, #f8fafc); color: var(--text); cursor: pointer; box-shadow: var(--shadow); }
    button:hover { border-color: var(--primary); box-shadow: 0 8px 20px rgba(38,139,210,0.15); }
    input { padding: 0.55rem 0.6rem; border:1px solid var(--border); border-radius:8px; background: var(--card); color: var(--text); }
    .muted { color: var(--muted); font-size: 0.92rem; }
    #cap-sel { position: absolute; left:0; top:0; width:100%; height:100%; pointer-events:none; }
    #cap-axes { position: absolute; left:0; top:0; width:100%; height:100%; pointer-events:none; }
    #cap-inline { display:block; }
    #cap-inline svg { width: 100%; height: auto; display:block; background:#fff; }
    .sel-box { position:absolute; border: 2px solid rgba(38,139,210,0.9); background: rgba(38,139,210,0.18); box-shadow: 0 4px 10px rgba(0,0,0,0.15); border-radius:8px; transition: background 120ms ease, transform 120ms ease; }
    #type-buttons button { background: linear-gradient(180deg,#fff,#f4f7fb); border:1px solid var(--border); border-radius:8px; cursor:pointer; }
    #type-buttons button.active { background: var(--primary); color:#fff; border-color: var(--primary-600); }
    .group { margin-top: 0.75rem; }
    .group h3 { margin: 0.4rem 0; font-size: 1rem; color: #333; }
    .group .buttons { display:flex; flex-wrap: wrap; gap: 0.5rem; }
    .badges { display:flex; gap:0.5rem; align-items:center; }
    .badge { display:inline-block; padding: 0.2rem 0.5rem; border-radius:999px; background: rgba(38,139,210,0.12); color: var(--primary-600); border: 1px solid rgba(38,139,210,0.2); font-size: 0.8rem; }
    @media (max-width: 900px) {
      .row { flex-direction: column; }
    }
    #agent-panel { margin-top: 0.6rem; background: var(--card); border:1px solid var(--border); border-radius:10px; padding:0.6rem; box-shadow: var(--shadow); max-height: 70vh; min-height: 420px; overflow:auto; }
    #agent-title { font-weight:600; margin:0 0 0.4rem 0; }
    .timeline { list-style:none; padding:0; margin:0; }
    .timeline li { position:relative; padding-left:1.2rem; margin:0.42rem 0; animation: fadeIn 260ms ease both; }
    .timeline li::before { content:""; position:absolute; left:0.4rem; top:0.45rem; width:6px; height:6px; background: var(--primary); border-radius:50%; box-shadow: 0 0 0 2px rgba(38,139,210,0.15); }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(2px);} to { opacity:1; transform: translateY(0);} }
    .sidebar { position: sticky; top: 1rem; }
    /* Switch */
    .switch { display:inline-flex; align-items:center; gap:0.4rem; font-size:0.9rem; color: var(--muted); }
    .switch input { width: 1rem; height:1rem; }
    /* Toast */
    #toast { position: fixed; right: 16px; bottom: 16px; z-index: 9999; display: none; background: var(--card); color: var(--text); border:1px solid var(--border); padding: 0.6rem 0.9rem; border-radius: 8px; box-shadow: var(--shadow); }
    #agent-skel .line{ height:10px; background: linear-gradient(90deg, rgba(0,0,0,0.06), rgba(0,0,0,0.12), rgba(0,0,0,0.06)); background-size:200% 100%; animation: shimmer 1s linear infinite; border-radius:6px; margin:8px 0; }
    @keyframes shimmer { from { background-position: 200% 0;} to { background-position: -200% 0;} }
    /* 详情视图 */
    .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 16px; }
    .kv { display: flex; gap: 8px; align-items: baseline; }
    .kv .k { min-width: 44px; color: var(--muted); }
    .kv .v { flex: 1; word-break: break-all; }
  </style>
  <script>
    let currentId = null;
    let currentType = null;
    let clickAnswer = null;
    let agentPoint = null; // 识别Agent建议的坐标（可视化用）
    let currentMeta = null;
    let gridSelected = new Set();
    let seqSelected = [];
    let axesGridN = 10; // 坐标网格密度（N×N）
    let ttlTimer = null;
    let ttlRemain = 0;
    let lastPayload = null; // 保存最近一次 /captcha 响应，供智能体读取图像等
    let agentResult = null; // 最近一次识别Agent的JSON结果
    let clickRetrying = false; // 点击类验证失败后的自动复审标记，避免递归

    const typeNames = {
      text:'字符型', math:'算术型', distort:'扭曲字符', subset:'指定位置', click:'点选图形', case:'区分大小写', sumdigits:'数字求和', count:'计数图形', grid:'网格多选', seq:'顺序点击', charseq:'字符顺序', odd:'找不同',
      gridvowel:'选择元音', arrowseq:'箭头顺序', anagram:'字母复原', compare:'比较大小', digitcount:'数字个数', vowelcount:'元音个数', palin:'回文判断', hex2dec:'十六进制转十进制',
      gridcolor:'按颜色多选', gridshape:'按形状多选', alphaseq:'字母顺序', numseq:'数字顺序'
    };

    function setActiveButton(t){
      const btns = document.querySelectorAll('#type-buttons button');
      btns.forEach(b=>b.classList.remove('active'));
      const active = document.querySelector(`#type-buttons button[data-type="${t}"]`);
      if (active) active.classList.add('active');
    }

    const CATS = [
      { key:'输入类', types:['text','math','case','distort','anagram'] },
      { key:'点选/九宫格', types:['click','odd','grid','gridcolor','gridshape'] },
      { key:'顺序点击', types:['seq','arrowseq','charseq','alphaseq','numseq'] },
      { key:'统计/规则', types:['sumdigits','digitcount','vowelcount','hex2dec','count'] }
    ];

    function renderTypeButtons(){
      const wrap = document.getElementById('type-buttons');
      wrap.innerHTML = '';
      CATS.forEach(cat => {
        const sec = document.createElement('div'); sec.className = 'group';
        const h = document.createElement('h3'); h.textContent = cat.key; sec.appendChild(h);
        const btns = document.createElement('div'); btns.className = 'buttons';
        cat.types.forEach(t => {
          const b = document.createElement('button');
          b.textContent = typeNames[t] || t; b.dataset.type = t; b.onclick = () => loadCaptcha(t);
          btns.appendChild(b);
        });
        sec.appendChild(btns);
        wrap.appendChild(sec);
      });
    }


    function setPrompt(t) {
      const el = document.getElementById('cap-prompt');
      el.textContent = t || '';
    }
    function setAgentCheck(text){
      const el = document.getElementById('cap-agent-check');
      if (el) el.textContent = text || '';
    }

    function setUIForType(t, meta) {
      const input = document.getElementById('cap-input');
      const slider = null;
      const clickTip = document.getElementById('cap-click-tip');
      const inline = null;
      input.style.display = 'none';
      // no slider
      clickTip.style.display = 'none';
      
      const imgEl = document.getElementById('cap-img');
      // 使用内嵌 SVG 展示，隐藏 <img>，避免出现重复空框
      imgEl.style.display = 'none';
      imgEl.onerror = () => { console.warn('图片加载失败'); showToast('图片加载失败'); };
      document.getElementById('cap-sel').innerHTML = '';
      gridSelected = new Set();
      seqSelected = [];
      if (t === 'click' || t === 'odd') {
        clickTip.style.display = '';
        // 重置提示文本与坐标，避免切换题型后仍显示上一次的坐标
        clickTip.textContent = '点击上方图片进行选择';
        clickAnswer = null;
        agentPoint = null;
        currentMeta = null;
        document.getElementById('grid-switch').style.display = 'none';
      } else if (t === 'grid') {
        clickTip.style.display = '';
        clickTip.textContent = '点击图片格子可选择/取消选择';
        currentMeta = meta || {};
        document.getElementById('grid-switch').style.display = '';
      } else if (t === 'gridcolor') {
        clickTip.style.display = '';
        clickTip.textContent = '按颜色选择对应格子，可多选';
        input.style.display = 'none';
        currentMeta = meta || {};
        document.getElementById('grid-switch').style.display = '';
      } else if (t === 'gridshape') {
        clickTip.style.display = '';
        clickTip.textContent = '按形状选择对应格子，可多选';
        input.style.display = 'none';
        currentMeta = meta || {};
        document.getElementById('grid-switch').style.display = '';
      } else if (t === 'gridvowel') {
        clickTip.style.display = '';
        clickTip.textContent = '选择所有元音字母';
        input.style.display = 'none';
        currentMeta = meta || {};
        document.getElementById('grid-switch').style.display = '';
      } else if (t === 'seq') {
        clickTip.style.display = '';
        clickTip.textContent = '按顺序点击 1→' + (meta && meta.k ? meta.k : 'N');
        currentMeta = meta || {};
        document.getElementById('grid-switch').style.display = '';
      } else if (t === 'charseq') {
        clickTip.style.display = '';
        clickTip.textContent = '按提示的字符顺序依次点击';
        input.style.display = 'none';
        currentMeta = meta || {};
        document.getElementById('grid-switch').style.display = '';
      } else if (t === 'arrowseq') {
        clickTip.style.display = '';
        clickTip.textContent = '按顺序点击：↑ → ↓ ←';
        input.style.display = 'none';
        currentMeta = meta || {};
        document.getElementById('grid-switch').style.display = '';
      } else if (t === 'alphaseq') {
        clickTip.style.display = '';
        clickTip.textContent = '按顺序点击：A → B → C → D → E';
        input.style.display = 'none';
        currentMeta = meta || {};
        document.getElementById('grid-switch').style.display = '';
      } else if (t === 'numseq') {
        clickTip.style.display = '';
        clickTip.textContent = '按顺序点击：1 → 2 → 3 → 4 → 5';
        input.style.display = 'none';
        currentMeta = meta || {};
        document.getElementById('grid-switch').style.display = '';
      } else {
        input.style.display = '';
        input.value = '';
        currentMeta = null;
        document.getElementById('grid-switch').style.display = 'none';
      }
      updateAxes();
    }

    function updateDetailsView(j){
      const elId = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v ?? ''; };
      elId('cap-id', j.id || '');
      elId('cap-type', j.type || '');
      elId('cap-norm', j.norm || '');
      elId('cap-prompt-inline', j.prompt || '');
      const rc = (j.meta && (j.meta.rows||j.meta.cols)) ? ((j.meta.rows||'?') + '×' + (j.meta.cols||'?')) : '-';
      elId('cap-rc', rc);
      const pre = document.getElementById('cap-json'); if (pre) pre.textContent = JSON.stringify(j, null, 2);
      // 从 SVG 或舞台计算尺寸
      setTimeout(() => {
        try {
          const sz = getStageSize();
          if (sz && sz.naturalW && sz.naturalH) elId('cap-size', sz.naturalW + '×' + sz.naturalH);
        } catch (e) {}
        updateHighlights();
      }, 0);
    }

    async function loadCaptcha(kind) {
      const r = await fetch(`/captcha?type=${encodeURIComponent(kind)}`);
      const j = await r.json();
      currentId = j.id;
      currentType = j.type;
      const inline = document.getElementById('cap-inline'); inline.style.display='block'; inline.innerHTML='';
      // 直接使用内嵌 SVG 展示
      try { inline.innerHTML = j.svg || ''; } catch(e){}
      lastPayload = j; // 缓存，识别Agent时可将 data_uri 一并传给服务端
      updateDetailsView(j);
      document.getElementById('cap-kind').textContent = (typeNames[j.type] || j.type);
      setActiveButton(j.type);
      if (ttlTimer) clearInterval(ttlTimer);
      ttlRemain = j.expires_in || 0;
      const expEl = document.getElementById('cap-exp');
      expEl.textContent = ttlRemain + 's';
      ttlTimer = setInterval(()=>{ ttlRemain = Math.max(0, ttlRemain-1); expEl.textContent = ttlRemain + 's'; if (ttlRemain<=0) clearInterval(ttlTimer); }, 1000);
      document.getElementById('cap-result').textContent = '';
      setPrompt(j.prompt || '');
      setUIForType(j.type, j.meta || {});
      // reset agent panel
      agentPoint = null;
      const ap = document.getElementById('agent-panel'); ap.style.display='';
      const ae = document.getElementById('agent-empty'); if (ae) ae.style.display='';
      document.getElementById('agent-summary').textContent='';
      document.getElementById('agent-steps').innerHTML='';
      if (j.debug_answer) {
        document.getElementById('cap-debug').textContent = '答案(调试): ' + j.debug_answer;
      } else {
      document.getElementById('cap-debug').textContent = '';
      setAgentCheck('');
      }
      // 初始化坐标工具为关闭
      axesGridN = 10;
      const axisSwitch = document.getElementById('axis-switch');
      axisSwitch.style.display = 'none';
      const ax = document.getElementById('toggle-axes'); ax.checked = false;
      updateAxes();
      // 加载失败回退：内嵌 SVG 展示
      const img = document.getElementById('cap-img');
      if (img) {
        img.onerror = () => {
          try { inline.innerHTML = j.svg || ''; inline.style.display='block'; img.style.display='none'; } catch(e){}
        };
      }
    }
    async function verify(silent=false) {
      let answer = '';
      if (currentType === 'click' || currentType === 'odd') {
        if (!clickAnswer) { if (!silent) alert('请点击图片中的目标图形'); return {ok:false, message:'no_click'}; }
        // 统一对外提交为“左下角为原点”的坐标，直接复用已记录的 bottom-left 值
        answer = clickAnswer.x + ',' + clickAnswer.y;
      } else if (currentType === 'grid' || currentType === 'gridcolor' || currentType === 'gridshape' || currentType === 'gridvowel') {
        if (!gridSelected.size) { if (!silent) alert('请至少选择一个格子'); return {ok:false, message:'no_grid_selection'}; }
        answer = Array.from(gridSelected).sort((a,b)=>a-b).join(',');
      } else if (currentType === 'seq' || currentType === 'charseq' || currentType === 'arrowseq' || currentType === 'alphaseq' || currentType === 'numseq') {
        if (!seqSelected.length) { if (!silent) alert('请按顺序点击图中数字'); return {ok:false, message:'no_seq_selection'}; }
        answer = seqSelected.join(',');
      } else {
        answer = document.getElementById('cap-input').value;
      }
      const sz1 = getStageSize();
      const r = await fetch('/verify', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ id: currentId, answer, origin: (currentType==='click'||currentType==='odd') ? 'bl' : undefined, image_height: sz1.naturalH })});
      const j = await r.json();
      const msg = j.ok ? '✅ 验证通过' : ('❌ 失败: ' + j.message);
      document.getElementById('cap-result').textContent = msg;
      showToast(msg);
      return j;
    }
    function addRecoStep(text){
      const list = document.getElementById('agent-steps');
      const li = document.createElement('li'); li.textContent = text; list.appendChild(li);
    }
    function addExecStep(text){
      const list = document.getElementById('exec-steps');
      const li = document.createElement('li'); li.textContent = text; list.appendChild(li);
    }
      // 兼容旧代码：时间线记录到“执行”面板
      function addTimeline(text){
        try{ addExecStep(text); }catch(e){}
      }

    // 调试日志：将每一步的请求/响应打印（省略大字段）
    function _safeObj(o){
      try{
        return JSON.parse(JSON.stringify(o, (k, v) => {
          if ((k === 'image_png' || k === 'data_uri') && typeof v === 'string') {
            return `<omitted ${v.length} chars>`;
          }
          return v;
        }));
      }catch(e){ return {}; }
    }
    function logJSON(label, obj){
      const pre = document.getElementById('agent-log');
      if (!pre) return;
      pre.textContent += `【${label}】\n` + JSON.stringify(_safeObj(obj||{}), null, 2) + '\n\n';
    }

    // 从识别Agent结果中尝试提取 point 坐标（仅用于可视化自检）
    function extractPointFromAgent(obj){
      try{
        if (!obj) return null;
        // 优先读取“标记坐标”数组
        if (Array.isArray(obj['标记坐标']) && obj['标记坐标'].length){
          const pt = obj['标记坐标'][0];
          if (pt && typeof pt.x === 'number' && typeof pt.y === 'number'){
            return { x: Math.round(pt.x), y: Math.round(pt.y) };
          }
        }
        // 优先结构化字段
        if (obj.calc_center && typeof obj.calc_center.x === 'number' && typeof obj.calc_center.y === 'number'){
          return { x: Math.round(obj.calc_center.x), y: Math.round(obj.calc_center.y) };
        }
        if (obj.point && typeof obj.point.x === 'number' && typeof obj.point.y === 'number'){
          return { x: Math.round(obj.point.x), y: Math.round(obj.point.y) };
        }
        const parseStr = (s) => {
          if (typeof s !== 'string') return null;
          // calc_center: {x: 72, y: 84}
          let m = s.match(/calc_center\s*[:：]?\s*\{\s*x\s*[:=]?\s*([0-9]+)\s*[,，]\s*y\s*[:=]?\s*([0-9]+)\s*\}/i);
          if (m) return { x: parseInt(m[1],10), y: parseInt(m[2],10) };
          // 所有形如 “123,45” 的坐标对（包含 point/x/y/括号等描述），统一提取再择优
          const all = [];
          const re = /([0-9]{1,4})\s*[,，]\s*([0-9]{1,4})/g;
          let mm;
          while ((mm = re.exec(s)) !== null){
            const x = parseInt(mm[1],10);
            const y = parseInt(mm[2],10);
            all.push({ x, y });
          }
          if (!all.length) return null;
          // 优先选择“像素级”坐标（至少一轴 > 10），避免误把 (2,2) 这种网格索引当成像素坐标
          const pixelLike = all.filter(p => Math.max(p.x, p.y) > 10);
          const chosen = (pixelLike.length ? pixelLike[pixelLike.length-1] : all[all.length-1]);
          return { x: chosen.x, y: chosen.y };
        };
        // 优先显式字段
        const direct = parseStr(obj.point) || parseStr(obj.answer) || null;
        if (direct) return direct;
        // 收集所有候选，优先“正确答案”标签，若无则取最后一个匹配
        const candidates = [];
        const steps = obj.steps || obj.plan || [];
        for (const st of steps){
          const text = st && (st.detail || st.action || st.step);
          const cand = parseStr(text);
          if (cand){
            const lbl = (st && st.label) || '';
            candidates.push({ cand, score: (typeof lbl === 'string' && lbl.includes('正确')) ? 2 : 1 });
          }
        }
        if (parseStr(obj.summary)) candidates.push({ cand: parseStr(obj.summary), score: 1 });
        if (!candidates.length) return null;
        // 先找最高分的最后一个，再 fallback 最后一个
        const maxScore = Math.max(...candidates.map(c=>c.score));
        for (let i=candidates.length-1; i>=0; i--){
          if (candidates[i].score === maxScore) return candidates[i].cand;
        }
        return candidates[candidates.length-1].cand;
      }catch(e){ return null; }
    }

    async function runAgent(){
      if (!currentId) return;
      const ap = document.getElementById('agent-panel');
      const sum = document.getElementById('agent-summary');
      const list = document.getElementById('agent-steps');
      ap.style.display='';
      const ae = document.getElementById('agent-empty'); if (ae) ae.style.display='none';
      sum.textContent = '识别Agent正在解答…';
      list.innerHTML = '';
      const exlist = document.getElementById('exec-steps'); if (exlist) exlist.innerHTML='';
      const logEl = document.getElementById('agent-log'); if (logEl) logEl.textContent = '';
      addRecoStep('流程：识别Agent → 执行Agent → 验证');
      setAgentCheck('识别Agent处理中...');
      try{
        // 将 SVG 渲染为 PNG dataURL，提升第三方接口兼容性
        const body = { id: currentId };
        if (lastPayload) {
          body.type = lastPayload.type;
          body.kind = lastPayload.type;
          body.norm = lastPayload.norm;
          body.prompt = lastPayload.prompt;
          body.meta = lastPayload.meta;
          body.data_uri = lastPayload.data_uri;
          // 生成 PNG 图像（在浏览器端完成矢量到位图转换）
          try {
            // 始终以 SVG 原始坐标空间尺寸为准（viewBox 或显式 width/height）
            const sz = getStageSize();
            let w = Math.max(1, Math.floor(sz.naturalW || 0));
            let h = Math.max(1, Math.floor(sz.naturalH || 0));
            if (!w || !h) {
              // 最后兜底：使用舞台显示尺寸
              const stage = document.getElementById('cap-stage');
              const rect = stage.getBoundingClientRect();
              w = Math.max(1, Math.floor(rect.width));
              h = Math.max(1, Math.floor(rect.height));
            }
            const canvas = document.createElement('canvas');
            canvas.width = Math.max(1, w);
            canvas.height = Math.max(1, h);
            const ctx = canvas.getContext('2d');
            // 若 img 未就绪，则新建离屏图像加载 data_uri
            const drawFromUri = () => new Promise((resolve)=>{
              const im = new Image();
              im.onload = () => { try { ctx.drawImage(im, 0, 0, canvas.width, canvas.height); resolve(); } catch(e){ resolve(); } };
              im.onerror = () => resolve();
              im.src = lastPayload.data_uri || '';
            });
            await drawFromUri();
            // 仅在需要坐标/索引的题型上绘制参考网格，避免 PNG 过大
            try {
              const t = (lastPayload && lastPayload.type) || '';
              // 只在需要网格索引/顺序的题型上叠加网格，点击题不叠加网格避免干扰识别
              const needGrid = (t === 'grid' || t === 'gridcolor' || t === 'gridshape' || t === 'gridvowel' || t === 'seq' || t === 'charseq' || t === 'arrowseq' || t === 'alphaseq' || t === 'numseq');
              if (needGrid){
                const N = 20;
                ctx.save();
                ctx.strokeStyle = 'rgba(0,0,0,0.18)';
                ctx.lineWidth = 1;
                for (let i=1;i<N;i++){
                  const x = Math.round(canvas.width * i/N);
                  ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,canvas.height); ctx.stroke();
                }
                for (let i=1;i<N;i++){
                  const y = Math.round(canvas.height * i/N);
                  ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(canvas.width,y); ctx.stroke();
                }
                ctx.restore();
              }
              if (t === 'click' || t === 'odd') {
                ctx.save();
                ctx.strokeStyle = 'rgba(0,0,0,0.55)';
                ctx.lineWidth = Math.max(1, Math.round(canvas.width / 240));
                const bottomY = canvas.height - ctx.lineWidth / 2;
                ctx.beginPath(); ctx.moveTo(0, bottomY); ctx.lineTo(canvas.width, bottomY); ctx.stroke();
                ctx.beginPath(); ctx.moveTo(ctx.lineWidth / 2, canvas.height); ctx.lineTo(ctx.lineWidth / 2, 0); ctx.stroke();
                ctx.fillStyle = 'rgba(255,255,255,0.65)';
                const legendHeight = Math.max(16, Math.round(canvas.height / 18));
                const legendWidth = Math.min(Math.round(canvas.width * 0.45), 160);
                ctx.fillRect(4, canvas.height - legendHeight - 4, legendWidth, legendHeight);
                ctx.fillStyle = 'rgba(0,0,0,0.75)';
                ctx.font = `${Math.max(12, Math.round(canvas.width / 40))}px -apple-system,Arial`;
                ctx.textBaseline = 'bottom';
                ctx.fillText('原点 (0,0)', 10, canvas.height - 6);
                ctx.strokeStyle = 'rgba(0,0,0,0.55)';
                ctx.textBaseline = 'bottom';
                ctx.textAlign = 'center';
                for (let i=0;i<=5;i++){
                  const x = Math.round(canvas.width * i/5);
                  ctx.beginPath(); ctx.moveTo(x, canvas.height - 8); ctx.lineTo(x, canvas.height); ctx.stroke();
                  const val = Math.round(w * i/5);
                  ctx.fillText(String(val), x, canvas.height - 10);
                }
                ctx.textAlign = 'left';
                ctx.textBaseline = 'middle';
                for (let i=1;i<=5;i++){
                  const y = canvas.height - Math.round(canvas.height * i/5);
                  ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(8, y); ctx.stroke();
                  const val = Math.round(h * i/5);
                  ctx.fillText(String(val), 10, y);
                }
                ctx.restore();
              }
            } catch(e) {}
            const png = canvas.toDataURL('image/png');
            if (png && png.startsWith('data:image/png')) body.image_png = png;
            // 坐标系尺寸必须与发送给 LLM 的 PNG 实际像素一致
            body.image_width = canvas.width;
            body.image_height = canvas.height;
          } catch (e) { /* 忽略客户端渲染失败 */ }
        }
        logJSON('请求 /agent', body);
        const r = await fetch('/agent', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
        let j = {};
        try { j = await r.json(); } catch (e) { j = {}; }
        logJSON('响应 /agent', j);
        if (!r.ok) {
          sum.textContent = j.error || ('识别Agent失败：HTTP ' + r.status);
          setAgentCheck('识别失败');
          return;
        }
        agentResult = j; // 保存供执行Agent使用
        sum.textContent = j.error || j.summary || j.text || '';
        (j.steps || []).forEach(s => {
          const li = document.createElement('li');
          li.textContent = (s.label ? (s.label+': ') : '') + (s.detail || s.action || '');
          list.appendChild(li);
        });
        // 可视化识别Agent推定的坐标，便于人工核查
        const ap = extractPointFromAgent(j);
        let revisions = 0;
        let reviewDone = false; // 是否完成“标记后复审”
        if (ap && Number.isFinite(ap.x) && Number.isFinite(ap.y)){
          agentPoint = { x: Math.round(ap.x), y: Math.round(ap.y) };
          addRecoStep('识别坐标: ' + agentPoint.x + ',' + agentPoint.y);
          setAgentCheck('已标记坐标，复审中…');
          updateAxes();
      // 自校正：为所有点击类题型，将已标记坐标叠加到图片上，再请求 LLM 校验/微调
          if (isClickTask(currentType)){
            // 空/零坐标直接放弃复审
            if (!(agentPoint && Number.isFinite(agentPoint.x) && Number.isFinite(agentPoint.y) && (agentPoint.x!==0 || agentPoint.y!==0))){
              addRecoStep('复审跳过：未获取有效坐标');
            } else {
            // 最多尝试四轮修正，每次都将标记后的图片返回给识别Agent检查；若确认无变化则直接进入执行
            while (revisions < 5 && agentPoint){
              const prevPoint = { ...agentPoint };
              revisions += 1;
              try{
                const sz = getStageSize();
                const w = Math.max(1, Math.floor(sz.naturalW || 0));
                const h = Math.max(1, Math.floor(sz.naturalH || 0));
                const canvas = document.createElement('canvas');
                canvas.width = w; canvas.height = h;
                const ctx = canvas.getContext('2d');
                const im = new Image();
                await new Promise((resolve)=>{ im.onload = () => { try{ ctx.drawImage(im,0,0,w,h); }catch(e){} resolve(); }; im.onerror=()=>resolve(); im.src = (lastPayload && lastPayload.data_uri) || ''; });
                // 叠加橙色十字，方便 LLM 看到当前点位（使用 BL 坐标，不翻转 prev_point）
                ctx.save();
                ctx.strokeStyle = 'rgba(255,126,0,0.95)';
                ctx.lineWidth = Math.max(3, Math.round(w/160));
                const cx = agentPoint.x;
                const cy = h - agentPoint.y; // 画布y轴向下，需要翻转
                const arm = Math.max(10, Math.round(w/40));
                ctx.beginPath(); ctx.moveTo(cx-arm, cy); ctx.lineTo(cx+arm, cy); ctx.stroke();
                ctx.beginPath(); ctx.moveTo(cx, cy-arm); ctx.lineTo(cx, cy+arm); ctx.stroke();
                ctx.restore();
                const png2 = canvas.toDataURL('image/png');
                const body2 = {
                  id: currentId,
                  type: lastPayload.type,
                  kind: lastPayload.type,
                  norm: lastPayload.norm,
                  prompt: lastPayload.prompt,
                  meta: lastPayload.meta,
                  data_uri: lastPayload.data_uri,
                  image_png: png2,
                  image_width: w,
                  image_height: h,
                  // 允许识别Agent直接确认当前坐标，无需强制调整
                  revise: { prev_point: agentPoint, require_change: false }
                };
                logJSON('请求 /agent(revise)', body2);
                const r2 = await fetch('/agent', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body2) });
                let j2 = {}; try { j2 = await r2.json(); } catch(e){ j2 = {}; }
                logJSON('响应 /agent(revise)', j2);
                if (!r2.ok) { setAgentCheck('复审失败'); agentPoint = null; break; }
                agentResult = j2;
                const ap2 = extractPointFromAgent(j2);
                if (ap2 && Number.isFinite(ap2.x) && Number.isFinite(ap2.y)){
                  const newPt = { x: Math.round(ap2.x), y: Math.round(ap2.y) };
                  const changed = (newPt.x !== prevPoint.x) || (newPt.y !== prevPoint.y);
                  if (changed) agentPoint = newPt; // 必须采纳模型给出的新点
                  reviewDone = true; // 至少完成了一次基于标记点的复审
                  addRecoStep(`复审坐标（第${revisions}次）: ${agentPoint.x},${agentPoint.y}`);
                  setAgentCheck(changed ? '复审完成：坐标已调整，继续复查新坐标…' : '复审完成：坐标已确认，准备执行');
                  updateAxes();
                  // 刷新步骤展示
                  list.innerHTML = '';
                  (j2.steps || []).forEach(s => {
                    const li = document.createElement('li');
                    li.textContent = (s.label ? (s.label+': ') : '') + (s.detail || s.action || '');
                    list.appendChild(li);
                  });
                  // 若模型给出新坐标，继续下一轮复审，直到稳定或达上限
                  if (!changed) break;
                  continue;
                } else {
                  agentPoint = null;
                  setAgentCheck('复审失败：未获取有效坐标');
                  break;
                }
              } catch(e){ agentPoint=null; break; }
            }
          }
          }
        } else {
          agentPoint = null;
          setAgentCheck('识别未返回坐标');
        }
        // 单点点击类题目必须完成“标记后复审”才允许后续执行
        if ((currentType === 'click' || currentType === 'odd') && (!reviewDone || !agentPoint)){
          addRecoStep('复审未完成或未取得有效坐标，停止执行Agent');
          return;
        }
        // 自动派发执行Agent（点击类需有有效坐标后才派发）
        let intent = 'input';
      if (!(isInputType(currentType))) intent = 'click';
      if (intent === 'click' && !agentPoint){
        addRecoStep('未获取有效坐标，暂停执行Agent，请检查标记点是否在目标内后重试');
        sum.textContent = '识别Agent未给出有效坐标，请重试';
        setAgentCheck('未获取有效坐标，停止执行');
        return;
      }
      addRecoStep('派发执行Agent：' + (intent==='input' ? '输入类' : '点击/网格/顺序'));
        let result = null;
        if (intent==='input') {
          result = await execInputAgent(true);
        } else {
          result = await execClickAgent(true);
        }
        // 展示执行Agent的思考与行动（如有）
        if (result && result._exec && (result._exec.summary || (result._exec.steps&&result._exec.steps.length))){
          addExecStep('执行Agent概述：' + (result._exec.summary||''));
          (result._exec.steps||[]).forEach(s=> addExecStep('执行步骤：' + (s.step||s.action||JSON.stringify(s))));
        }
        if (result && typeof result.ok === 'boolean') {
          addExecStep(result.ok ? '验证结果：通过' : ('验证结果：失败 - ' + (result.message||'')));
        }
      }catch(e){
        sum.textContent = '识别Agent失败：' + (e && e.message ? e.message : '未知错误');
      }
    }

    // —— 执行Agent：仅通过后端 LLM 执行 ——（本地备用解析已移除）

    function isInputType(t){
      return !(t === 'click' || t === 'odd' || t === 'grid' || t === 'gridcolor' || t === 'gridshape' || t === 'gridvowel' || t === 'seq' || t === 'charseq' || t === 'arrowseq' || t === 'alphaseq' || t === 'numseq');
    }
    function isClickTask(t){
      // 仅 click/odd 视为“单点点击题”，网格/顺序仍走点击流但不做点坐标直提提交
      return (t === 'click' || t === 'odd');
    }

    async function execInputAgent(autoMode){
      if (!agentResult){ showToast('请先运行识别Agent'); return; }
      if (!isInputType(currentType)){ showToast('当前题型非输入类'); return; }
      // 使用 SVG 原始尺寸，避免 <img> 隐藏导致的 0 值
      const sz0 = getStageSize();
      const w = sz0.naturalW;
      const h = sz0.naturalH;
      const body = { id: currentId, intent: 'input', reco: agentResult };
      addExecStep('准备执行输入类任务');
      if (lastPayload) {
        body.type = lastPayload.type; body.kind = lastPayload.type; body.norm = lastPayload.norm;
        body.prompt = lastPayload.prompt; body.meta = lastPayload.meta; body.data_uri = lastPayload.data_uri;
      }
      body.image_width = w; body.image_height = h;
      addExecStep('请求 /exec(input)：准备从识别结果抽取输入值');
      logJSON('请求 /exec (input)', body);
      const r = await fetch('/exec', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
      let j = {}; try { j = await r.json(); } catch (e) { j = {}; }
      logJSON('响应 /exec (input)', j);
      if (!r.ok || j.error){ addExecStep('执行失败：' + (j.error || ('HTTP '+r.status))); showToast('执行Agent(输入)失败: ' + (j.error || ('HTTP '+r.status))); return { ok:false, message: j.error || ('HTTP '+r.status) }; }
      if (j.action !== 'input' || typeof j.value !== 'string'){ showToast('执行Agent(输入)返回无效'); return { ok:false, message: 'invalid_action' }; }
      const input = document.getElementById('cap-input');
      addExecStep('填入值：' + j.value);
      input.style.display = '';
      input.value = j.value;
      // 记录 verify 请求
      logJSON('请求 /verify', { id: currentId, answer: j.value });
      addExecStep('提交验证');
      const res = await verify(Boolean(autoMode));
      res._exec = { summary: j.summary || '', steps: j.steps || j.plan || [] };
      logJSON('响应 /verify', res || {});
      return res;
    }

    async function execClickAgent(autoMode){
      if (!agentResult){ showToast('请先运行识别Agent'); return; }
      // 点击/网格/顺序均由执行Agent处理
      const img = document.getElementById('cap-img');
      // 始终基于舞台/SVG的自然尺寸进行坐标换算，避免隐藏<img>导致的 0 尺寸
      const sz0 = getStageSize();
      const w = sz0.naturalW;
      const h = sz0.naturalH;
      // 开启坐标工具（高密度细分网格），并记录到流程
      const maxDim = Math.max(sz0.naturalW || 0, sz0.naturalH || 0) || Math.max(w||0, h||0) || 300;
      const gridN = Math.max(40, Math.min(120, Math.round(maxDim / 14))); // ~14px 单元
      enableAxes(true, gridN);
      addTimeline('已开启坐标工具：' + gridN + '×' + gridN + ' 网格 + 坐标轴');
      const body = { id: currentId, intent: 'click', reco: agentResult };
      if (lastPayload) {
        body.type = lastPayload.type; body.kind = lastPayload.type; body.norm = lastPayload.norm;
        body.prompt = lastPayload.prompt; body.meta = lastPayload.meta; body.data_uri = lastPayload.data_uri;
      }
      // 尽量提供 PNG 以便模型参考（可选）。若 <img> 不可用，改用 data_uri 离屏加载
      try {
        const canvas = document.createElement('canvas');
        canvas.width = Math.max(1, w); canvas.height = Math.max(1, h);
        const ctx = canvas.getContext('2d');
        const drawFromUri = () => new Promise((resolve)=>{
          const im = new Image();
          im.onload = () => { try { ctx.drawImage(im, 0, 0, canvas.width, canvas.height); resolve(); } catch(e){ resolve(); } };
          im.onerror = () => resolve();
          im.src = (lastPayload && lastPayload.data_uri) || '';
        });
        if (img && (img.naturalWidth||img.width)) {
          try { ctx.drawImage(img, 0, 0, canvas.width, canvas.height); } catch(e) { await drawFromUri(); }
        } else {
          await drawFromUri();
        }
        const png = canvas.toDataURL('image/png');
        if (png && png.startsWith('data:image/png')) body.image_png = png;
      } catch (e) {}
      body.image_width = w; body.image_height = h;
      logJSON('请求 /exec (click)', body);
      const r = await fetch('/exec', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
      let j = {}; try { j = await r.json(); } catch (e) { j = {}; }
      logJSON('响应 /exec (click)', j);
      if (!r.ok || j.error){ showToast('执行Agent(点击)失败: ' + (j.error || ('HTTP '+r.status))); return { ok:false, message: j.error || ('HTTP '+r.status) }; }
      const meta = currentMeta || {};
      const rows = meta.rows || 3;
      const cols = meta.cols || 3;
      function pointToIndex(px, py){
        const col = Math.max(0, Math.min(cols-1, Math.floor((px / w) * cols)));
        const row = Math.max(0, Math.min(rows-1, Math.floor(((h - py) / h) * rows)));
        return row * cols + col;
      }
      if (j.action === 'click' && typeof j.x === 'number' && typeof j.y === 'number'){
        // 若当前题型是网格/顺序，优先将点换算为索引集合/序列
        if (currentType === 'grid' || currentType === 'gridcolor' || currentType === 'gridshape' || currentType === 'gridvowel') {
          const idx = pointToIndex(j.x, j.y);
          const answerStr = String(idx);
          logJSON('请求 /verify (grid-from-click)', { id: currentId, answer: answerStr });
          const r2 = await fetch('/verify', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ id: currentId, answer: answerStr }) });
          let res = {}; try { res = await r2.json(); } catch(e) { res = { ok:false, message:'bad_json' }; }
          enableAxes(false, axesGridN);
          res._exec = { summary: j.summary || '', steps: j.steps || j.plan || [] };
          logJSON('响应 /verify', res || {});
          return res;
        }
        if (currentType === 'seq' || currentType === 'charseq' || currentType === 'arrowseq' || currentType === 'alphaseq' || currentType === 'numseq') {
          const idx = pointToIndex(j.x, j.y);
          const answerStr = String(idx);
          logJSON('请求 /verify (seq-from-click)', { id: currentId, answer: answerStr });
          const r2 = await fetch('/verify', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ id: currentId, answer: answerStr }) });
          let res = {}; try { res = await r2.json(); } catch(e) { res = { ok:false, message:'bad_json' }; }
          enableAxes(false, axesGridN);
          res._exec = { summary: j.summary || '', steps: j.steps || j.plan || [] };
          logJSON('响应 /verify', res || {});
          return res;
        }
        // 单点点击：LLM 坐标为左下原点。为避免“描红暴露答案”，不在画面上标注，只后台提交验证。
        const blx = Math.max(0, Math.min(w, Math.round(j.x)));
        const bly = Math.max(0, Math.min(h, Math.round(j.y)));
        const answerStrBL = blx + ',' + bly;
        // 记录执行Agent的坐标，便于前端坐标轴/tool 提示保持一致
        clickAnswer = { x: blx, y: bly };
        const tip = document.getElementById('cap-click-tip');
        if (tip) tip.textContent = '执行Agent坐标(左下原点): ' + blx + ',' + bly;
        updateAxes();
        addExecStep('提交验证：answer=' + answerStrBL);
        logJSON('请求 /verify', { id: currentId, answer: answerStrBL, origin:'bl', image_height: h });
        const r2 = await fetch('/verify', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ id: currentId, answer: answerStrBL, origin:'bl', image_height: h }) });
        let res = {}; try { res = await r2.json(); } catch(e) { res = { ok:false, message:'bad_json' }; }
        addExecStep('已关闭坐标工具');
        enableAxes(false, axesGridN);
        res._exec = { summary: j.summary || '', steps: j.steps || j.plan || [] };
        logJSON('响应 /verify', res || {});
        return res;
      }
      if (j.action === 'grid'){
        // 网格多选：仅根据执行Agent给出的索引/坐标确定格子，不再在复审阶段移动到其它格子，避免破坏原本正确的选择
        let setIdx = new Set();
        const idxs = Array.isArray(j.indices) ? j.indices : null;
        if (idxs && idxs.length){
          idxs.forEach(i => { if (Number.isFinite(i)) setIdx.add(Number(i)); });
        } else if (Array.isArray(j.points) && j.points.length){
          j.points.forEach(p => {
            if (p && typeof p.x === 'number' && typeof p.y === 'number'){
              setIdx.add(pointToIndex(p.x, p.y));
            }
          });
        }
        if (!setIdx.size){ showToast('执行Agent未提供有效网格索引/坐标'); return { ok:false, message: 'no_grid' }; }
        const answerStr = Array.from(setIdx).sort((a,b)=>a-b).join(',');
        // 在 UI 上反映执行结果：高亮选中的网格
        gridSelected = new Set(setIdx);
        updateHighlights();
        const tip = document.getElementById('cap-click-tip');
        if (tip) tip.textContent = '执行Agent已选择: ' + answerStr;
        logJSON('请求 /verify', { id: currentId, answer: answerStr });
        const r2 = await fetch('/verify', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ id: currentId, answer: answerStr }) });
        let res = {}; try { res = await r2.json(); } catch(e) { res = { ok:false, message:'bad_json' }; }
        addTimeline('已关闭坐标工具');
        enableAxes(false, axesGridN);
        res._exec = { summary: j.summary || '', steps: j.steps || j.plan || [] };
        logJSON('响应 /verify', res || {});
        setAgentCheck('');
        return res;
      }
      if (j.action === 'seq'){
        // 顺序点击：仅根据执行Agent给出的索引/坐标确定顺序，不再在复审阶段移动到其它格子，避免破坏原本正确的顺序
        let seq = [];
        const idxs = Array.isArray(j.indices) ? j.indices : null;
        if (idxs && idxs.length){
          idxs.forEach(i => { if (Number.isFinite(i)) seq.push(Number(i)); });
        } else if (Array.isArray(j.points) && j.points.length){
          j.points.forEach(p => {
            if (p && typeof p.x === 'number' && typeof p.y === 'number'){
              seq.push(pointToIndex(p.x, p.y));
            }
          });
        }
        if (!seq.length){ showToast('执行Agent未提供有效顺序索引/坐标'); return { ok:false, message: 'no_seq' }; }
        const answerStr = seq.join(',');
        // 在 UI 上反映执行结果：高亮顺序点击
        seqSelected = seq.slice();
        updateHighlights();
        logJSON('请求 /verify', { id: currentId, answer: answerStr });
        const r2 = await fetch('/verify', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ id: currentId, answer: answerStr }) });
        let res = {}; try { res = await r2.json(); } catch(e) { res = { ok:false, message:'bad_json' }; }
        addTimeline('已关闭坐标工具');
        enableAxes(false, axesGridN);
        res._exec = { summary: j.summary || '', steps: j.steps || j.plan || [] };
        logJSON('响应 /verify', res || {});
        setAgentCheck('');
        return res;
      }
      showToast('执行Agent(点击)返回无效动作');
      return { ok:false, message: 'invalid_action' };
    }
    function getStageSize(){
      const img = document.getElementById('cap-img');
      const stage = document.getElementById('cap-stage');
      const rect = stage.getBoundingClientRect();
      let natW = img && img.naturalWidth ? img.naturalWidth : 0;
      let natH = img && img.naturalHeight ? img.naturalHeight : 0;
      if ((!natW || !natH) && lastPayload && lastPayload.svg){
        try{
          const m1 = lastPayload.svg.match(/viewBox\s*=\s*['\"]\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*['\"]/i);
          const m2 = lastPayload.svg.match(/width=['\"]([0-9.]+)['\"][^>]*height=['\"]([0-9.]+)['\"]/i);
          if (m1){ natW = parseFloat(m1[1]); natH = parseFloat(m1[2]); }
          else if (m2){ natW = parseFloat(m2[1]); natH = parseFloat(m2[2]); }
        }catch(e){}
      }
      return { displayW: rect.width, displayH: rect.height, naturalW: natW || rect.width, naturalH: natH || rect.height, rect };
    }

    function onImageClick(ev) {
      const stage = document.getElementById('cap-stage');
      const sz = getStageSize();
      const rect = sz.rect;
      const x = ev.clientX - rect.left;
      const y = ev.clientY - rect.top;
      if (currentType === 'grid' || currentType === 'gridcolor' || currentType === 'gridshape' || currentType === 'gridvowel' || currentType === 'seq' || currentType === 'charseq' || currentType === 'arrowseq' || currentType === 'alphaseq' || currentType === 'numseq') {
        const meta = currentMeta || {};
        const rows = meta.rows || 3;
        const cols = meta.cols || 3;
        const col = Math.max(0, Math.min(cols-1, Math.floor((x / rect.width) * cols)));
        const row = Math.max(0, Math.min(rows-1, Math.floor((y / rect.height) * rows)));
        const idx = row * cols + col;
        if (currentType === 'grid' || currentType === 'gridcolor' || currentType === 'gridshape' || currentType === 'gridvowel') {
          if (gridSelected.has(idx)) gridSelected.delete(idx); else gridSelected.add(idx);
          updateHighlights();
          document.getElementById('cap-click-tip').textContent = '已选择: ' + Array.from(gridSelected).sort((a,b)=>a-b).join(',');
        } else {
          // seq: toggle selection to allow修正/撤销，保持顺序（再次点击已选格子则移除它）
          const pos = seqSelected.indexOf(idx);
          if (pos >= 0) {
            seqSelected = seqSelected.filter(v => v !== idx);
          } else {
            seqSelected.push(idx);
          }
          updateHighlights();
          document.getElementById('cap-click-tip').textContent = '已点击顺序: ' + (seqSelected.length ? seqSelected.join('→') : '(空)');
        }
        return;
      }
      if (currentType !== 'click' && currentType !== 'odd') return;
      const scaleX = sz.naturalW / rect.width;
      const scaleY = sz.naturalH / rect.height;
      const xp = Math.round(x * scaleX);
      const yp_top = Math.round(y * scaleY); // top-left 原点
      const y_bl = Math.max(0, Math.min(sz.naturalH, Math.round(sz.naturalH - yp_top)));
      clickAnswer = {x: xp, y: y_bl}; // 直接记录“左下角原点”的数值，便于与 Agent 坐标对齐
      document.getElementById('cap-click-tip').textContent = '已选择坐标(左下原点): ' + xp + ',' + y_bl + '（再次点击可重选）';
      updateAxes();
    }

    function showToast(text){
      const t = document.getElementById('toast');
      if (!t) return;
      t.textContent = text;
      t.style.display = '';
      clearTimeout(showToast._timer);
      showToast._timer = setTimeout(()=>{ t.style.display='none'; }, 2600);
    }

    function updateHighlights() {
      const meta = currentMeta || {};
      const rows = meta.rows || 3;
      const cols = meta.cols || 3;
      const stage = document.getElementById('cap-stage');
      const sz = getStageSize();
      const rect = sz.rect;
      const cont = document.getElementById('cap-sel');
      cont.style.width = rect.width + 'px';
      cont.style.height = rect.height + 'px';
      cont.innerHTML = '';
      updateAxes();
      // subtle grid lines for grid-like types
      const isGrid = (currentType === 'grid' || currentType === 'gridcolor' || currentType === 'gridshape' || currentType === 'gridvowel' || currentType === 'seq' || currentType === 'charseq' || currentType === 'arrowseq' || currentType === 'alphaseq' || currentType === 'numseq');
      const gridToggle = document.getElementById('toggle-gridlines');
      if (isGrid && gridToggle && gridToggle.checked) {
        const cellW = rect.width / cols, cellH = rect.height / rows;
        cont.style.backgroundImage = `linear-gradient(to right, rgba(0,0,0,0.08) 1px, transparent 1px), linear-gradient(to bottom, rgba(0,0,0,0.08) 1px, transparent 1px)`;
        cont.style.backgroundSize = `${cellW}px ${cellH}px, ${cellW}px ${cellH}px`;
      } else {
        cont.style.backgroundImage = 'none';
      }
      const addBox = (idx, label) => {
        const r = Math.floor(idx / cols), c = idx % cols;
        const div = document.createElement('div');
        div.className = 'sel-box';
        div.style.left = (c * rect.width / cols) + 'px';
        div.style.top = (r * rect.height / rows) + 'px';
        div.style.width = (rect.width / cols) + 'px';
        div.style.height = (rect.height / rows) + 'px';
        if (label) { div.textContent = label; div.style.color = '#333'; div.style.fontWeight = 'bold'; div.style.textAlign='center'; div.style.lineHeight = (rect.height/rows) + 'px'; }
        cont.appendChild(div);
      };
      if (currentType === 'grid' || currentType === 'gridcolor' || currentType === 'gridshape' || currentType === 'gridvowel') {
        Array.from(gridSelected).forEach(i => addBox(i));
      } else if (currentType === 'seq' || currentType === 'charseq' || currentType === 'arrowseq' || currentType === 'alphaseq' || currentType === 'numseq') {
        seqSelected.forEach((i, idx) => addBox(i, String(idx+1)) );
      }
    }

    function enableAxes(show=true, gridN=20){
      axesGridN = Math.max(2, parseInt(gridN||20,10));
      const axisSwitch = document.getElementById('axis-switch');
      axisSwitch.style.display = '';
      const ax = document.getElementById('toggle-axes');
      ax.checked = !!show;
      updateAxes();
    }

    function updateAxes(){
      const axes = document.getElementById('cap-axes'); if (!axes) return;
      const show = document.getElementById('toggle-axes');
      const sz = getStageSize();
      axes.width = Math.max(1, Math.floor(sz.rect.width));
      axes.height = Math.max(1, Math.floor(sz.rect.height));
      const ctx = axes.getContext('2d');
      ctx.clearRect(0,0,axes.width,axes.height);
      if (!show || !show.checked) return;
      // draw axes and fine grid: origin bottom-left
      ctx.save();
      ctx.strokeStyle = 'rgba(0,0,0,0.18)';
      ctx.lineWidth = 1;
      // fine gridlines
      const N = axesGridN; // N divisions
      for(let i=1;i<N;i++){
        const x = Math.round(axes.width * i/N);
        ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,axes.height); ctx.stroke();
      }
      for(let i=1;i<N;i++){
        const y = Math.round(axes.height * i/N);
        ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(axes.width,y); ctx.stroke();
      }
      // darker axes
      ctx.strokeStyle = 'rgba(0,0,0,0.4)';
      // x-axis (bottom)
      ctx.beginPath(); ctx.moveTo(0, axes.height-0.5); ctx.lineTo(axes.width, axes.height-0.5); ctx.stroke();
      // y-axis (left)
      ctx.beginPath(); ctx.moveTo(0.5, axes.height); ctx.lineTo(0.5, 0); ctx.stroke();
      // ticks every 20%
      ctx.fillStyle = 'rgba(0,0,0,0.55)';
      ctx.font = '10px -apple-system,Arial';
      for (let i=1;i<=5;i++){
        const x = Math.round(axes.width * i/5);
        ctx.beginPath(); ctx.moveTo(x, axes.height-6); ctx.lineTo(x, axes.height); ctx.stroke();
        const val = Math.round(sz.naturalW * i/5);
        ctx.fillText(String(val), x-8, axes.height-8);
      }
      for (let i=1;i<=5;i++){
        const y = Math.round(axes.height * (1 - i/5));
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(6, y); ctx.stroke();
        const val = Math.round(sz.naturalH * i/5);
        ctx.fillText(String(val), 8, y+3);
      }
      // highlight selected point：将“左下角原点”的像素值转换成画布坐标，保持与 Agent 坐标一致
      if (clickAnswer && typeof clickAnswer.x==='number' && typeof clickAnswer.y==='number'){
        const scaleX = sz.naturalW ? (axes.width / sz.naturalW) : 1;
        const scaleY = sz.naturalH ? (axes.height / sz.naturalH) : 1;
        const cx = clickAnswer.x * scaleX;
        const cy = axes.height - (clickAnswer.y * scaleY);
        ctx.fillStyle = 'rgba(38,139,210,0.9)';
        ctx.beginPath(); ctx.arc(cx, cy, 5, 0, Math.PI*2); ctx.fill();
      }
      // 识别Agent建议的坐标（橙色十字标记，仅作参考，不提交验证）
      if (agentPoint && typeof agentPoint.x==='number' && typeof agentPoint.y==='number'){
        const scaleX = sz.naturalW ? (axes.width / sz.naturalW) : 1;
        const scaleY = sz.naturalH ? (axes.height / sz.naturalH) : 1;
        const cx = agentPoint.x * scaleX;
        const cy = axes.height - (agentPoint.y * scaleY);
        ctx.strokeStyle = 'rgba(255,126,0,0.95)';
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(cx-6, cy); ctx.lineTo(cx+6, cy); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(cx, cy-6); ctx.lineTo(cx, cy+6); ctx.stroke();
      }
      ctx.restore();
    }

    // 将当前点标记到图上并请求识别Agent再次确认，返回新的点或原样
    // positionPiece removed (slider captcha deleted)
    window.addEventListener('load', () => {
      document.getElementById('cap-stage').addEventListener('click', onImageClick);
      // 开关事件：坐标轴与网格线
      const ax = document.getElementById('toggle-axes');
      if (ax) ax.addEventListener('change', () => { updateAxes(); });
      const gl = document.getElementById('toggle-gridlines');
      if (gl) gl.addEventListener('change', () => { updateHighlights(); });
      // 窗口尺寸变化时，重绘坐标与高亮，避免错位
      window.addEventListener('resize', () => { updateHighlights(); });
      renderTypeButtons();
      loadCaptcha('text');
    });
  </script>
</head>
<body>
  <div class=\"container\">
  <h1 style=\"margin:0 0 0.5rem 0; font-size:1.6rem;\">多类型验证码演示</h1>
  <div class=\"row\" style=\"align-items:flex-start\"> 
    <div class=\"card\" style=\"flex:1; min-width: 560px;\"> 
      <div class=\"row\">
        <div class=\"stage\" id=\"cap-stage\"> 
          <img id=\"cap-img\" src=\"\" alt=\"captcha\" style=\"display:none\"/>
          <div id=\"cap-inline\"></div>
          <canvas id=\"cap-axes\"></canvas>
          <div id=\"cap-sel\"></div>
        </div>
        <div style=\"min-width:300px; display:flex; flex-direction:column; gap:0.4rem\"> 
          <div class=\"badges\"> 
            <span class=\"badge\">类型: <span id=\"cap-kind\"></span></span> 
            <span class=\"badge\">有效期: <span id=\"cap-exp\"></span></span> 
          </div> 
          <div id=\"cap-prompt\" class=\"muted\" style=\"margin-top:0.4rem\"></div> 
          <div id=\"cap-debug\" class=\"muted\"></div>
          <div id=\"cap-agent-check\" class=\"muted\"></div>
        </div> 
      </div> 
      <div style=\"margin-top: 0.6rem; align-items:center\" class=\"row\"> 
        <input id=\"cap-input\" placeholder=\"输入验证码\" /> 
        <span id=\"cap-click-tip\" class=\"muted\" style=\"display:none\">点击上方图片进行选择</span> 
        <button onclick=\"verify()\">验证</button> 
        <button onclick=\"loadCaptcha(currentType || 'text')\">刷新</button> 
        <button id=\"btn-agent\" onclick=\"runAgent()\">识别Agent</button> 
        <label class=\"switch\" id=\"grid-switch\" style=\"display:none\"><input type=\"checkbox\" id=\"toggle-gridlines\" checked />显示网格</label>
        <label class=\"switch\" id=\"axis-switch\" style=\"display:none\"><input type=\"checkbox\" id=\"toggle-axes\" />显示坐标轴</label>
      </div> 
      <div id=\"cap-info\" class=\"card\" style=\"margin-top:0.6rem\"> 
        <div style=\"display:flex; justify-content:space-between; align-items:center\"> 
          <div style=\"font-weight:600\">验证码详情</div>
          <div class=\"muted\">尺寸: <span id=\"cap-size\"></span></div>
        </div>
        <div class=\"grid2\" style=\"margin-top:0.4rem\"> 
          <div class=\"kv\"><span class=\"k\">ID</span><span class=\"v\" id=\"cap-id\"></span></div>
          <div class=\"kv\"><span class=\"k\">类型</span><span class=\"v\" id=\"cap-type\"></span></div>
          <div class=\"kv\"><span class=\"k\">规范</span><span class=\"v\" id=\"cap-norm\"></span></div>
          <div class=\"kv\"><span class=\"k\">网格</span><span class=\"v\" id=\"cap-rc\"></span></div>
        </div>
        <div class=\"kv\" style=\"margin-top:0.4rem\"><span class=\"k\">提示</span><span class=\"v\" id=\"cap-prompt-inline\"></span></div>
        <details style=\"margin-top:0.4rem\"> 
          <summary class=\"muted\">原始响应 JSON</summary>
          <pre id=\"cap-json\" style=\"white-space:pre-wrap; word-break:break-word; background:rgba(0,0,0,0.03); padding:8px; border-radius:8px;\"></pre>
        </details>
      </div>
    </div>
    <div class=\"card sidebar\" style=\"width:420px; flex:0 0 420px;\"> 
      <div id=\"agent-panel\" style=\"display:block\"> 
        <div id=\"agent-title\">识别Agent</div> 
        <div id=\"agent-summary\" class=\"muted\"></div>
        <ul id=\"agent-steps\" class=\"timeline\"></ul> 
        <div id=\"exec-title\" style=\"margin-top:0.6rem; font-weight:600;\">执行Agent</div>
        <ul id=\"exec-steps\" class=\"timeline\"></ul>
        <details id=\"agent-debug\" style=\"margin-top:0.5rem\" open> 
          <summary class=\"muted\">调试日志（请求/响应）</summary>
          <pre id=\"agent-log\" style=\"white-space:pre-wrap; word-break:break-word; background:rgba(0,0,0,0.03); padding:8px; border-radius:8px;\"></pre>
        </details>
      </div> 
      <div id=\"agent-empty\" class=\"muted\">点击“识别Agent”查看详细步骤</div>
      <div id=\"agent-skel\" style=\"display:none\"> <div class=\"line\"></div><div class=\"line\"></div><div class=\"line\"></div> </div>
    </div> 
  </div>
  <div id=\"type-buttons\" style=\"margin-top: 1rem; width:100%\"></div> 
  <p id=\"cap-result\" style=\"margin-top: 1rem;\"></p> 
  <div id=\"toast\"></div>
  </div>
  </body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "CaptchaServer/1.0"

    def _set_common_headers(self, code: int = 200, content_type: str = "application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.end_headers()

    def do_OPTIONS(self):  # CORS preflight
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._set_common_headers(200, "text/html; charset=utf-8")
            html = INDEX_HTML
            # 修正历史遗留的反斜杠转义，确保属性如 id="cap-img" 被浏览器正确解析
            # 反复替换直至没有残留
            while '\\"' in html:
                html = html.replace('\\"', '"')
            self.wfile.write(html.encode("utf-8"))
            return

        if parsed.path.startswith("/img/") and parsed.path.endswith(".png"):
            img_id = parsed.path[len("/img/"):-len(".png")]
            entry = _STORE.get(img_id)
            if not entry:
                self._set_common_headers(404)
                self.wfile.write(_json_bytes({"error": "not_found"}))
                return
            data_uri = entry.get("image_png")
            if not data_uri or not isinstance(data_uri, str) or "," not in data_uri:
                self._set_common_headers(404)
                self.wfile.write(_json_bytes({"error": "no_image"}))
                return
            try:
                import base64
                b64 = data_uri.split(",", 1)[1]
                content = base64.b64decode(b64)
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(content)
                return
            except Exception:
                self._set_common_headers(500)
                self.wfile.write(_json_bytes({"error": "decode_failed"}))
                return

        if parsed.path == "/captcha":
            params = parse_qs(parsed.query)
            kind = (params.get("type", ["text"]) or ["text"])[0]
            gen = _FACTORY.get(kind)
            c = gen.generate(ttl_seconds=_TTL_SECONDS)
            _STORE.put(
                c.id,
                c.answer,
                c.expires_at,
                c.norm,
                kind=getattr(c, "kind", None),
                prompt=getattr(c, "prompt", None),
                meta=getattr(c, "meta", None),
            )

            payload = {
                "id": c.id,
                "type": c.kind,
                "expires_in": _TTL_SECONDS,
                "svg": c.svg,
                "data_uri": c.data_uri,
                "norm": c.norm,
            }
            if getattr(c, "prompt", None):
                payload["prompt"] = c.prompt
            if getattr(c, "meta", None):
                payload["meta"] = c.meta
            if _DEBUG:
                payload["debug_answer"] = c.answer

            self._set_common_headers(200)
            self.wfile.write(_json_bytes(payload))
            return

        self._set_common_headers(404)
        self.wfile.write(_json_bytes({"error": "not_found"}))

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/verify":
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                body = json.loads(raw.decode("utf-8"))
            except Exception:
                body = {}
            id_ = str(body.get("id", ""))
            answer = str(body.get("answer", ""))
            # 兼容外部传入“左下角为原点”的 point 坐标：origin=bl
            entry = _STORE.get(id_) or {}
            origin = str(body.get("origin", "")).lower()
            if origin == 'bl':
                try:
                    norm = (entry.get("norm") or "").lower()
                    if norm.startswith("point:"):
                        # 解析 y_bl，并转换为 y_top 以与存储答案保持一致
                        ax, ay = [float(v) for v in answer.split(",", 1)]
                        h = 0
                        # 优先使用显式传入的 image_height；否则尝试 meta.height
                        try:
                            h = int(float(body.get("image_height") or 0))
                        except Exception:
                            h = 0
                        if not h:
                            try:
                                meta = entry.get("meta") or {}
                                h = int(float(meta.get("height") or 0))
                            except Exception:
                                h = 0
                        if h:
                            ay_top = h - ay
                            answer = f"{int(round(ax))},{int(round(ay_top))}"
                except Exception:
                    pass
            ok, msg = _STORE.verify(id_, answer)
            self._set_common_headers(200)
            self.wfile.write(_json_bytes({"ok": ok, "message": msg}))
            return
        if parsed.path == "/agent":
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                body = json.loads(raw.decode("utf-8"))
            except Exception:
                body = {}
            id_ = str(body.get("id", ""))
            info = _STORE.get(id_) or {}
            # 如果 GET /captcha 的响应未包含某些字段，尝试补齐
            for k in ("type","kind","prompt","meta","norm","data_uri","image_png","image_width","image_height"):
                if k in body and k not in info:
                    info[k] = body[k]
            if "revise" in body and "revise" not in info:
                info["revise"] = body["revise"]
            # 如果前端传了 image_png，则保存，并生成可被外网访问的 URL（基于 Host）
            if body.get("image_png"):
                _STORE.update(id_, image_png=body.get("image_png"))
                host = (self.headers.get("X-Forwarded-Host") or self.headers.get("Host") or "localhost").lower()
                proto = (self.headers.get("X-Forwarded-Proto") or ("https" if self.server.server_address[1] == 443 else "http")).lower()
                # 仅当 Host 不是本机地址时，才生成对外 URL
                if not (host.startswith("127.") or host.startswith("localhost") or host.startswith("0.0.0.0")):
                    info["image_url"] = f"{proto}://{host}/img/{id_}.png"
            try:
                data = _agent_guidance(info)
                self._set_common_headers(200)
                self.wfile.write(_json_bytes(data))
            except Exception as e:
                self._set_common_headers(500)
                self.wfile.write(_json_bytes({"error": str(e)}))
            return
        if parsed.path == "/exec":
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                body = json.loads(raw.decode("utf-8"))
            except Exception:
                body = {}
            id_ = str(body.get("id", ""))
            intent = str(body.get("intent", "")).strip().lower()
            reco = body.get("reco") or {}
            info = _STORE.get(id_) or {}
            for k in ("type","kind","prompt","meta","norm","data_uri","image_png","image_width","image_height"):
                if k in body and k not in info:
                    info[k] = body[k]
            try:
                if intent == "input":
                    data = _agent_exec_input_llm(info, reco)
                elif intent == "click":
                    data = _agent_exec_click_llm(info, reco)
                else:
                    raise RuntimeError("invalid_intent")
                self._set_common_headers(200)
                self.wfile.write(_json_bytes(data))
            except Exception as e:
                self._set_common_headers(500)
                self.wfile.write(_json_bytes({"error": str(e)}))
            return
        

        self._set_common_headers(404)
        self.wfile.write(_json_bytes({"error": "not_found"}))


def _cleanup_task():
    while True:
        try:
            _STORE.cleanup()
        except Exception:
            pass
        time.sleep(30)


def run_server(host: str = "127.0.0.1", port: int = 8000, ttl_seconds: int = 120, debug: bool = False):
    global _TTL_SECONDS, _DEBUG
    _TTL_SECONDS = max(10, int(ttl_seconds))
    _DEBUG = bool(debug)

    th = threading.Thread(target=_cleanup_task, daemon=True)
    th.start()

    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Serving on http://{host}:{port} (ttl={_TTL_SECONDS}s, debug={_DEBUG}) …")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
