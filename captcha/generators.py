import time
import uuid
import random
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

from .svg import (
    text_captcha_svg,
    math_captcha_svg,
    distorted_text_captcha_svg,
    shapes_click_svg,
    shapes_count_svg,
    odd_shape_svg,
    grid_color_svg,
    sequence_chars_grid_svg,
    grid_shape_only_svg,
    grid_vowel_letters_svg,
    to_data_uri,
)


@dataclass
class CaptchaData:
    id: str
    kind: str
    svg: str
    data_uri: str
    answer: str
    expires_at: float
    norm: str  # 'lower', 'exact', 'int', 'point:<tol>', 'approx:<tol>'
    prompt: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


class TextCaptcha:
    def __init__(self, length: int = 5, width: int = 160, height: int = 60, charset: Optional[str] = None):
        self.length = length
        self.width = width
        self.height = height
        self.charset = charset or "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # avoid ambiguous chars

    def _random_text(self) -> str:
        return "".join(random.choice(self.charset) for _ in range(self.length))

    def generate(self, ttl_seconds: int = 120) -> CaptchaData:
        text = self._random_text()
        svg = text_captcha_svg(text, self.width, self.height)
        return CaptchaData(
            id=uuid.uuid4().hex,
            kind="text",
            svg=svg,
            data_uri=to_data_uri(svg),
            answer=text.lower(),
            expires_at=time.time() + ttl_seconds,
            norm="lower",
            prompt="请输入图中字符（不区分大小写）",
        )


class MathCaptcha:
    def __init__(self, width: int = 200, height: int = 70, value_range: Tuple[int, int] = (1, 20)):
        self.width = width
        self.height = height
        self.value_range = value_range

    def _rand_val(self) -> int:
        a, b = self.value_range
        return random.randint(a, b)

    def _expression(self) -> tuple[str, int]:
        a = self._rand_val()
        b = self._rand_val()
        op = random.choice(["+", "-", "×"])  # use multiplication sign
        if op == "+":
            ans = a + b
        elif op == "-":
            if b > a:
                a, b = b, a
            ans = a - b
        else:
            # keep products modest
            a = random.randint(2, 9)
            b = random.randint(2, 9)
            ans = a * b
        expr = f"{a} {op} {b} = ?"
        return expr, ans

    def generate(self, ttl_seconds: int = 120) -> CaptchaData:
        expr, ans = self._expression()
        svg = math_captcha_svg(expr, self.width, self.height)
        return CaptchaData(
            id=uuid.uuid4().hex,
            kind="math",
            svg=svg,
            data_uri=to_data_uri(svg),
            answer=str(ans),
            expires_at=time.time() + ttl_seconds,
            norm="int",
            prompt="请输入算式的结果",
        )


class CaptchaFactory:
    def __init__(self):
        self._registry = {
            "text": TextCaptcha(),
            "math": MathCaptcha(),
            "distort": None,  # lazy
            "subset": None,
            "click": None,
            "case": None,
            "sumdigits": None,
            "count": None,
            "grid": None,
            "seq": None,
            "odd": None,
            "gridcolor": None,
            "charseq": None,
            "gridshape": None,
            "gridvowel": None,
            "arrowseq": None,
            "alphaseq": None,
            "numseq": None,
            "anagram": None,
            "compare": None,
            "digitcount": None,
            "vowelcount": None,
            "palin": None,
            "hex2dec": None,
        }

    def get(self, kind: str):
        kind = (kind or "text").lower()
        if kind not in self._registry:
            kind = "text"
        if self._registry[kind] is None:
            # Lazy init for heavier types
            if kind == "distort":
                self._registry[kind] = DistortedTextCaptcha()
            elif kind == "subset":
                self._registry[kind] = SubsetTextCaptcha()
            elif kind == "click":
                self._registry[kind] = ClickShapeCaptcha()
            elif kind == "case":
                self._registry[kind] = CaseSensitiveTextCaptcha()
            elif kind == "sumdigits":
                self._registry[kind] = SumDigitsCaptcha()
            elif kind == "count":
                self._registry[kind] = CountShapeCaptcha()
            elif kind == "grid":
                self._registry[kind] = GridSelectCaptcha()
            elif kind == "seq":
                self._registry[kind] = SequenceGridCaptcha()
            elif kind == "odd":
                self._registry[kind] = OddOneCaptcha()
            elif kind == "gridcolor":
                self._registry[kind] = GridColorSelectCaptcha()
            elif kind == "charseq":
                self._registry[kind] = CharSequenceCaptcha()
            elif kind == "gridshape":
                self._registry[kind] = GridShapeOnlyCaptcha()
            elif kind == "gridvowel":
                self._registry[kind] = GridVowelSelectCaptcha()
            elif kind == "arrowseq":
                self._registry[kind] = ArrowSequenceCaptcha()
            elif kind == "alphaseq":
                self._registry[kind] = AlphaSequenceCaptcha()
            elif kind == "numseq":
                self._registry[kind] = NumSequenceCaptcha()
            elif kind == "anagram":
                self._registry[kind] = AnagramCaptcha()
            elif kind == "compare":
                self._registry[kind] = CompareCaptcha()
            elif kind == "digitcount":
                self._registry[kind] = DigitCountCaptcha()
            elif kind == "vowelcount":
                self._registry[kind] = VowelCountCaptcha()
            elif kind == "palin":
                self._registry[kind] = PalindromeCaptcha()
            elif kind == "hex2dec":
                self._registry[kind] = HexToDecCaptcha()
        return self._registry[kind]


class DistortedTextCaptcha(TextCaptcha):
    def __init__(self, length: int = 5, width: int = 180, height: int = 70, charset: Optional[str] = None):
        super().__init__(length, width, height, charset)

    def generate(self, ttl_seconds: int = 120) -> CaptchaData:
        text = self._random_text()
        svg = distorted_text_captcha_svg(text, self.width, self.height)
        return CaptchaData(
            id=uuid.uuid4().hex,
            kind="distort",
            svg=svg,
            data_uri=to_data_uri(svg),
            answer=text.lower(),
            expires_at=time.time() + ttl_seconds,
            norm="lower",
            prompt="请输入图中字符",
        )


class SubsetTextCaptcha(TextCaptcha):
    def __init__(self, length: int = 6, width: int = 180, height: int = 70, charset: Optional[str] = None):
        super().__init__(length, width, height, charset)

    def generate(self, ttl_seconds: int = 120) -> CaptchaData:
        text = self._random_text()
        # Choose 2-3 distinct positions (1-based)
        k = random.choice([2, 3])
        idxs = sorted(random.sample(range(1, len(text) + 1), k))
        # Chinese index representation: 第1、3、5位
        idx_str = "、".join(str(i) for i in idxs)
        ans = "".join(text[i - 1] for i in idxs).lower()
        svg = text_captcha_svg(text, self.width, self.height)
        return CaptchaData(
            id=uuid.uuid4().hex,
            kind="subset",
            svg=svg,
            data_uri=to_data_uri(svg),
            answer=ans,
            expires_at=time.time() + ttl_seconds,
            norm="lower",
            prompt=f"请输入第{idx_str}位字符",
            meta={"full_text_len": len(text)},
        )


class ClickShapeCaptcha:
    def __init__(self, width: int = 240, height: int = 120, count: int = 6):
        self.width = width
        self.height = height
        self.count = count

    def generate(self, ttl_seconds: int = 120) -> CaptchaData:
        svg, target_label, cx, cy, tol = shapes_click_svg(self.width, self.height, self.count)
        ans = f"{int(cx)},{int(cy)}"  # store as "x,y"
        return CaptchaData(
            id=uuid.uuid4().hex,
            kind="click",
            svg=svg,
            data_uri=to_data_uri(svg),
            answer=ans,
            expires_at=time.time() + ttl_seconds,
            norm=f"point:{tol}",
            prompt=f"请点击：{target_label}",
            meta={"width": self.width, "height": self.height, "tolerance": tol},
        )


class CaseSensitiveTextCaptcha(TextCaptcha):
    def __init__(self, length: int = 5, width: int = 180, height: int = 70):
        super().__init__(length, width, height, charset="ABCDEFGHJKLMNPQRSTUVWXYZ23456789")

    def _random_text(self) -> str:
        # mix random cases
        base = super()._random_text()
        out = []
        for ch in base:
            if ch.isalpha() and random.random() < 0.5:
                out.append(ch.lower())
            else:
                out.append(ch)
        return "".join(out)

    def generate(self, ttl_seconds: int = 120) -> CaptchaData:
        text = self._random_text()
        svg = text_captcha_svg(text, self.width, self.height)
        return CaptchaData(
            id=uuid.uuid4().hex,
            kind="case",
            svg=svg,
            data_uri=to_data_uri(svg),
            answer=text,  # exact
            expires_at=time.time() + ttl_seconds,
            norm="exact",
            prompt="大小写敏感：按图输入字符（区分大小写）",
        )


class ReverseTextCaptcha(TextCaptcha):
    def __init__(self, length: int = 6, width: int = 200, height: int = 70, charset: Optional[str] = None):
        super().__init__(length, width, height, charset or "ABCDEFGHJKLMNPQRSTUVWXYZ23456789")

    def generate(self, ttl_seconds: int = 120) -> CaptchaData:
        text = super()._random_text()
        svg = text_captcha_svg(text, self.width, self.height)
        return CaptchaData(
            id=uuid.uuid4().hex,
            kind="reverse",
            svg=svg,
            data_uri=to_data_uri(svg),
            answer=text[::-1].lower(),
            expires_at=time.time() + ttl_seconds,
            norm="lower",
            prompt="请将图中字符逆序输入",
        )


class SortTextCaptcha(TextCaptcha):
    def __init__(self, length: int = 6, width: int = 200, height: int = 70):
        # use only letters and ensure uniqueness by sampling
        self.length = length
        self.width = width
        self.height = height
        self.charset = "ABCDEFGHJKLMNPQRSTUVWXYZ"

    def _random_text(self) -> str:
        # sample without replacement for uniqueness
        return "".join(random.sample(self.charset, self.length))

    def generate(self, ttl_seconds: int = 120) -> CaptchaData:
        text = self._random_text()
        svg = text_captcha_svg(text, self.width, self.height)
        sorted_text = "".join(sorted(text.lower()))
        return CaptchaData(
            id=uuid.uuid4().hex,
            kind="sort",
            svg=svg,
            data_uri=to_data_uri(svg),
            answer=sorted_text,
            expires_at=time.time() + ttl_seconds,
            norm="lower",
            prompt="请按字母表顺序输入这些字符",
        )


class SumDigitsCaptcha(TextCaptcha):
    def __init__(self, length: int = 6, width: int = 200, height: int = 70):
        super().__init__(length, width, height, charset="ABCDEFGHJKLMNPQRSTUVWXYZ0123456789")

    def generate(self, ttl_seconds: int = 120) -> CaptchaData:
        text = super()._random_text()
        svg = text_captcha_svg(text, self.width, self.height)
        s = sum(int(ch) for ch in text if ch.isdigit())
        return CaptchaData(
            id=uuid.uuid4().hex,
            kind="sumdigits",
            svg=svg,
            data_uri=to_data_uri(svg),
            answer=str(s),
            expires_at=time.time() + ttl_seconds,
            norm="int",
            prompt="请输入图中所有数字之和",
        )


class CountShapeCaptcha:
    def __init__(self, width: int = 260, height: int = 120, count: int = 12):
        self.width = width
        self.height = height
        self.count = count

    def generate(self, ttl_seconds: int = 120) -> CaptchaData:
        svg, label, num = shapes_count_svg(self.width, self.height, self.count)
        return CaptchaData(
            id=uuid.uuid4().hex,
            kind="count",
            svg=svg,
            data_uri=to_data_uri(svg),
            answer=str(int(num)),
            expires_at=time.time() + ttl_seconds,
            norm="int",
            prompt=f"请输入图中{label}的数量",
        )


class GridSelectCaptcha:
    def __init__(self, rows: int = 3, cols: int = 3, width: int = 240, height: int = 160):
        self.rows = rows
        self.cols = cols
        self.width = width
        self.height = height

    def generate(self, ttl_seconds: int = 120) -> CaptchaData:
        from .svg import grid_shapes_svg

        svg, rows, cols, label, answers = grid_shapes_svg(self.rows, self.cols, self.width, self.height)
        ans = ",".join(str(int(i)) for i in sorted(answers))
        return CaptchaData(
            id=uuid.uuid4().hex,
            kind="grid",
            svg=svg,
            data_uri=to_data_uri(svg),
            answer=ans,
            expires_at=time.time() + ttl_seconds,
            norm="set",
            prompt=f"请选择所有：{label}",
            meta={"rows": rows, "cols": cols},
        )


class SequenceGridCaptcha:
    def __init__(self, k: int = 5, rows: int = 3, cols: int = 3, width: int = 240, height: int = 160):
        self.k = k
        self.rows = rows
        self.cols = cols
        self.width = width
        self.height = height

    def generate(self, ttl_seconds: int = 120) -> CaptchaData:
        from .svg import sequence_grid_svg

        svg, rows, cols, order = sequence_grid_svg(self.k, self.rows, self.cols, self.width, self.height)
        ans = ",".join(str(int(i)) for i in order)
        return CaptchaData(
            id=uuid.uuid4().hex,
            kind="seq",
            svg=svg,
            data_uri=to_data_uri(svg),
            answer=ans,
            expires_at=time.time() + ttl_seconds,
            norm="list",
            prompt=f"请按顺序点击：1 → {self.k}",
            meta={"rows": rows, "cols": cols, "k": self.k},
        )


class OddOneCaptcha:
    def __init__(self, width: int = 240, height: int = 120, count: int = 7):
        self.width = width
        self.height = height
        self.count = count

    def generate(self, ttl_seconds: int = 120) -> CaptchaData:
        svg, cx, cy, tol = odd_shape_svg(self.width, self.height, self.count)
        ans = f"{int(cx)},{int(cy)}"
        return CaptchaData(
            id=uuid.uuid4().hex,
            kind="odd",
            svg=svg,
            data_uri=to_data_uri(svg),
            answer=ans,
            expires_at=time.time() + ttl_seconds,
            norm=f"point:{tol}",
            prompt="请点击与众不同的那个图形",
            meta={"width": self.width, "height": self.height, "tolerance": tol},
        )


class GridColorSelectCaptcha:
    def __init__(self, rows: int = 3, cols: int = 3, width: int = 240, height: int = 160):
        self.rows = rows
        self.cols = cols
        self.width = width
        self.height = height

    def generate(self, ttl_seconds: int = 120) -> CaptchaData:
        svg, rows, cols, label, answers = grid_color_svg(self.rows, self.cols, self.width, self.height)
        ans = ",".join(str(int(i)) for i in sorted(answers))
        return CaptchaData(
            id=uuid.uuid4().hex,
            kind="gridcolor",
            svg=svg,
            data_uri=to_data_uri(svg),
            answer=ans,
            expires_at=time.time() + ttl_seconds,
            norm="set",
            prompt=f"请选择所有：{label}",
            meta={"rows": rows, "cols": cols},
        )


class CharSequenceCaptcha:
    def __init__(self, text: str = "验证码", rows: int = 2, cols: int = 4, width: int = 280, height: int = 140):
        self.text = text
        self.rows = rows
        self.cols = cols
        self.width = width
        self.height = height

    def generate(self, ttl_seconds: int = 120) -> CaptchaData:
        svg, rows, cols, order, label = sequence_chars_grid_svg(self.text, self.rows, self.cols, self.width, self.height)
        ans = ",".join(str(int(i)) for i in order)
        return CaptchaData(
            id=uuid.uuid4().hex,
            kind="charseq",
            svg=svg,
            data_uri=to_data_uri(svg),
            answer=ans,
            expires_at=time.time() + ttl_seconds,
            norm="list",
            prompt=f"按顺序点击：{label}",
            meta={"rows": rows, "cols": cols, "k": len(order)},
        )


class GridShapeOnlyCaptcha:
    def __init__(self, rows: int = 3, cols: int = 3, width: int = 240, height: int = 160):
        self.rows = rows
        self.cols = cols
        self.width = width
        self.height = height

    def generate(self, ttl_seconds: int = 120) -> CaptchaData:
        svg, rows, cols, label, answers = grid_shape_only_svg(self.rows, self.cols, self.width, self.height)
        ans = ",".join(str(int(i)) for i in sorted(answers))
        return CaptchaData(
            id=uuid.uuid4().hex,
            kind="gridshape",
            svg=svg,
            data_uri=to_data_uri(svg),
            answer=ans,
            expires_at=time.time() + ttl_seconds,
            norm="set",
            prompt=f"请选择所有：{label}",
            meta={"rows": rows, "cols": cols},
        )


class GridVowelSelectCaptcha:
    def __init__(self, rows: int = 3, cols: int = 3, width: int = 240, height: int = 160):
        self.rows = rows
        self.cols = cols
        self.width = width
        self.height = height

    def generate(self, ttl_seconds: int = 120) -> CaptchaData:
        svg, rows, cols, label, answers = grid_vowel_letters_svg(self.rows, self.cols, self.width, self.height)
        ans = ",".join(str(int(i)) for i in sorted(answers))
        return CaptchaData(
            id=uuid.uuid4().hex,
            kind="gridvowel",
            svg=svg,
            data_uri=to_data_uri(svg),
            answer=ans,
            expires_at=time.time() + ttl_seconds,
            norm="set",
            prompt=f"请选择所有：{label}",
            meta={"rows": rows, "cols": cols},
        )


class ArrowSequenceCaptcha:
    def __init__(self, rows: int = 3, cols: int = 3, width: int = 240, height: int = 160):
        self.rows = rows
        self.cols = cols
        self.width = width
        self.height = height

    def generate(self, ttl_seconds: int = 120) -> CaptchaData:
        # Use four arrows sequence
        svg, rows, cols, order, label = sequence_chars_grid_svg("↑→↓←", self.rows, self.cols, self.width, self.height)
        ans = ",".join(str(int(i)) for i in order)
        return CaptchaData(
            id=uuid.uuid4().hex,
            kind="arrowseq",
            svg=svg,
            data_uri=to_data_uri(svg),
            answer=ans,
            expires_at=time.time() + ttl_seconds,
            norm="list",
            prompt="按顺序点击：↑ → ↓ ←",
            meta={"rows": rows, "cols": cols, "k": len(order)},
        )


class AlphaSequenceCaptcha:
    def __init__(self, rows: int = 3, cols: int = 3, width: int = 240, height: int = 160):
        self.rows = rows; self.cols = cols; self.width = width; self.height = height

    def generate(self, ttl_seconds: int = 120) -> CaptchaData:
        svg, rows, cols, order, label = sequence_chars_grid_svg("ABCDE", self.rows, self.cols, self.width, self.height)
        ans = ",".join(str(int(i)) for i in order)
        return CaptchaData(
            id=uuid.uuid4().hex,
            kind="alphaseq",
            svg=svg,
            data_uri=to_data_uri(svg),
            answer=ans,
            expires_at=time.time() + ttl_seconds,
            norm="list",
            prompt="按顺序点击：A → B → C → D → E",
            meta={"rows": rows, "cols": cols, "k": len(order)},
        )


class NumSequenceCaptcha:
    def __init__(self, rows: int = 3, cols: int = 3, width: int = 240, height: int = 160):
        self.rows = rows; self.cols = cols; self.width = width; self.height = height

    def generate(self, ttl_seconds: int = 120) -> CaptchaData:
        svg, rows, cols, order, label = sequence_chars_grid_svg("12345", self.rows, self.cols, self.width, self.height)
        ans = ",".join(str(int(i)) for i in order)
        return CaptchaData(
            id=uuid.uuid4().hex,
            kind="numseq",
            svg=svg,
            data_uri=to_data_uri(svg),
            answer=ans,
            expires_at=time.time() + ttl_seconds,
            norm="list",
            prompt="按顺序点击：1 → 2 → 3 → 4 → 5",
            meta={"rows": rows, "cols": cols, "k": len(order)},
        )


class AnagramCaptcha:
    def __init__(self, words: Optional[list[str]] = None, width: int = 200, height: int = 70):
        self.words = words or ["APPLE", "MANGO", "BERRY", "GRAPE", "LEMON", "PEACH"]
        self.width = width
        self.height = height

    def generate(self, ttl_seconds: int = 120) -> CaptchaData:
        word = random.choice(self.words)
        letters = list(word)
        random.shuffle(letters)
        scrambled = "".join(letters)
        svg = text_captcha_svg(scrambled, self.width, self.height)
        return CaptchaData(
            id=uuid.uuid4().hex,
            kind="anagram",
            svg=svg,
            data_uri=to_data_uri(svg),
            answer=word.lower(),
            expires_at=time.time() + ttl_seconds,
            norm="lower",
            prompt="请将字母复原为正确英文单词",
        )


class CompareCaptcha:
    def __init__(self, width: int = 200, height: int = 70, value_range: Tuple[int, int] = (1, 50)):
        self.width = width
        self.height = height
        self.value_range = value_range

    def generate(self, ttl_seconds: int = 120) -> CaptchaData:
        a = random.randint(*self.value_range)
        b = random.randint(*self.value_range)
        op = '>' if a > b else ('<' if a < b else '=')
        expr = f"{a} ? {b}"
        svg = math_captcha_svg(expr, self.width, self.height)
        return CaptchaData(
            id=uuid.uuid4().hex,
            kind="compare",
            svg=svg,
            data_uri=to_data_uri(svg),
            answer=op,
            expires_at=time.time() + ttl_seconds,
            norm="exact",
            prompt="请在 ? 处填入 >、< 或 =",
        )


class DigitCountCaptcha(TextCaptcha):
    def __init__(self, length: int = 6, width: int = 200, height: int = 70):
        super().__init__(length, width, height, charset="ABCDEFGHJKLMNPQRSTUVWXYZ0123456789")

    def generate(self, ttl_seconds: int = 120) -> CaptchaData:
        text = super()._random_text()
        svg = text_captcha_svg(text, self.width, self.height)
        cnt = sum(1 for ch in text if ch.isdigit())
        return CaptchaData(
            id=uuid.uuid4().hex,
            kind="digitcount",
            svg=svg,
            data_uri=to_data_uri(svg),
            answer=str(cnt),
            expires_at=time.time() + ttl_seconds,
            norm="int",
            prompt="请输入图中数字的个数",
        )


class VowelCountCaptcha(TextCaptcha):
    def __init__(self, length: int = 6, width: int = 200, height: int = 70):
        super().__init__(length, width, height, charset="ABCDEFGHJKLMNPQRSTUVWXYZ0123456789")

    def generate(self, ttl_seconds: int = 120) -> CaptchaData:
        text = super()._random_text()
        svg = text_captcha_svg(text, self.width, self.height)
        vowels = set('AEIOU')
        cnt = sum(1 for ch in text if ch.upper() in vowels)
        return CaptchaData(
            id=uuid.uuid4().hex,
            kind="vowelcount",
            svg=svg,
            data_uri=to_data_uri(svg),
            answer=str(cnt),
            expires_at=time.time() + ttl_seconds,
            norm="int",
            prompt="请输入图中元音字母的个数",
        )


class PalindromeCaptcha(TextCaptcha):
    def __init__(self, length: int = 5, width: int = 200, height: int = 70):
        super().__init__(length, width, height, charset="ABCDEFGHJKLMNPQRSTUVWXYZ0123456789")

    def _make_text(self) -> tuple[str, bool]:
        # 50% palindrome
        is_palin = random.random() < 0.5
        half = ''.join(random.choice(self.charset) for _ in range(self.length // 2))
        mid = random.choice(self.charset)
        if self.length % 2 == 0:
            base = half + (half[::-1] if is_palin else ''.join(random.choice(self.charset) for _ in range(len(half))))
        else:
            base = half + mid + (half[::-1] if is_palin else ''.join(random.choice(self.charset) for _ in range(len(half))))
        return base, is_palin

    def generate(self, ttl_seconds: int = 120) -> CaptchaData:
        text, is_palin = self._make_text()
        svg = text_captcha_svg(text, self.width, self.height)
        return CaptchaData(
            id=uuid.uuid4().hex,
            kind="palin",
            svg=svg,
            data_uri=to_data_uri(svg),
            answer="YES" if is_palin else "NO",
            expires_at=time.time() + ttl_seconds,
            norm="exact",
            prompt="该字符串是否回文？输入 YES 或 NO",
        )


class HexToDecCaptcha(TextCaptcha):
    def __init__(self, width: int = 220, height: int = 70):
        super().__init__(length=4, width=width, height=height, charset="0123456789ABCDEF")

    def generate(self, ttl_seconds: int = 120) -> CaptchaData:
        hex_str = ''.join(random.choice(self.charset) for _ in range(self.length))
        svg = text_captcha_svg(f"HEX {hex_str}", self.width, self.height)
        val = int(hex_str, 16)
        return CaptchaData(
            id=uuid.uuid4().hex,
            kind="hex2dec",
            svg=svg,
            data_uri=to_data_uri(svg),
            answer=str(val),
            expires_at=time.time() + ttl_seconds,
            norm="int",
            prompt="将十六进制数字转为十进制",
        )


# SliderCaptcha removed per request
