# 项目实现说明（多类型 SVG 验证码 + HTTP 服务 + 智能体）

本文详细介绍项目的整体架构、工作原理、接口协议与前端交互，实现细节尽量还原源码逻辑，便于未阅读代码的同学快速理解与二次开发。

- 运行入口：`main.py`
- 核心包：`captcha/`
  - 题面生成：`captcha/generators.py`
  - SVG 渲染：`captcha/svg.py`
  - 内存存储与校验：`captcha/store.py`
  - HTTP 服务与演示页：`captcha/server.py`


## 快速上手

- 环境要求：Python 3.10+
- 启动服务：
  - 命令：`python main.py --host 127.0.0.1 --port 8000 --ttl 120 --debug`
  - 访问演示页：`http://127.0.0.1:8000/`
  - `--ttl`：验证码有效期秒数（默认 120）。
  - `--debug`：在接口响应中回显正确答案（仅用于开发调试，不要在生产开启）。

- 可选（启用智能体/LLM 功能）：
  - 环境变量：
    - `OPENAI_API_KEY`（必填，否则仅使用本地回退策略）
    - `OPENAI_BASE_URL`（可选，默认 `https://api.qingyuntop.top/v1`）
    - `OPENAI_MODEL`（可选，默认 `gpt-4o`）


## 项目结构与职责

- `main.py`
  - 解析命令行参数，调用 `captcha.server.run_server(...)` 启动线程化 HTTP 服务。

- `captcha/svg.py`
  - 负责所有 SVG 题面的绘制与扰动（背景、噪声线/点、扭曲滤镜等）。
  - 重要方法：
    - `text_captcha_svg`/`math_captcha_svg`/`distorted_text_captcha_svg` 等生成字符类、算术类和扭曲字符题面。
    - `shapes_click_svg`/`odd_shape_svg`/`grid_shapes_svg` 等生成点选、找不同、网格多选题面。
    - `sequence_grid_svg`/`sequence_chars_grid_svg` 生成顺序点击题面（数字/指定字符）。
    - 返回值通常包含 SVG 字符串以及答案所需的结构化信息（如目标坐标、网格行列、正确索引集合等）。

- `captcha/generators.py`
  - 将 SVG 题面封装为统一的数据结构 `CaptchaData`，并给出校验规范 `norm` 与提示文案 `prompt`。
  - `CaptchaFactory` 统一管理和按需（懒加载）实例化多种题型生成器。
  - 每次生成都会写入 `CaptchaStore`（答案、过期时间、校验规范、辅助 meta 等）。

- `captcha/store.py`
  - 线程安全的内存存储，负责：
    - `put(id, answer, expires_at, norm, kind, prompt, meta)` 存入一条验证码元数据。
    - `verify(id, user_answer)` 校验，并在成功后一次性删除该条记录。
    - `cleanup()` 周期清理过期条目。
  - 支持多种 `norm` 规范（见“校验规范”一节）。

- `captcha/server.py`
  - 提供 HTTP API 与内置演示页面（纯前端 HTML/JS 注入字符串）。
  - 端点：
    - `GET /`：演示页面（类型选择、坐标/网格叠加工具、智能体面板）。
    - `GET /captcha?type=...`：生成题面，返回 JSON（包含 `svg` 与 `data_uri`）。
    - `POST /verify`：提交答案校验（支持输入类/点选类/网格类/顺序类）。
    - `POST /agent`：识别 Agent（LLM 优先，本地规则兜底），输出“讲解+结论”的 JSON。
    - `POST /exec`：执行 Agent（严格基于识别结果给出可执行的输入或点击/索引）。


## 统一数据结构：CaptchaData

- 字段：
  - `id`：唯一标识。
  - `kind`：题型标识（如 `text`、`grid`、`seq` 等）。
  - `svg`：原始 SVG 文本。
  - `data_uri`：`svg` 的 `data:image/svg+xml;base64,...` 表示，便于直接 `<img src>` 或前端内嵌。
  - `answer`：标准答案（服务端保存，不默认下发）。
  - `expires_at`：过期时间戳。
  - `norm`：校验规范（见下）。
  - `prompt`：界面提示文案（用户可读）。
 - `meta`：辅助信息（如网格行列 `rows/cols`、容差 `tolerance`、画布尺寸等）。


## 题型与生成逻辑

以下题型在 `CaptchaFactory` 中注册，通过 `GET /captcha?type=<kind>` 触发生成：

- 输入类（返回 `norm` 一般为 `lower`/`exact`/`int`）：
  - `text`：随机字符，大小写不敏感。
  - `math`：简单加减乘算式，答案是整数。
  - `distort`：字符 + 扭曲滤镜。
  - `case`：区分大小写的字符题。
  - `anagram`：打乱字母顺序，要求还原为正确英文单词。

- 点选与九宫格：
  - `click`：点选“颜色+形状”的目标图形，答案为像素坐标，带容差（`norm=point:<tol>`）。
  - `odd`：找出与众不同的那一个图形（不同颜色或形状），答案为像素坐标，带容差。
  - `grid`：网格内“颜色+形状”多选，答案是索引集合（`norm=set`）。
  - `gridcolor`：仅按颜色多选（形状任意）。
  - `gridshape`：仅按形状多选（颜色任意）。
  - `gridvowel`：9 宫格随机字母，选择所有元音（A/E/I/O/U），答案为索引集合。

- 顺序点击：
  - `seq`：1→k 的数字顺序点击（答案是有序索引列表，`norm=list`）。
  - `charseq`：按给定字符串（如“验证码”）的字符顺序点击。
  - `arrowseq`：箭头序列（↑→↓→←）。
  - `alphaseq`：字母序列（A→B→C→D→E）。
  - `numseq`：数字序列（1→2→3→4→5）。

- 统计/规则类：
  - `sumdigits`：字符中所有数字求和。
  - `digitcount`：统计数字字符的个数。
  - `vowelcount`：统计元音字母个数。
  - `hex2dec`：十六进制转十进制。
  - `count`：统计网格中满足“颜色+形状”的数量。

说明：源码还包含 `ReverseTextCaptcha`/`SortTextCaptcha` 等示例实现，但未在工厂默认注册，可按需扩展。


## 生成到校验的完整链路

1) 客户端请求 `GET /captcha?type=<kind>`。
   - 服务端创建对应生成器，得到 `CaptchaData`：包含 `id/kind/svg/data_uri/answer/expires_at/norm/prompt/meta`。
   - 将 `answer/expires_at/norm` 等写入内存存储 `CaptchaStore`。
   - 返回 JSON：包含 `id/type/expires_in/svg/data_uri/norm`，以及可选的 `prompt/meta`。若开启 `--debug`，附带 `debug_answer`。

2) 用户提交 `POST /verify`：
   - 请求体 JSON：`{ id, answer, origin? }`。
   - `origin='bl'` 表示答案坐标以“左下角”为原点（像素），服务端会按图像高度换算到“左上角为原点”的坐标再比对（与存储真值一致）。
   - `CaptchaStore.verify(...)` 根据 `norm` 执行相等/集合/序列/容差等校验，通过即“一次性删除该条”防复用；失败返回具体原因。

3) 过期清理：后台线程周期 `cleanup()` 删除过期条目。


## 校验规范（norm）

- `exact`：字符串完全相等。
- `lower`：对用户输入执行 `.lower()` 后比较。
- `int`：两边都转换为整数再比较（格式非法返回 `invalid_format`）。
- `point:<tol>`：二维坐标欧氏距离不超过 `<tol>` 视为正确，格式非法返回 `invalid_format`。
- `approx:<tol>`：允许整数误差在 `<tol>` 内。
- `set`：解析为整数集合，顺序忽略；格式非法返回 `invalid_format`。
- `list`：解析为整数有序列表，顺序必须一致；格式非法返回 `invalid_format`。

返回消息（`POST /verify`）：`ok|mismatch|invalid_format|expired|not_found`。


## 坐标系与索引约定

- 题面 SVG 坐标原点为左上角，y 轴向下；服务端存储的点选题答案亦使用该坐标系。
- 为便于人类标注/LLM 交互，接口允许上传“左下角为原点（像素）”的坐标：在 `POST /verify` 传 `origin='bl'` 即可由服务端换算。
- 网格类题目返回/校验使用“0 基索引”：`row * cols + col`。
- 顺序点击题返回/校验使用“索引的有序列表”（不去重/不排序）。


## HTTP API 说明

1) 生成题面：`GET /captcha?type=<kind>`

- 入参：`type` 可选（缺省 `text`）。
- 响应示例（字段随题型略有差异）：
```json
{
  "id": "dfe1b8...",
  "type": "grid",
  "expires_in": 120,
  "svg": "<svg ...>",
  "data_uri": "data:image/svg+xml;base64,....",
  "norm": "set",
  "prompt": "请选择所有：红色三角形",
  "meta": {"rows":3, "cols":3}
}
```

2) 校验答案：`POST /verify`

- 请求体：
```json
{ "id": "dfe1b8...", "answer": "0,2,5", "origin": "bl" }
```
- 响应：`{ "ok": true, "message": "ok" }` 或 `{ "ok": false, "message": "mismatch" }` 等。

- 点选题坐标举例：
  - 若直接使用 SVG 坐标（左上角原点），传 `answer="120,64"`。
  - 若使用左下角原点坐标，传 `answer="120,56"` 且附 `origin="bl"`，服务端会按图高换算。

3) 识别 Agent：`POST /agent`

- 作用：让 LLM 生成“题型解析+解题步骤+结论”的 JSON；若 LLM 不可用则使用本地兜底规则给出非坐标/非索引的文字指导。
- 请求体（最简）：`{ "id": "<captcha_id>" }`
- 建议追加（利于 LLM 推理/尺寸换算）：`image_width`、`image_height`、或提供 `image_png`（base64）以供回显；若部署在公网，服务端会在响应中补上 `image_url`（通过 `GET /img/<id>.png`）。
- 响应：`{ "summary": "...", "steps": [{"label":"...","detail":"..."}, ...] }`

4) 执行 Agent：`POST /exec`

- 作用：严格“复述/结构化”识别 Agent 的结论，给出可直接执行的动作，不得自行推理图片内容。
- 入参：
```json
{
  "id": "<captcha_id>",
  "intent": "input" | "click",
  "reco": { "summary": "来自 /agent 的结果...", "steps": [...] },
  "image_width": 240, "image_height": 160  // 可选，用于点坐标到索引换算
}
```
- 返回：
  - 输入类：`{"action":"input","value":"<字符串>"}`
  - 点选类（单点）：`{"action":"click","x":120,"y":56}`（左下角为原点像素）
  - 网格多选：`{"action":"grid","indices":[0,2,5]}` 或 `{"action":"grid","points":[{"x":..,"y":..}]}`
  - 顺序点击：`{"action":"seq","indices":[1,4,8,2,0]}` 或 `{"action":"seq","points":[...]}`

- 说明：服务端在执行前会尝试“无模型直提”（从 `reco` 中解析最终输入值/索引/坐标），仅在缺失时才调用 LLM。


## 前端演示页要点

- 点击 `GET /` 返回的演示页即可使用（内嵌 HTML/JS/CSS）。
- 功能：
  - 题型分组与切换、倒计时、提示文案；
  - 坐标轴与网格线叠加工具（辅助定位/点击）；
  - 图像既支持内嵌 `data_uri` 的 SVG，也支持外链 PNG；
  - 智能体面板：展示 `/agent` 的“讲解步骤”，并可触发 `/exec` 自动操作（输入/点击/索引）。
- 坐标换算：前端采集到的点击通常是视口相对的，需要换算到题面像素坐标；服务端在 `/verify` 也支持 `origin='bl'` 的统一换算。


## 扩展与二次开发

- 新增题型：
 1. 在 `captcha/generators.py` 新增生成器类，返回 `CaptchaData`；
 2. 将类注册到 `CaptchaFactory` 的 `_registry`（键为对外 `type` 名）；
 3. （可选）在前端演示页类型清单中加入新类型显示名称。

- 自定义校验：
  - 可在 `CaptchaStore.verify` 中新增自定义 `norm` 分支（例如 `regex:<pat>`、`range:<min,max>` 等），并在对应题型的生成器中设置。

- 自定义渲染：
  - `captcha/svg.py` 中的所有绘制函数均为纯函数，便于替换/扩展（加入更多形状、颜色策略、噪声模型、滤镜等）。


## 安全与注意事项

- 本项目为“教学/演示”级实现：
  - 题面随机性有限，扰动/噪声非针对性防攻击；
  - 存储为内存级别，单机有效，重启即失；
  - `--debug` 仅限本地开发，生产环境务必关闭。
- 生产落地建议：
  - 独立的持久化与限流；
  - 结合风控/行为分析；
  - 更强的渲染与破题对抗策略。


## 常见问题（FAQ）

- 点选题总是 `mismatch`？
  - 请确认传入坐标原点（左上角 vs. 左下角），必要时加上 `origin='bl'`；
  - 题面有容差（`point:<tol>`），但若点击超出阈值也会失败。

- 为什么 /agent 或 /exec 报错？
  - 未配置 `OPENAI_API_KEY` 或网络不可用；
  - 可先使用“本地回退”模式（/agent 会返回思路与非坐标的结论）。

- `gridvowel` 在演示页没按钮？
  - 该类型已实现并在 `typeNames` 中命名，但演示页分组清单里未加入，可自行补充以显示入口。


## 关键实现索引（便于快速定位源码）

- 运行入口：`main.py`
- 题型工厂：`captcha/generators.py`
- 存储与校验：`captcha/store.py`
- HTTP 端点与页面：`captcha/server.py`
- SVG 渲染：`captcha/svg.py`

