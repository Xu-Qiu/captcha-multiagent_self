import base64
import random
from typing import Iterable, Tuple, Optional
import random


def rand_int(a: int, b: int) -> int:
    return random.randint(a, b)


def rand_float(a: float, b: float) -> float:
    return random.random() * (b - a) + a


def rand_color(light: bool = False) -> Tuple[int, int, int]:
    if light:
        # Pastel-ish range
        return (
            rand_int(200, 255),
            rand_int(200, 255),
            rand_int(200, 255),
        )
    return (
        rand_int(0, 180),
        rand_int(0, 180),
        rand_int(0, 180),
    )


def rgb_str(c: Tuple[int, int, int]) -> str:
    return f"rgb({c[0]},{c[1]},{c[2]})"


def escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def svg_header(width: int, height: int) -> str:
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>"
        f"<defs>"
        f"<filter id='noise'><feTurbulence type='fractalNoise' baseFrequency='0.{rand_int(2,7)}' numOctaves='1' stitchTiles='stitch'/></filter>"
        f"</defs>"
    )


def svg_footer() -> str:
    return "</svg>"


def add_background(width: int, height: int) -> str:
    bg = rgb_str(rand_color(light=True))
    return f"<rect width='{width}' height='{height}' fill='{bg}'/>"


def add_noise(width: int, height: int, lines: int = 3, dots: int = 60) -> str:
    parts = []
    for _ in range(lines):
        color = rgb_str(rand_color(light=False))
        sw = rand_float(0.8, 1.8)
        y1 = rand_int(5, height - 5)
        y2 = rand_int(5, height - 5)
        cx = rand_int(0, width)
        cy = rand_int(0, height)
        parts.append(
            f"<path d='M0,{y1} Q {cx},{cy} {width},{y2}' fill='none' stroke='{color}' stroke-opacity='0.35' stroke-width='{sw}'/>"
        )
    for _ in range(dots):
        color = rgb_str(rand_color(light=False))
        r = rand_float(0.5, 1.2)
        x = rand_int(0, width)
        y = rand_int(0, height)
        parts.append(
            f"<circle cx='{x}' cy='{y}' r='{r}' fill='{color}' fill-opacity='0.3'/>"
        )
    return "".join(parts)


def text_captcha_svg(text: str, width: int = 160, height: int = 60) -> str:
    # Draw each character with jitter positioning and rotation
    margin = 8
    step = (width - margin * 2) / max(1, len(text))
    parts = [svg_header(width, height), add_background(width, height), add_noise(width, height)]

    for i, ch in enumerate(text):
        font_size = rand_int(int(height * 0.5), int(height * 0.72))
        x = margin + i * step + step * 0.2 + rand_float(-step * 0.15, step * 0.15)
        y = rand_float(height * 0.65, height * 0.85)
        rotate = rand_int(-25, 25)
        fill = rgb_str(rand_color(light=False))
        parts.append(
            "<g transform='translate(0,0)'>"
            f"<text x='{x:.1f}' y='{y:.1f}' fill='{fill}' font-size='{font_size}' font-family='Verdana,Arial,sans-serif' "
            f"transform='rotate({rotate} {x:.1f} {y:.1f})' letter-spacing='{rand_float(-1.0, 1.2):.1f}'"
            f" stroke='black' stroke-opacity='0.25' stroke-width='0.6'>"
            f"{escape_xml(ch)}"
            "</text>"
            "</g>"
        )

    parts.append(svg_footer())
    return "".join(parts)


def math_captcha_svg(expr: str, width: int = 200, height: int = 70) -> str:
    parts = [svg_header(width, height), add_background(width, height), add_noise(width, height, lines=4, dots=80)]
    font_size = int(height * 0.55)
    x = rand_float(width * 0.1, width * 0.18)
    y = rand_float(height * 0.6, height * 0.8)
    fill = rgb_str(rand_color(light=False))
    parts.append(
        f"<text x='{x:.1f}' y='{y:.1f}' fill='{fill}' font-size='{font_size}' font-family='Verdana,Arial,sans-serif' "
        f" transform='skewX({rand_int(-10,10)}) rotate({rand_int(-3,3)} {x:.1f} {y:.1f})' "
        f" stroke='black' stroke-opacity='0.25' stroke-width='0.8'>"
        f"{escape_xml(expr)}"
        "</text>"
    )
    parts.append(svg_footer())
    return "".join(parts)


def to_data_uri(svg: str) -> str:
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def distorted_text_captcha_svg(text: str, width: int = 180, height: int = 70) -> str:
    margin = 10
    step = (width - margin * 2) / max(1, len(text))
    warp_freq = f"0.{rand_int(2,7)} {rand_float(0.5, 1.2):.2f}"
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<defs>",
        f"<filter id='warp'><feTurbulence type='turbulence' baseFrequency='{warp_freq}' numOctaves='1' result='turb' />",
        f"<feDisplacementMap in2='turb' in='SourceGraphic' scale='{rand_int(8,16)}' xChannelSelector='R' yChannelSelector='G'/></filter>",
        "</defs>",
        add_background(width, height),
        add_noise(width, height, lines=5, dots=90),
        "<g filter='url(#warp)'>",
    ]
    for i, ch in enumerate(text):
        font_size = rand_int(int(height * 0.5), int(height * 0.75))
        x = margin + i * step + step * 0.2 + rand_float(-step * 0.15, step * 0.15)
        y = rand_float(height * 0.62, height * 0.85)
        rotate = rand_int(-25, 25)
        fill = rgb_str(rand_color(light=False))
        parts.append(
            f"<text x='{x:.1f}' y='{y:.1f}' fill='{fill}' font-size='{font_size}' font-family='Verdana,Arial,sans-serif' transform='rotate({rotate} {x:.1f} {y:.1f})' stroke='black' stroke-opacity='0.25' stroke-width='0.6'>{escape_xml(ch)}</text>"
        )
    parts.append("</g>")
    parts.append(svg_footer())
    return "".join(parts)


def _polygon_points(cx: float, cy: float, r: float, sides: int, rotation_deg: float = -90) -> str:
    import math
    pts = []
    rot = math.radians(rotation_deg)
    for i in range(sides):
        ang = rot + 2 * math.pi * i / sides
        x = cx + r * math.cos(ang)
        y = cy + r * math.sin(ang)
        pts.append(f"{x:.1f},{y:.1f}")
    return " ".join(pts)


def shapes_click_svg(width: int = 240, height: int = 120, count: int = 6):
    # Returns (svg, target_label, cx, cy, tolerance)
    # Prepare palette and shape kinds
    colors = [
        ("红色", (220, 50, 47)),
        ("蓝色", (38, 139, 210)),
        ("绿色", (133, 153, 0)),
        ("紫色", (108, 113, 196)),
        ("橙色", (203, 75, 22)),
    ]
    kinds = ["圆形", "方形", "三角形", "五角星"]
    # Layout slots
    cols = 3
    rows = max(2, (count + cols - 1) // cols)
    pad = 18
    cell_w = (width - pad * 2) / cols
    cell_h = (height - pad * 2) / rows
    centers = []
    for r in range(rows):
        for c in range(cols):
            if len(centers) >= count:
                break
            cx = pad + cell_w * (c + 0.5) + rand_float(-cell_w * 0.15, cell_w * 0.15)
            cy = pad + cell_h * (r + 0.5) + rand_float(-cell_h * 0.15, cell_h * 0.15)
            centers.append((cx, cy))
    random.shuffle(centers)
    # 先选出目标的形状与颜色，后续填充其他图形时避免出现同款同色的“第二个目标”
    target_kind = random.choice(kinds)
    target_color_name, target_color = random.choice(colors)
    target_index = random.randrange(count)

    items = []
    target_tuple = None
    for i in range(count):
        if i == target_index:
            k = target_kind
            color_name, color = target_color_name, target_color
        else:
            # 生成非目标图形：允许同形不同色或同色不同形，但避免同形同色
            while True:
                k = random.choice(kinds)
                color_name, color = random.choice(colors)
                if not (k == target_kind and color_name == target_color_name):
                    break
        cx, cy = centers[i]
        size = rand_float(min(cell_w, cell_h) * 0.22, min(cell_w, cell_h) * 0.28)
        item = (k, color_name, color, cx, cy, size)
        items.append(item)
        if i == target_index:
            target_tuple = item

    tk, tcn, _, tcx, tcy, tsize = target_tuple
    target_label = f"{tcn}{tk}"
    # 容差随目标尺寸收紧，仍允许小幅偏差；目标半径≈size，方形半边≈0.9*size
    tol = max(8, tsize * 0.95)

    parts = [svg_header(width, height), add_background(width, height), add_noise(width, height, lines=4, dots=60)]
    for k, color_name, color, cx, cy, size in items:
        fill = rgb_str(color)
        stroke = "#333"
        if k == "圆形":
            parts.append(f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='{size:.1f}' fill='{fill}' stroke='{stroke}' stroke-opacity='0.6' />")
        elif k == "方形":
            s = size * 1.8
            x = cx - s / 2
            y = cy - s / 2
            rot = rand_int(-12, 12)
            parts.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{s:.1f}' height='{s:.1f}' fill='{fill}' stroke='{stroke}' stroke-opacity='0.6' transform='rotate({rot} {cx:.1f} {cy:.1f})' />")
        elif k == "三角形":
            pts = _polygon_points(cx, cy, size * 1.3, 3, rotation_deg=-90)
            parts.append(f"<polygon points='{pts}' fill='{fill}' stroke='{stroke}' stroke-opacity='0.6' />")
        else:  # 五角星 (true star, not pentagon)
            pts = _star_points(cx, cy, outer_r=size * 1.5, inner_r=size * 0.65, points=5, rotation_deg=-90)
            parts.append(f"<polygon points='{pts}' fill='{fill}' stroke='{stroke}' stroke-opacity='0.6' />")

    parts.append(svg_footer())
    return "".join(parts), target_label, tcx, tcy, tol


def _star_points(cx: float, cy: float, outer_r: float, inner_r: Optional[float] = None, points: int = 5, rotation_deg: float = -90) -> str:
    import math
    if inner_r is None:
        inner_r = outer_r * 0.5
    pts = []
    rot = math.radians(rotation_deg)
    step = math.pi / points  # half-angle step for alternating inner/outer
    for i in range(points * 2):
        r = outer_r if i % 2 == 0 else inner_r
        ang = rot + i * step
        x = cx + r * math.cos(ang)
        y = cy + r * math.sin(ang)
        pts.append(f"{x:.1f},{y:.1f}")
    return " ".join(pts)


def shapes_count_svg(width: int = 260, height: int = 120, count: int = 12):
    # Returns (svg, target_label, target_count)
    colors = [
        ("红色", (220, 50, 47)),
        ("蓝色", (38, 139, 210)),
        ("绿色", (133, 153, 0)),
        ("紫色", (108, 113, 196)),
        ("橙色", (203, 75, 22)),
    ]
    kinds = ["圆形", "方形", "三角形"]

    cols = 6
    rows = max(2, (count + cols - 1) // cols)
    pad = 14
    cell_w = (width - pad * 2) / cols
    cell_h = (height - pad * 2) / rows
    centers = []
    for r in range(rows):
        for c in range(cols):
            if len(centers) >= count:
                break
            cx = pad + cell_w * (c + 0.5) + rand_float(-cell_w * 0.18, cell_w * 0.18)
            cy = pad + cell_h * (r + 0.5) + rand_float(-cell_h * 0.18, cell_h * 0.18)
            centers.append((cx, cy))
    parts = [svg_header(width, height), add_background(width, height), add_noise(width, height, lines=3, dots=40)]

    # choose target
    target_kind = random.choice(kinds)
    target_color_name, target_color = random.choice(colors)
    target_label = f"{target_color_name}{target_kind}"
    target_count = 0

    for i in range(count):
        k = random.choice(kinds)
        color_name, color = random.choice(colors)
        cx, cy = centers[i]
        size = rand_float(min(cell_w, cell_h) * 0.2, min(cell_w, cell_h) * 0.28)

        if k == target_kind and color_name == target_color_name:
            target_count += 1
        fill = rgb_str(color)
        stroke = "#333"
        if k == "圆形":
            parts.append(f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='{size:.1f}' fill='{fill}' stroke='{stroke}' stroke-opacity='0.5' />")
        elif k == "方形":
            s = size * 1.6
            x = cx - s / 2
            y = cy - s / 2
            rot = rand_int(-10, 10)
            parts.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{s:.1f}' height='{s:.1f}' fill='{fill}' stroke='{stroke}' stroke-opacity='0.5' transform='rotate({rot} {cx:.1f} {cy:.1f})' />")
        else:
            pts = _polygon_points(cx, cy, size * 1.2, 3, rotation_deg=-90)
            parts.append(f"<polygon points='{pts}' fill='{fill}' stroke='{stroke}' stroke-opacity='0.5' />")

    # Ensure at least one target by overlaying one if zero
    if target_count == 0:
        cx, cy = random.choice(centers)
        size = rand_float(min(cell_w, cell_h) * 0.2, min(cell_w, cell_h) * 0.28)
        fill = rgb_str(target_color)
        stroke = "#333"
        if target_kind == "圆形":
            parts.append(f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='{size:.1f}' fill='{fill}' stroke='{stroke}' stroke-opacity='0.5' />")
        elif target_kind == "方形":
            s = size * 1.6
            x = cx - s / 2
            y = cy - s / 2
            parts.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{s:.1f}' height='{s:.1f}' fill='{fill}' stroke='{stroke}' stroke-opacity='0.5' />")
        else:
            pts = _polygon_points(cx, cy, size * 1.2, 3, rotation_deg=-90)
            parts.append(f"<polygon points='{pts}' fill='{fill}' stroke='{stroke}' stroke-opacity='0.5' />")
        target_count = 1

    parts.append(svg_footer())
    return "".join(parts), target_label, target_count


def grid_shapes_svg(rows: int = 3, cols: int = 3, width: int = 240, height: int = 160):
    # Returns (svg, rows, cols, target_label, answer_indices)
    colors = [
        ("红色", (220, 50, 47)),
        ("蓝色", (38, 139, 210)),
        ("绿色", (133, 153, 0)),
        ("紫色", (108, 113, 196)),
        ("橙色", (203, 75, 22)),
    ]
    kinds = ["圆形", "方形", "三角形"]
    while True:
        parts = [svg_header(width, height), add_background(width, height), add_noise(width, height, lines=3, dots=40)]
        cell_w = width / cols
        cell_h = height / rows
        target_kind = random.choice(kinds)
        target_color_name, target_color = random.choice(colors)
        target_label = f"{target_color_name}{target_kind}"
        answers = []
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                cx = (c + 0.5) * cell_w
                cy = (r + 0.5) * cell_h
                # choose shape & color
                k = random.choice(kinds)
                color_name, color = random.choice(colors)
                size = min(cell_w, cell_h) * 0.25
                if k == target_kind and color_name == target_color_name:
                    answers.append(idx)
                fill = rgb_str(color)
                stroke = "#333"
                if k == "圆形":
                    parts.append(f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='{size:.1f}' fill='{fill}' stroke='{stroke}' stroke-opacity='0.5' />")
                elif k == "方形":
                    s = size * 1.6
                    x = cx - s / 2
                    y = cy - s / 2
                    parts.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{s:.1f}' height='{s:.1f}' fill='{fill}' stroke='{stroke}' stroke-opacity='0.5' />")
                else:
                    pts = _polygon_points(cx, cy, size * 1.3, 3, rotation_deg=-90)
                    parts.append(f"<polygon points='{pts}' fill='{fill}' stroke='{stroke}' stroke-opacity='0.5' />")
        if answers:
            parts.append(svg_footer())
            return "".join(parts), rows, cols, target_label, answers


def sequence_grid_svg(k: int = 5, rows: int = 3, cols: int = 3, width: int = 240, height: int = 160):
    # Place numbers 1..k in random distinct grid cells; user must click in ascending order
    # Returns (svg, rows, cols, order_indices)
    assert k <= rows * cols
    parts = [svg_header(width, height), add_background(width, height), add_noise(width, height, lines=2, dots=30)]
    cell_w = width / cols
    cell_h = height / rows
    indices = random.sample(range(rows * cols), k)
    order = list(indices)  # order indices correspond to ascending numbers
    random.shuffle(order)
    # We'll label visually with numbers, but their order is explicit: we will map number n to the nth element of sorted by number
    # To make it consistent, assign numbers by ascending sequence positions
    for n, idx in enumerate(order, start=1):
        r = idx // cols
        c = idx % cols
        cx = (c + 0.5) * cell_w
        cy = (r + 0.5) * cell_h
        size = min(cell_w, cell_h) * 0.22
        parts.append(f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='{size:.1f}' fill='rgb(255,255,255)' fill-opacity='0.5' stroke='#333' stroke-opacity='0.6' />")
        parts.append(f"<text x='{cx:.1f}' y='{cy + size*0.35:.1f}' text-anchor='middle' font-size='{size*1.4:.1f}' fill='#333' font-family='Verdana,Arial,sans-serif'>{n}</text>")
    parts.append(svg_footer())
    # The correct order is the order list's indices in number ascending, i.e., [order[0], order[1], ...]
    return "".join(parts), rows, cols, order


def odd_shape_svg(width: int = 240, height: int = 120, count: int = 7):
    # Returns (svg, cx, cy, tol)
    colors = [
        ("红色", (220, 50, 47)),
        ("蓝色", (38, 139, 210)),
        ("绿色", (133, 153, 0)),
        ("紫色", (108, 113, 196)),
        ("橙色", (203, 75, 22)),
    ]
    kinds = ["圆形", "方形", "三角形", "五角星"]
    parts = [svg_header(width, height), add_background(width, height), add_noise(width, height, lines=3, dots=50)]
    pad = 18
    cols = 4
    rows = max(2, (count + cols - 1) // cols)
    cell_w = (width - pad * 2) / cols
    cell_h = (height - pad * 2) / rows

    base_kind = random.choice(kinds[:3])
    base_color_name, base_color = random.choice(colors)
    odd_kind = random.choice([k for k in kinds if k != base_kind])
    odd_color_name, odd_color = random.choice([c for c in colors if c[0] != base_color_name])
    odd_index = random.randrange(count)
    target_cx = target_cy = None

    target_size = None
    for i in range(count):
        r = i // cols
        c = i % cols
        cx = pad + cell_w * (c + 0.5) + rand_float(-cell_w * 0.15, cell_w * 0.15)
        cy = pad + cell_h * (r + 0.5) + rand_float(-cell_h * 0.15, cell_h * 0.15)
        size = rand_float(min(cell_w, cell_h) * 0.22, min(cell_w, cell_h) * 0.28)
        if i == odd_index:
            k = odd_kind
            color_name, color = odd_color_name, odd_color
            target_cx, target_cy = cx, cy
            target_size = size
        else:
            k = base_kind
            color_name, color = base_color_name, base_color
        fill = rgb_str(color)
        stroke = "#333"
        if k == "圆形":
            parts.append(f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='{size:.1f}' fill='{fill}' stroke='{stroke}' stroke-opacity='0.6' />")
        elif k == "方形":
            s = size * 1.8
            x = cx - s / 2
            y = cy - s / 2
            parts.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{s:.1f}' height='{s:.1f}' fill='{fill}' stroke='{stroke}' stroke-opacity='0.6' />")
        elif k == "三角形":
            pts = _polygon_points(cx, cy, size * 1.3, 3, rotation_deg=-90)
            parts.append(f"<polygon points='{pts}' fill='{fill}' stroke='{stroke}' stroke-opacity='0.6' />")
        else:
            pts = _star_points(cx, cy, outer_r=size * 1.5, inner_r=size * 0.65, points=5, rotation_deg=-90)
            parts.append(f"<polygon points='{pts}' fill='{fill}' stroke='{stroke}' stroke-opacity='0.6' />")

    parts.append(svg_footer())
    # 收紧容差，取目标尺寸的比例（落点需更靠近形心）
    tol = max(8, (target_size or min(cell_w, cell_h) * 0.22) * 0.95)
    return "".join(parts), target_cx, target_cy, tol


def grid_color_svg(rows: int = 3, cols: int = 3, width: int = 240, height: int = 160):
    # Returns (svg, rows, cols, color_label, answer_indices)
    colors = [
        ("红色", (220, 50, 47)),
        ("蓝色", (38, 139, 210)),
        ("绿色", (133, 153, 0)),
        ("紫色", (108, 113, 196)),
        ("橙色", (203, 75, 22)),
    ]
    kinds = ["圆形", "方形", "三角形", "五角星"]
    while True:
        parts = [svg_header(width, height), add_background(width, height), add_noise(width, height, lines=3, dots=40)]
        cell_w = width / cols
        cell_h = height / rows
        target_color_name, target_color = random.choice(colors)
        target_label = f"{target_color_name}"
        answers = []
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                cx = (c + 0.5) * cell_w
                cy = (r + 0.5) * cell_h
                k = random.choice(kinds)
                color_name, color = random.choice(colors)
                size = min(cell_w, cell_h) * 0.25
                if color_name == target_color_name:
                    answers.append(idx)
                fill = rgb_str(color)
                stroke = "#333"
                if k == "圆形":
                    parts.append(f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='{size:.1f}' fill='{fill}' stroke='{stroke}' stroke-opacity='0.5' />")
                elif k == "方形":
                    s = size * 1.6
                    x = cx - s / 2
                    y = cy - s / 2
                    parts.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{s:.1f}' height='{s:.1f}' fill='{fill}' stroke='{stroke}' stroke-opacity='0.5' />")
                elif k == "三角形":
                    pts = _polygon_points(cx, cy, size * 1.3, 3, rotation_deg=-90)
                    parts.append(f"<polygon points='{pts}' fill='{fill}' stroke='{stroke}' stroke-opacity='0.5' />")
                else:
                    pts = _star_points(cx, cy, outer_r=size * 1.5, inner_r=size * 0.65, points=5, rotation_deg=-90)
                    parts.append(f"<polygon points='{pts}' fill='{fill}' stroke='{stroke}' stroke-opacity='0.5' />")
        if answers:
            parts.append(svg_footer())
            return "".join(parts), rows, cols, target_label, answers


def sequence_chars_grid_svg(chars: str = "验证码", rows: int = 2, cols: int = 4, width: int = 280, height: int = 140):
    # Place given characters randomly in grid; ask user to click in order of the given string
    # Returns (svg, rows, cols, order_indices, label)
    n = len(chars)
    tot = max(n, rows * cols)
    parts = [svg_header(width, height), add_background(width, height), add_noise(width, height, lines=2, dots=30)]
    cell_w = width / cols
    cell_h = height / rows
    indices = random.sample(range(rows * cols), n)
    order = list(indices)
    for ch, idx in zip(chars, order):
        r = idx // cols
        c = idx % cols
        cx = (c + 0.5) * cell_w
        cy = (r + 0.6) * cell_h
        size = min(cell_w, cell_h) * 0.5
        parts.append(f"<text x='{cx:.1f}' y='{cy:.1f}' text-anchor='middle' font-size='{size:.1f}' fill='#333' font-family='Verdana,Arial,sans-serif'>{escape_xml(ch)}</text>")
    parts.append(svg_footer())
    return "".join(parts), rows, cols, order, chars


def grid_shape_only_svg(rows: int = 3, cols: int = 3, width: int = 240, height: int = 160):
    # Returns (svg, rows, cols, shape_label, answer_indices)
    kinds = ["圆形", "方形", "三角形", "五角星"]
    colors = [
        ("红色", (220, 50, 47)),
        ("蓝色", (38, 139, 210)),
        ("绿色", (133, 153, 0)),
        ("紫色", (108, 113, 196)),
        ("橙色", (203, 75, 22)),
    ]
    while True:
        parts = [svg_header(width, height), add_background(width, height), add_noise(width, height, lines=3, dots=40)]
        cell_w = width / cols
        cell_h = height / rows
        target_kind = random.choice(kinds)
        target_label = f"{target_kind}"
        answers = []
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                cx = (c + 0.5) * cell_w
                cy = (r + 0.5) * cell_h
                k = random.choice(kinds)
                color_name, color = random.choice(colors)
                size = min(cell_w, cell_h) * 0.25
                if k == target_kind:
                    answers.append(idx)
                fill = rgb_str(color)
                stroke = "#333"
                if k == "圆形":
                    parts.append(f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='{size:.1f}' fill='{fill}' stroke='{stroke}' stroke-opacity='0.5' />")
                elif k == "方形":
                    s = size * 1.6
                    x = cx - s / 2
                    y = cy - s / 2
                    parts.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{s:.1f}' height='{s:.1f}' fill='{fill}' stroke='{stroke}' stroke-opacity='0.5' />")
                elif k == "三角形":
                    pts = _polygon_points(cx, cy, size * 1.3, 3, rotation_deg=-90)
                    parts.append(f"<polygon points='{pts}' fill='{fill}' stroke='{stroke}' stroke-opacity='0.5' />")
                else:
                    pts = _star_points(cx, cy, outer_r=size * 1.5, inner_r=size * 0.65, points=5, rotation_deg=-90)
                    parts.append(f"<polygon points='{pts}' fill='{fill}' stroke='{stroke}' stroke-opacity='0.5' />")
        if answers:
            parts.append(svg_footer())
            return "".join(parts), rows, cols, target_label, answers


def grid_vowel_letters_svg(rows: int = 3, cols: int = 3, width: int = 240, height: int = 160):
    # Returns (svg, rows, cols, target_label, answer_indices) where answers are vowels A/E/I/O/U
    vowels = set("AEIOU")
    while True:
        parts = [svg_header(width, height), add_background(width, height), add_noise(width, height, lines=2, dots=30)]
        cell_w = width / cols
        cell_h = height / rows
        answers = []
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                ch = chr(random.randint(ord('A'), ord('Z')))
                cx = (c + 0.5) * cell_w
                cy = (r + 0.6) * cell_h
                size = min(cell_w, cell_h) * 0.6
                parts.append(f"<text x='{cx:.1f}' y='{cy:.1f}' text-anchor='middle' font-size='{size:.1f}' fill='#333' font-family='Verdana,Arial,sans-serif'>{escape_xml(ch)}</text>")
                if ch in vowels:
                    answers.append(idx)
        if answers:
            parts.append(svg_footer())
            return "".join(parts), rows, cols, "元音字母", answers
