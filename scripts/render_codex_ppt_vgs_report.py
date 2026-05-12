#!/usr/bin/env python3
"""Render a clean professional full-image PPT deck for the VGS report."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "outputs" / "codex_ppt_vgs_clean_20260509"
IMAGE_DIR = PROJECT / "origin_image"

W, H = 1920, 1080

FONT_REG = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_MED = "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

BLUE = "#2563EB"
BLUE_DARK = "#1E3A8A"
TEAL = "#0F766E"
CYAN = "#0891B2"
AMBER = "#F59E0B"
SLATE = "#334155"
TEXT = "#0F172A"
MUTED = "#64748B"
BORDER = "#D9E2EF"
BG = "#F6F9FF"
CARD = "#FFFFFF"
SOFT = "#EFF6FF"
TEAL_SOFT = "#ECFDF5"
AMBER_SOFT = "#FFFBEB"
RED_SOFT = "#FEF2F2"
RED = "#DC2626"
GREEN = "#16A34A"


def font(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    path = {"regular": FONT_REG, "medium": FONT_MED, "bold": FONT_BOLD}[weight]
    return ImageFont.truetype(path, size)


F = {
    "kicker": font(25, "medium"),
    "title": font(64, "bold"),
    "subtitle": font(34, "medium"),
    "h1": font(49, "bold"),
    "h2": font(34, "bold"),
    "h3": font(27, "bold"),
    "body": font(25, "regular"),
    "body_med": font(25, "medium"),
    "small": font(21, "regular"),
    "small_med": font(21, "medium"),
    "mini": font(17, "regular"),
    "metric": font(58, "bold"),
    "metric_sm": font(42, "bold"),
}


def new_canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, W, H), fill=BG)
    # Subtle technical background lines.
    for x in range(80, W, 160):
        draw.line((x, 0, x, H), fill="#EEF4FC", width=1)
    for y in range(90, H, 150):
        draw.line((0, y, W, y), fill="#EEF4FC", width=1)
    return img, draw


def text_bbox(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fnt) -> tuple[int, int, int, int]:
    return draw.textbbox(xy, text, font=fnt)


def tw(draw: ImageDraw.ImageDraw, text: str, fnt) -> int:
    box = text_bbox(draw, (0, 0), text, fnt)
    return box[2] - box[0]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, fnt, max_width: int) -> list[str]:
    def tokenize(s: str) -> list[str]:
        out: list[str] = []
        cur = ""
        for ch in s:
            is_ascii_word = ch.isascii() and (ch.isalnum() or ch in "-_+/.,:()=%")
            if ch == " ":
                if cur:
                    out.append(cur)
                    cur = ""
                out.append(ch)
            elif is_ascii_word:
                cur += ch
            else:
                if cur:
                    out.append(cur)
                    cur = ""
                out.append(ch)
        if cur:
            out.append(cur)
        return out

    lines: list[str] = []
    for raw in text.split("\n"):
        if not raw:
            lines.append("")
            continue
        cur = ""
        for token in tokenize(raw):
            trial = cur + token
            if cur and tw(draw, trial.rstrip(), fnt) > max_width:
                lines.append(cur.rstrip())
                cur = "" if token == " " else token.lstrip()
                if tw(draw, cur, fnt) > max_width:
                    broken = ""
                    for ch in cur:
                        trial_ch = broken + ch
                        if broken and tw(draw, trial_ch, fnt) > max_width:
                            lines.append(broken)
                            broken = ch
                        else:
                            broken = trial_ch
                    cur = broken
            else:
                cur = trial
        if cur:
            lines.append(cur.rstrip())
    return lines


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fnt,
    fill: str = TEXT,
    max_width: int | None = None,
    line_spacing: int = 8,
    anchor: str | None = None,
) -> int:
    x, y = xy
    if max_width is None:
        draw.text((x, y), text, font=fnt, fill=fill, anchor=anchor)
        box = text_bbox(draw, (x, y), text, fnt)
        return box[3] - box[1]
    total = 0
    for line in wrap_text(draw, text, fnt, max_width):
        draw.text((x, y + total), line, font=fnt, fill=fill)
        box = text_bbox(draw, (x, y + total), line, fnt)
        total += (box[3] - box[1]) + line_spacing
    return total


def rounded(draw: ImageDraw.ImageDraw, box, fill, outline=BORDER, width=2, radius=8):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def header(draw: ImageDraw.ImageDraw, section: str, title: str, subtitle: str | None = None):
    draw_text(draw, (92, 58), section, F["kicker"], fill=BLUE)
    draw.rectangle((92, 93, 172, 98), fill=TEAL)
    title_h = draw_text(draw, (92, 116), title, F["h1"], fill=TEXT, max_width=1420, line_spacing=4)
    if subtitle:
        draw_text(draw, (92, 116 + title_h + 8), subtitle, F["small"], fill=MUTED, max_width=1420)


def bullet_list(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    items: Sequence[str],
    width: int,
    fnt=None,
    fill: str = TEXT,
    gap: int = 17,
    dot: str = BLUE,
) -> int:
    fnt = fnt or F["body"]
    cy = y
    for item in items:
        draw.ellipse((x, cy + 9, x + 12, cy + 21), fill=dot)
        used = draw_text(draw, (x + 28, cy), item, fnt, fill=fill, max_width=width - 28, line_spacing=5)
        cy += used + gap
    return cy - y


def card_title(draw, box, title, subtitle=None, accent=BLUE, fill=CARD):
    rounded(draw, box, fill=fill, outline=BORDER, width=2, radius=8)
    x1, y1, x2, _ = box
    draw.rectangle((x1, y1, x1 + 7, y1 + 88), fill=accent)
    draw_text(draw, (x1 + 28, y1 + 24), title, F["h3"], fill=TEXT, max_width=x2 - x1 - 56)
    if subtitle:
        draw_text(draw, (x1 + 28, y1 + 62), subtitle, F["small"], fill=MUTED, max_width=x2 - x1 - 56)


def arrow(draw, start, end, color=BLUE, width=5):
    x1, y1 = start
    x2, y2 = end
    draw.line((x1, y1, x2, y2), fill=color, width=width)
    angle = math.atan2(y2 - y1, x2 - x1)
    size = 16
    pts = [
        (x2, y2),
        (x2 - size * math.cos(angle - 0.48), y2 - size * math.sin(angle - 0.48)),
        (x2 - size * math.cos(angle + 0.48), y2 - size * math.sin(angle + 0.48)),
    ]
    draw.polygon(pts, fill=color)


def metric_card(draw, box, metric, label, color=BLUE, sub=None, fill=CARD):
    rounded(draw, box, fill=fill, outline=BORDER, width=2, radius=8)
    x1, y1, x2, _ = box
    draw_text(draw, (x1 + 30, y1 + 28), metric, F["metric"], fill=color, max_width=x2 - x1 - 60)
    label_h = draw_text(draw, (x1 + 32, y1 + 102), label, F["body_med"], fill=TEXT, max_width=x2 - x1 - 64)
    if sub:
        draw_text(draw, (x1 + 32, y1 + 102 + label_h + 10), sub, F["small"], fill=MUTED, max_width=x2 - x1 - 64)


def bar_chart(draw, box, labels, values, max_value=None, colors=None, title=None, suffix="", baseline=None):
    x1, y1, x2, y2 = box
    if title:
        draw_text(draw, (x1, y1 - 52), title, F["h3"], fill=TEXT)
    max_value = max_value or max(values) * 1.12
    colors = colors or [BLUE] * len(values)
    chart_h = y2 - y1 - 76
    n = len(values)
    gap = 24
    bar_w = max(30, int((x2 - x1 - gap * (n - 1)) / n))
    axis_y = y1 + chart_h
    draw.line((x1, axis_y, x2, axis_y), fill=BORDER, width=2)
    if baseline is not None:
        by = axis_y - chart_h * baseline / max_value
        draw.line((x1, by, x2, by), fill="#CBD5E1", width=2)
        draw_text(draw, (x2 - 94, int(by) - 28), f"{baseline:.2f}", F["mini"], fill=MUTED)
    for i, (lab, val) in enumerate(zip(labels, values)):
        bx1 = x1 + i * (bar_w + gap)
        bh = int(chart_h * val / max_value)
        by1 = axis_y - bh
        rounded(draw, (bx1, by1, bx1 + bar_w, axis_y), fill=colors[i], outline=colors[i], width=1, radius=6)
        draw_text(draw, (bx1, by1 - 34), f"{val:.3f}{suffix}", F["small_med"], fill=colors[i])
        draw_text(draw, (bx1, axis_y + 18), lab, F["mini"], fill=MUTED, max_width=bar_w + 20)


def line_chart(draw, box, xs, series, title=None, y_min=None, y_max=None, suffix=""):
    x1, y1, x2, y2 = box
    if title:
        draw_text(draw, (x1, y1 - 48), title, F["h3"], fill=TEXT)
    all_vals = [v for _, vals, _ in series for v in vals]
    y_min = min(all_vals) if y_min is None else y_min
    y_max = max(all_vals) if y_max is None else y_max
    pad = 50
    cx1, cy1, cx2, cy2 = x1 + pad, y1 + 20, x2 - 32, y2 - 72
    draw.line((cx1, cy2, cx2, cy2), fill=BORDER, width=2)
    draw.line((cx1, cy1, cx1, cy2), fill=BORDER, width=2)
    for frac in [0, 0.5, 1]:
        yy = cy2 - (cy2 - cy1) * frac
        draw.line((cx1, yy, cx2, yy), fill="#E8EEF6", width=1)
        val = y_min + (y_max - y_min) * frac
        draw_text(draw, (x1, int(yy) - 12), f"{val:.2f}{suffix}", F["mini"], fill=MUTED)
    min_x, max_x = min(xs), max(xs)
    for x in xs:
        xx = cx1 + (cx2 - cx1) * (x - min_x) / (max_x - min_x)
        draw_text(draw, (int(xx) - 15, cy2 + 22), str(x), F["mini"], fill=MUTED)
    for label, vals, color in series:
        pts = []
        for x, y in zip(xs, vals):
            xx = cx1 + (cx2 - cx1) * (x - min_x) / (max_x - min_x)
            yy = cy2 - (cy2 - cy1) * (y - y_min) / (y_max - y_min)
            pts.append((xx, yy))
        draw.line(pts, fill=color, width=5, joint="curve")
        for xx, yy in pts:
            draw.ellipse((xx - 7, yy - 7, xx + 7, yy + 7), fill=color)
        draw_text(draw, (int(pts[-1][0]) + 14, int(pts[-1][1]) - 18), label, F["mini"], fill=color)


def table(draw, box, headers, rows, col_widths=None, header_fill=SOFT, font_size="small"):
    fnt = F[font_size]
    x1, y1, x2, y2 = box
    rounded(draw, box, fill=CARD, outline=BORDER, width=2, radius=8)
    n = len(headers)
    if col_widths is None:
        col_widths = [(x2 - x1) / n] * n
    col_x = [x1]
    for w_ in col_widths:
        col_x.append(col_x[-1] + w_)
    row_h = (y2 - y1) / (len(rows) + 1)
    draw.rectangle((x1, y1, x2, y1 + row_h), fill=header_fill)
    for i, h in enumerate(headers):
        draw_text(draw, (int(col_x[i] + 16), int(y1 + 18)), h, F["small_med"], fill=SLATE, max_width=int(col_widths[i] - 28))
    for r, row in enumerate(rows):
        yy = y1 + row_h * (r + 1)
        draw.line((x1, yy, x2, yy), fill="#E6EDF5", width=1)
        for c, val in enumerate(row):
            color = TEXT
            if isinstance(val, tuple):
                val, color = val
            draw_text(draw, (int(col_x[c] + 16), int(yy + 16)), str(val), fnt, fill=color, max_width=int(col_widths[c] - 28), line_spacing=3)
    for x in col_x[1:-1]:
        draw.line((x, y1, x, y2), fill="#E6EDF5", width=1)


def save_slide(idx: int, img: Image.Image):
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    img.save(IMAGE_DIR / f"slide_{idx:02d}.png", "PNG", optimize=True)


def slide1():
    img, draw = new_canvas()
    draw.rectangle((0, 0, W, H), fill="#F8FBFF")
    draw.ellipse((1280, 90, 2100, 850), fill="#EAF4FF", outline=None)
    draw.ellipse((1420, 210, 1980, 980), fill="#E6FFFA", outline=None)
    draw_text(draw, (92, 98), "VGS 实验汇报", F["kicker"], fill=TEAL)
    draw_text(draw, (92, 184), "基于盲参考差分的 VLM\n幻觉校正几何分析", F["title"], fill=TEXT, max_width=980, line_spacing=14)
    draw_text(draw, (96, 405), "Blind-Reference Differencing\nReveals Layered Correction Geometry", F["subtitle"], fill=SLATE, max_width=950)
    rounded(draw, (96, 752, 1120, 928), fill=CARD, outline="#D8E6F8", width=2, radius=8)
    draw_text(draw, (132, 786), "中心论点", F["h3"], fill=BLUE)
    draw_text(draw, (132, 835), "最大的 visual correction directions 并不是 hallucination decision directions；关键信号更多藏在 residual/tail evidence-sensitive coordinates 中。", F["body_med"], fill=TEXT, max_width=920, line_spacing=6)

    # Right-side layered geometry diagram.
    for i, (dx, dy, color) in enumerate([(0, 0, "#DBEAFE"), (60, 70, "#CCFBF1"), (120, 140, "#FEF3C7")]):
        pts = [(1250 + dx, 260 + dy), (1660 + dx, 170 + dy), (1810 + dx, 450 + dy), (1395 + dx, 555 + dy)]
        draw.polygon(pts, fill=color, outline="#B6C7D9")
        draw_text(draw, (1300 + dx, 310 + dy), ["z_img", "z_blind", "d = z_blind - z_img"][i], F["h3"], fill=[BLUE, TEAL, AMBER][i])
    arrow(draw, (1380, 725), (1640, 625), color=BLUE, width=7)
    draw_text(draw, (1284, 760), "hidden-state\ncorrection space", F["small_med"], fill=SLATE, max_width=360)
    return img


def slide2():
    img, draw = new_canvas()
    header(draw, "研究问题", "视觉证据如何改变 VLM 的内部表示？", "从输出 token 转向 hidden-state correction geometry")
    card_title(draw, (96, 292, 890, 690), "传统疑问", "模型为什么在不存在物体时仍回答 Yes？", accent=AMBER, fill=AMBER_SOFT)
    bullet_list(draw, 136, 410, [
        "只看最终回答，难以定位内部机制",
        "置信度和 margin 能预警，但解释力有限",
        "需要比较有图和无图时的内部状态变化",
    ], 690, dot=AMBER)
    card_title(draw, (1030, 292, 1824, 690), "本工作切入点", "构造 blind-reference difference", accent=TEAL, fill=TEAL_SOFT)
    draw_text(draw, (1080, 418), "z_img   = hidden_state(image + question)", F["body_med"], fill=SLATE)
    draw_text(draw, (1080, 480), "z_blind = hidden_state(question only)", F["body_med"], fill=SLATE)
    rounded(draw, (1074, 556, 1768, 622), fill="#FFFFFF", outline="#A7F3D0", radius=8)
    draw_text(draw, (1108, 570), "d = z_blind - z_img", F["h2"], fill=TEAL)
    rounded(draw, (240, 792, 1680, 934), fill=CARD, outline="#CFE1F6", width=2, radius=8)
    draw_text(draw, (300, 826), "核心问题", F["h3"], fill=BLUE)
    draw_text(draw, (500, 822), "difference space 中，哪些方向与 hallucination 有关？", F["h2"], fill=TEXT, max_width=1030)
    return img


def slide3():
    img, draw = new_canvas()
    header(draw, "任务边界", "先区分三个不同问题", "避免把机制分析、部署检测和因果修正混在一起")
    xs = [96, 672, 1248]
    cards = [
        ("FP vs TN", "ground-truth=No 中，错误 Yes vs 正确 No", "机制分析显微镜", "不能直接等同部署检测"),
        ("FP vs TP", "predicted-Yes 中，错误 Yes vs 正确 Yes", "部署风险识别", "不等价于 FP/TN"),
        ("FP rescue", "通过干预把 FP 改成 No", "因果修正探索", "目前不能证明可靠 mitigation"),
    ]
    fills = [SOFT, TEAL_SOFT, RED_SOFT]
    accents = [BLUE, TEAL, RED]
    for x, (title, definition, role, caveat), fill, accent in zip(xs, cards, fills, accents):
        rounded(draw, (x, 300, x + 520, 850), fill=fill, outline=BORDER, width=2, radius=8)
        draw.rectangle((x, 300, x + 520, 312), fill=accent)
        draw_text(draw, (x + 34, 344), title, F["h2"], fill=accent)
        draw_text(draw, (x + 34, 428), "定义", F["h3"], fill=TEXT)
        draw_text(draw, (x + 34, 470), definition, F["body"], fill=SLATE, max_width=450)
        draw_text(draw, (x + 34, 595), "作用", F["h3"], fill=TEXT)
        draw_text(draw, (x + 34, 637), role, F["body_med"], fill=TEXT, max_width=450)
        draw_text(draw, (x + 34, 735), "不能说明", F["h3"], fill=TEXT)
        draw_text(draw, (x + 34, 777), caveat, F["small_med"], fill=RED if accent == RED else MUTED, max_width=450)
    return img


def slide4():
    img, draw = new_canvas()
    header(draw, "方法总览", "Blind-Reference Differencing", "同一样本两次前向，比较 hidden state correction")
    # Two input lanes.
    lanes = [
        (210, 322, "Image + Question", "z_img at layer L", BLUE),
        (210, 570, "Question only", "z_blind at layer L", TEAL),
    ]
    for x, y, t, z, color in lanes:
        rounded(draw, (x, y, x + 410, y + 96), fill=CARD, outline=color, width=3, radius=8)
        draw_text(draw, (x + 32, y + 28), t, F["body_med"], fill=TEXT)
        arrow(draw, (x + 410, y + 48), (x + 610, y + 48), color=color)
        rounded(draw, (x + 640, y, x + 1020, y + 96), fill="#F8FAFC", outline=color, width=3, radius=8)
        draw_text(draw, (x + 672, y + 28), z, F["body_med"], fill=color)
    arrow(draw, (1230, 370), (1380, 492), color=SLATE)
    arrow(draw, (1230, 618), (1380, 542), color=SLATE)
    rounded(draw, (1400, 444, 1756, 596), fill=CARD, outline="#A5B4FC", width=3, radius=8)
    draw_text(draw, (1444, 486), "d = z_blind - z_img", F["h2"], fill=BLUE)
    modules = [("SVD", 330), ("tail bands", 600), ("intervention", 870), ("selective gate", 1140)]
    for label, x in modules:
        rounded(draw, (x, 760, x + 230, 838), fill=CARD, outline=BORDER, width=2, radius=8)
        draw_text(draw, (x + 26, 783), label, F["small_med"], fill=TEXT)
        arrow(draw, (1560, 596), (x + 115, 760), color="#94A3B8", width=3)
    rounded(draw, (140, 906, 1780, 980), fill="#FFFFFF", outline="#DCEBFA", radius=8)
    draw_text(draw, (176, 926), "主设置：L16 / L20 / L24 / L28 / L32；readout: last prompt token 与 prompt mean；主任务：POPE FP vs TN；外部：AMBER；跨模型：LLaVA / Qwen / InternVL", F["small_med"], fill=SLATE, max_width=1580)
    return img


def slide5():
    img, draw = new_canvas()
    header(draw, "Finding 1", "Correction space 有强低秩 backbone", "Top-4 SVD directions 解释大部分 blind-image difference 方差")
    metric_card(draw, (108, 292, 510, 510), "88.6%", "L8 top-4 explained variance", BLUE)
    metric_card(draw, (555, 292, 957, 510), "87.7%", "L24 top-4 explained variance", TEAL)
    metric_card(draw, (1002, 292, 1404, 510), "72.7%", "L32 top-4 explained variance", AMBER)
    rounded(draw, (98, 604, 818, 904), fill=CARD, outline=BORDER, width=2, radius=8)
    draw_text(draw, (138, 642), "解释", F["h3"], fill=TEXT)
    bullet_list(draw, 142, 700, [
        "视觉条件引起的 hidden-state 变化高度集中",
        "Split-half 稳定性在小 K 时最强",
        "但低秩 backbone 本身还不能说明 paired visual grounding",
    ], 620, fnt=F["body"], dot=BLUE)
    rounded(draw, (900, 604, 1818, 904), fill=CARD, outline=BORDER, width=2, radius=8)
    bar_chart(draw, (978, 710, 1748, 870), ["L8", "L24", "L32"], [88.6, 87.7, 72.7], max_value=100, colors=[BLUE, TEAL, AMBER], title="Top-4 explained variance", suffix="%")
    return img


def slide6():
    img, draw = new_canvas()
    header(draw, "Finding 2", "Variance is not discrimination", "主方差方向不是 hallucination decision direction")
    rounded(draw, (96, 274, 832, 912), fill=CARD, outline=BORDER, width=2, radius=8)
    bar_chart(draw, (170, 462, 760, 835), ["L24\ntop-4", "L20\ntop-256", "L24\nfull"], [0.471, 0.677, 0.721], max_value=0.82, colors=[RED, TEAL, BLUE], title="Seed-robust FP/TN AUROC", baseline=0.50)
    draw_text(draw, (146, 334), "高方差不等于高判别", F["h2"], fill=TEXT)
    draw_text(draw, (146, 386), "Top-4 解释 >80% 方差，但 L24 top-4 AUROC 仅 0.471。", F["small_med"], fill=MUTED, max_width=610)
    rounded(draw, (900, 274, 1818, 912), fill=CARD, outline=BORDER, width=2, radius=8)
    line_chart(draw, (966, 430, 1744, 790), [4, 64, 128, 256], [
        ("L20", [0.5570, 0.6338, 0.6846, 0.6948], BLUE),
        ("L24", [0.4637, 0.6192, 0.6539, 0.6496], TEAL),
        ("L32", [0.5005, 0.5652, 0.5900, 0.6185], AMBER),
    ], title="Top-K AUROC 随 K 增长", y_min=0.44, y_max=0.72)
    table(draw, (1000, 805, 1710, 888), ["Bootstrap delta", "95% CI"], [
        [("top-256 > top-4: +0.193", BLUE), "0.155 - 0.227"],
        [("full > top-256: +0.053", TEAL), "0.032 - 0.075"],
    ], col_widths=[430, 280], font_size="mini")
    return img


def slide7():
    img, draw = new_canvas()
    header(draw, "机制解释", "Top backbone 更像“有没有图”", "不是证据是否支持回答；SVD 抓到大变化，FP/TN 标签依赖细粒度 evidence signal")
    rounded(draw, (128, 302, 1792, 436), fill=CARD, outline="#BFD7F5", width=2, radius=8)
    draw_text(draw, (196, 342), "d = a · v_image + b · v_evidence + epsilon", F["h2"], fill=BLUE)
    cols = [
        (128, 548, 610, "v_image", "图像条件带来的大变化", "方差大，SVD 容易优先捕获", BLUE, SOFT),
        (720, 548, 1202, "v_evidence", "证据是否支持回答", "方差较小，但与 FP/TN 更相关", TEAL, TEAL_SOFT),
        (1312, 548, 1794, "epsilon / tail", "残差校正坐标", "判别信号分散在 tail / full difference", AMBER, AMBER_SOFT),
    ]
    for x, y, x2, title, body, sub, color, fill in cols:
        rounded(draw, (x, y, x2, 846), fill=fill, outline=BORDER, width=2, radius=8)
        draw_text(draw, (x + 34, y + 34), title, F["h2"], fill=color)
        draw_text(draw, (x + 34, y + 104), body, F["body_med"], fill=TEXT, max_width=x2 - x - 68)
        draw_text(draw, (x + 34, y + 192), sub, F["small_med"], fill=MUTED, max_width=x2 - x - 68)
    arrow(draw, (610, 700), (720, 700), color="#94A3B8", width=4)
    arrow(draw, (1202, 700), (1312, 700), color="#94A3B8", width=4)
    rounded(draw, (300, 900, 1620, 990), fill="#FFFFFF", outline="#DDEAF8", radius=8)
    draw_text(draw, (344, 920), "解释 Stage B：Top-backbone 分离 image-conditioned vs blind；tail / supervised view 更能区分 matched、random、adversarial evidence。", F["small_med"], fill=SLATE, max_width=1220)
    return img


def slide8():
    img, draw = new_canvas()
    header(draw, "Finding 3", "Residual / Tail 更接近 evidence signal", "matched vs mismatch 的差异主要出现在 tail band 与 supervised decision view")
    rounded(draw, (96, 304, 960, 910), fill=CARD, outline=BORDER, width=2, radius=8)
    draw_text(draw, (150, 354), "matched-specific tail 增强", F["h2"], fill=TEXT)
    draw_text(draw, (150, 410), "TN 在 matched evidence 下的 tail 增强更明显；L32 matched-random: FP -10.2 vs TN +44.2。", F["small_med"], fill=MUTED, max_width=720)
    draw_text(draw, (150, 492), "Tail 257-1024 condition delta", F["small_med"], fill=SLATE)
    bar_chart(draw, (168, 548, 880, 830), ["L20\nM-R", "L24\nM-R", "L32\nM-R", "L20\nM-A", "L24\nM-A", "L32\nM-A"], [5.7, 14.2, 17.0, 11.8, 25.7, 39.1], max_value=45, colors=[BLUE, BLUE, BLUE, TEAL, TEAL, TEAL], title=None)
    rounded(draw, (1030, 304, 1818, 910), fill=CARD, outline=BORDER, width=2, radius=8)
    obs = [
        ("Top-backbone energy", "区分 image-conditioned vs blind", BLUE),
        ("Tail band 257-1024", "对 matched vs mismatch 更敏感", TEAL),
        ("Supervised decision score", "只在 matched evidence 下明显区分 FP/TN", AMBER),
        ("结论", "幻觉关联的是 matched evidence 下扭曲的条件特定 correction geometry", SLATE),
    ]
    y = 360
    for t, b, color in obs:
        rounded(draw, (1084, y, 1768, y + 104), fill="#F8FAFC", outline="#E2E8F0", radius=8)
        draw_text(draw, (1118, y + 20), t, F["small_med"], fill=color)
        draw_text(draw, (1118, y + 55), b, F["small"], fill=TEXT, max_width=600)
        y += 126
    return img


def slide9():
    img, draw = new_canvas()
    header(draw, "Finding 4", "Tail ablation 会因果性破坏正确 negative decision", "L24 tail slice 对 TN 正确拒绝具有必要性")
    rounded(draw, (96, 290, 950, 910), fill=CARD, outline=BORDER, width=2, radius=8)
    bar_chart(draw, (175, 508, 850, 830), ["α4", "α5", "α6", "α7", "α8"], [0.000, 0.125, 0.562, 0.938, 1.000], max_value=1.05, colors=[BLUE, BLUE, TEAL, AMBER, RED], title="L24 tail ablation: Yes rate")
    draw_text(draw, (146, 350), "剂量依赖翻转", F["h2"], fill=TEXT)
    draw_text(draw, (146, 402), "alpha 从 4 到 8 时，TN 的 Yes rate 从 0 增至 1.0。", F["small_med"], fill=MUTED, max_width=700)
    rounded(draw, (1030, 290, 1818, 910), fill=CARD, outline=BORDER, width=2, radius=8)
    line_chart(draw, (1100, 482, 1728, 790), [4, 5, 6, 7, 8], [
        ("median margin", [-0.750, -0.328, 0.016, 0.391, 0.934], TEAL),
    ], title="Median logit(No)-logit(Yes) margin", y_min=-0.9, y_max=1.05)
    rounded(draw, (1100, 812, 1728, 872), fill=TEAL_SOFT, outline="#A7F3D0", radius=8)
    draw_text(draw, (1130, 827), "Norm-matched random tail control 在 last-token 设置下保持 0 Yes rate。", F["small_med"], fill=TEAL, max_width=560)
    return img


def slide10():
    img, draw = new_canvas()
    header(draw, "负结果", "FP rescue 很弱，不能包装成 reliable mitigation", "Tail coordinates 对 TN necessary，但不足以稳健修复 FP")
    metric_card(draw, (120, 300, 548, 560), "3/64", "FP rescue 翻转", RED, "decoded；均为 borderline", fill=RED_SOFT)
    metric_card(draw, (612, 300, 1040, 560), "2/32", "Stage M rescue", AMBER, "baseline margin 0.0156 / 0.0313", fill=AMBER_SOFT)
    metric_card(draw, (1104, 300, 1532, 560), "30/32", "margin 改善但未翻转", TEAL, "logit 改善不等于 decoded 翻转", fill=TEAL_SOFT)
    rounded(draw, (160, 662, 1760, 860), fill=CARD, outline=BORDER, width=2, radius=8)
    draw_text(draw, (210, 706), "正确表述", F["h3"], fill=BLUE)
    draw_text(draw, (410, 700), "Tail coordinates are necessary for correct negative decisions, but not sufficient for robust FP rescue.", F["h2"], fill=TEXT, max_width=1190)
    rounded(draw, (420, 902, 1500, 974), fill="#FFFFFF", outline="#FAD7D7", radius=8)
    draw_text(draw, (460, 920), "汇报中主动讲这个负结果，可以把主张收敛为机制分析与选择性路由，而不是“修正幻觉”。", F["small_med"], fill=SLATE, max_width=1000)
    return img


def slide11():
    img, draw = new_canvas()
    header(draw, "部署视角", "Selective warning / routing：geometry 是互补信号", "Fixed-trigger 对照显示：low-margin 很强，geometry 在低预算和 TP-preserving routing 上提供增量价值")
    rounded(draw, (88, 272, 1840, 612), fill=CARD, outline=BORDER, width=2, radius=8)
    table(draw, (126, 338, 1800, 590), ["Target", "Gate", "FP recall", "TP damage", "Warning precision", "ICD FP reduction", "TP preserved", "Acc delta"], [
        ["10%", "Margin-only", "0.396", "0.072", "0.350", "0.245", "0.958", "-0.007"],
        ["10%", ("Margin + PLS/full/tail", BLUE), ("0.434", BLUE), ("0.068", BLUE), ("0.383", BLUE), "0.208-0.264", "0.982-0.993", "+0.001~+0.005"],
        ["20%", "Margin-only", "0.660", "0.157", "0.292", "0.321", "0.930", "-0.016"],
        ["20%", ("Margin + PLS/full", TEAL), ("0.679", TEAL), ("0.155", TEAL), ("0.300", TEAL), "0.321-0.340", "0.954-0.965", "-0.005~-0.001"],
        ["30%", "Margin + PLS", "0.849", "0.247", "0.251", "0.340", "0.932", "-0.014"],
    ], col_widths=[120, 300, 165, 165, 230, 230, 200, 180], font_size="mini")
    metric_card(draw, (126, 690, 520, 905), "0.089", "predicted-Yes FP 基准占比", MUTED, fill="#F8FAFC")
    metric_card(draw, (590, 690, 984, 905), "0.383", "10% 预算 warning precision", BLUE, "margin + geometry", fill=SOFT)
    metric_card(draw, (1054, 690, 1448, 905), "0.965", "20% 预算 TP preserved", TEAL, "icd_blind + low-margin + PLS", fill=TEAL_SOFT)
    metric_card(draw, (1518, 690, 1818, 905), "-0.001", "Acc delta", AMBER, "near-neutral", fill=AMBER_SOFT)
    return img


def slide12():
    img, draw = new_canvas()
    header(draw, "跨模型边界", "FP/TN 可分不等于部署风险可分", "跨架构审计支持谨慎泛化，而不是 universal claim")
    table(draw, (110, 292, 1810, 700), ["Model", "Mechanistic FP/TN signal", "Deployment FP/TP view", "Interpretation"], [
        ["LLaVA-1.5-7B", "主实验成立；full diff mean AUROC 0.721", "Stage T 可做 selective routing", ("主证据链", BLUE)],
        ["LLaVA-1.5-13B", "Full diff AUROC 0.736；top-4 0.549", "方差-判别 mismatch 复现", ("checkpoint replication", TEAL)],
        ["Qwen2-VL / Qwen2.5-VL", "Best diff AUROC 0.772 / 0.771", "margin entropy 0.869 / 0.883 更强", ("中等 geometry 信号", AMBER)],
        ["InternVL2 / InternVL2.5", "FP/TN AUROC 0.999 / 0.998", "FP/TP AUROC 0.187 / 0.121", ("warning failure case", RED)],
    ], col_widths=[340, 430, 430, 500], font_size="mini")
    rounded(draw, (220, 792, 1700, 918), fill=RED_SOFT, outline="#FECACA", width=2, radius=8)
    draw_text(draw, (270, 826), "关键边界", F["h3"], fill=RED)
    draw_text(draw, (470, 818), "Internal separability can reflect answer-state geometry; it does not automatically become deployable hallucination-risk separability.", F["h2"], fill=TEXT, max_width=1120)
    return img


def slide13():
    img, draw = new_canvas()
    header(draw, "Takeaways", "五个需要听众记住的结论", "把大量 Stage 压缩成一条机制故事线")
    points = [
        ("1", "Blind-reference differencing 是研究 visual-evidence correction 的有效机制视角。"),
        ("2", "Dominant correction directions 解释方差，但不解释 hallucination decisions。"),
        ("3", "幻觉相关信号主要在 residual/tail 与 evidence-sensitive coordinates。"),
        ("4", "Tail coordinates 对正确拒绝有因果必要性，但 FP rescue 很弱。"),
        ("5", "Geometry 最适合作为机制证据和 complementary selective-routing signal。"),
    ]
    y = 276
    for num, text in points:
        rounded(draw, (170, y, 1750, y + 104), fill=CARD, outline=BORDER, width=2, radius=8)
        draw.ellipse((210, y + 26, 262, y + 78), fill=BLUE)
        draw_text(draw, (230, y + 32), num, F["small_med"], fill="#FFFFFF")
        draw_text(draw, (298, y + 30), text, F["body_med"], fill=TEXT, max_width=1340)
        y += 124
    return img


def slide14():
    img, draw = new_canvas()
    header(draw, "下一步", "围绕强基线、因果证据和跨架构边界收口", "不盲目增加 Stage，而是回答最可能被质疑的问题")
    steps = [
        ("统一三类 gate 对照", "已有 margin-only / geometry-only / margin+geometry fixed-trigger 结果可作为主线；继续报告 geometry 的增量而非全面胜出。"),
        ("扩大 tail ablation 因果实验", "从 pilot 变成主文级证据：更多 TN 样本，并补 LLaVA-13B 复现。"),
        ("进一步审计 InternVL failure case", "解释为什么 FP/TN 可分但 FP/TP 部署失败，形成清晰 architecture limitation。"),
        ("补开放式生成幻觉评估", "避免工作被理解为只适用于 POPE yes/no artifact。"),
    ]
    xys = [(118, 292), (1018, 292), (118, 612), (1018, 612)]
    for (title, body), (x, y), color in zip(steps, xys, [BLUE, TEAL, AMBER, RED]):
        rounded(draw, (x, y, x + 784, y + 240), fill=CARD, outline=BORDER, width=2, radius=8)
        draw.rectangle((x, y, x + 784, y + 10), fill=color)
        draw_text(draw, (x + 34, y + 42), title, F["h3"], fill=color, max_width=700)
        draw_text(draw, (x + 34, y + 104), body, F["body"], fill=TEXT, max_width=690)
    rounded(draw, (270, 910, 1650, 984), fill="#FFFFFF", outline="#DCEBFA", radius=8)
    draw_text(draw, (315, 928), "一句话中心论点：VLM 中最大的 visual correction directions 不是 hallucination decision directions。", F["small_med"], fill=SLATE, max_width=1300)
    return img


SLIDES = [
    slide1,
    slide2,
    slide3,
    slide4,
    slide5,
    slide6,
    slide7,
    slide8,
    slide9,
    slide10,
    slide11,
    slide12,
    slide13,
    slide14,
]


OUTLINE = """# 基于盲参考差分的 VLM 幻觉校正几何分析

## Outline

### Slide 1: 标题页
- Blind-reference differencing reveals layered correction geometry.
- 中心论点：最大的 visual correction directions 不是 hallucination decision directions。

### Slide 2: 研究问题
- 视觉证据如何改变 VLM 的内部表示？
- 用 z_blind - z_img 表示视觉证据带来的 correction。

### Slide 3: 任务边界
- FP vs TN 是机制分析显微镜。
- FP vs TP 才是部署风险识别。
- FP rescue 是更强的因果修正问题，目前很弱。

### Slide 4: 方法总览
- 两次前向：image+question 与 question-only。
- 对 difference matrix 做 SVD、tail analysis、intervention 与 selective gate。

### Slide 5: Finding 1
- correction space 有强低秩结构。
- Top-4 SVD directions 解释 72.7%-88.6% 方差。

### Slide 6: Finding 2
- 方差与判别不同步。
- L24 top-4 AUROC 仅约 0.471，full difference 多种子约 0.721。

### Slide 7: 机制解释
- top backbone 更像 image-conditioning signal。
- evidence correctness signal 方差较小，分布在 residual/tail coordinates。

### Slide 8: Finding 3
- residual/tail coordinates 更 evidence-sensitive。
- matched vs mismatch 差异在 tail 与 supervised view 中更明显。

### Slide 9: Finding 4
- L24 tail ablation 会剂量依赖地把 TN 推向 Yes。
- norm-matched random tail control 保持 0 Yes rate。

### Slide 10: 负结果
- FP rescue 只在 borderline case 上弱成功。
- 不能把工作包装成 reliable hallucination mitigation。

### Slide 11: 部署视角
- geometry 更适合作为 selective warning / routing 的互补信号。
- fixed-trigger 对照显示低预算下 margin+geometry 有增量。

### Slide 12: 跨模型边界
- LLaVA-13B 复现方差-判别解耦。
- Qwen 有中等 geometry 信号；InternVL 是部署 gate failure case。

### Slide 13: Takeaways
- 五个核心结论，强调机制定位与路由定位。

### Slide 14: 下一步
- 强基线对照、因果证据扩大、InternVL 审计、开放式生成评估。
"""


SPEECH = """# Speaker Notes

## Slide 1: 标题页

今天汇报的是最近围绕 VLM 幻觉内部机制的一组实验。核心问题是：当模型看到图像以后，相比只看到问题，它的 hidden state 会发生怎样的 correction，这些 correction 里哪些方向和幻觉有关。

整场汇报的中心结论是：图像带来的最大变化方向，并不是幻觉判别方向。真正与正确拒绝和幻觉风险相关的信号，更多出现在 residual/tail evidence-sensitive coordinates 中。

## Slide 2: 研究问题

传统做法往往从最终回答或置信度出发，但这很难解释模型内部到底发生了什么。这里我把同一个样本分别在 image+question 和 question-only 条件下前向，然后比较指定层的 hidden state。

差分 d = z_blind - z_img 可以被理解为视觉证据对内部表示施加的 correction。后面的所有实验都围绕这个 difference space 展开。

## Slide 3: 任务边界

这一页非常重要。FP vs TN 是机制分析显微镜，因为它固定 ground-truth=No，看模型为什么有时错误接受、有时正确拒绝。

真实部署场景更接近 FP vs TP，也就是模型已经预测 Yes 时，哪些 Yes 是错的。FP rescue 则是更强的干预问题，目前结果很弱，所以不能把这项工作说成可靠修正幻觉。

## Slide 4: 方法总览

方法本身很简单：对同一个问题做两次前向，分别得到 z_img 和 z_blind，再取差分。然后在多个层和读出位置上分析这个 difference matrix。

分析路径包括谱结构、top-K 和 full difference probe、tail bands、条件几何、因果干预、跨模型审计，以及最后的 selective warning 和 gated ICD/VCD。

## Slide 5: Finding 1

第一步发现 correction space 有很强的低秩结构。Top-4 SVD directions 在不同层可以解释 72.7% 到 88.6% 的方差。

这说明模型从 question-only 到 image-question 的变化不是均匀分散的，而是被少数主方向支配。但这个发现本身还不能说明这些方向就是幻觉方向。

## Slide 6: Finding 2

这一页是核心结果。虽然 top-4 解释了大部分方差，但它对 FP/TN 的判别接近随机，L24 top-4 多种子 AUROC 只有约 0.471。

随着 K 增长到 64、128、256，判别性能才逐渐出现；full difference 在多种子结果里最强，约 0.721。这就是方差-判别解耦。

## Slide 7: 机制解释

一个直观解释是：difference 里同时包含 image-conditioning signal 和 evidence-correctness signal。前者方差大，所以 SVD 优先抓到；后者方差小，但更接近 FP/TN 标签。

因此 top backbone 更像是在区分有没有图、是否进入视觉条件模式，而不是判断图像证据是否支持当前回答。

## Slide 8: Finding 3

条件几何支持这个解释。matched、random、adversarial 和 blind 条件对比显示，top-backbone 更偏 image-conditioned vs blind，而 residual/tail 更敏感于 matched vs mismatch。

尤其是 tail 257-1024 的 condition delta 在 L24 和 L32 上更明显，TN 的 matched-specific tail 增强也远强于 FP。

## Slide 9: Finding 4

这里是目前最干净的因果证据。对 L24 tail slice 做 ablation，随着 alpha 增大，原本正确回答 No 的 TN 会逐渐翻转为 Yes。

alpha=8 时 L24 Yes rate 达到 1.0，而 norm-matched random tail control 在 last-token 设置下保持 0 Yes rate。这说明 tail coordinates 对正确 negative decision 是 necessary。

## Slide 10: 负结果

反方向把 FP 救回来就弱得多。64 个 FP 样本中只有 3 个 decoded rescue，Stage M 的 32 个样本里只有 2 个，且都是 baseline margin 很小的 borderline case。

所以更准确的表述是：tail 对 TN 正确拒绝有因果必要性，但不等于可以可靠修正 FP。这个负结果需要主动讲清楚。

## Slide 11: 部署视角

部署上更稳的故事是 selective warning 和 routing。fixed-trigger 对照显示，low-margin 是强基线，但 margin+geometry 在 10% 和 20% 低预算下能小幅提升 FP recall 和 warning precision。

在 gated ICD 里，low-margin+PLS 在 20% 预算下可以做到接近 always-on 的 FP reduction，同时更好保护 TP，accuracy delta 接近 0。这支持 geometry 作为互补路由信号。

## Slide 12: 跨模型边界

跨模型结果提醒我们不要过度泛化。LLaVA-13B 复现了方差-判别解耦，Qwen 上也有中等 geometry 信号，但 margin entropy 更强。

InternVL 是很重要的失败案例：它在 FP/TN 上几乎完美可分，但在 predicted-Yes 的 FP/TP 部署任务上失败。这说明内部可分性不自动等于可部署风险可分性。

## Slide 13: Takeaways

总结来说，blind-reference differencing 是一个有用的机制视角。它揭示了 visual correction space 的分层结构：主方差方向解释图文条件差异，但不是幻觉判别方向。

真正与正确拒绝和幻觉风险相关的信号更接近 residual/tail 和 evidence-sensitive coordinates。它更适合作为机制证据和 selective routing 的互补信号。

## Slide 14: 下一步

下一步建议围绕最容易被质疑的地方收口：第一，继续统一 margin-only、geometry-only、margin+geometry 对照；第二，把 tail ablation 从 pilot 扩成主文级因果证据。

第三，深入解释 InternVL failure case；第四，补一个开放式生成幻觉评估，避免结果被认为只适用于 POPE yes/no artifact。
"""


def main() -> None:
    PROJECT.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    for i, slide in enumerate(SLIDES, 1):
        save_slide(i, slide())
    (PROJECT / "outline.md").write_text(OUTLINE, encoding="utf-8")
    (PROJECT / "speech.md").write_text(SPEECH, encoding="utf-8")
    print(f"Rendered {len(SLIDES)} slides to {IMAGE_DIR}")
    print(f"Wrote {PROJECT / 'outline.md'}")
    print(f"Wrote {PROJECT / 'speech.md'}")


if __name__ == "__main__":
    main()
