"""Render deterministic PerfRunbench submission visuals.

The chart data is intentionally duplicated here as public submission constants.
Do not replace these values with private artifact reads or live benchmark runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Iterable, Sequence

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - exercised only without Pillow.
    raise SystemExit(
        "Pillow is required to render PNG submission assets. "
        "The script does not use network access."
    ) from exc


ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = ROOT / "docs" / "submission" / "assets"

WIDTH = 1280
HEIGHT = 720
BG = "#f7f9fb"
INK = "#16202a"
MUTED = "#576574"
GRID = "#d7dee8"
TEAL = "#00796b"
BLUE = "#1f6fb2"
GOLD = "#b98216"
GREEN = "#2e7d32"
RED = "#b23b3b"
PANEL = "#ffffff"


@dataclass(frozen=True)
class BarItem:
    label: str
    value: float
    value_label: str
    color: str


@dataclass(frozen=True)
class BarChart:
    filename_stem: str
    title: str
    subtitle: str
    unit: str
    annotation: Sequence[str]
    footnotes: Sequence[str]
    bars: Sequence[BarItem]


LLAMA_RUNTIME = BarChart(
    filename_stem="c7g_llama_runtime_before_after",
    title="Arm inference runtime tuning on AWS C7g",
    subtitle="llama.cpp runtime-search protocol, Qwen2.5-7B Q4_K_M",
    unit="tokens/s",
    annotation=("+28.25% observed decode throughput",),
    footnotes=(
        "Protocol-specific preliminary evidence.",
        "Do not combine with build-profile or server-latency results.",
    ),
    bars=(
        BarItem("Baseline", 13.630, "13.630 tokens/s", BLUE),
        BarItem("Selected", 17.480, "17.480 tokens/s", TEAL),
    ),
)

C7G_PYTORCH = BarChart(
    filename_stem="c7g_pytorch_before_after",
    title="PyTorch training tuning on AWS C7g",
    subtitle="Synthetic CPU-only workload, effective batch size preserved",
    unit="samples/s",
    annotation=(
        "Synthetic CPU-only workload",
        "Effective batch size preserved at 128",
    ),
    footnotes=(
        "Synthetic validation of tuning and guard behavior.",
        "Not a production training-throughput claim.",
    ),
    bars=(
        BarItem("Baseline", 77859.454508, "77,859.455 samples/s", GOLD),
        BarItem("Selected", 203215.038822, "203,215.039 samples/s", TEAL),
    ),
)

VALIDATION_MATRIX = {
    "C7g Arm64": {
        "llama.cpp tuning": "passed_preliminary",
        "PyTorch tuning": "passed_preliminary",
        "guarded apply": "passed",
    },
    "T4g Arm64": {
        "low-resource guards": "passed",
        "real PyTorch tuning": "passed_preliminary",
        "cross-target drift refusal": "passed",
    },
    "Windows x86_64": {
        "golden-path CLI": "passed",
        "real CPU PyTorch tuning": "passed_preliminary",
        "Arm recommendation refusal": "passed",
    },
}

ARCHITECTURE_STEPS = [
    "User command and config",
    "Doctor and workload identity",
    "Workload adapter",
    "Candidate generator",
    "Bounded tuning slices",
    "Metrics, progress, and resource guards",
    "Statistical confirmation",
    "Performance Passport",
    "RecommendationBundle and cache",
    "Environment validation",
    "Canary and process-scoped apply",
    "Full workload",
    "Cleanup or rollback",
]

ADAPTER_EXAMPLES = ["llama.cpp", "PyTorch training", "generic command"]
TARGET_EXAMPLES = ["Linux Arm64", "Windows x86_64", "SSH remote target"]


def main() -> int:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    for chart in (LLAMA_RUNTIME, C7G_PYTORCH):
        (ASSET_DIR / f"{chart.filename_stem}.svg").write_text(
            render_bar_chart_svg(chart), encoding="utf-8"
        )
        render_bar_chart_png(chart, ASSET_DIR / f"{chart.filename_stem}.png")

    (ASSET_DIR / "multitarget_validation_summary.svg").write_text(
        render_matrix_svg(), encoding="utf-8"
    )
    render_matrix_png(ASSET_DIR / "multitarget_validation_summary.png")

    (ASSET_DIR / "perfrunbench_architecture.svg").write_text(
        render_architecture_svg(), encoding="utf-8"
    )
    render_architecture_png(ASSET_DIR / "perfrunbench_architecture.png")
    return 0


def render_bar_chart_svg(chart: BarChart) -> str:
    max_value = max(item.value for item in chart.bars) * 1.12
    chart_left = 230
    chart_top = 170
    chart_width = 780
    bar_h = 88
    gap = 72
    lines = [
        svg_header(),
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{BG}"/>',
        f'<text x="70" y="80" class="title">{escape(chart.title)}</text>',
        f'<text x="70" y="118" class="subtitle">{escape(chart.subtitle)}</text>',
    ]
    for idx, item in enumerate(chart.bars):
        y = chart_top + idx * (bar_h + gap)
        w = int(chart_width * item.value / max_value)
        lines.extend(
            [
                f'<text x="70" y="{y + 54}" class="label">{escape(item.label)}</text>',
                f'<rect x="{chart_left}" y="{y}" width="{chart_width}" height="{bar_h}" rx="8" fill="{GRID}"/>',
                f'<rect x="{chart_left}" y="{y}" width="{w}" height="{bar_h}" rx="8" fill="{item.color}"/>',
                f'<text x="{chart_left + w + 18}" y="{y + 54}" class="value">{escape(item.value_label)}</text>',
            ]
        )
    ann_y = 500
    for line in chart.annotation:
        lines.append(f'<text x="70" y="{ann_y}" class="annotation">{escape(line)}</text>')
        ann_y += 34
    foot_y = 635
    for line in chart.footnotes:
        lines.append(f'<text x="70" y="{foot_y}" class="footnote">{escape(line)}</text>')
        foot_y += 24
    lines.append(svg_footer())
    return "\n".join(lines)


def render_matrix_svg() -> str:
    panel_w = 360
    panel_h = 330
    top = 160
    left = 70
    gap = 40
    lines = [
        svg_header(),
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{BG}"/>',
        '<text x="70" y="78" class="title">Multi-target validation summary</text>',
        '<text x="70" y="116" class="subtitle">Capability evidence only; no cross-machine throughput ranking</text>',
    ]
    for panel_idx, (platform, capabilities) in enumerate(VALIDATION_MATRIX.items()):
        x = left + panel_idx * (panel_w + gap)
        lines.append(
            f'<rect x="{x}" y="{top}" width="{panel_w}" height="{panel_h}" rx="10" '
            f'fill="{PANEL}" stroke="{GRID}" stroke-width="2"/>'
        )
        lines.append(f'<text x="{x + 24}" y="{top + 46}" class="label">{escape(platform)}</text>')
        y = top + 92
        for capability, status in capabilities.items():
            fill = GREEN if status == "passed" else TEAL
            lines.append(
                f'<rect x="{x + 24}" y="{y}" width="190" height="32" rx="16" fill="{fill}"/>'
            )
            lines.append(f'<text x="{x + 40}" y="{y + 22}" class="matrixTextSmall">{escape(status)}</text>')
            lines.append(f'<text x="{x + 24}" y="{y + 60}" class="small">{escape(capability)}</text>')
            y += 78
    footnotes = [
        "Synthetic PyTorch results are guard/workflow evidence, not production training claims.",
        "C7g, T4g, and Windows protocols are intentionally not placed on one performance axis.",
    ]
    y = 626
    for line in footnotes:
        lines.append(f'<text x="70" y="{y}" class="footnote">{escape(line)}</text>')
        y += 24
    lines.append(svg_footer())
    return "\n".join(lines)


def render_architecture_svg() -> str:
    box_w = 245
    box_h = 48
    x1 = 80
    x2 = 405
    x3 = 730
    top = 95
    gap = 62
    positions: list[tuple[str, int, int]] = []
    for idx, label in enumerate(ARCHITECTURE_STEPS):
        col = idx // 5
        row = idx % 5
        x = [x1, x2, x3][col]
        y = top + row * gap
        positions.append((label, x, y))
    lines = [
        svg_header(),
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{BG}"/>',
        '<text x="70" y="58" class="title">PerfRunbench guarded autotuning architecture</text>',
    ]
    for idx, (label, x, y) in enumerate(positions):
        lines.extend(
            [
                f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="8" fill="{PANEL}" stroke="{BLUE}" stroke-width="2"/>',
                f'<text x="{x + 14}" y="{y + 30}" class="boxText">{escape(label)}</text>',
            ]
        )
        if idx + 1 < len(positions):
            _, nx, ny = positions[idx + 1]
            if nx == x:
                lines.append(arrow_svg(x + box_w / 2, y + box_h, x + box_w / 2, ny))
            else:
                lines.append(arrow_svg(x + box_w, y + box_h / 2, nx, ny + box_h / 2))
    lines.append('<text x="80" y="455" class="subtitle">Adapter examples</text>')
    for i, text in enumerate(ADAPTER_EXAMPLES):
        x = 80 + i * 180
        lines.append(f'<rect x="{x}" y="478" width="160" height="42" rx="8" fill="#e7f4f2"/>')
        lines.append(f'<text x="{x + 16}" y="505" class="small">{escape(text)}</text>')
    lines.append('<text x="730" y="455" class="subtitle">Target examples</text>')
    for i, text in enumerate(TARGET_EXAMPLES):
        x = 730
        y = 478 + i * 50
        lines.append(f'<rect x="{x}" y="{y}" width="220" height="42" rx="8" fill="#eef3f9"/>')
        lines.append(f'<text x="{x + 16}" y="{y + 27}" class="small">{escape(text)}</text>')
    lines.append('<text x="70" y="675" class="footnote">Default workflow shown. Optional external workloads use the generic command adapter boundary.</text>')
    lines.append(svg_footer())
    return "\n".join(lines)


def render_bar_chart_png(chart: BarChart, path: Path) -> None:
    img, draw = new_canvas()
    title(draw, chart.title, 70, 62)
    subtitle(draw, chart.subtitle, 70, 100)
    max_value = max(item.value for item in chart.bars) * 1.12
    chart_left = 230
    chart_top = 170
    chart_width = 780
    bar_h = 88
    gap = 72
    for idx, item in enumerate(chart.bars):
        y = chart_top + idx * (bar_h + gap)
        w = int(chart_width * item.value / max_value)
        text(draw, item.label, 70, y + 32, 22, INK)
        draw.rounded_rectangle([chart_left, y, chart_left + chart_width, y + bar_h], radius=8, fill=GRID)
        draw.rounded_rectangle([chart_left, y, chart_left + w, y + bar_h], radius=8, fill=item.color)
        text(draw, item.value_label, chart_left + w + 18, y + 32, 22, INK)
    y = 470
    for line in chart.annotation:
        text(draw, line, 70, y, 28, TEAL)
        y += 38
    y = 614
    for line in chart.footnotes:
        text(draw, line, 70, y, 18, MUTED)
        y += 26
    img.save(path)


def render_matrix_png(path: Path) -> None:
    img, draw = new_canvas()
    title(draw, "Multi-target validation summary", 70, 60)
    subtitle(draw, "Capability evidence only; no cross-machine throughput ranking", 70, 98)
    panel_w = 360
    panel_h = 330
    left = 70
    top = 160
    gap = 40
    for panel_idx, (platform, capabilities) in enumerate(VALIDATION_MATRIX.items()):
        x = left + panel_idx * (panel_w + gap)
        draw.rounded_rectangle([x, top, x + panel_w, top + panel_h], radius=10, fill=PANEL, outline=GRID, width=2)
        text(draw, platform, x + 24, top + 24, 23, INK, bold=True)
        y = top + 92
        for capability, status in capabilities.items():
            fill = GREEN if status == "passed" else TEAL
            draw.rounded_rectangle([x + 24, y, x + 214, y + 32], radius=16, fill=fill)
            text(draw, status, x + 40, y + 9, 13, "#ffffff")
            text(draw, capability, x + 24, y + 44, 18, INK, max_width=panel_w - 48)
            y += 78
    text(draw, "Synthetic PyTorch results are guard/workflow evidence, not production training claims.", 70, 608, 18, MUTED)
    text(draw, "C7g, T4g, and Windows protocols are intentionally not placed on one performance axis.", 70, 634, 18, MUTED)
    img.save(path)


def render_architecture_png(path: Path) -> None:
    img, draw = new_canvas()
    title(draw, "PerfRunbench guarded autotuning architecture", 70, 42)
    box_w = 245
    box_h = 48
    xs = [80, 405, 730]
    top = 95
    gap = 62
    positions: list[tuple[str, int, int]] = []
    for idx, label in enumerate(ARCHITECTURE_STEPS):
        col = idx // 5
        row = idx % 5
        positions.append((label, xs[col], top + row * gap))
    for idx, (label, x, y) in enumerate(positions):
        draw.rounded_rectangle([x, y, x + box_w, y + box_h], radius=8, fill=PANEL, outline=BLUE, width=2)
        text(draw, label, x + 14, y + 15, 15, INK, max_width=box_w - 24)
        if idx + 1 < len(positions):
            _, nx, ny = positions[idx + 1]
            if nx == x:
                arrow_png(draw, x + box_w // 2, y + box_h, x + box_w // 2, ny)
            else:
                arrow_png(draw, x + box_w, y + box_h // 2, nx, ny + box_h // 2)
    subtitle(draw, "Adapter examples", 80, 440)
    for i, label in enumerate(ADAPTER_EXAMPLES):
        x = 80 + i * 180
        draw.rounded_rectangle([x, 478, x + 160, 520], radius=8, fill="#e7f4f2")
        text(draw, label, x + 16, 490, 16, INK)
    subtitle(draw, "Target examples", 730, 440)
    for i, label in enumerate(TARGET_EXAMPLES):
        y = 478 + i * 50
        draw.rounded_rectangle([730, y, 950, y + 42], radius=8, fill="#eef3f9")
        text(draw, label, 746, y + 12, 16, INK)
    text(draw, "Default workflow shown. Optional external workloads use the generic command adapter boundary.", 70, 648, 18, MUTED)
    img.save(path)


def svg_header() -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">
<defs>
  <style>
    .title {{ font: 700 34px Arial, sans-serif; fill: {INK}; }}
    .subtitle {{ font: 400 20px Arial, sans-serif; fill: {MUTED}; }}
    .label {{ font: 700 22px Arial, sans-serif; fill: {INK}; }}
    .value {{ font: 700 22px Arial, sans-serif; fill: {INK}; }}
    .annotation {{ font: 700 28px Arial, sans-serif; fill: {TEAL}; }}
    .footnote {{ font: 400 18px Arial, sans-serif; fill: {MUTED}; }}
    .small {{ font: 600 16px Arial, sans-serif; fill: {INK}; }}
    .matrixText {{ font: 700 17px Arial, sans-serif; fill: white; }}
    .matrixTextSmall {{ font: 700 14px Arial, sans-serif; fill: white; }}
    .boxText {{ font: 600 15px Arial, sans-serif; fill: {INK}; }}
  </style>
  <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L0,6 L9,3 z" fill="{BLUE}" />
  </marker>
</defs>"""


def svg_footer() -> str:
    return "</svg>\n"


def arrow_svg(x1: float, y1: float, x2: float, y2: float) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{BLUE}" stroke-width="2.5" marker-end="url(#arrow)"/>'
    )


def new_canvas() -> tuple["Image.Image", "ImageDraw.ImageDraw"]:
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    return img, ImageDraw.Draw(img)


def font(size: int, bold: bool = False) -> "ImageFont.FreeTypeFont | ImageFont.ImageFont":
    names = (
        "arialbd.ttf" if bold else "arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def title(draw: "ImageDraw.ImageDraw", value: str, x: int, y: int) -> None:
    text(draw, value, x, y, 34, INK, bold=True)


def subtitle(draw: "ImageDraw.ImageDraw", value: str, x: int, y: int) -> None:
    text(draw, value, x, y, 20, MUTED)


def text(
    draw: "ImageDraw.ImageDraw",
    value: str,
    x: int,
    y: int,
    size: int,
    fill: str,
    *,
    bold: bool = False,
    max_width: int | None = None,
) -> None:
    chosen = font(size, bold=bold)
    if max_width is None:
        draw.text((x, y), value, fill=fill, font=chosen)
        return
    words = value.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if text_width(draw, candidate, chosen) <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    line_h = size + 3
    for idx, line in enumerate(lines[:3]):
        draw.text((x, y + idx * line_h), line, fill=fill, font=chosen)


def text_width(draw: "ImageDraw.ImageDraw", value: str, chosen: "ImageFont.ImageFont") -> int:
    box = draw.textbbox((0, 0), value, font=chosen)
    return int(box[2] - box[0])


def arrow_png(draw: "ImageDraw.ImageDraw", x1: int, y1: int, x2: int, y2: int) -> None:
    draw.line([x1, y1, x2, y2], fill=BLUE, width=3)
    if x2 == x1:
        points = [(x2, y2), (x2 - 7, y2 - 12), (x2 + 7, y2 - 12)]
    else:
        points = [(x2, y2), (x2 - 12, y2 - 7), (x2 - 12, y2 + 7)]
    draw.polygon(points, fill=BLUE)


if __name__ == "__main__":
    raise SystemExit(main())
