from __future__ import annotations

from typing import Any


def metric_bar_chart(title: str, rows: list[tuple[str, float | int | None]], *, unit: str = "") -> str:
    values = [abs(float(value)) for _label, value in rows if isinstance(value, (int, float))]
    maximum = max(values) if values else 1.0
    width = 720
    row_height = 34
    label_width = 210
    chart_width = width - label_width - 40
    height = 54 + row_height * len(rows)
    parts = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{_esc(title)}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="20" y="30" font-family="Segoe UI, sans-serif" font-size="18" font-weight="700" fill="#0f172a">{_esc(title)}</text>',
    ]
    for index, (label, value) in enumerate(rows):
        y = 54 + index * row_height
        numeric = float(value) if isinstance(value, (int, float)) else None
        bar_width = int((abs(numeric) / maximum) * chart_width) if numeric is not None and maximum > 0 else 0
        fill = "#2563eb" if numeric is None or numeric >= 0 else "#dc2626"
        value_text = "n/a" if numeric is None else f"{numeric:.3f}{unit}"
        parts.extend(
            [
                f'<text x="20" y="{y + 20}" font-family="Segoe UI, sans-serif" font-size="13" fill="#334155">{_esc(label)}</text>',
                f'<rect x="{label_width}" y="{y + 7}" width="{chart_width}" height="18" rx="9" fill="#e2e8f0"/>',
                f'<rect x="{label_width}" y="{y + 7}" width="{bar_width}" height="18" rx="9" fill="{fill}"/>',
                f'<text x="{label_width + chart_width + 10}" y="{y + 21}" font-family="Segoe UI, sans-serif" font-size="12" fill="#0f172a">{_esc(value_text)}</text>',
            ]
        )
    parts.append("</svg>")
    return "\n".join(parts)


def sparkline_svg(title: str, values: list[Any], *, width: int = 720, height: int = 160) -> str:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    if not numeric:
        return metric_bar_chart(title, [("samples", None)])
    low = min(numeric)
    high = max(numeric)
    spread = high - low or 1.0
    step = (width - 40) / max(1, len(numeric) - 1)
    points = []
    for index, value in enumerate(numeric):
        x = 20 + index * step
        y = height - 28 - ((value - low) / spread) * (height - 58)
        points.append(f"{x:.2f},{y:.2f}")
    return "\n".join(
        [
            f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{_esc(title)}">',
            '<rect width="100%" height="100%" fill="#f8fafc"/>',
            f'<text x="20" y="24" font-family="Segoe UI, sans-serif" font-size="16" font-weight="700" fill="#0f172a">{_esc(title)}</text>',
            f'<polyline fill="none" stroke="#2563eb" stroke-width="3" points="{" ".join(points)}"/>',
            f'<text x="20" y="{height - 8}" font-family="Segoe UI, sans-serif" font-size="12" fill="#475569">min {low:.3f} max {high:.3f}</text>',
            "</svg>",
        ]
    )


def _esc(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
