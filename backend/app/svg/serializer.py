"""Write clean SVG output from resolved spatial intent."""

from __future__ import annotations

from typing import Any


def serialize_svg(
    elements: list[dict[str, Any]],
    canvas_w: float = 24.0,
    canvas_h: float = 24.0,
    title: str = "",
    description: str = "",
    styles: dict[str, str] | None = None,
) -> str:
    """Generate clean SVG markup from element definitions."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg viewBox="0 0 {canvas_w} {canvas_h}" xmlns="http://www.w3.org/2000/svg"'
        f' role="img">',
    ]

    if title:
        lines.append(f"  <title>{title}</title>")
    if description:
        lines.append(f"  <desc>{description}</desc>")

    if styles:
        lines.append("  <style>")
        for selector, props in styles.items():
            lines.append(f"    {selector} {{ {props} }}")
        lines.append("  </style>")

    for elem in elements:
        tag = elem.get("tag", "path")
        attrs = {k: v for k, v in elem.items() if k not in ("tag", "children")}
        attr_str = " ".join(f'{k}="{v}"' for k, v in attrs.items())
        lines.append(f"  <{tag} {attr_str} />")

    lines.append("</svg>")
    return "\n".join(lines)
