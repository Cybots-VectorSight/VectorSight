"""Generate SVG from resolved coordinates."""

from __future__ import annotations

import math
from typing import Any

from app.svg.serializer import serialize_svg


def elements_to_svg_dicts(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert resolved element coordinates to SVG element dictionaries."""
    svg_elements: list[dict[str, Any]] = []

    for elem in elements:
        if elem.get("tag") == "raw":
            # Raw SVG passthrough — handled separately
            continue

        shape = elem.get("shape", "")
        cx = elem.get("cx", 12)
        cy = elem.get("cy", 12)
        w = elem.get("w", 10)
        h = elem.get("h", 10)
        fill = elem.get("fill", "none")
        stroke = elem.get("stroke", "currentColor")

        if shape == "circular" or shape == "circle":
            r = min(w, h) / 2
            svg_elements.append({
                "tag": "circle",
                "cx": str(round(cx, 2)),
                "cy": str(round(cy, 2)),
                "r": str(round(r, 2)),
                "fill": fill,
                "stroke": stroke,
                "stroke-width": "2",
            })
        elif shape == "rectangular" or shape == "rect":
            svg_elements.append({
                "tag": "rect",
                "x": str(round(cx - w / 2, 2)),
                "y": str(round(cy - h / 2, 2)),
                "width": str(round(w, 2)),
                "height": str(round(h, 2)),
                "fill": fill,
                "stroke": stroke,
                "stroke-width": "2",
            })
        elif shape == "line":
            svg_elements.append({
                "tag": "line",
                "x1": str(round(cx - w / 2, 2)),
                "y1": str(round(cy, 2)),
                "x2": str(round(cx + w / 2, 2)),
                "y2": str(round(cy, 2)),
                "stroke": stroke,
                "stroke-width": "2",
            })
        elif shape == "ellipse":
            svg_elements.append({
                "tag": "ellipse",
                "cx": str(round(cx, 2)),
                "cy": str(round(cy, 2)),
                "rx": str(round(w / 2, 2)),
                "ry": str(round(h / 2, 2)),
                "fill": fill,
                "stroke": stroke,
                "stroke-width": "2",
            })
        elif shape == "triangular" or shape == "triangle":
            # Equilateral triangle pointing up
            pts = _triangle_points(cx, cy, w, h)
            svg_elements.append({
                "tag": "polygon",
                "points": " ".join(f"{x:.2f},{y:.2f}" for x, y in pts),
                "fill": fill,
                "stroke": stroke,
                "stroke-width": "2",
            })
        elif shape == "arc":
            # Simple arc element
            r = min(w, h) / 2
            svg_elements.append({
                "tag": "path",
                "d": f"M {cx - r:.2f} {cy:.2f} A {r:.2f} {r:.2f} 0 0 1 {cx + r:.2f} {cy:.2f}",
                "fill": fill,
                "stroke": stroke,
                "stroke-width": "2",
            })
        else:
            # Generic: use a circle as fallback
            r = min(w, h) / 2
            svg_elements.append({
                "tag": "circle",
                "cx": str(round(cx, 2)),
                "cy": str(round(cy, 2)),
                "r": str(round(r, 2)),
                "fill": fill,
                "stroke": stroke,
                "stroke-width": "2",
            })

    return svg_elements


def _triangle_points(
    cx: float, cy: float, w: float, h: float,
) -> list[tuple[float, float]]:
    """Generate equilateral triangle vertices."""
    top = (cx, cy - h / 2)
    bottom_left = (cx - w / 2, cy + h / 2)
    bottom_right = (cx + w / 2, cy + h / 2)
    return [top, bottom_left, bottom_right]


def generate_svg(
    elements: list[dict[str, Any]],
    canvas_w: float = 24.0,
    canvas_h: float = 24.0,
    styles: dict[str, str] | None = None,
) -> str:
    """Full pipeline: resolved elements → SVG string."""
    # Check for raw SVG passthrough
    for elem in elements:
        if elem.get("tag") == "raw" and "svg" in elem:
            return elem["svg"]

    svg_dicts = elements_to_svg_dicts(elements)
    return serialize_svg(svg_dicts, canvas_w, canvas_h, styles=styles)
