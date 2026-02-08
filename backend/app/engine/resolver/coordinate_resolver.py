"""Resolve spatial intent descriptions into concrete coordinates.

Converts relative spatial descriptions (e.g., "60% of canvas", "center")
into absolute (x, y, w, h) values.
"""

from __future__ import annotations

import re
from typing import Any

from app.models.intent import SpatialElement, SpatialIntent


def resolve_coordinates(intent: SpatialIntent) -> list[dict[str, Any]]:
    """Convert SpatialIntent elements into positioned SVG element dicts."""
    cw = intent.canvas_width
    ch = intent.canvas_height
    resolved: list[dict[str, Any]] = []
    named_elements: dict[str, dict[str, Any]] = {}

    for elem in intent.elements:
        # If raw SVG path was given, pass through
        if elem.path and elem.shape == "raw":
            resolved.append({"tag": "raw", "svg": elem.path})
            continue

        # Parse size
        w, h = _parse_size(elem.size, cw, ch)

        # Parse position
        cx, cy = _parse_position(elem.position, cw, ch, named_elements)

        info: dict[str, Any] = {
            "name": elem.name,
            "shape": elem.shape,
            "cx": round(cx, 2),
            "cy": round(cy, 2),
            "w": round(w, 2),
            "h": round(h, 2),
            "fill": elem.fill,
            "stroke": elem.stroke,
        }
        named_elements[elem.name] = info
        resolved.append(info)

    return resolved


def _parse_size(size_str: str, cw: float, ch: float) -> tuple[float, float]:
    """Parse size description into (width, height)."""
    if not size_str:
        return (cw * 0.5, ch * 0.5)

    # Match "X% of canvas"
    pct_match = re.search(r"(\d+(?:\.\d+)?)%\s*(?:of\s+canvas)?", size_str)
    if pct_match:
        pct = float(pct_match.group(1)) / 100
        return (cw * pct, ch * pct)

    # Match absolute "WxH" or "W"
    abs_match = re.search(r"(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)", size_str)
    if abs_match:
        return (float(abs_match.group(1)), float(abs_match.group(2)))

    single_match = re.search(r"(\d+(?:\.\d+)?)", size_str)
    if single_match:
        val = float(single_match.group(1))
        return (val, val)

    return (cw * 0.5, ch * 0.5)


def _parse_position(
    pos_str: str,
    cw: float,
    ch: float,
    named: dict[str, Any],
) -> tuple[float, float]:
    """Parse position description into (x, y) center coordinates."""
    if not pos_str:
        return (cw / 2, ch / 2)

    pos = pos_str.lower()

    if "center" in pos:
        return (cw / 2, ch / 2)

    # Match percentage coordinates "(X%, Y%)"
    pct_match = re.search(r"(\d+(?:\.\d+)?)%\s*[x,]\s*(\d+(?:\.\d+)?)%", pos)
    if pct_match:
        return (
            cw * float(pct_match.group(1)) / 100,
            ch * float(pct_match.group(2)) / 100,
        )

    # Match absolute "(X, Y)"
    abs_match = re.search(r"\(?\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)?", pos)
    if abs_match:
        return (float(abs_match.group(1)), float(abs_match.group(2)))

    # Quadrant positions
    if "upper-left" in pos or "top-left" in pos:
        return (cw * 0.25, ch * 0.25)
    if "upper-right" in pos or "top-right" in pos:
        return (cw * 0.75, ch * 0.25)
    if "lower-left" in pos or "bottom-left" in pos:
        return (cw * 0.25, ch * 0.75)
    if "lower-right" in pos or "bottom-right" in pos:
        return (cw * 0.75, ch * 0.75)
    if "top" in pos or "upper" in pos:
        return (cw / 2, ch * 0.25)
    if "bottom" in pos or "lower" in pos:
        return (cw / 2, ch * 0.75)
    if "left" in pos:
        return (cw * 0.25, ch / 2)
    if "right" in pos:
        return (cw * 0.75, ch / 2)

    return (cw / 2, ch / 2)
