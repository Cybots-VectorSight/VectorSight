"""Resolve mirror specifications — duplicate elements across axis."""

from __future__ import annotations

from typing import Any

from app.models.intent import MirrorSpec, SpatialIntent


def apply_mirrors(
    elements: list[dict[str, Any]],
    mirrors: list[MirrorSpec],
    canvas_w: float,
    canvas_h: float,
) -> list[dict[str, Any]]:
    """Duplicate mirrored elements across specified axes."""
    result = list(elements)
    named = {e["name"]: e for e in elements if "name" in e}

    for mirror in mirrors:
        axis = mirror.axis.lower()

        # Determine mirror axis position
        if "vertical" in axis:
            mirror_x = canvas_w / 2
            for elem_name in mirror.elements:
                if elem_name not in named:
                    continue
                original = named[elem_name]
                mirrored = dict(original)
                mirrored["name"] = f"{elem_name}_mirror"
                mirrored["cx"] = round(2 * mirror_x - original["cx"], 2)
                result.append(mirrored)

        elif "horizontal" in axis:
            mirror_y = canvas_h / 2
            for elem_name in mirror.elements:
                if elem_name not in named:
                    continue
                original = named[elem_name]
                mirrored = dict(original)
                mirrored["name"] = f"{elem_name}_mirror"
                mirrored["cy"] = round(2 * mirror_y - original["cy"], 2)
                result.append(mirrored)

    return result
