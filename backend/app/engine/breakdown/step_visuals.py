"""Per-step SVG visual previews for the breakdown pipeline.

Converts Shapely polygons at each pipeline stage into standalone SVG documents
for live preview in the frontend. No new dependencies — uses only Shapely coords
and string formatting.
"""

from __future__ import annotations

from shapely.geometry import Polygon, MultiPolygon, GeometryCollection

from app.engine.breakdown.separate import GroupData
from app.engine.breakdown.silhouette import SilhouetteResult

# 12 distinct colors for coloring individual subpaths / groups
_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
    "#fabed4", "#469990", "#dcbeff", "#9A6324",
]


def _polygon_to_svg_paths(
    geom,
    fill: str = "#888888",
    opacity: float = 0.7,
) -> str:
    """Convert a Shapely geometry to SVG <path> elements."""
    if geom is None or geom.is_empty:
        return ""

    parts: list[str] = []

    def _render_polygon(poly: Polygon) -> str:
        if poly.is_empty or poly.area < 0.5:
            return ""
        coords = list(poly.exterior.coords)
        if len(coords) < 3:
            return ""
        d = f"M {coords[0][0]:.1f},{coords[0][1]:.1f}"
        for x, y in coords[1:]:
            d += f" L {x:.1f},{y:.1f}"
        d += " Z"
        return (
            f'<path d="{d}" fill="{fill}" fill-opacity="{opacity}" '
            f'stroke="{fill}" stroke-width="0.5" stroke-opacity="0.9"/>'
        )

    if geom.geom_type == "Polygon":
        parts.append(_render_polygon(geom))
    elif geom.geom_type == "MultiPolygon":
        for poly in geom.geoms:
            parts.append(_render_polygon(poly))
    elif geom.geom_type == "GeometryCollection":
        for g in geom.geoms:
            if g.geom_type == "Polygon":
                parts.append(_render_polygon(g))
            elif g.geom_type == "MultiPolygon":
                for poly in g.geoms:
                    parts.append(_render_polygon(poly))

    return "\n".join(p for p in parts if p)


def _svg_wrap(content: str, cw: float, ch: float) -> str:
    """Wrap SVG content in a standalone SVG document."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {cw:.1f} {ch:.1f}"'
        f' width="{cw:.1f}" height="{ch:.1f}"'
        ' style="background:#1a1a2e">'
        f'\n{content}\n</svg>'
    )


def render_raw_subpaths(
    elements: list[dict],
    cw: float,
    ch: float,
) -> str:
    """B0.02: Each raw subpath gets a unique palette color."""
    paths: list[str] = []
    for i, el in enumerate(elements):
        color = _PALETTE[i % len(_PALETTE)]
        paths.append(_polygon_to_svg_paths(el["polygon"], fill=color, opacity=0.6))
    return _svg_wrap("\n".join(p for p in paths if p), cw, ch)


def render_merged_features(
    features: list[dict],
    cw: float,
    ch: float,
) -> str:
    """B1.02: Consolidated shapes after overlap merging, using actual fills."""
    paths: list[str] = []
    for i, f in enumerate(features):
        fills = f.get("fills", [])
        color = fills[0] if fills else _PALETTE[i % len(_PALETTE)]
        paths.append(_polygon_to_svg_paths(f["polygon"], fill=color, opacity=0.7))
    return _svg_wrap("\n".join(p for p in paths if p), cw, ch)


def render_grouped_features(
    groups: list[GroupData],
    cw: float,
    ch: float,
) -> str:
    """B1.03: Containment groups, each group gets a distinct color range."""
    paths: list[str] = []
    for gi, g in enumerate(groups):
        color = _PALETTE[gi % len(_PALETTE)]
        paths.append(_polygon_to_svg_paths(g.polygon, fill=color, opacity=0.65))
    return _svg_wrap("\n".join(p for p in paths if p), cw, ch)


def render_silhouettes(
    groups: list[GroupData],
    silhouettes: list[SilhouetteResult | None],
    cw: float,
    ch: float,
) -> str:
    """B2.01: Smooth Bezier outlines from SilhouetteResult.svg_d."""
    paths: list[str] = []
    for gi, (g, sr) in enumerate(zip(groups, silhouettes)):
        color = _PALETTE[gi % len(_PALETTE)]
        if sr is not None and sr.svg_d:
            paths.append(
                f'<path d="{sr.svg_d}" fill="none" '
                f'stroke="{color}" stroke-width="1.5" stroke-opacity="0.9"/>'
            )
        elif g.polygon is not None and not g.polygon.is_empty:
            # Fallback: render polygon outline if no silhouette
            paths.append(
                _polygon_to_svg_paths(g.polygon, fill=color, opacity=0.3)
            )
    return _svg_wrap("\n".join(p for p in paths if p), cw, ch)


def render_final_composite(
    groups: list[GroupData],
    cw: float,
    ch: float,
    silhouettes: list[SilhouetteResult | None] | None = None,
) -> str:
    """B4.01: All groups with G0, G1, ... labels at centroids.

    Uses silhouette Bezier paths when available for smooth outlines,
    falls back to raw polygons otherwise.
    """
    paths: list[str] = []
    labels: list[str] = []
    for gi, g in enumerate(groups):
        color = _PALETTE[gi % len(_PALETTE)]
        sr = silhouettes[gi] if silhouettes and gi < len(silhouettes) else None
        if sr is not None and sr.svg_d:
            paths.append(
                f'<path d="{sr.svg_d}" fill="{color}" fill-opacity="0.55" '
                f'stroke="{color}" stroke-width="0.5" stroke-opacity="0.9"/>'
            )
        else:
            paths.append(_polygon_to_svg_paths(g.polygon, fill=color, opacity=0.55))
        cx, cy = g.centroid
        labels.append(
            f'<text x="{cx:.1f}" y="{cy:.1f}" '
            f'text-anchor="middle" dominant-baseline="central" '
            f'font-family="monospace" font-size="{max(8, min(cw, ch) / 20):.1f}" '
            f'fill="white" font-weight="bold">G{gi}</text>'
        )
    content = "\n".join(p for p in paths if p) + "\n" + "\n".join(labels)
    return _svg_wrap(content, cw, ch)
