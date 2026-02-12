"""T1.25 — Color Analysis.

Extract fill/stroke colors from SVG source tags. Group elements by color similarity.
Identify dominant palette, color contrast between contained elements, and visual salience.
"""

from __future__ import annotations

import re
from collections import Counter

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform

# Regex patterns for extracting color attributes
_FILL_RE = re.compile(r'fill\s*=\s*"([^"]+)"')
_STROKE_RE = re.compile(r'stroke\s*=\s*"([^"]+)"')
_OPACITY_RE = re.compile(r'(?:fill-)?opacity\s*=\s*"([^"]+)"')
_STYLE_FILL_RE = re.compile(r'fill\s*:\s*([^;"\s]+)')
_STYLE_STROKE_RE = re.compile(r'stroke\s*:\s*([^;"\s]+)')

# Munsell Value scale mapped to BT.601 [0,1]:
_LIGHTNESS_BLACK = 0.15     # Munsell Value <= 1.5
_LIGHTNESS_WHITE = 0.85     # Munsell Value >= 8.5
_LIGHTNESS_DARK = 0.35      # Munsell Value <= 3.5
_LIGHTNESS_LIGHT = 0.65     # Munsell Value >= 6.5
# HSV saturation: achromatic below Munsell Chroma 1.
_SAT_ACHROMATIC = 0.15      # Below this, hue is indiscriminable
# W3C WCAG: saturation > 0.4 = "colorful" for accessibility.
_SAT_COLORFUL = 0.4
# Yellow in sRGB: both R,G channels > 78% (200/255).
_YELLOW_CHANNEL_MIN = 200   # = floor(0.784 * 255)
# Weber contrast JND ~= 0.15-0.20. Report contrasts above 0.20.
_CONTRAST_REPORT_MIN = 0.20

# Named colors to hex
_NAMED_COLORS = {
    "black": "#000000", "white": "#ffffff", "red": "#ff0000",
    "green": "#008000", "blue": "#0000ff", "yellow": "#ffff00",
    "none": None, "transparent": None, "currentColor": "#000000",
}


def _parse_hex(color: str) -> tuple[int, int, int] | None:
    """Parse a hex color string to (r, g, b)."""
    if not color:
        return None
    color = color.strip().lower()
    if color in _NAMED_COLORS:
        mapped = _NAMED_COLORS[color]
        if mapped is None:
            return None
        color = mapped
    if not color.startswith("#"):
        return None
    color = color[1:]
    if len(color) == 3:
        color = color[0]*2 + color[1]*2 + color[2]*2
    if len(color) != 6:
        return None
    try:
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
        return (r, g, b)
    except ValueError:
        return None


def _lightness(rgb: tuple[int, int, int]) -> float:
    """Perceived lightness 0-1 (ITU-R BT.601)."""
    return (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255.0


def _color_label(rgb: tuple[int, int, int]) -> str:
    """Classify color into a human-readable label."""
    r, g, b = rgb
    lightness = _lightness(rgb)
    if lightness < _LIGHTNESS_BLACK:
        return "black"
    if lightness > _LIGHTNESS_WHITE:
        return "white"
    if lightness < _LIGHTNESS_DARK:
        return "dark"
    if lightness > _LIGHTNESS_LIGHT:
        return "light"
    # Check saturation
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    sat = (max_c - min_c) / max_c if max_c > 0 else 0
    if sat < _SAT_ACHROMATIC:
        return "gray"
    # Determine hue
    if r > g and r > b:
        return "red" if sat > _SAT_COLORFUL else "warm-gray"
    if g > r and g > b:
        return "green"
    if b > r and b > g:
        return "blue"
    if r > _YELLOW_CHANNEL_MIN and g > _YELLOW_CHANNEL_MIN:
        return "yellow"
    return "mid-tone"


def _extract_color(tag: str, attr_re: re.Pattern, style_re: re.Pattern) -> str | None:
    """Extract color from a source tag, checking both attribute and style."""
    m = attr_re.search(tag)
    if m:
        return m.group(1).strip()
    m = style_re.search(tag)
    if m:
        return m.group(1).strip()
    return None


@transform(
    id="T1.25",
    layer=Layer.SHAPE_ANALYSIS,
    dependencies=["T0.04"],
    description="Extract fill/stroke colors and compute color groups",
)
def color_analysis(ctx: PipelineContext) -> None:
    palette: Counter[str] = Counter()
    color_groups: dict[str, list[str]] = {}  # label → [element_ids]

    for sp in ctx.subpaths:
        tag = sp.source_tag or ""
        fill_str = _extract_color(tag, _FILL_RE, _STYLE_FILL_RE)
        stroke_str = _extract_color(tag, _STROKE_RE, _STYLE_STROKE_RE)

        fill_rgb = _parse_hex(fill_str) if fill_str else None
        stroke_rgb = _parse_hex(stroke_str) if stroke_str else None

        # Store raw colors
        sp.features["fill_color"] = fill_str or ""
        sp.features["stroke_color"] = stroke_str or ""
        sp.features["fill_rgb"] = fill_rgb
        sp.features["stroke_rgb"] = stroke_rgb

        # Determine dominant color (fill takes priority for fill-based)
        dominant = fill_rgb or stroke_rgb
        if dominant:
            label = _color_label(dominant)
            sp.features["color_label"] = label
            lightness = _lightness(dominant)
            sp.features["color_lightness"] = round(lightness, 2)
            hex_str = fill_str or stroke_str or ""
            palette[hex_str.lower()] += 1
            color_groups.setdefault(label, []).append(sp.id)
        else:
            sp.features["color_label"] = "none"
            sp.features["color_lightness"] = 0.0

    # Compute contrast between contained elements
    containment_matrix = ctx.containment_matrix
    if containment_matrix is not None:
        n = len(ctx.subpaths)
        for i in range(n):
            parent = ctx.subpaths[i]
            parent_light = parent.features.get("color_lightness", 0.5)
            children_contrasts = []
            for j in range(n):
                if containment_matrix[i][j]:
                    child = ctx.subpaths[j]
                    child_light = child.features.get("color_lightness", 0.5)
                    contrast = abs(parent_light - child_light)
                    if contrast > _CONTRAST_REPORT_MIN:
                        children_contrasts.append((ctx.subpaths[j].id, round(contrast, 2)))
            if children_contrasts:
                parent.features["color_contrasts"] = children_contrasts

    # Store global color data on first subpath for enrichment formatter
    if ctx.subpaths:
        # Top colors by frequency
        top_colors = palette.most_common(8)
        ctx.subpaths[0].features["palette"] = top_colors
        ctx.subpaths[0].features["color_groups"] = color_groups
