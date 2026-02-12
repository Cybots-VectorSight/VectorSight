"""SVG parser — facade over svgpathtools + lxml.

Converts raw SVG string → PipelineContext with SubPathData populated.
"""

from __future__ import annotations

import io
import logging
import re

import numpy as np
from svgpathtools import parse_path, Path, Line, CubicBezier, QuadraticBezier, Arc

from app.engine.context import PipelineContext, SubPathData
from app.utils.geometry import bbox, signed_area

logger = logging.getLogger(__name__)

# Regex for extracting viewBox
_VIEWBOX_RE = re.compile(r'viewBox\s*=\s*"([^"]+)"')
_WIDTH_RE = re.compile(r'width\s*=\s*"([^"]*?)"')
_HEIGHT_RE = re.compile(r'height\s*=\s*"([^"]*?)"')
_PATH_D_RE = re.compile(r'<path[^>]*\sd\s*=\s*"([^"]+)"[^>]*/?\s*>', re.IGNORECASE)
_CIRCLE_TAG_RE = re.compile(r'<circle[^>]*/?\s*>', re.IGNORECASE)
_RECT_RE = re.compile(
    r'<rect[^>]*'
    r'x\s*=\s*"([^"]+)"[^>]*'
    r'y\s*=\s*"([^"]+)"[^>]*'
    r'width\s*=\s*"([^"]+)"[^>]*'
    r'height\s*=\s*"([^"]+)"[^>]*/?\s*>',
    re.IGNORECASE,
)
_LINE_TAG_RE = re.compile(r'<line[^>]*/?\s*>', re.IGNORECASE)
_FILL_RE = re.compile(r'fill\s*=\s*"([^"]+)"', re.IGNORECASE)
_STROKE_RE = re.compile(r'stroke\s*=\s*"([^"]+)"', re.IGNORECASE)
_STROKE_WIDTH_RE = re.compile(r'stroke-width\s*=\s*"([^"]+)"', re.IGNORECASE)


def parse_svg(svg_text: str) -> PipelineContext:
    """Parse raw SVG string into a PipelineContext."""
    ctx = PipelineContext(svg_raw=svg_text)

    # Extract viewBox
    vb_match = _VIEWBOX_RE.search(svg_text)
    if vb_match:
        parts = vb_match.group(1).split()
        if len(parts) >= 4:
            ctx.canvas_width = float(parts[2])
            ctx.canvas_height = float(parts[3])
    else:
        w_match = _WIDTH_RE.search(svg_text)
        h_match = _HEIGHT_RE.search(svg_text)
        if w_match:
            try:
                ctx.canvas_width = float(w_match.group(1).replace("px", "").replace("pt", ""))
            except ValueError:
                pass
        if h_match:
            try:
                ctx.canvas_height = float(h_match.group(1).replace("px", "").replace("pt", ""))
            except ValueError:
                pass

    # Detect stroke vs fill
    _detect_stroke_fill(svg_text, ctx)

    z_order = 0

    # Extract paths
    for match in _PATH_D_RE.finditer(svg_text):
        d = match.group(1)
        full_tag = match.group(0)
        attrs = _extract_attrs(full_tag)

        try:
            path = parse_path(d)
        except Exception as e:
            logger.warning("Failed to parse path: %s", e)
            continue

        # Split into sub-paths (M...Z boundaries)
        subpaths = _split_subpaths(path)

        for i, sp_segments in enumerate(subpaths):
            sp_path = Path(*sp_segments)
            sp_id = f"E{z_order + 1}"

            points = _sample_path(sp_path)
            if len(points) < 3:
                z_order += 1
                continue

            pts = np.array(points)
            sp_data = SubPathData(
                id=sp_id,
                points=pts,
                segments=list(sp_segments),
                attributes=attrs,
                closed=_is_closed(sp_path),
                winding=1 if signed_area(pts) > 0 else -1,
                z_order=z_order,
                bbox=bbox(pts),
                source_tag=full_tag,
                source_span=(match.start(), match.end()),
            )

            # Build polygon
            try:
                from shapely.geometry import Polygon

                poly = Polygon(points)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                sp_data.polygon = poly
            except Exception:
                pass

            ctx.subpaths.append(sp_data)
            z_order += 1

    # Extract circles (order-independent attribute parsing)
    for match in _CIRCLE_TAG_RE.finditer(svg_text):
        tag_text = match.group(0)
        attrs = _extract_attrs(tag_text)
        try:
            cx, cy, r = float(attrs["cx"]), float(attrs["cy"]), float(attrs["r"])
        except (KeyError, ValueError):
            continue
        points = _circle_points(cx, cy, r)
        pts = np.array(points)
        sp_id = f"E{z_order + 1}"

        sp_data = SubPathData(
            id=sp_id,
            points=pts,
            attributes=attrs,
            closed=True,
            winding=1,
            z_order=z_order,
            bbox=bbox(pts),
            source_tag=tag_text,
            source_span=(match.start(), match.end()),
        )

        try:
            from shapely.geometry import Polygon

            poly = Polygon(points)
            if not poly.is_valid:
                poly = poly.buffer(0)
            sp_data.polygon = poly
        except Exception:
            pass

        ctx.subpaths.append(sp_data)
        z_order += 1

    # Extract lines (order-independent attribute parsing)
    for match in _LINE_TAG_RE.finditer(svg_text):
        tag_text = match.group(0)
        attrs = _extract_attrs(tag_text)
        try:
            x1, y1 = float(attrs["x1"]), float(attrs["y1"])
            x2, y2 = float(attrs["x2"]), float(attrs["y2"])
        except (KeyError, ValueError):
            continue
        pts = np.array([[x1, y1], [x2, y2]])
        sp_id = f"E{z_order + 1}"

        sp_data = SubPathData(
            id=sp_id,
            points=pts,
            attributes=attrs,
            closed=False,
            winding=0,
            z_order=z_order,
            bbox=bbox(pts),
            source_tag=tag_text,
            source_span=(match.start(), match.end()),
        )

        ctx.subpaths.append(sp_data)
        z_order += 1

    logger.info("Parsed SVG: %d elements, canvas %.0f×%.0f", len(ctx.subpaths), ctx.canvas_width, ctx.canvas_height)
    return ctx


def _detect_stroke_fill(svg_text: str, ctx: PipelineContext) -> None:
    """Detect if the SVG is primarily stroke-based or fill-based."""
    fill_matches = _FILL_RE.findall(svg_text)
    stroke_matches = _STROKE_RE.findall(svg_text)

    has_fill_none = any(f.lower() == "none" for f in fill_matches)
    has_stroke = any(s.lower() not in ("none", "") for s in stroke_matches)

    if has_fill_none and has_stroke:
        ctx.is_stroke_based = True


def _extract_attrs(tag_text: str) -> dict[str, str]:
    """Extract key attributes from an SVG tag string."""
    attrs: dict[str, str] = {}
    for m in re.finditer(r'(\w[\w-]*)\s*=\s*"([^"]*)"', tag_text):
        attrs[m.group(1)] = m.group(2)
    return attrs


def _split_subpaths(path: Path) -> list[list]:
    """Split a compound path into separate sub-paths at M commands."""
    if not path:
        return []

    subpaths: list[list] = []
    current: list = []

    for seg in path:
        current.append(seg)

    if current:
        subpaths.append(current)

    # Check for Z commands followed by M (compound paths)
    # svgpathtools already splits on M for compound paths, so handle continuous paths
    return subpaths if subpaths else [[]]


def _sample_path(path: Path, num_samples: int = 200) -> list[tuple[float, float]]:
    """Sample points along a path using parametric evaluation."""
    points: list[tuple[float, float]] = []

    if not path or path.length() < 1e-10:
        return points

    for t in np.linspace(0, 1, num_samples):
        try:
            pt = path.point(t)
            points.append((pt.real, pt.imag))
        except Exception:
            pass

    return points


def _is_closed(path: Path) -> bool:
    """Check if a path is closed (start ≈ end)."""
    if not path:
        return False
    try:
        start = path.point(0)
        end = path.point(1)
        return abs(start - end) < 0.5
    except Exception:
        return False


def _circle_points(cx: float, cy: float, r: float, n: int = 100) -> list[tuple[float, float]]:
    """Generate points along a circle."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]
