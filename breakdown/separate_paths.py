"""separate_paths.py -- Split SVG into individual subpaths, merge overlapping layers.

Step 1: Split compound paths at moveto discontinuities.
Step 2: Merge paths that overlap significantly (shadow/highlight layers of same feature).

No magic numbers. Overlap ratio is IoU (intersection over smaller area).
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path
from collections import defaultdict

import cairosvg
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid
import svgpathtools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
import matplotlib.colors as mcolors

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent

_BG = "#1a1a2e"
_TEXT = "#e0e0e0"
_GRID = "#2a2a4a"
_ACCENT = "#4ECDC4"

_PALETTE = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#F7DC6F", "#BB8FCE", "#85C1E9", "#F0B27A",
    "#AED6F1", "#A3E4D7", "#FAD7A0", "#D2B4DE", "#A9CCE3",
]


# == Step 1: Load + Split =====================================================

def _split_subpaths(path):
    """Split compound SVG path at moveto discontinuities."""
    subpaths = []
    current = []
    for seg in path:
        if current and abs(seg.start - current[-1].end) > 1e-6:
            subpaths.append(svgpathtools.Path(*current))
            current = []
        current.append(seg)
    if current:
        subpaths.append(svgpathtools.Path(*current))
    return subpaths


def _path_to_polygon(path):
    """Sample points along SVG path -> Shapely Polygon."""
    try:
        length = path.length()
        if length < 1e-6:
            return None
        n = max(20, min(2000, int(length)))
        pts = [(path.point(i / n).real, path.point(i / n).imag)
               for i in range(n)]
        if len(pts) < 3:
            return None
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = make_valid(poly)
        if poly.is_empty or poly.area < 1e-6:
            return None
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)
        elif poly.geom_type == "GeometryCollection":
            polys = [g for g in poly.geoms
                     if g.geom_type in ("Polygon", "MultiPolygon")]
            if not polys:
                return None
            poly = max(polys, key=lambda g: g.area)
            if poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda g: g.area)
        return poly
    except Exception:
        return None


def load_and_split(svg_path):
    """Load SVG, split compound paths into subpaths."""
    doc = svgpathtools.Document(str(svg_path))
    paths_t = doc.paths()
    _, attrs_list = svgpathtools.svg2paths(str(svg_path))

    elements = []
    n_compound = 0

    for i, path in enumerate(paths_t):
        fill = "#888888"
        if i < len(attrs_list):
            f = attrs_list[i].get("fill", "")
            if f and f.lower() not in ("none", "transparent", ""):
                fill = f

        subpaths = _split_subpaths(path)
        if len(subpaths) > 1:
            n_compound += 1

        for si, sp in enumerate(subpaths):
            label = f"{i}" if len(subpaths) == 1 else f"{i}.{si}"
            poly = _path_to_polygon(sp)
            if poly is not None:
                elements.append({
                    "label": label, "polygon": poly, "fill": fill,
                    "area": poly.area,
                    "centroid": (poly.centroid.x, poly.centroid.y),
                })

    return elements, n_compound, len(paths_t)


# == Step 2: Merge Overlapping Layers =========================================

def merge_overlapping(elements):
    """Merge paths that overlap significantly (shadow/highlight/outline layers).

    Two paths belong to the same feature if they're nearly the same shape
    at different depths (shadow, base, highlight).

    Metric: IoU (Intersection over Union). Only high when shapes are
    similar SIZE and POSITION. A tiny eye inside a huge face → IoU ≈ 0.
    A shadow layer behind a base layer → IoU ≈ 0.8+.

    Threshold: Q3 (75th percentile) of nonzero IoU values — shadow/highlight
    pairs sit in the upper quartile of the overlap distribution.

    Clustering: Complete-linkage — a candidate only joins a group if its
    MINIMUM IoU with every existing member exceeds the threshold. Prevents
    chain-merge snowball (A→B→C where A and C don't actually overlap).
    """
    n = len(elements)
    if n <= 1:
        return elements

    # Compute pairwise IoU (only for bbox-overlapping pairs)
    iou_matrix = {}  # (i, j) -> IoU, i < j
    nonzero_ious = []

    for i in range(n):
        pi = elements[i]["polygon"]
        bi = pi.bounds
        for j in range(i + 1, n):
            pj = elements[j]["polygon"]
            bj = pj.bounds
            # Quick bbox rejection
            if bi[2] < bj[0] or bj[2] < bi[0] or bi[3] < bj[1] or bj[3] < bi[1]:
                continue
            try:
                inter = pi.intersection(pj)
                if inter.is_empty:
                    continue
                inter_area = inter.area
                union_area = pi.area + pj.area - inter_area
                if union_area < 1e-6:
                    continue
                iou = inter_area / union_area
                iou_matrix[(i, j)] = iou
                if iou > 0:
                    nonzero_ious.append(iou)
            except Exception:
                continue

    if not nonzero_ious:
        return elements

    # Threshold: Q3 (75th percentile) of nonzero IoU values
    threshold = float(np.percentile(nonzero_ious, 75))
    n_above = sum(1 for r in nonzero_ious if r >= threshold)
    print(f"    Overlap pairs: {len(iou_matrix)}")
    print(f"    IoU distribution: min={min(nonzero_ious):.3f} "
          f"Q1={np.percentile(nonzero_ious, 25):.3f} "
          f"median={np.median(nonzero_ious):.3f} "
          f"Q3={threshold:.3f} "
          f"max={max(nonzero_ious):.3f}")
    print(f"    Threshold (Q3): {threshold:.3f}")
    print(f"    Above threshold: {n_above}")

    # Average-linkage clustering: merge if MEAN IoU across all cross-pairs
    # exceeds threshold. Balances between single-linkage (snowball) and
    # complete-linkage (too strict — one weak pair blocks everything).
    groups = [[i] for i in range(n)]

    def _get_iou(a, b):
        key = (min(a, b), max(a, b))
        return iou_matrix.get(key, 0.0)

    changed = True
    while changed:
        changed = False
        best_merge = None
        best_avg_iou = threshold

        for gi in range(len(groups)):
            for gj in range(gi + 1, len(groups)):
                # Average-linkage: mean IoU across all cross-pairs
                total_iou = 0.0
                n_pairs = 0
                for a in groups[gi]:
                    for b in groups[gj]:
                        total_iou += _get_iou(a, b)
                        n_pairs += 1
                avg_iou = total_iou / n_pairs if n_pairs > 0 else 0.0

                if avg_iou >= best_avg_iou:
                    best_avg_iou = avg_iou
                    best_merge = (gi, gj)

        if best_merge is not None:
            gi, gj = best_merge
            groups[gi].extend(groups[gj])
            groups.pop(gj)
            changed = True

    # Build merged features
    features = []
    for member_indices in groups:
        members = [elements[i] for i in member_indices]
        labels = [m["label"] for m in members]
        colors = list(set(m["fill"] for m in members))
        geom = unary_union([m["polygon"] for m in members])
        features.append({
            "labels": labels,
            "polygon": geom,
            "fills": colors,
            "area": geom.area,
            "centroid": (geom.centroid.x, geom.centroid.y),
            "n_layers": len(members),
        })

    features.sort(key=lambda f: f["area"], reverse=True)
    return features


# == Rendering ================================================================

def _extract_polygons(geom):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    if geom.geom_type == "GeometryCollection":
        result = []
        for g in geom.geoms:
            result.extend(_extract_polygons(g))
        return result
    return []


def _render_geom(ax, geom, **kw):
    for poly in _extract_polygons(geom):
        if poly.geom_type != "Polygon":
            continue
        ext = np.array(poly.exterior.coords)
        codes = ([MplPath.MOVETO]
                 + [MplPath.LINETO] * (len(ext) - 2)
                 + [MplPath.CLOSEPOLY])
        ax.add_patch(PathPatch(MplPath(ext, codes), **kw))


def _render_svg(ax, svg_path, cw, ch):
    png = cairosvg.svg2png(url=str(svg_path), output_width=512,
                           output_height=int(512 * ch / cw))
    img = Image.open(io.BytesIO(png)).convert("RGBA")
    ax.imshow(img, extent=[0, cw, ch, 0])
    ax.set_xlim(0, cw)
    ax.set_ylim(ch, 0)
    ax.set_aspect("equal")


# == Main =====================================================================

def main():
    svg_name = sys.argv[1] if len(sys.argv) > 1 else "faker.svg"
    svg_path = ROOT / "samples" / "test" / svg_name
    out_png = OUT_DIR / f"separate_{svg_path.stem}.png"

    print(f"Loading: {svg_path.name}")
    svg_text = svg_path.read_text()
    vb = re.search(r'viewBox="([^"]+)"', svg_text)
    cw, ch = (float(vb.group(1).split()[2]),
              float(vb.group(1).split()[3])) if vb else (256, 308)

    # Step 1: Load + split
    print("\n--- Step 1: Split Subpaths ---")
    elements, n_compound, n_orig = load_and_split(svg_path)
    print(f"  {n_orig} paths -> {len(elements)} subpaths "
          f"({n_compound} compound split)")

    # Step 2: Merge overlapping layers
    print("\n--- Step 2: Merge Overlapping Layers ---")
    features = merge_overlapping(elements)
    n_merged = len(elements) - len(features)
    print(f"  {len(elements)} subpaths -> {len(features)} features "
          f"({n_merged} merged as shadow/highlight layers)")

    for i, f in enumerate(features):
        labels = ",".join(f["labels"][:5])
        if len(f["labels"]) > 5:
            labels += f"...+{len(f['labels'])-5}"
        fills = ",".join(f["fills"][:3])
        if len(f["fills"]) > 3:
            fills += f"...+{len(f['fills'])-3}"
        print(f"    F{i:2d}: {f['n_layers']} layers  area={f['area']:8.0f}  "
              f"[{labels}]  {fills}")

    # -- Render --
    print("\nRendering...")
    n = len(features)
    show_n = min(n, 25)
    det_cols = min(show_n, 5)
    det_rows = max(1, (show_n + det_cols - 1) // det_cols)

    fig = plt.figure(figsize=(22, 5 + 4 * det_rows), facecolor=_BG)
    gs_top = fig.add_gridspec(1, 3, hspace=0.35, wspace=0.2,
                              left=0.03, right=0.97, top=0.90, bottom=0.52)
    gs_det = fig.add_gridspec(det_rows, det_cols, hspace=0.35, wspace=0.2,
                              left=0.03, right=0.97, top=0.48, bottom=0.02)

    fig.suptitle(
        f"Separated Features -- {svg_path.stem}: "
        f"{n_orig} paths -> {len(elements)} subpaths -> {n} features",
        color=_ACCENT, fontsize=13, fontweight="bold", y=0.96)

    # Top left: Original
    ax0 = fig.add_subplot(gs_top[0, 0])
    ax0.set_facecolor(_BG)
    _render_svg(ax0, svg_path, cw, ch)
    ax0.set_title("Original", color=_TEXT, fontsize=9)
    ax0.tick_params(colors=_GRID, labelsize=5)

    # Top middle: All features colored
    ax1 = fig.add_subplot(gs_top[0, 1])
    ax1.set_facecolor(_BG)
    for i, f in enumerate(features):
        c = _PALETTE[i % len(_PALETTE)]
        rgb = mcolors.to_rgb(c)
        _render_geom(ax1, f["polygon"],
                     facecolor=(*rgb, 0.3), edgecolor=(*rgb, 0.8),
                     linewidth=0.5)
    ax1.set_xlim(0, cw)
    ax1.set_ylim(ch, 0)
    ax1.set_aspect("equal")
    ax1.set_title(f"Features ({n})", color=_TEXT, fontsize=9)
    ax1.tick_params(colors=_GRID, labelsize=5)

    # Top right: Stats
    ax_s = fig.add_subplot(gs_top[0, 2])
    ax_s.set_facecolor(_BG)
    ax_s.axis("off")
    lines = [
        f"Separated Features: {svg_path.stem}",
        "-" * 50,
        f"Paths: {n_orig} -> Subpaths: {len(elements)} -> Features: {n}",
        f"Compound paths split: {n_compound}",
        f"Layers merged: {n_merged}",
        "-" * 50,
    ]
    for i, f in enumerate(features[:25]):
        labels = ",".join(f["labels"][:4])
        if len(f["labels"]) > 4:
            labels += "..."
        lines.append(f"F{i:2d}: {f['n_layers']}L  area={f['area']:7.0f}  [{labels}]")
    if n > 25:
        lines.append(f"  ... +{n-25} more")
    ax_s.text(0.02, 0.98, "\n".join(lines), transform=ax_s.transAxes,
              fontsize=5.5, fontfamily="monospace", color=_TEXT, va="top",
              bbox=dict(boxstyle="round,pad=0.4", facecolor=_GRID, alpha=0.5))

    # Detail panels
    for gi in range(min(show_n, det_rows * det_cols)):
        f = features[gi]
        row, col = gi // det_cols, gi % det_cols
        ax = fig.add_subplot(gs_det[row, col])
        ax.set_facecolor(_BG)
        c = _PALETTE[gi % len(_PALETTE)]
        rgb = mcolors.to_rgb(c)

        _render_svg(ax, svg_path, cw, ch)
        ax.imshow(np.full((10, 10, 4), [26, 26, 46, 200], dtype=np.uint8),
                  extent=[0, cw, ch, 0], aspect="auto", zorder=1)
        _render_geom(ax, f["polygon"],
                     facecolor=(*rgb, 0.5), edgecolor="white",
                     linewidth=1.0, zorder=2)

        ax.set_xlim(0, cw)
        ax.set_ylim(ch, 0)
        ax.set_aspect("equal")
        labels = ",".join(f["labels"][:3])
        if len(f["labels"]) > 3:
            labels += "..."
        ax.set_title(f"F{gi}: {f['n_layers']}L [{labels}]\n"
                     f"{','.join(f['fills'][:2])}",
                     color=c, fontsize=5, pad=2)
        ax.tick_params(colors=_GRID, labelsize=3)

    for pi in range(show_n, det_rows * det_cols):
        row, col = pi // det_cols, pi % det_cols
        fig.add_subplot(gs_det[row, col]).set_visible(False)

    fig.savefig(str(out_png), dpi=150, facecolor=_BG)
    plt.close(fig)
    print(f"\nSaved: {out_png}")
    print(f"\n{'=' * 50}")
    print(f"SUMMARY: {svg_path.stem}")
    print(f"  {n_orig} paths -> {len(elements)} subpaths -> {n} features")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
