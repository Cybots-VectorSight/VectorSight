"""08_grouping.py -- Visible-region contour grouping.

Algorithm (fully deterministic, zero parameters):
  1. Compute visible region per path: polygon[i] - union(everything above)
  2. Merge same-color adjacent visible regions into "visual patches"
  3. Build contour hierarchy (which patches sit inside which)
  4. Groups = branches of the contour tree

The key insight: SVG layers hide geometry underneath. By computing what's
actually VISIBLE, we get non-overlapping regions that represent the final
rendered image -- in vector space, no rasterization needed.
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
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from shapely.validation import make_valid
import svgpathtools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D
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


# == SVG Parsing ==============================================================

def _path_to_polygon(path):
    try:
        length = path.length()
        if length < 1e-6:
            return None
        n = max(20, min(2000, int(length)))
        pts = [(path.point(i / n).real, path.point(i / n).imag) for i in range(n)]
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
            polys = [g for g in poly.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
            if not polys:
                return None
            poly = max(polys, key=lambda g: g.area)
            if poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda g: g.area)
        return poly
    except Exception:
        return None


def _split_subpaths(path):
    """Split a compound SVG path into individual continuous subpaths."""
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


def _compound_path_to_geometry(path):
    """Convert a compound SVG path to proper geometry with holes.

    Strategy:
      1. Largest subpath polygon = outer boundary
      2. Smaller subpaths INSIDE outer = holes (let colors below show through)
      3. Smaller subpaths OUTSIDE outer = separate shapes (wand, diamonds, etc.)
      4. Returns MultiPolygon: [outer_with_holes, separate1, separate2, ...]
    """
    subpaths = _split_subpaths(path)
    if len(subpaths) <= 1:
        return _path_to_polygon(path)

    # Build polygon for each subpath
    sub_polys = []
    for sp in subpaths:
        poly = _path_to_polygon(sp)
        if poly is not None:
            sub_polys.append(poly)

    if not sub_polys:
        return None

    # Sort by area descending — largest is the outer boundary
    sub_polys.sort(key=lambda p: p.area, reverse=True)
    outer = sub_polys[0]

    holes = []
    separates = []

    for sp in sub_polys[1:]:
        try:
            # Check if this subpath is inside the outer boundary
            overlap = outer.intersection(sp).area
            # >50% inside outer = it's a hole
            if overlap > sp.area * 0.5:
                holes.append(sp)
            else:
                separates.append(sp)
        except Exception:
            separates.append(sp)

    # Cut holes from outer
    if holes:
        try:
            hole_union = unary_union(holes)
            outer = outer.difference(hole_union)
            if not outer.is_valid:
                outer = make_valid(outer)
        except Exception:
            pass

    # Combine outer (with holes) + separate shapes
    all_parts = _extract_polygons(outer) + separates
    all_parts = [p for p in all_parts if p.area > 1e-6]

    if not all_parts:
        return None
    if len(all_parts) == 1:
        return all_parts[0]
    from shapely.geometry import MultiPolygon
    return MultiPolygon(all_parts)


def load_elements(svg_path):
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

        # Handle compound paths (multiple subpaths) with proper holes
        subpaths = _split_subpaths(path)
        if len(subpaths) > 1:
            n_compound += 1
            poly = _compound_path_to_geometry(path)
        else:
            poly = _path_to_polygon(path)

        if poly is None:
            try:
                mid = path.point(0.5)
                centroid = (mid.real, mid.imag)
            except Exception:
                centroid = (0, 0)
            elements.append({"index": i, "polygon": None, "fill": fill,
                             "centroid": centroid, "area": 0,
                             "n_subpaths": len(subpaths)})
        else:
            elements.append({"index": i, "polygon": poly, "fill": fill,
                             "centroid": (poly.centroid.x, poly.centroid.y),
                             "area": poly.area,
                             "n_subpaths": len(subpaths)})

    print(f"  {len(paths_t)} paths ({n_compound} compound with holes)")
    return elements


# == Step 1: Visible Regions ==================================================

def compute_visible_regions(elements):
    """Compute what's actually visible per path.

    visible[i] = polygon[i] - union(all polygons painted after i)

    Iterates top-down for O(n) cumulative union (not O(n^2)).
    """
    n = len(elements)
    above_union = None

    for i in range(n - 1, -1, -1):
        poly = elements[i]["polygon"]
        if poly is None:
            elements[i]["visible_poly"] = None
            elements[i]["visible_area"] = 0.0
            continue

        if above_union is None:
            elements[i]["visible_poly"] = poly
            elements[i]["visible_area"] = poly.area
        else:
            try:
                visible = poly.difference(above_union)
                if visible.is_empty:
                    elements[i]["visible_poly"] = None
                    elements[i]["visible_area"] = 0.0
                else:
                    elements[i]["visible_poly"] = visible
                    elements[i]["visible_area"] = visible.area
            except Exception:
                elements[i]["visible_poly"] = poly
                elements[i]["visible_area"] = poly.area

        if above_union is None:
            above_union = poly
        else:
            try:
                above_union = above_union.union(poly)
                if not above_union.is_valid:
                    above_union = make_valid(above_union)
            except Exception:
                pass

    return elements


# == Step 2: Merge Same-Color Adjacent Regions ================================

def _extract_polygons(geom):
    """Get all Polygon objects from any geometry type."""
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


def merge_by_color(elements):
    """Merge visible regions with identical fill that are adjacent.

    Two same-color regions are adjacent if distance == 0 (shared boundary).
    Connected components via union-find. Fully deterministic.
    """
    color_groups = defaultdict(list)
    for el in elements:
        if el["visible_poly"] is not None and el["visible_area"] > 1e-6:
            color_groups[el["fill"]].append(el)

    patches = []

    for color, els in color_groups.items():
        if len(els) == 1:
            patches.append({
                "color": color,
                "geometry": els[0]["visible_poly"],
                "source_indices": [els[0]["index"]],
                "area": els[0]["visible_area"],
            })
            continue

        # Union-find for connected components
        idx_list = [el["index"] for el in els]
        parent = {idx: idx for idx in idx_list}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for ai in range(len(els)):
            for bi in range(ai + 1, len(els)):
                try:
                    if els[ai]["visible_poly"].distance(els[bi]["visible_poly"]) < 1e-6:
                        union(els[ai]["index"], els[bi]["index"])
                except Exception:
                    pass

        components = defaultdict(list)
        for el in els:
            root = find(el["index"])
            components[root].append(el)

        for comp_els in components.values():
            geom = unary_union([el["visible_poly"] for el in comp_els])
            patches.append({
                "color": color,
                "geometry": geom,
                "source_indices": sorted(el["index"] for el in comp_els),
                "area": sum(el["visible_area"] for el in comp_els),
            })

    patches.sort(key=lambda p: p["area"], reverse=True)
    return patches


# == Step 3: Contour Hierarchy ================================================

def build_contour_hierarchy(patches):
    """Build parent-child tree from visual patches.

    Patch B is inside patch A if B's representative point falls within
    A's filled exterior (outer ring, ignoring holes where children sit).

    Immediate parent = smallest enclosing patch.
    """
    n = len(patches)

    # Filled exteriors for containment testing
    filled = []
    for p in patches:
        polys = _extract_polygons(p["geometry"])
        if polys:
            fexts = [Polygon(pp.exterior) for pp in polys]
            filled.append(unary_union(fexts))
        else:
            filled.append(None)

    # Representative points
    rep_pts = []
    for p in patches:
        try:
            rep_pts.append(p["geometry"].representative_point())
        except Exception:
            polys = _extract_polygons(p["geometry"])
            rep_pts.append(polys[0].centroid if polys else Point(0, 0))

    parent_of = [None] * n
    for i in range(n):
        best_parent = None
        best_area = float("inf")
        for j in range(n):
            if i == j or filled[j] is None:
                continue
            if patches[j]["area"] <= patches[i]["area"]:
                continue
            try:
                if filled[j].contains(rep_pts[i]):
                    if patches[j]["area"] < best_area:
                        best_parent = j
                        best_area = patches[j]["area"]
            except Exception:
                continue
        parent_of[i] = best_parent

    children = defaultdict(list)
    for i, p in enumerate(parent_of):
        if p is not None:
            children[p].append(i)

    roots = [i for i in range(n) if parent_of[i] is None]
    return parent_of, children, roots


# == Step 4: Hybrid Color Grouping ============================================

def _hex_to_rgb(h):
    """Hex string to (R,G,B) tuple. None for gradients/invalid."""
    if not h or h.startswith("url("):
        return None
    h = h.lstrip("#")
    if len(h) == 3:
        h = h[0]*2 + h[1]*2 + h[2]*2
    if len(h) != 6:
        return None
    try:
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return None


def _sample_gradient_colors(svg_path, patches, cw, ch):
    """Pixel-sample colors only for gradient-filled patches.

    For solid fills, _hex_to_rgb gives exact colors.
    For gradients (url(...)), we rasterize and sample at the centroid.
    Returns dict: patch_index -> (R,G,B) for gradient patches only.
    """
    gradient_indices = [i for i, p in enumerate(patches)
                        if _hex_to_rgb(p["color"]) is None]
    if not gradient_indices:
        return {}

    render_w = 512
    render_h = int(512 * ch / cw)
    png = cairosvg.svg2png(url=str(svg_path), output_width=render_w,
                           output_height=render_h)
    img = np.array(Image.open(io.BytesIO(png)).convert("RGB"))
    h_px, w_px = img.shape[:2]

    sampled = {}
    for i in gradient_indices:
        try:
            pt = patches[i]["geometry"].representative_point()
            px = int(pt.x / cw * w_px)
            py = int(pt.y / ch * h_px)
            px = max(0, min(w_px - 1, px))
            py = max(0, min(h_px - 1, py))
            sampled[i] = tuple(int(c) for c in img[py, px])
        except Exception:
            sampled[i] = (128, 128, 128)
    return sampled


def _get_rgb(patches, gradient_colors, i):
    """Get (R,G,B) for patch i: exact hex or pixel-sampled for gradients."""
    if i in gradient_colors:
        return gradient_colors[i]
    return _hex_to_rgb(patches[i]["color"])


def _rgb_dist(c1, c2):
    """Euclidean RGB distance between two (R,G,B) tuples."""
    return float(np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2))))


def _is_related(parent_of, i, j):
    """True if i is ancestor of j or j is ancestor of i."""
    curr = j
    while parent_of[curr] is not None:
        if parent_of[curr] == i:
            return True
        curr = parent_of[curr]
    curr = i
    while parent_of[curr] is not None:
        if parent_of[curr] == j:
            return True
        curr = parent_of[curr]
    return False


def group_by_color_adjacency(patches, parent_of, gradient_colors):
    """Group patches by color similarity + spatial adjacency.

    Merge rule (ALL three must hold):
      1. Patches share a boundary (geometrically adjacent)
      2. RGB distance < median of adjacent pair distances
      3. Neither patch is ancestor/descendant of the other

    Hybrid color: exact hex for solid fills, pixel-sampled for gradients.
    Every pair gets a computable distance — no more None gaps.
    """
    n = len(patches)
    if n == 0:
        return []

    # Collect all adjacent pairs with color distances
    adj_pairs = []
    all_dists = []

    for i in range(n):
        gi = patches[i]["geometry"]
        if gi is None or gi.is_empty:
            continue
        rgb_i = _get_rgb(patches, gradient_colors, i)
        if rgb_i is None:
            continue
        for j in range(i + 1, n):
            gj = patches[j]["geometry"]
            if gj is None or gj.is_empty:
                continue
            try:
                if gi.distance(gj) < 1e-6:
                    rgb_j = _get_rgb(patches, gradient_colors, j)
                    if rgb_j is None:
                        continue
                    d = _rgb_dist(rgb_i, rgb_j)
                    adj_pairs.append((i, j, d))
                    all_dists.append(d)
            except Exception:
                pass

    # Derive threshold: median of adjacent color distances
    if len(all_dists) >= 2:
        threshold = float(np.median(all_dists))
    else:
        threshold = 0.0

    print(f"    Adjacent pairs: {len(adj_pairs)}")
    if all_dists:
        print(f"    Color distance threshold (median): {threshold:.1f}")
        print(f"    Distance range: {min(all_dists):.1f} - {max(all_dists):.1f}")
        print(f"    Below threshold: "
              f"{sum(1 for d in all_dists if d < threshold)}/{len(all_dists)}")

    # Union-find
    uf = list(range(n))

    def find(x):
        while uf[x] != x:
            uf[x] = uf[uf[x]]
            x = uf[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            uf[ra] = rb

    for i, j, d in adj_pairs:
        if d < threshold and not _is_related(parent_of, i, j):
            union(i, j)

    # Build groups
    components = defaultdict(list)
    for i in range(n):
        components[find(i)].append(i)

    groups = []
    for comp in components.values():
        src = []
        colors = set()
        for pi in comp:
            src.extend(patches[pi]["source_indices"])
            colors.add(patches[pi]["color"])
        groups.append({
            "patch_indices": comp,
            "source_indices": sorted(set(src)),
            "area": sum(patches[pi]["area"] for pi in comp),
            "n_patches": len(comp),
            "colors": colors,
        })

    groups.sort(key=lambda g: g["area"], reverse=True)
    return groups


# == Step 5: Absorb Tiny Groups ==============================================

def absorb_tiny_groups(groups, patches):
    """Merge tiny isolated groups into their nearest larger neighbor.

    Threshold: groups with area < median_group_area are candidates.
    Each tiny group merges into the nearest group (by centroid distance)
    that is larger than it. This is a cleanup pass, not the main grouping.
    """
    if len(groups) <= 2:
        return groups

    areas = [g["area"] for g in groups]
    median_area = float(np.median(areas))

    # Compute centroid per group
    centroids = []
    for g in groups:
        all_geoms = [patches[pi]["geometry"] for pi in g["patch_indices"]
                     if patches[pi]["geometry"] is not None]
        if all_geoms:
            u = unary_union(all_geoms)
            pt = u.representative_point()
            centroids.append((pt.x, pt.y))
        else:
            centroids.append((0, 0))

    # Find tiny groups and their nearest larger neighbor
    absorbed = {}  # tiny_idx -> target_idx
    for i, g in enumerate(groups):
        if g["area"] >= median_area:
            continue
        best_j = None
        best_dist = float("inf")
        for j, g2 in enumerate(groups):
            if j == i or g2["area"] <= g["area"]:
                continue
            d = np.sqrt((centroids[i][0] - centroids[j][0]) ** 2 +
                        (centroids[i][1] - centroids[j][1]) ** 2)
            if d < best_dist:
                best_dist = d
                best_j = j
        if best_j is not None:
            absorbed[i] = best_j

    if not absorbed:
        return groups

    # Resolve chains: if A->B and B->C, then A->C
    def resolve(idx):
        visited = set()
        while idx in absorbed and idx not in visited:
            visited.add(idx)
            idx = absorbed[idx]
        return idx

    for k in list(absorbed.keys()):
        absorbed[k] = resolve(k)

    # Merge
    merged = defaultdict(lambda: {"patch_indices": [], "source_indices": [],
                                   "area": 0, "n_patches": 0, "colors": set()})
    for i, g in enumerate(groups):
        target = absorbed.get(i, i)
        merged[target]["patch_indices"].extend(g["patch_indices"])
        merged[target]["source_indices"].extend(g["source_indices"])
        merged[target]["area"] += g["area"]
        merged[target]["n_patches"] += g["n_patches"]
        merged[target]["colors"].update(g["colors"])

    result = []
    for g in merged.values():
        g["source_indices"] = sorted(set(g["source_indices"]))
        result.append(g)
    result.sort(key=lambda g: g["area"], reverse=True)

    print(f"    Absorbed {len(absorbed)} tiny groups "
          f"(area < median {median_area:.0f})")
    print(f"    {len(groups)} -> {len(result)} groups")
    return result


# == Visualization ============================================================

def _poly_to_patch(poly, **kw):
    if poly is None or poly.geom_type != "Polygon":
        return None
    ext = np.array(poly.exterior.coords)
    codes = ([MplPath.MOVETO]
             + [MplPath.LINETO] * (len(ext) - 2)
             + [MplPath.CLOSEPOLY])
    return PathPatch(MplPath(ext, codes), **kw)


def _render_geom(ax, geom, **kw):
    for poly in _extract_polygons(geom):
        p = _poly_to_patch(poly, **kw)
        if p:
            ax.add_patch(p)


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
    out_png = OUT_DIR / f"08_grouping_{svg_path.stem}.png"

    print(f"Loading: {svg_path.name}")
    svg_text = svg_path.read_text()
    vb = re.search(r'viewBox="([^"]+)"', svg_text)
    cw, ch = (float(vb.group(1).split()[2]),
              float(vb.group(1).split()[3])) if vb else (256, 308)

    # -- Step 1: Parse + visible regions --
    print("\n--- Step 1: Visible Regions ---")
    elements = load_elements(svg_path)
    n_total = len(elements)
    elements = compute_visible_regions(elements)

    vis_count = sum(1 for el in elements if el["visible_area"] > 1e-6)
    hid_count = sum(1 for el in elements
                    if el["polygon"] is not None and el["visible_area"] <= 1e-6)
    full_area = sum(el["area"] for el in elements)
    vis_area = sum(el["visible_area"] for el in elements)
    pct = 100 * vis_area / full_area if full_area else 0
    print(f"  Visible: {vis_count} paths")
    print(f"  Fully hidden: {hid_count} paths (dropped)")
    print(f"  Area: {full_area:.0f} full -> {vis_area:.0f} visible ({pct:.0f}%)")

    # -- Step 2: Explode COMPOUND visible regions only --
    # Only compound paths (wand + body + hat packed in one <path>) get split.
    # Simple paths stay as one piece even if their visible region fragments.
    print("\n--- Step 2: Explode Compound Paths ---")
    patches = []
    n_exploded = 0
    for el in elements:
        if el["visible_poly"] is None or el["visible_area"] <= 1e-6:
            continue
        if el["n_subpaths"] > 1:
            # Compound path: split into separate visual pieces
            pieces = _extract_polygons(el["visible_poly"])
            pieces = [p for p in pieces if p.area > 1e-6]
            if pieces:
                n_exploded += 1
                for piece in pieces:
                    patches.append({
                        "color": el["fill"],
                        "geometry": piece,
                        "source_indices": [el["index"]],
                        "area": piece.area,
                        "compound": True,
                    })
            else:
                patches.append({
                    "color": el["fill"],
                    "geometry": el["visible_poly"],
                    "source_indices": [el["index"]],
                    "area": el["visible_area"],
                    "compound": True,
                })
        else:
            # Simple path: keep as one piece
            patches.append({
                "color": el["fill"],
                "geometry": el["visible_poly"],
                "source_indices": [el["index"]],
                "area": el["visible_area"],
                "compound": False,
            })
    patches.sort(key=lambda p: p["area"], reverse=True)
    print(f"  {vis_count} visible paths -> {len(patches)} patches "
          f"({n_exploded} compound paths exploded)")
    for pi, p in enumerate(patches[:30]):
        tag = " *" if p["compound"] else ""
        print(f"    P{pi:2d}: area={p['area']:7.0f}  {p['color']:>15}  "
              f"path={p['source_indices']}{tag}")
    if len(patches) > 30:
        print(f"    ... +{len(patches)-30} more")

    # -- Step 3: Contour hierarchy --
    print("\n--- Step 3: Contour Hierarchy ---")
    parent_of, children_map, roots = build_contour_hierarchy(patches)
    print(f"  {len(roots)} root patches")

    def print_tree(idx, depth=0):
        p = patches[idx]
        indent = "    " + "  " * depth
        srcs = str(p["source_indices"][:4])
        tail = "..." if len(p["source_indices"]) > 4 else ""
        print(f"{indent}P{idx}: {p['color']:>15}  area={p['area']:7.0f}  "
              f"paths={srcs}{tail}")
        for child in children_map.get(idx, []):
            print_tree(child, depth + 1)

    for root in roots:
        print_tree(root)

    # -- Step 4: Patches = Groups (no merging) --
    # Each visual patch from color merge IS a feature.
    # The contour tree gives us parent-child relationships.
    print("\n--- Step 4: Patches as Groups ---")
    groups = []
    for pi, p in enumerate(patches):
        parent = parent_of[pi]
        kids = children_map.get(pi, [])
        groups.append({
            "patch_indices": [pi],
            "source_indices": p["source_indices"],
            "area": p["area"],
            "n_patches": 1,
            "colors": {p["color"]},
            "parent": parent,
            "children": kids,
            "depth": 0,
        })
    # Compute depth in contour tree
    for gi, g in enumerate(groups):
        d = 0
        curr = gi
        while parent_of[curr] is not None:
            d += 1
            curr = parent_of[curr]
        g["depth"] = d

    print(f"  {len(groups)} groups (1 per patch)")
    for gi, g in enumerate(groups):
        p = patches[gi]
        depth_str = "  " * g["depth"]
        parent_str = f"<-P{g['parent']}" if g['parent'] is not None else "(root)"
        kids_str = f" ->P{g['children']}" if g['children'] else ""
        print(f"    {depth_str}G{gi}: area={g['area']:7.0f}  "
              f"{p['color']:>15}  paths={p['source_indices']}  "
              f"{parent_str}{kids_str}")

    # == Visualization =========================================================
    print("\nRendering...")

    ng = len(groups)
    show_n = min(ng, 25)  # show top 25 by area in detail panels
    det_cols = min(show_n, 5)
    det_rows = max(1, (show_n + det_cols - 1) // det_cols)

    fig = plt.figure(figsize=(22, 5 + 4 * det_rows), facecolor=_BG)
    gs_top = fig.add_gridspec(1, 3, hspace=0.35, wspace=0.2,
                              left=0.03, right=0.97, top=0.90, bottom=0.52)
    gs_det = fig.add_gridspec(det_rows, det_cols, hspace=0.35, wspace=0.2,
                              left=0.03, right=0.97, top=0.48, bottom=0.02)

    fig.suptitle(
        f"Visible Regions + Color Grouping -- {svg_path.stem} "
        f"-> {ng} Groups",
        color=_ACCENT, fontsize=14, fontweight="bold", y=0.96)

    # -- Top left: Original --
    ax0 = fig.add_subplot(gs_top[0, 0])
    ax0.set_facecolor(_BG)
    _render_svg(ax0, svg_path, cw, ch)
    ax0.set_title(f"Original ({n_total} paths)", color=_TEXT, fontsize=9)
    ax0.tick_params(colors=_GRID, labelsize=5)

    # -- Top middle: All groups colored --
    ax1 = fig.add_subplot(gs_top[0, 1])
    ax1.set_facecolor(_BG)
    for gi, g in enumerate(groups):
        c = _PALETTE[gi % len(_PALETTE)]
        rgb = mcolors.to_rgb(c)
        for pi in g["patch_indices"]:
            p = patches[pi]
            try:
                _render_geom(ax1, p["geometry"],
                             facecolor=(*rgb, 0.35),
                             edgecolor=(*rgb, 0.9), linewidth=0.6)
            except Exception:
                pass
        # Label at group centroid
        all_geoms = [patches[pi]["geometry"] for pi in g["patch_indices"]
                     if patches[pi]["geometry"] is not None]
        if all_geoms:
            try:
                u = unary_union(all_geoms)
                pt = u.representative_point()
                ax1.text(pt.x, pt.y, f"G{gi}", fontsize=5, ha="center",
                         va="center", color=c, fontweight="bold",
                         path_effects=[pe.withStroke(linewidth=2,
                                                     foreground="black")])
            except Exception:
                pass
    ax1.set_xlim(0, cw)
    ax1.set_ylim(ch, 0)
    ax1.set_aspect("equal")
    ax1.set_title(f"Color-Similarity Groups ({ng})", color=_TEXT, fontsize=9)
    ax1.tick_params(colors=_GRID, labelsize=5)

    # Legend
    handles = [Line2D([0], [0], color=_PALETTE[gi % len(_PALETTE)],
                      linewidth=4, label=f"G{gi}({len(g['source_indices'])})")
               for gi, g in enumerate(groups[:12])]
    ax1.legend(handles=handles, fontsize=5, facecolor=_BG, edgecolor=_GRID,
               labelcolor=_TEXT, loc="upper left", handlelength=1,
               handletextpad=0.3, ncol=2)

    # -- Top right: Stats --
    ax_s = fig.add_subplot(gs_top[0, 2])
    ax_s.set_facecolor(_BG)
    ax_s.axis("off")
    lines = [
        f"Color-Similarity Grouping: {svg_path.stem}",
        "-" * 55,
        f"Paths: {n_total} total, {vis_count} visible, {hid_count} hidden",
        f"Patches: {len(patches)}  ->  Groups: {ng}",
        f"Contour tree roots: {len(roots)}",
        "-" * 55,
    ]
    for gi, g in enumerate(groups):
        cols = sorted(g["colors"])[:2]
        col_s = ", ".join(cols) + ("..." if len(g["colors"]) > 2 else "")
        lines.append(f"G{gi:2d}: {len(g['source_indices']):2d}p  "
                     f"area={g['area']:7.0f}  {col_s}")
    lines.extend([
        "-" * 55,
        "visible -> color merge -> contour tree",
        "Each patch = one group (no merging)",
        "Tree depth shows nesting",
    ])
    ax_s.text(0.02, 0.98, "\n".join(lines), transform=ax_s.transAxes,
              fontsize=5.5, fontfamily="monospace", color=_TEXT, va="top",
              bbox=dict(boxstyle="round,pad=0.4", facecolor=_GRID, alpha=0.5))

    # -- Detail panels: one per group (top N by area) --
    for gi in range(min(show_n, det_rows * det_cols)):
        g = groups[gi]
        row, col = gi // det_cols, gi % det_cols
        ax = fig.add_subplot(gs_det[row, col])
        ax.set_facecolor(_BG)
        c = _PALETTE[gi % len(_PALETTE)]
        rgb = mcolors.to_rgb(c)

        for pi in g["patch_indices"]:
            p = patches[pi]
            try:
                _render_geom(ax, p["geometry"],
                             facecolor=(*rgb, 0.3),
                             edgecolor=(*rgb, 0.9), linewidth=0.8)
            except Exception:
                pass
            # Path index labels
            for si in p["source_indices"]:
                el = elements[si]
                if el["visible_area"] > 1e-6:
                    cx, cy = el["centroid"]
                    ax.text(cx, cy, str(si), fontsize=3, ha="center",
                            va="center", color="white", fontweight="bold",
                            path_effects=[pe.withStroke(linewidth=1,
                                                        foreground=c)])

        ax.set_xlim(0, cw)
        ax.set_ylim(ch, 0)
        ax.set_aspect("equal")
        short = ",".join(str(i) for i in g["source_indices"][:8])
        if len(g["source_indices"]) > 8:
            short += f"...+{len(g['source_indices']) - 8}"
        cols = sorted(g["colors"])[:2]
        col_s = ", ".join(cols) + ("..." if len(g["colors"]) > 2 else "")
        ax.set_title(f"G{gi}: {len(g['source_indices'])}p  "
                     f"{g['n_patches']} patches\n"
                     f"[{short}]\n{col_s}",
                     color=c, fontsize=5, pad=2)
        ax.tick_params(colors=_GRID, labelsize=3)

    for pi in range(show_n, det_rows * det_cols):
        row, col = pi // det_cols, pi % det_cols
        fig.add_subplot(gs_det[row, col]).set_visible(False)

    fig.savefig(str(out_png), dpi=150, facecolor=_BG)
    plt.close(fig)
    print(f"\nSaved: {out_png}")

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {svg_path.stem} -> {ng} groups")
    print(f"  visible -> color merge -> contour tree")
    print(f"  each patch = one group")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
