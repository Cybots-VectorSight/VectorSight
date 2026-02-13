"""Path separation -- split SVG into individual subpaths, merge overlapping layers.

Step 1: Split compound paths at moveto discontinuities.
Step 2: Merge paths that overlap significantly (shadow/highlight layers of same feature).
Step 3: Group features by containment tree + adjacency.

No magic numbers. Overlap ratio is Ochiai coefficient.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from io import StringIO

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid
import svgpathtools


@dataclass
class GroupData:
    """A merged+grouped visual feature from the SVG."""

    labels: list[str] = field(default_factory=list)
    polygon: Polygon | None = None
    fills: list[str] = field(default_factory=list)
    area: float = 0.0
    centroid: tuple[float, float] = (0.0, 0.0)
    n_layers: int = 1
    n_merged: int = 1


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
        pts = [
            (path.point(i / n).real, path.point(i / n).imag) for i in range(n)
        ]
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
            polys = [
                g
                for g in poly.geoms
                if g.geom_type in ("Polygon", "MultiPolygon")
            ]
            if not polys:
                return None
            poly = max(polys, key=lambda g: g.area)
            if poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda g: g.area)
        return poly
    except Exception:
        return None


def load_and_split(svg_text: str) -> tuple[list[dict], int, int]:
    """Parse SVG text, split compound paths into subpaths.

    Returns (elements, n_compound, n_original_paths).
    Each element is a dict with: label, polygon, fill, area, centroid.
    """
    import logging
    _logger = logging.getLogger(__name__)

    # Parse attributes list (fill colors etc.)
    try:
        _, attrs_list = svgpathtools.svg2paths(StringIO(svg_text))
    except Exception as e:
        _logger.warning("svg2paths failed: %s — trying Document-only parse", e)
        attrs_list = []

    # Parse paths via Document for consistent handling
    try:
        doc = svgpathtools.Document(StringIO(svg_text))
        paths_t = doc.paths()
    except Exception as e:
        _logger.error("SVG Document parse failed: %s", e)
        # Return empty — pipeline will report 0 elements but won't crash
        return [], 0, 0

    elements = []
    n_compound = 0

    for i, path in enumerate(paths_t):
        fill = "#888888"
        if i < len(attrs_list):
            f = attrs_list[i].get("fill", "")
            if f and f.lower() not in ("none", "transparent", ""):
                fill = f

        try:
            subpaths = _split_subpaths(path)
        except Exception as e:
            _logger.warning("split_subpaths failed for path %d: %s", i, e)
            continue

        if len(subpaths) > 1:
            n_compound += 1

        for si, sp in enumerate(subpaths):
            label = f"{i}" if len(subpaths) == 1 else f"{i}.{si}"
            poly = _path_to_polygon(sp)
            if poly is not None:
                elements.append(
                    {
                        "label": label,
                        "polygon": poly,
                        "fill": fill,
                        "area": poly.area,
                        "centroid": (poly.centroid.x, poly.centroid.y),
                    }
                )

    return elements, n_compound, len(paths_t)


def extract_viewbox(svg_text: str) -> tuple[float, float]:
    """Extract canvas width and height from SVG viewBox or width/height attrs.

    Returns (canvas_width, canvas_height). Falls back to W3C defaults (300x150).
    """
    vb = re.search(r'viewBox="([^"]+)"', svg_text)
    if vb:
        parts = vb.group(1).split()
        if len(parts) >= 4:
            return float(parts[2]), float(parts[3])

    # Fallback: try width/height attributes
    w_match = re.search(r'width="([^"]+)"', svg_text)
    h_match = re.search(r'height="([^"]+)"', svg_text)
    if w_match and h_match:
        try:
            w = float(re.sub(r"[^0-9.]", "", w_match.group(1)))
            h = float(re.sub(r"[^0-9.]", "", h_match.group(1)))
            if w > 0 and h > 0:
                return w, h
        except ValueError:
            pass

    # W3C CSS 2.1 default
    return 300.0, 150.0


# == Step 2: Merge Overlapping Layers =========================================


def merge_overlapping(elements: list[dict]) -> list[dict]:
    """Merge paths that overlap significantly (shadow/highlight/outline layers).

    Metric: Ochiai coefficient -- intersection / sqrt(area_a * area_b).
    Threshold: median of nonzero Ochiai values.
    Clustering: Average-linkage.
    """
    n = len(elements)
    if n <= 1:
        return elements

    # Compute pairwise Ochiai coefficient
    score_matrix = {}
    nonzero_scores = []

    for i in range(n):
        pi = elements[i]["polygon"]
        bi = pi.bounds
        for j in range(i + 1, n):
            pj = elements[j]["polygon"]
            bj = pj.bounds
            # Quick bbox rejection
            if (
                bi[2] < bj[0]
                or bj[2] < bi[0]
                or bi[3] < bj[1]
                or bj[3] < bi[1]
            ):
                continue
            try:
                inter = pi.intersection(pj)
                if inter.is_empty:
                    continue
                inter_area = inter.area
                denom = math.sqrt(pi.area * pj.area)
                if denom < 1e-6:
                    continue
                ochiai = inter_area / denom
                score_matrix[(i, j)] = ochiai
                if ochiai > 0:
                    nonzero_scores.append(ochiai)
            except Exception:
                continue

    if not nonzero_scores:
        return elements

    # Threshold: median of nonzero Ochiai values
    threshold = float(np.median(nonzero_scores))

    # Average-linkage clustering
    groups = [[i] for i in range(n)]

    def _get_score(a, b):
        key = (min(a, b), max(a, b))
        return score_matrix.get(key, 0.0)

    changed = True
    while changed:
        changed = False
        best_merge = None
        best_avg = threshold

        for gi in range(len(groups)):
            for gj in range(gi + 1, len(groups)):
                total_score = 0.0
                n_pairs = 0
                for a in groups[gi]:
                    for b in groups[gj]:
                        total_score += _get_score(a, b)
                        n_pairs += 1
                avg_score = total_score / n_pairs if n_pairs > 0 else 0.0

                if avg_score >= best_avg:
                    best_avg = avg_score
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
        features.append(
            {
                "labels": labels,
                "polygon": geom,
                "fills": colors,
                "area": geom.area,
                "centroid": (geom.centroid.x, geom.centroid.y),
                "n_layers": len(members),
            }
        )

    features.sort(key=lambda f: f["area"], reverse=True)
    return features


# == Step 3: Group by Containment + Adjacency ==================================


def group_by_proximity(features: list[dict]) -> list[GroupData]:
    """Group features by containment tree + adjacency.

    1. Identify outer contour (largest feature containing most others).
    2. Build containment tree: each feature's parent is the SMALLEST
       feature that contains >50% of its area.
    3. Root features (no parent except outer contour) + their descendants
       = one visual group.
    4. Orphan features not inside any root get merged with their nearest
       root group.
    5. Merge adjacent root groups — root groups whose polygons touch or
       are very close are likely parts of the same visual feature
       (e.g. tail segments, wing pieces).

    Returns list of GroupData dataclasses.
    """
    n = len(features)
    if n <= 2:
        return [_dict_to_group(f) for f in features]

    # Step A: Identify outer contour
    contain_counts = [0] * n
    for i in range(n):
        pi = features[i]["polygon"]
        for j in range(n):
            if i == j:
                continue
            pj = features[j]["polygon"]
            try:
                inter = pi.intersection(pj).area
                if features[j]["area"] > 0 and inter / features[j]["area"] > 0.5:
                    contain_counts[i] += 1
            except Exception:
                continue

    outer_idx = max(range(n), key=lambda i: contain_counts[i])

    # Step B: Build containment tree
    inner = [i for i in range(n) if i != outer_idx]
    parent = {}

    for i in inner:
        pi = features[i]["polygon"]
        ai = features[i]["area"]
        if ai <= 0:
            continue
        best_parent = None
        best_parent_area = float("inf")
        for j in inner:
            if i == j:
                continue
            pj = features[j]["polygon"]
            aj = features[j]["area"]
            if aj <= ai:
                continue
            try:
                inter = pi.intersection(pj).area
                coverage = inter / ai
                if coverage > 0.5 and aj < best_parent_area:
                    best_parent = j
                    best_parent_area = aj
            except Exception:
                continue
        if best_parent is not None:
            parent[i] = best_parent

    # Step C: Find root features
    roots = [i for i in inner if i not in parent]

    children = defaultdict(list)
    for child, par in parent.items():
        children[par].append(child)

    def _get_descendants(idx):
        desc = []
        queue = list(children.get(idx, []))
        while queue:
            node = queue.pop(0)
            desc.append(node)
            queue.extend(children.get(node, []))
        return desc

    # Step D: Build groups from roots + descendants
    grouped_indices = set()
    root_groups = []
    for r in roots:
        desc = _get_descendants(r)
        members = [r] + desc
        root_groups.append(members)
        grouped_indices.update(members)

    # Orphans: merge into nearest root group
    orphans = [i for i in inner if i not in grouped_indices]
    for orph in orphans:
        po = features[orph]["polygon"]
        best_group = None
        best_dist = float("inf")
        for gi, members in enumerate(root_groups):
            geom = unary_union([features[m]["polygon"] for m in members])
            try:
                d = po.distance(geom)
                if d < best_dist:
                    best_dist = d
                    best_group = gi
            except Exception:
                continue
        if best_group is not None:
            root_groups[best_group].append(orph)

    # Step D2: Merge adjacent root groups
    # Root groups that are touching or very close are likely parts of the
    # same visual feature (e.g. tail segments, wing pieces). Merge them.
    outer_poly = features[outer_idx]["polygon"]
    canvas_diag = math.sqrt(outer_poly.area) if outer_poly.area > 0 else 100
    adjacency_threshold = canvas_diag * 0.05  # 5% of sqrt(outer area)

    merged_any = True
    while merged_any and len(root_groups) > 1:
        merged_any = False
        # Pre-compute union geometries for each group
        group_geoms = []
        for members in root_groups:
            group_geoms.append(unary_union([features[m]["polygon"] for m in members]))

        best_pair = None
        best_dist = adjacency_threshold

        for gi in range(len(root_groups)):
            for gj in range(gi + 1, len(root_groups)):
                try:
                    d = group_geoms[gi].distance(group_geoms[gj])
                    if d < best_dist:
                        best_dist = d
                        best_pair = (gi, gj)
                except Exception:
                    continue

        if best_pair is not None:
            gi, gj = best_pair
            root_groups[gi].extend(root_groups[gj])
            root_groups.pop(gj)
            merged_any = True

    # Step E: Build output
    def _build_group(member_indices: list[int]) -> GroupData:
        members = [features[i] for i in member_indices]
        all_labels = []
        all_fills: set[str] = set()
        total_layers = 0
        for m in members:
            all_labels.extend(m["labels"])
            all_fills.update(m["fills"])
            total_layers += m["n_layers"]
        geom = unary_union([m["polygon"] for m in members])
        return GroupData(
            labels=all_labels,
            polygon=geom,
            fills=list(all_fills),
            area=geom.area,
            centroid=(geom.centroid.x, geom.centroid.y),
            n_layers=total_layers,
            n_merged=len(member_indices),
        )

    result = []
    # Outer contour
    f = features[outer_idx]
    result.append(
        GroupData(
            labels=f["labels"],
            polygon=f["polygon"],
            fills=f["fills"],
            area=f["area"],
            centroid=f["centroid"],
            n_layers=f["n_layers"],
            n_merged=1,
        )
    )
    # Root groups
    for members in root_groups:
        result.append(_build_group(members))

    result.sort(key=lambda g: g.area, reverse=True)
    return result


def _dict_to_group(f: dict) -> GroupData:
    """Convert a raw feature dict to GroupData."""
    return GroupData(
        labels=f.get("labels", [f.get("label", "?")]),
        polygon=f.get("polygon"),
        fills=f.get("fills", [f.get("fill", "#888888")]),
        area=f.get("area", 0.0),
        centroid=f.get("centroid", (0.0, 0.0)),
        n_layers=f.get("n_layers", 1),
        n_merged=f.get("n_merged", 1),
    )
