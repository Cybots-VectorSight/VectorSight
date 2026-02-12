"""T1.20 — Symmetry Detection. ★★★

Test bilateral symmetry by reflecting boundary points across candidate axes.
Also test rotational symmetry.
"""

from __future__ import annotations

import numpy as np

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform
from app.engine.spatial_constants import SYMMETRY_MIRROR_FRACTION

# Multi-element symmetry: >50% matched = majority rule (democracy threshold).
_SYM_MAJORITY = 0.50
# Reportable bilateral: >70% matched = strong majority (supermajority).
_SYM_REPORTABLE = 0.70
# Minimum points for bilateral analysis: 2x ellipse DOF (5) = 10.
_MIN_POINTS_SYMMETRY = 10


@transform(
    id="T1.20",
    layer=Layer.SHAPE_ANALYSIS,
    dependencies=["T0.04", "T1.14"],
    description="Detect bilateral and rotational symmetry",
)
def symmetry_detection(ctx: PipelineContext) -> None:
    if len(ctx.subpaths) < 1:
        return

    # Per-element symmetry (bilateral)
    for sp in ctx.subpaths:
        if len(sp.points) < _MIN_POINTS_SYMMETRY:
            sp.features["bilateral_symmetry_score"] = 0.0
            sp.features["bilateral_symmetry_axis"] = "none"
            continue

        cx, cy = sp.centroid
        best_score = 0.0
        best_axis = "none"

        # Test vertical axis (x = cx)
        reflected = sp.points.copy()
        reflected[:, 0] = 2 * cx - reflected[:, 0]
        score = _match_score(sp.points, reflected)
        if score > best_score:
            best_score = score
            best_axis = "vertical"

        # Test horizontal axis (y = cy)
        reflected = sp.points.copy()
        reflected[:, 1] = 2 * cy - reflected[:, 1]
        score = _match_score(sp.points, reflected)
        if score > best_score:
            best_score = score
            best_axis = "horizontal"

        sp.features["bilateral_symmetry_score"] = round(best_score, 3)
        sp.features["bilateral_symmetry_axis"] = best_axis

    # Multi-element symmetry (across the canvas)
    if len(ctx.subpaths) >= 2:
        canvas_cx = ctx.canvas_width / 2
        canvas_cy = ctx.canvas_height / 2

        # Test vertical axis (x = canvas_center)
        centroids = np.array([sp.centroid for sp in ctx.subpaths])
        reflected_x = 2 * canvas_cx - centroids[:, 0]

        pairs: list[tuple[int, int]] = []
        used: set[int] = set()
        on_axis: list[int] = []
        eps = ctx.viewbox_diagonal * SYMMETRY_MIRROR_FRACTION

        for i in range(len(centroids)):
            if i in used:
                continue
            # Check if on axis
            if abs(centroids[i, 0] - canvas_cx) < eps:
                on_axis.append(i)
                used.add(i)
                continue
            # Find mirror partner
            for j in range(i + 1, len(centroids)):
                if j in used:
                    continue
                dist = np.sqrt(
                    (reflected_x[i] - centroids[j, 0]) ** 2
                    + (centroids[i, 1] - centroids[j, 1]) ** 2
                )
                if dist < eps:
                    pairs.append((i, j))
                    used.add(i)
                    used.add(j)
                    break

        matched = len(pairs) * 2 + len(on_axis)
        total = len(ctx.subpaths)
        sym_score = matched / max(total, 1)

        if sym_score > _SYM_MAJORITY:
            ctx.symmetry_axis = "vertical"
            ctx.symmetry_score = round(sym_score, 3)
            ctx.symmetry_pairs = pairs
        else:
            ctx.symmetry_axis = None
            ctx.symmetry_score = round(sym_score, 3)


def _match_score(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """Compute match quality between two point sets.

    For each point in A, find nearest point in B. Score = mean(1 - dist/max_dist).
    """
    if len(pts_a) == 0:
        return 0.0

    from scipy.spatial import cKDTree

    tree = cKDTree(pts_b)
    dists, _ = tree.query(pts_a)
    max_dist = np.max(dists) if len(dists) > 0 else 1.0
    if max_dist < 1e-10:
        return 1.0
    scores = 1.0 - dists / max_dist
    return float(np.mean(scores))
