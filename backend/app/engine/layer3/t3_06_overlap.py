"""T3.06 — Overlap Detection. ★★

Check bounding box intersection and pointcloud overlap between pairs.
"""

from __future__ import annotations

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform


def _bbox_intersects(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    """Check if two bboxes intersect."""
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _bbox_overlap_area(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Compute overlap area of two bboxes."""
    x_overlap = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    y_overlap = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return x_overlap * y_overlap


@transform(
    id="T3.06",
    layer=Layer.RELATIONSHIPS,
    dependencies=["T1.14"],
    description="Detect overlap between element bounding boxes",
)
def overlap(ctx: PipelineContext) -> None:
    n = ctx.num_elements
    if n < 2:
        return

    for i, sp in enumerate(ctx.subpaths):
        overlaps = []
        bbox_i = sp.bbox
        area_i = (bbox_i[2] - bbox_i[0]) * (bbox_i[3] - bbox_i[1])

        for j in range(n):
            if i == j:
                continue
            bbox_j = ctx.subpaths[j].bbox

            if _bbox_intersects(bbox_i, bbox_j):
                overlap_area = _bbox_overlap_area(bbox_i, bbox_j)
                iou = 0.0
                if area_i > 0:
                    area_j = (bbox_j[2] - bbox_j[0]) * (bbox_j[3] - bbox_j[1])
                    union = area_i + area_j - overlap_area
                    if union > 0:
                        iou = overlap_area / union

                overlaps.append({
                    "element": ctx.subpaths[j].id,
                    "bbox_overlap": True,
                    "overlap_area": round(overlap_area, 2),
                    "iou": round(iou, 3),
                })

        sp.features["overlaps"] = overlaps
        sp.features["overlapping_count"] = len(overlaps)
