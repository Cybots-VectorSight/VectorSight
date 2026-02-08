"""T3.07 — DBSCAN Grouping. ★★★ CRITICAL

Cluster elements using DBSCAN on centroids.
eps = 8-12% of viewBox diagonal, min_samples=2.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform


@transform(
    id="T3.07",
    layer=Layer.RELATIONSHIPS,
    dependencies=["T1.14", "T1.23"],
    description="DBSCAN clustering of elements by centroid proximity",
)
def dbscan_grouping(ctx: PipelineContext) -> None:
    n = ctx.num_elements
    if n < 2:
        return

    centroids = np.array([sp.centroid for sp in ctx.subpaths])
    diag = ctx.viewbox_diagonal
    eps = diag * 0.10  # 10% of diagonal

    db = DBSCAN(eps=eps, min_samples=2).fit(centroids)
    labels = db.labels_
    ctx.cluster_labels = labels

    n_clusters = int(labels.max()) + 1 if len(labels) > 0 and labels.max() >= 0 else 0

    for i, sp in enumerate(ctx.subpaths):
        cluster_id = int(labels[i])
        sp.features["cluster_id"] = cluster_id
        sp.features["is_isolated"] = cluster_id == -1

    # Build cluster summaries
    cluster_summaries = []
    for cid in range(n_clusters):
        members = [i for i in range(n) if labels[i] == cid]
        member_centroids = centroids[members]
        center = member_centroids.mean(axis=0)
        dists = np.sqrt(np.sum((member_centroids - center) ** 2, axis=1))
        radius = float(np.max(dists)) if len(dists) > 0 else 0.0

        # Dominant shape class in cluster
        shape_classes = [ctx.subpaths[m].features.get("shape_class", "unknown") for m in members]
        from collections import Counter
        dominant = Counter(shape_classes).most_common(1)[0][0]

        summary = {
            "cluster_id": cid,
            "member_count": len(members),
            "members": [ctx.subpaths[m].id for m in members],
            "center": (round(float(center[0]), 2), round(float(center[1]), 2)),
            "radius": round(radius, 2),
            "dominant_shape": dominant,
        }
        cluster_summaries.append(summary)

        # Set on each member
        for m in members:
            ctx.subpaths[m].features["cluster_summary"] = summary

    # Set on all elements
    for sp in ctx.subpaths:
        sp.features["total_clusters"] = n_clusters
        sp.features["total_isolated"] = int(np.sum(labels == -1))
