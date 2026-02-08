"""T3.21 — Visual Stacking Tree. ★★★ CRITICAL

Construction:
1. Sort by z-order.
2. For each element, find containing parent via containment matrix (T3.01).
3. If no parent → root-level.
4. If contained by multiple → parent is smallest container (nearest ancestor).
5. Order children by z-order.
"""

from __future__ import annotations

from app.engine.context import PipelineContext
from app.engine.registry import Layer, transform


@transform(
    id="T3.21",
    layer=Layer.RELATIONSHIPS,
    dependencies=["T3.01", "T1.14"],
    description="Build visual stacking tree from z-order and containment",
)
def visual_stacking_tree(ctx: PipelineContext) -> None:
    n = ctx.num_elements
    if n < 2:
        if n == 1:
            ctx.stacking_tree = []
            ctx.subpaths[0].features["stacking_parent"] = None
            ctx.subpaths[0].features["stacking_children"] = []
            ctx.subpaths[0].features["stacking_depth"] = 0
        return

    cmat = ctx.containment_matrix
    if cmat is None:
        return

    # Sort elements by z-order
    sorted_indices = sorted(range(n), key=lambda i: ctx.subpaths[i].z_order)

    # For each element, find its parent (smallest container)
    parent: dict[int, int | None] = {}
    children: dict[int, list[int]] = {i: [] for i in range(n)}

    for i in range(n):
        # Find all elements that contain i
        containers = [j for j in range(n) if j != i and cmat[j][i]]

        if not containers:
            parent[i] = None
        else:
            # Pick the smallest container (nearest ancestor)
            containers.sort(key=lambda j: ctx.subpaths[j].area)
            parent[i] = containers[0]

    # Build tree edges
    tree_edges: list[tuple[int, int]] = []
    for child_idx, parent_idx in parent.items():
        if parent_idx is not None:
            tree_edges.append((parent_idx, child_idx))
            children[parent_idx].append(child_idx)

    # Sort children by z-order
    for p_idx in children:
        children[p_idx].sort(key=lambda c: ctx.subpaths[c].z_order)

    ctx.stacking_tree = tree_edges

    # Compute depth
    def _depth(idx: int) -> int:
        p = parent[idx]
        if p is None:
            return 0
        return 1 + _depth(p)

    # Build tree text representation
    def _tree_text(idx: int, indent: int = 0) -> str:
        sp = ctx.subpaths[idx]
        prefix = "  " * indent
        shape = sp.features.get("shape_class", "?")
        size = sp.features.get("size_tier", "?")
        line = f"{prefix}{sp.id} [{shape}, {size}, z={sp.z_order}]"
        lines = [line]
        for child in children[idx]:
            lines.append(_tree_text(child, indent + 1))
        return "\n".join(lines)

    # Find root nodes
    roots = [i for i in range(n) if parent[i] is None]
    roots.sort(key=lambda i: ctx.subpaths[i].z_order)

    tree_text = "\n".join(_tree_text(r) for r in roots)

    for i, sp in enumerate(ctx.subpaths):
        sp.features["stacking_parent"] = ctx.subpaths[parent[i]].id if parent[i] is not None else None
        sp.features["stacking_children"] = [ctx.subpaths[c].id for c in children[i]]
        sp.features["stacking_depth"] = _depth(i)
        sp.features["stacking_tree_text"] = tree_text
