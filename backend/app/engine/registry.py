"""Transform registry — every transform is a standalone function registered via decorator.

Usage:
    @transform(id="T1.03", layer=Layer.SHAPE_ANALYSIS, dependencies=["T0.04"], tags={"always"})
    def turning_function(ctx: PipelineContext) -> None:
        for sp in ctx.subpaths:
            sp.features["turning_total"] = compute(sp.points)

Adding a new transform = creating one file with the decorator. Nothing else changes.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from app.engine.context import PipelineContext

logger = logging.getLogger(__name__)


class Layer(enum.IntEnum):
    PARSING = 0
    SHAPE_ANALYSIS = 1
    VISUALIZATION = 2
    RELATIONSHIPS = 3
    VALIDATION = 4


@dataclass
class TransformSpec:
    id: str
    layer: Layer
    fn: Callable[["PipelineContext"], None]
    dependencies: list[str] = field(default_factory=list)
    tags: set[str] = field(default_factory=set)
    description: str = ""


class TransformRegistry:
    """Singleton registry of all transforms."""

    def __init__(self) -> None:
        self._transforms: dict[str, TransformSpec] = {}

    def register(self, spec: TransformSpec) -> None:
        if spec.id in self._transforms:
            raise ValueError(f"Duplicate transform ID: {spec.id}")
        self._transforms[spec.id] = spec
        logger.debug("Registered transform %s (%s)", spec.id, spec.layer.name)

    def get(self, transform_id: str) -> TransformSpec:
        return self._transforms[transform_id]

    def get_layer(self, layer: Layer) -> list[TransformSpec]:
        specs = [s for s in self._transforms.values() if s.layer == layer]
        return sorted(specs, key=lambda s: s.id)

    def all(self) -> list[TransformSpec]:
        return sorted(self._transforms.values(), key=lambda s: (s.layer, s.id))

    def resolve_order(self, requested_ids: set[str] | None = None) -> list[TransformSpec]:
        """Topological sort respecting dependencies. If requested_ids is None, run all."""
        pool = self._transforms
        if requested_ids is not None:
            # Expand with transitive dependencies
            expanded: set[str] = set()
            stack = list(requested_ids)
            while stack:
                tid = stack.pop()
                if tid in expanded:
                    continue
                expanded.add(tid)
                spec = pool.get(tid)
                if spec:
                    stack.extend(spec.dependencies)
            pool = {k: v for k, v in pool.items() if k in expanded}

        # Kahn's algorithm
        in_degree: dict[str, int] = {tid: 0 for tid in pool}
        for tid, spec in pool.items():
            for dep in spec.dependencies:
                if dep in pool:
                    in_degree[tid] += 1

        queue = sorted([tid for tid, d in in_degree.items() if d == 0])
        ordered: list[TransformSpec] = []

        while queue:
            tid = queue.pop(0)
            ordered.append(pool[tid])
            for other_id, other_spec in pool.items():
                if tid in other_spec.dependencies:
                    in_degree[other_id] -= 1
                    if in_degree[other_id] == 0:
                        queue.append(other_id)
                        queue.sort()

        if len(ordered) != len(pool):
            missing = set(pool.keys()) - {s.id for s in ordered}
            raise ValueError(f"Circular dependency detected among: {missing}")

        return ordered

    @property
    def count(self) -> int:
        return len(self._transforms)


# Module-level singleton
_registry = TransformRegistry()


def get_registry() -> TransformRegistry:
    return _registry


def transform(
    *,
    id: str,
    layer: Layer,
    dependencies: list[str] | None = None,
    tags: set[str] | None = None,
    description: str = "",
):
    """Decorator to register a transform function."""

    def decorator(fn: Callable[["PipelineContext"], None]):
        spec = TransformSpec(
            id=id,
            layer=layer,
            fn=fn,
            dependencies=dependencies or [],
            tags=tags or set(),
            description=description,
        )
        _registry.register(spec)
        return fn

    return decorator
