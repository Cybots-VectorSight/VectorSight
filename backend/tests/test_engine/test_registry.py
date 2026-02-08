"""Tests for the transform registry."""

from app.engine.context import PipelineContext
from app.engine.registry import Layer, TransformRegistry, TransformSpec


def _noop(ctx: PipelineContext) -> None:
    pass


def test_register_and_get():
    reg = TransformRegistry()
    spec = TransformSpec(id="T0.01", layer=Layer.PARSING, fn=_noop)
    reg.register(spec)
    assert reg.get("T0.01") is spec
    assert reg.count == 1


def test_get_layer():
    reg = TransformRegistry()
    s0 = TransformSpec(id="T0.01", layer=Layer.PARSING, fn=_noop)
    s1 = TransformSpec(id="T1.01", layer=Layer.SHAPE_ANALYSIS, fn=_noop)
    reg.register(s0)
    reg.register(s1)
    layer0 = reg.get_layer(Layer.PARSING)
    assert len(layer0) == 1
    assert layer0[0].id == "T0.01"


def test_resolve_order_with_deps():
    reg = TransformRegistry()
    s1 = TransformSpec(id="T0.04", layer=Layer.PARSING, fn=_noop)
    s2 = TransformSpec(id="T1.03", layer=Layer.SHAPE_ANALYSIS, fn=_noop, dependencies=["T0.04"])
    reg.register(s1)
    reg.register(s2)
    order = reg.resolve_order({"T1.03"})
    ids = [s.id for s in order]
    assert ids.index("T0.04") < ids.index("T1.03")


def test_resolve_order_all():
    reg = TransformRegistry()
    for i in range(5):
        reg.register(TransformSpec(id=f"T0.0{i+1}", layer=Layer.PARSING, fn=_noop))
    order = reg.resolve_order(None)
    assert len(order) == 5
