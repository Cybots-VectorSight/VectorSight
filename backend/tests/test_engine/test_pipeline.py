"""Tests for the pipeline orchestrator."""

from app.engine.context import PipelineContext
from app.engine.pipeline import Pipeline
from app.engine.registry import Layer, TransformRegistry, TransformSpec


def test_pipeline_runs_transforms():
    reg = TransformRegistry()
    results = []

    def t1(ctx: PipelineContext) -> None:
        results.append("t1")

    def t2(ctx: PipelineContext) -> None:
        results.append("t2")

    reg.register(TransformSpec(id="T0.01", layer=Layer.PARSING, fn=t1))
    reg.register(TransformSpec(id="T0.02", layer=Layer.PARSING, fn=t2, dependencies=["T0.01"]))

    pipeline = Pipeline(registry=reg)
    ctx = PipelineContext()
    pipeline.run(ctx)

    assert results == ["t1", "t2"]
    assert "T0.01" in ctx.completed_transforms
    assert "T0.02" in ctx.completed_transforms


def test_pipeline_handles_errors():
    reg = TransformRegistry()

    def fail(ctx: PipelineContext) -> None:
        raise ValueError("test error")

    reg.register(TransformSpec(id="T0.01", layer=Layer.PARSING, fn=fail))

    pipeline = Pipeline(registry=reg)
    ctx = PipelineContext()
    pipeline.run(ctx)

    assert "T0.01" in ctx.errors
    assert "test error" in ctx.errors["T0.01"]
