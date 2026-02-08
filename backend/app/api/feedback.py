"""POST /api/feedback — Record learning feedback from analysis sessions."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.learning.memory import AnalysisSession, get_memory_store

router = APIRouter()


class FeedbackRequest(BaseModel):
    svg: str = Field("", description="Original SVG code (used for hashing)")
    svg_hash: str = Field("", description="SVG hash (alternative to sending full SVG)")
    prediction: str = Field(..., description="What the system predicted")
    actual: str = Field(..., description="What the SVG actually depicts")
    learnings: list[str] = Field(
        default_factory=list,
        description="Derived insights from this session",
    )
    # Optional session context
    element_count: int = 0
    symmetry_score: float = 0.0
    fill_pct: float = 0.0
    composition_type: str = ""
    pose: str = ""
    shape_distribution: dict[str, int] = Field(default_factory=dict)


class FeedbackResponse(BaseModel):
    status: str = "ok"
    patterns_count: int = 0
    message: str = ""


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(req: FeedbackRequest) -> FeedbackResponse:
    """Record analysis feedback to improve future predictions."""
    store = get_memory_store()

    # Compute hash if SVG provided
    svg_hash = req.svg_hash
    if not svg_hash and req.svg:
        svg_hash = store.svg_hash(req.svg)
    if not svg_hash:
        svg_hash = "unknown"

    # Auto-generate learnings if prediction != actual and none provided
    learnings = req.learnings
    if not learnings and req.prediction != req.actual:
        learnings = [
            f"Predicted '{req.prediction}' but actual was '{req.actual}' — "
            f"need better feature interpretation for this type of SVG."
        ]

    # Record session
    session = AnalysisSession(
        svg_hash=svg_hash,
        element_count=req.element_count,
        symmetry_score=req.symmetry_score,
        fill_pct=req.fill_pct,
        composition_type=req.composition_type,
        pose=req.pose,
        shape_distribution=req.shape_distribution,
        prediction=req.prediction,
        actual=req.actual,
        learnings=learnings,
    )
    store.record_session(session)

    # Record feedback (may generate new patterns)
    store.record_feedback(
        svg_hash=svg_hash,
        prediction=req.prediction,
        actual=req.actual,
        learnings=learnings,
    )

    patterns = store.get_patterns()
    return FeedbackResponse(
        status="ok",
        patterns_count=len(patterns),
        message=f"Recorded feedback: '{req.prediction}' → '{req.actual}'. "
        f"{len(learnings)} learnings stored. {len(patterns)} total patterns.",
    )


class ReflectRequest(BaseModel):
    svg: str = Field(..., description="SVG code to self-reflect on")


class ReflectResponse(BaseModel):
    status: str = "ok"
    visual_description: str = ""
    enrichment_accuracy: str = ""
    gaps: list[str] = Field(default_factory=list)
    patterns_derived: int = 0


@router.post("/feedback/reflect", response_model=ReflectResponse)
async def self_reflect(req: ReflectRequest) -> ReflectResponse:
    """Render SVG, show to LLM vision, auto-derive patterns.

    This is the self-learning endpoint: the LLM sees the image and
    compares what it sees with the enrichment data.
    """
    from app.engine.pipeline import create_pipeline
    from app.learning.self_reflect import reflect_on_svg
    from app.llm.enrichment_formatter import context_to_enrichment_text
    from app.svg.parser import parse_svg

    ctx = parse_svg(req.svg)
    pipeline = create_pipeline()
    ctx = pipeline.run(ctx)

    enrichment_text = context_to_enrichment_text(ctx)
    result = await reflect_on_svg(req.svg, enrichment_text, ctx)

    if result is None:
        return ReflectResponse(status="failed", visual_description="Could not reflect")

    return ReflectResponse(
        status="ok",
        visual_description=result.get("visual_description", ""),
        enrichment_accuracy=result.get("enrichment_accuracy", ""),
        gaps=result.get("gaps", []),
        patterns_derived=len(result.get("patterns", [])),
    )


class PatternsResponse(BaseModel):
    patterns: list[dict] = Field(default_factory=list)
    count: int = 0


@router.get("/feedback/patterns", response_model=PatternsResponse)
async def get_patterns() -> PatternsResponse:
    """Get all learned patterns."""
    store = get_memory_store()
    patterns = store.get_patterns()
    return PatternsResponse(
        patterns=[
            {
                "id": p.id,
                "condition": p.condition,
                "insight": p.insight,
                "confidence": p.confidence,
                "tags": p.tags,
            }
            for p in patterns
        ],
        count=len(patterns),
    )
