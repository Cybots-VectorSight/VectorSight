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
