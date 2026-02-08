"""API response models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.models.enrichment import EnrichmentOutput


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    transforms_registered: int = 0


class AnalyzeResponse(BaseModel):
    enrichment: EnrichmentOutput
    processing_time_ms: float = 0.0
    transforms_completed: int = 0
    transforms_failed: int = 0
    errors: dict[str, str] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    answer: str
    enrichment_used: bool = True


class ModifyResponse(BaseModel):
    svg: str
    changes: list[str] = Field(default_factory=list)


class CreateResponse(BaseModel):
    svg: str
    intent: str = ""
    validation_passed: bool = False


class IconSetRulesResponse(BaseModel):
    rules: dict[str, str] = Field(default_factory=dict)
    common_properties: dict[str, float] = Field(default_factory=dict)


class IconSetGenerateResponse(BaseModel):
    svg: str
    rules_applied: list[str] = Field(default_factory=list)


class PlaygroundResponse(BaseModel):
    svg: str
    action: str = ""
    element_clicked: str | None = None
