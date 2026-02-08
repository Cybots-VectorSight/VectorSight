"""API request models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    svg: str = Field(..., description="Raw SVG code")
    options: dict[str, bool] = Field(
        default_factory=dict,
        description="Optional feature flags (e.g., skip_layer3=True)",
    )


class ChatRequest(BaseModel):
    svg: str = Field(..., description="Raw SVG code")
    question: str = Field(..., description="User's question about the SVG")
    history: list[dict[str, str]] = Field(
        default_factory=list,
        description="Chat history (role/content pairs)",
    )


class ModifyRequest(BaseModel):
    svg: str = Field(..., description="Raw SVG code")
    instruction: str = Field(..., description="Modification instruction in natural language")


class CreateRequest(BaseModel):
    description: str = Field(..., description="Description of icon to create")
    style: str = Field(default="outline", description="Icon style (outline, filled, etc.)")
    canvas_size: int = Field(default=24, description="Canvas width/height")


class IconSetAnalyzeRequest(BaseModel):
    svgs: list[str] = Field(..., description="List of SVG codes in the set")


class IconSetGenerateRequest(BaseModel):
    svgs: list[str] = Field(..., description="Reference SVG codes")
    description: str = Field(..., description="Description of new icon to generate")


class PlaygroundClickRequest(BaseModel):
    svg: str = Field(..., description="Raw SVG code")
    x: float = Field(..., description="Click x coordinate")
    y: float = Field(..., description="Click y coordinate")
