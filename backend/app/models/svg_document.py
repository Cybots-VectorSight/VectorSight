"""Parsed SVG document model."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SvgElement(BaseModel):
    tag: str
    attributes: dict[str, str] = Field(default_factory=dict)
    path_data: str | None = None


class SvgDocument(BaseModel):
    """Represents a parsed SVG file."""

    viewbox: tuple[float, float, float, float] = (0.0, 0.0, 24.0, 24.0)
    width: float = 24.0
    height: float = 24.0
    elements: list[SvgElement] = Field(default_factory=list)
    raw_svg: str = ""
    fill_rule: str = "nonzero"
    default_fill: str = "black"
    default_stroke: str = "none"
    stroke_width: float = 0.0
