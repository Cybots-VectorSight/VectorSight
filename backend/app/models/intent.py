"""Spatial intent model — for create/modify flows."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SpatialElement(BaseModel):
    name: str
    shape: str  # circular, organic, rectangular, triangular, arc, line
    size: str = ""  # e.g. "60% of canvas", "4% of head area"
    position: str = ""  # e.g. "center of canvas", "inside head, upper-forward (75% x, 35% y)"
    padding: str = ""  # e.g. "10% from head boundary"
    depth: str = ""  # e.g. "above head"
    fill: str = ""
    stroke: str = ""
    protrude: str = ""
    extends: str = ""
    path: str | None = None


class MirrorSpec(BaseModel):
    elements: list[str]
    axis: str  # e.g. "vertical center of head"


class SpatialIntent(BaseModel):
    canvas_width: float = 24.0
    canvas_height: float = 24.0
    elements: list[SpatialElement] = Field(default_factory=list)
    mirrors: list[MirrorSpec] = Field(default_factory=list)
    depth_order: list[str] = Field(default_factory=list)
    style: dict[str, str] = Field(default_factory=dict)
