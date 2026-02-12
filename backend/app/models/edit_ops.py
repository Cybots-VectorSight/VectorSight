"""Edit operation models for surgical SVG modification."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class EditOp(BaseModel):
    """A single surgical edit operation on an SVG element."""

    action: Literal["add", "delete", "modify"]
    target: str | None = None  # Element ID, e.g. "E5" (required for delete/modify)
    position: str | None = None  # For add: "after:E3", "before:E3", "end"
    svg_fragment: str | None = None  # For add: raw SVG tag(s) to insert
    attributes: dict[str, str] | None = None  # For modify: attrs to set/override


class EditPlan(BaseModel):
    """LLM-generated plan of surgical edit operations."""

    reasoning: str = ""  # LLM's spatial reasoning (for debugging/reflection)
    operations: list[EditOp]  # Ordered list of edit operations
