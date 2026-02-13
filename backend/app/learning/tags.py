"""Knowledge & tags — the agent's self-evolving understanding of SVG patterns.

tag_rules.md is a living knowledge document the reflection system reads
and edits. Tags are assigned by the LLM using its judgment, not rigid rules.
Code only extracts factual tags (shape classes from the pipeline).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_KNOWLEDGE_PATH = Path(__file__).parent / "tag_rules.md"
_REFERENCE_PATH = Path(__file__).parent / "spatial_reference.md"


def get_spatial_reference() -> str:
    """Read the static spatial correlation reference for prompt injection.

    Returns a compressed version (~2000 tokens) suitable for inclusion in
    reflection prompts. The full version lives in spatial_reference.md.
    """
    if not _REFERENCE_PATH.exists():
        return ""
    return _REFERENCE_PATH.read_text(encoding="utf-8")


def get_knowledge() -> str:
    """Read the knowledge document for injection into LLM prompts."""
    if not _KNOWLEDGE_PATH.exists():
        return "(No prior knowledge yet.)"
    return _KNOWLEDGE_PATH.read_text(encoding="utf-8")


def update_knowledge(update_text: str) -> None:
    """Append a new learning to the knowledge document.

    The reflection system calls this when it discovers something worth
    remembering. The update is appended under a 'Learned' section.
    """
    if not update_text or not update_text.strip():
        return

    current = get_knowledge()

    # Append under a "## Learned" section
    if "## Learned" not in current:
        current += "\n\n## Learned\n"

    current += f"\n- {update_text.strip()}\n"

    _KNOWLEDGE_PATH.write_text(current, encoding="utf-8")
    logger.info("Knowledge updated: %s", update_text[:80])


def build_factual_tags(
    shape_distribution: dict[str, int] | None = None,
    composition_type: str = "",
) -> set[str]:
    """Extract only factual, pipeline-derived tags. No judgment calls."""
    tags: set[str] = set()

    if shape_distribution:
        for shape, count in shape_distribution.items():
            if count > 0:
                tags.add(shape)

    if composition_type:
        tags.add(composition_type.split()[0].lower())

    return tags


def build_factual_tags_from_breakdown(result: Any) -> set[str]:
    """Extract factual tags from a BreakdownResult."""
    shape_dist: dict[str, int] = {}
    for g in getattr(result, "groups", []):
        if g.polygon and not g.polygon.is_empty:
            try:
                from app.engine.breakdown.prompt_builder import _shape_descriptor

                sd = _shape_descriptor(g.polygon)
                s = sd["shape"] if sd else "organic"
                shape_dist[s] = shape_dist.get(s, 0) + 1
            except Exception:
                shape_dist["organic"] = shape_dist.get("organic", 0) + 1

    return build_factual_tags(
        shape_distribution=shape_dist,
        composition_type="breakdown",
    )
