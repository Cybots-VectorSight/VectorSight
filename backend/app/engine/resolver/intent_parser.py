"""Parse LLM spatial intent output into SpatialIntent model."""

from __future__ import annotations

import json
import re

from app.models.intent import MirrorSpec, SpatialElement, SpatialIntent


def parse_intent(text: str) -> SpatialIntent:
    """Parse LLM output (JSON or structured text) into SpatialIntent.

    Supports:
    1. Direct JSON output
    2. JSON embedded in markdown code block
    3. Freeform text (best-effort extraction)
    """
    # Try JSON code block
    json_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1)

    # Try direct JSON parse
    try:
        data = json.loads(text.strip())
        return SpatialIntent.model_validate(data)
    except (json.JSONDecodeError, Exception):
        pass

    # Freeform text fallback: extract SVG if present
    svg_match = re.search(r"<svg[^>]*>.*?</svg>", text, re.DOTALL | re.IGNORECASE)
    if svg_match:
        # Return a minimal intent — the SVG itself is the output
        return SpatialIntent(
            elements=[SpatialElement(name="raw_svg", shape="raw", path=svg_match.group(0))],
        )

    # Couldn't parse — return empty intent
    return SpatialIntent()
