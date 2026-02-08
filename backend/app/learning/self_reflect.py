"""Self-Reflection — LLM vision sees the rendered SVG and auto-learns.

Instead of relying on user feedback ("that was a squirrel"), the system:
1. Renders SVG → PNG via cairosvg
2. Sends the image + enrichment text to Claude vision (Haiku for cost)
3. LLM compares what it SEES vs what the enrichment DATA says
4. Auto-derives patterns from the gap → stored in learning memory

This is NOT fine-tuning. The LLM's image understanding is pretrained.
We're just using it to generate structured experience records.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Prompt for the vision model
_REFLECT_PROMPT = """You are VectorSight's self-reflection system. You will see:
1. A rendered image of an SVG
2. The spatial enrichment data that our geometry engine produced from that SVG

Your job: Compare what you SEE in the image with what the enrichment DATA describes.

ENRICHMENT DATA:
{enrichment}

INSTRUCTIONS:
1. First, describe what you see in the image in 1-2 sentences. Be specific (species, pose, objects, style).
2. Then evaluate: did the enrichment data capture the key visual features correctly?
3. Identify GAPS — what important visual information is missing or misleading in the enrichment?
4. Generate 1-5 PATTERNS (lessons learned) in this exact JSON format:

```json
{{
  "visual_description": "what you see in the image",
  "enrichment_accuracy": "good/partial/poor",
  "gaps": ["gap 1", "gap 2"],
  "patterns": [
    {{
      "condition": "when you observe X in the data",
      "insight": "it likely means Y in the image",
      "tags": ["tag1", "tag2"]
    }}
  ]
}}
```

Tags should describe relevant features: shape types (organic, circular), complexity (simple, complex),
symmetry (symmetric, asymmetric, side-profile, front-facing), fill density (dense-fill, sparse),
specific shapes present in the SVG. These tags are used for pattern matching on future SVGs.

Be concise. Focus on ACTIONABLE patterns that would help identify similar SVGs in the future."""


def render_svg_to_png(svg: str, width: int = 256, height: int = 256) -> bytes:
    """Render SVG string to PNG bytes using cairosvg."""
    import cairosvg

    try:
        png_bytes = cairosvg.svg2png(
            bytestring=svg.encode("utf-8"),
            output_width=width,
            output_height=height,
        )
        return png_bytes
    except Exception as e:
        logger.warning("Failed to render SVG to PNG: %s", e)
        raise


async def reflect_on_svg(
    svg: str,
    enrichment_text: str,
    ctx: Any | None = None,
) -> dict | None:
    """Render SVG, show to LLM vision, auto-derive patterns.

    Returns the reflection result dict or None on failure.
    """
    from app.config import settings

    if not settings.anthropic_api_key:
        logger.debug("No API key — skipping self-reflection")
        return None

    try:
        # 1. Render SVG → PNG
        png_bytes = render_svg_to_png(svg)
        png_b64 = base64.b64encode(png_bytes).decode("ascii")

        # 2. Send to Claude vision (Haiku — cheap tier)
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage

        llm = ChatAnthropic(
            model=settings.model_cheap,
            api_key=settings.anthropic_api_key,
            max_tokens=1024,
        )

        prompt = _REFLECT_PROMPT.format(enrichment=enrichment_text[:3000])

        message = HumanMessage(
            content=[
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": png_b64,
                    },
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ]
        )

        response = await llm.ainvoke([message])
        result_text = str(response.content)

        # 3. Parse the JSON from the response
        reflection = _parse_reflection(result_text)
        if reflection is None:
            logger.warning("Could not parse reflection JSON")
            return None

        # 4. Store patterns in learning memory
        _store_reflection(reflection, svg, ctx)

        logger.info(
            "Self-reflection complete: saw '%s', %d patterns derived",
            reflection.get("visual_description", "?")[:50],
            len(reflection.get("patterns", [])),
        )
        return reflection

    except Exception as e:
        logger.warning("Self-reflection failed: %s", e)
        return None


def _parse_reflection(text: str) -> dict | None:
    """Extract JSON from LLM response text."""
    import json
    import re

    # Try to find JSON block in markdown fences
    match = re.search(r"```(?:json)?\s*\n?({[\s\S]*?})\s*\n?```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON
    match = re.search(r"({[\s\S]*\"patterns\"[\s\S]*})", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    return None


def _store_reflection(reflection: dict, svg: str, ctx: Any | None) -> None:
    """Store reflection patterns in the learning memory."""
    from app.learning.memory import MemoryStore, Pattern, get_memory_store

    store = get_memory_store()
    svg_h = store.svg_hash(svg)

    visual_desc = reflection.get("visual_description", "")
    patterns_data = reflection.get("patterns", [])

    existing_patterns = store.get_patterns()
    next_num = len(existing_patterns) + 1

    for p in patterns_data:
        condition = p.get("condition", "")
        insight = p.get("insight", "")
        tags = p.get("tags", [])

        if not condition or not insight:
            continue

        pattern = Pattern(
            id=f"v{next_num:03d}",
            condition=f"Vision: {condition} (from {visual_desc[:60]})",
            insight=insight,
            confidence=0.8,  # Vision-derived patterns start high
            times_confirmed=1,
            tags=tags,
        )
        store.add_pattern(pattern)
        next_num += 1

    # Also update the most recent session with what was actually seen
    sessions = store.get_sessions(limit=10)
    for session in reversed(sessions):
        if session.svg_hash == svg_h:
            session.actual = visual_desc[:200]
            session.learnings.extend(
                [p.get("insight", "") for p in patterns_data if p.get("insight")]
            )
            # Rewrite
            store._save_sessions(store._load_sessions())
            break

    logger.info("Stored %d vision-derived patterns for SVG %s", len(patterns_data), svg_h)


async def reflect_background(
    svg: str,
    enrichment_text: str,
    ctx: Any | None = None,
) -> None:
    """Fire-and-forget self-reflection. Runs in background, never blocks."""
    try:
        await reflect_on_svg(svg, enrichment_text, ctx)
    except Exception as e:
        logger.debug("Background reflection failed (non-critical): %s", e)
