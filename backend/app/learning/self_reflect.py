"""Self-Reflection — automatic learning after every user interaction.

Closed-loop workflow (no user feedback needed):
1. User asks a question or requests a modification
2. LLM responds (chat answer or modified SVG)
3. Background: render SVG → PNG, send to vision model
4. Vision compares: what it SEES vs enrichment vs LLM's answer vs user's intent
5. Auto-derives patterns from mismatches → stored in learning memory
6. Existing patterns confirmed/contradicted → confidence adjusted

The user's decision (their question or instruction) drives what gets learned.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

logger = logging.getLogger(__name__)

_CHAT_REFLECT_PROMPT = """You are VectorSight's self-learning system. After every user interaction, you verify what happened.

SPATIAL REFERENCE (research-based correlations — use to interpret geometry):
{spatial_reference}

PRIOR KNOWLEDGE (learned from experience — add to this):
{knowledge}

THE USER ASKED: {user_input}

THE LLM ANSWERED: {llm_output}

ENRICHMENT DATA (geometry engine output, truncated):
{enrichment}

Look at the rendered SVG image above. Compare THREE things:
1. What you SEE in the image
2. What the ENRICHMENT DATA describes (use the spatial reference to interpret)
3. What the LLM TOLD the user

Use the spatial reference to understand what geometry measurements mean visually.
Use prior knowledge to inform your analysis. Then derive NEW patterns — things
the spatial reference doesn't already cover. Assign tags freely — use whatever
words best describe the combination of spatial traits that matter for this SVG.

RESPOND in this exact JSON format:

```json
{{
  "visual_description": "1-2 sentences: what you actually see in the image",
  "llm_was_correct": true/false,
  "llm_correction": "if incorrect, what should the LLM have said instead (empty if correct)",
  "enrichment_gaps": ["what the enrichment missed that would have helped"],
  "patterns": [
    {{
      "condition": "when the enrichment shows X",
      "insight": "the image likely depicts Y",
      "tags": ["tag1", "tag2"]
    }}
  ],
  "knowledge_update": "if you learned something that should be added to or changed in the prior knowledge document, describe it here (empty if nothing new)"
}}
```"""

_MODIFY_REFLECT_PROMPT = """You are VectorSight's self-learning system. The user requested an SVG modification.

SPATIAL REFERENCE (research-based correlations — use to interpret geometry):
{spatial_reference}

PRIOR KNOWLEDGE (learned from experience — add to this):
{knowledge}

THE USER ASKED FOR: {user_input}

ENRICHMENT DATA (from the ORIGINAL SVG before modification):
{enrichment}

Look at the rendered MODIFIED SVG image above. Evaluate:
1. What you SEE in the modified image
2. Did the modification achieve what the user wanted?
3. What spatial patterns made this modification succeed or fail?

Use the spatial reference to understand geometry measurements. Use prior
knowledge to inform your analysis. Derive NEW patterns beyond what the
reference covers. Assign tags freely — use whatever words best describe
what matters about this SVG spatially.

RESPOND in this exact JSON format:

```json
{{
  "visual_description": "what the modified SVG looks like now",
  "modification_succeeded": true/false,
  "what_worked": "what spatial reasoning led to a good result (empty if failed)",
  "what_failed": "what went wrong spatially (empty if succeeded)",
  "patterns": [
    {{
      "condition": "when modifying SVGs with X spatial property",
      "insight": "Y approach works well / should be avoided",
      "tags": ["tag1", "tag2"]
    }}
  ],
  "knowledge_update": "if you learned something new about spatial modification strategies, describe it here (empty if nothing new)"
}}
```"""


def render_svg_to_png(svg: str, width: int = 256, height: int = 256) -> bytes:
    """Render SVG string to PNG bytes using cairosvg."""
    import cairosvg

    try:
        return cairosvg.svg2png(
            bytestring=svg.encode("utf-8"),
            output_width=width,
            output_height=height,
        )
    except Exception as e:
        logger.warning("Failed to render SVG to PNG: %s", e)
        raise


async def reflect_on_chat(
    svg: str,
    enrichment_text: str,
    user_question: str,
    llm_answer: str,
    ctx: Any | None = None,
) -> dict | None:
    """After a chat: vision verifies what the LLM told the user."""
    from app.learning.tags import get_knowledge, get_spatial_reference

    prompt = _CHAT_REFLECT_PROMPT.format(
        user_input=user_question[:500],
        llm_output=llm_answer[:1000],
        enrichment=enrichment_text[:2500],
        knowledge=get_knowledge(),
        spatial_reference=get_spatial_reference(),
    )
    return await _reflect(svg, prompt, ctx, task="chat",
                          user_input=user_question, llm_output=llm_answer)


async def reflect_on_modify(
    original_svg: str,
    modified_svg: str,
    enrichment_text: str,
    user_instruction: str,
    ctx: Any | None = None,
) -> dict | None:
    """After a modify: vision verifies the modification result."""
    from app.learning.tags import get_knowledge, get_spatial_reference

    prompt = _MODIFY_REFLECT_PROMPT.format(
        user_input=user_instruction[:500],
        enrichment=enrichment_text[:2500],
        knowledge=get_knowledge(),
        spatial_reference=get_spatial_reference(),
    )
    # Show the MODIFIED svg to vision, not the original
    return await _reflect(modified_svg, prompt, ctx, task="modify",
                          user_input=user_instruction, llm_output="")


async def _reflect(
    svg: str,
    prompt: str,
    ctx: Any | None,
    task: str,
    user_input: str,
    llm_output: str,
) -> dict | None:
    """Core reflection: render → vision → parse → store."""
    from app.config import settings

    if not settings.anthropic_api_key:
        return None

    try:
        png_bytes = render_svg_to_png(svg)
        png_b64 = base64.b64encode(png_bytes).decode("ascii")

        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage

        llm = ChatAnthropic(
            model=settings.model_cheap,
            api_key=settings.anthropic_api_key,
            max_tokens=1024,
        )

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
                {"type": "text", "text": prompt},
            ]
        )

        response = await llm.ainvoke([message])
        result_text = str(response.content)

        reflection = _parse_reflection(result_text)
        if reflection is None:
            logger.warning("Could not parse reflection JSON from: %s", result_text[:200])
            return None

        # Store patterns + adjust existing ones
        _store_and_adjust(reflection, svg, ctx, task, user_input)

        visual = reflection.get("visual_description", "?")[:50]
        n_patterns = len(reflection.get("patterns", []))
        logger.info("Self-reflection (%s): saw '%s', %d patterns", task, visual, n_patterns)
        return reflection

    except Exception as e:
        logger.warning("Self-reflection failed: %s", e)
        return None


def _parse_reflection(text: str) -> dict | None:
    """Extract JSON from LLM response text."""
    import json
    import re

    # Try markdown fences first
    match = re.search(r"```(?:json)?\s*\n?({[\s\S]*?})\s*\n?```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try raw JSON
    match = re.search(r"({[\s\S]*\"patterns\"[\s\S]*})", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    return None


def _store_and_adjust(
    reflection: dict,
    svg: str,
    ctx: Any | None,
    task: str,
    user_input: str,
) -> None:
    """Store new patterns and adjust confidence on existing ones."""
    from app.learning.memory import Pattern, get_memory_store

    store = get_memory_store()
    svg_h = store.svg_hash(svg)

    visual_desc = reflection.get("visual_description", "")
    patterns_data = reflection.get("patterns", [])

    # --- Adjust existing patterns ---
    # If vision confirms the LLM was correct, boost patterns that matched
    # If vision contradicts, downgrade patterns that led to the wrong answer
    existing = store.get_patterns()
    llm_correct = reflection.get("llm_was_correct", True)
    mod_succeeded = reflection.get("modification_succeeded", True)
    was_correct = llm_correct if task == "chat" else mod_succeeded

    if existing:
        # Build tags from current SVG using shared rules
        current_tags: set[str] = set()
        if ctx:
            from app.learning.tags import build_factual_tags_from_context

            current_tags = build_factual_tags_from_context(ctx)

        changed = False
        for pat in existing:
            overlap = len(current_tags.intersection(set(pat.tags)))
            if overlap == 0:
                continue

            if was_correct:
                pat.times_confirmed += 1
                # Nudge confidence up (max 0.95)
                pat.confidence = min(0.95, pat.confidence + 0.05)
            else:
                pat.times_contradicted += 1
                # Nudge confidence down (min 0.1)
                pat.confidence = max(0.1, pat.confidence - 0.1)
            changed = True

        if changed:
            store._save_patterns(existing)
            store._patterns_cache = existing

    # --- Store new patterns ---
    next_num = len(existing) + 1
    prefix = "vc" if task == "chat" else "vm"  # vision-chat / vision-modify

    for p in patterns_data:
        condition = p.get("condition", "")
        insight = p.get("insight", "")
        tags = p.get("tags", [])

        if not condition or not insight:
            continue

        source = f"[{task}] user: {user_input[:40]}"
        pattern = Pattern(
            id=f"{prefix}{next_num:03d}",
            condition=f"{condition} (from: {visual_desc[:50]})",
            insight=insight,
            confidence=0.75,
            times_confirmed=1,
            tags=tags,
            source_svg_hash=svg_h,
        )
        store.add_pattern(pattern)
        next_num += 1

    # Update session record with vision's description
    sessions = store.get_sessions(limit=10)
    for session in reversed(sessions):
        if session.svg_hash == svg_h:
            session.actual = visual_desc[:200]
            session.learnings.extend(
                [p.get("insight", "") for p in patterns_data if p.get("insight")]
            )
            store._save_sessions(store._load_sessions())
            break

    # Self-edit knowledge document if the LLM learned something new
    knowledge_update = reflection.get("knowledge_update", "")
    if knowledge_update:
        from app.learning.tags import update_knowledge

        update_knowledge(knowledge_update)

    logger.info(
        "Stored %d patterns, adjusted %d existing (correct=%s) for SVG %s",
        len(patterns_data), len(existing), was_correct, svg_h,
    )


async def reflect_background_chat(
    svg: str,
    enrichment_text: str,
    user_question: str,
    llm_answer: str,
    ctx: Any | None = None,
) -> None:
    """Fire-and-forget after chat. Never blocks the response."""
    try:
        await reflect_on_chat(svg, enrichment_text, user_question, llm_answer, ctx)
    except Exception as e:
        logger.debug("Background chat reflection failed: %s", e)


async def reflect_background_modify(
    original_svg: str,
    modified_svg: str,
    enrichment_text: str,
    user_instruction: str,
    ctx: Any | None = None,
) -> None:
    """Fire-and-forget after modify. Never blocks the response."""
    try:
        await reflect_on_modify(
            original_svg, modified_svg, enrichment_text, user_instruction, ctx
        )
    except Exception as e:
        logger.debug("Background modify reflection failed: %s", e)
