"""Surgical SVG edit applier — splices JSON edit operations into original SVG."""

from __future__ import annotations

import logging
import re

from app.engine.context import PipelineContext
from app.models.edit_ops import EditOp

logger = logging.getLogger(__name__)

_CLOSING_SVG_RE = re.compile(r"</svg\s*>", re.IGNORECASE)


def apply_edits(svg_raw: str, ops: list[EditOp], ctx: PipelineContext) -> str:
    """Apply a list of edit operations to an SVG string, returning the modified SVG.

    Operations reference elements by their IDs (e.g. "E5"). The original SVG text is
    spliced at the character offsets stored during parsing, so untouched elements remain
    byte-identical.
    """
    element_map = {sp.id: sp for sp in ctx.subpaths}

    # Resolve conflicts: if the same target has both delete and modify, delete wins
    targets_to_delete: set[str] = set()
    for op in ops:
        if op.action == "delete" and op.target:
            targets_to_delete.add(op.target)

    # Build splice instructions: (offset, length, replacement)
    splices: list[tuple[int, int, str]] = []

    for op in ops:
        if op.action == "delete":
            if not op.target or op.target not in element_map:
                logger.warning("Edit op delete: unknown target %r, skipping", op.target)
                continue
            sp = element_map[op.target]
            if sp.source_span == (0, 0):
                logger.warning("Edit op delete: no source_span for %s, skipping", op.target)
                continue
            start, end = sp.source_span
            splices.append((start, end - start, ""))

        elif op.action == "modify":
            if not op.target or op.target not in element_map:
                logger.warning("Edit op modify: unknown target %r, skipping", op.target)
                continue
            if op.target in targets_to_delete:
                logger.info("Edit op modify on %s skipped — target is being deleted", op.target)
                continue
            sp = element_map[op.target]
            if sp.source_span == (0, 0):
                logger.warning("Edit op modify: no source_span for %s, skipping", op.target)
                continue
            if not op.attributes:
                continue
            new_tag = _apply_attr_overrides(sp.source_tag, op.attributes)
            start, end = sp.source_span
            splices.append((start, end - start, new_tag))

        elif op.action == "add":
            fragment = op.svg_fragment or ""
            if not fragment.strip():
                logger.warning("Edit op add: empty svg_fragment, skipping")
                continue
            insert_offset = _resolve_add_position(op.position, element_map, svg_raw)
            if insert_offset < 0:
                logger.warning("Edit op add: could not resolve position %r, skipping", op.position)
                continue
            # Insert with a newline for readability
            splices.append((insert_offset, 0, "\n  " + fragment.strip()))

    if not splices:
        return svg_raw

    # Sort by offset descending so earlier splices don't shift later offsets
    splices.sort(key=lambda s: s[0], reverse=True)

    result = svg_raw
    for offset, length, replacement in splices:
        result = result[:offset] + replacement + result[offset + length:]

    return result


def _resolve_add_position(
    position: str | None,
    element_map: dict,
    svg_raw: str,
) -> int:
    """Resolve an add position string to a character offset in svg_raw.

    Supported formats:
    - "after:E3" — insert after the element's closing tag
    - "before:E3" — insert before the element's opening tag
    - "end" or None — insert before </svg>
    """
    if not position or position.strip().lower() == "end":
        m = _CLOSING_SVG_RE.search(svg_raw)
        return m.start() if m else -1

    if ":" in position:
        directive, element_id = position.split(":", 1)
        directive = directive.strip().lower()
        element_id = element_id.strip()

        sp = element_map.get(element_id)
        if sp is None or sp.source_span == (0, 0):
            return -1

        if directive == "after":
            return sp.source_span[1]
        elif directive == "before":
            return sp.source_span[0]

    return -1


def _apply_attr_overrides(source_tag: str, overrides: dict[str, str]) -> str:
    """Apply attribute overrides to an SVG tag string.

    Existing attributes are updated, new attributes are added before the closing
    /> or >. The tag structure (self-closing vs not) is preserved.
    """
    existing = _extract_attrs(source_tag)
    merged = {**existing, **overrides}

    # Determine the tag name
    tag_match = re.match(r"<(\w+)", source_tag)
    if not tag_match:
        return source_tag
    tag_name = tag_match.group(1)

    # Determine closing style
    is_self_closing = source_tag.rstrip().endswith("/>")

    # Rebuild the tag — escape any quotes in attribute values
    parts = []
    for k, v in merged.items():
        # Escape embedded double quotes (shouldn't happen in SVG but be safe)
        safe_v = v.replace('"', "&quot;")
        parts.append(f'{k}="{safe_v}"')
    attrs_str = " ".join(parts)

    if is_self_closing:
        return f"<{tag_name} {attrs_str} />"
    else:
        return f"<{tag_name} {attrs_str}>"


def _extract_attrs(tag_text: str) -> dict[str, str]:
    """Extract key=value attributes from an SVG tag string."""
    attrs: dict[str, str] = {}
    for m in re.finditer(r'(\w[\w-]*)\s*=\s*"([^"]*)"', tag_text):
        attrs[m.group(1)] = m.group(2)
    return attrs
