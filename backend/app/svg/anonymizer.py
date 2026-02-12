"""SVG Anonymizer — strips identity-revealing metadata before LLM sees SVG.

Two levels:
- sanitize_for_llm(): Light-touch — removes <title>, <desc>, comments, descriptive
  attributes (id, class, aria-*, data-*) while keeping ALL visual attributes (fill,
  stroke, d, cx, cy, etc.).  Used before injecting SVG into LLM prompts.
- anonymize(): Heavy — rebuilds SVG with only geometry attributes.  Not used in the
  main pipeline but available for research/testing.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

# ── Light-touch sanitization (used in main pipeline) ──────────────────────

# Attributes that reveal identity — strip these before LLM sees the SVG
_IDENTITY_ATTRS_RE = re.compile(
    r"""\s+(?:
        id | class | name
        | aria-\w+
        | data-\w+
        | inkscape:\w+
        | sodipodi:\w+
    )\s*=\s*"[^"]*" """,
    re.VERBOSE,
)

# Metadata tags to remove entirely (with their content)
_META_TAG_RE = re.compile(
    r"<\s*(?:title|desc|metadata)[^>]*>.*?</\s*(?:title|desc|metadata)\s*>",
    re.DOTALL | re.IGNORECASE,
)
_META_SELFCLOSE_RE = re.compile(
    r"<\s*(?:title|desc|metadata)[^/]*/\s*>",
    re.IGNORECASE,
)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def sanitize_for_llm(svg_text: str) -> str:
    """Strip identity-revealing metadata from SVG while keeping all visual attributes.

    Removes: <title>, <desc>, <metadata>, HTML comments, id/class/aria-*/data-* attrs.
    Keeps: fill, stroke, d, cx, cy, r, viewBox, transform, etc.
    """
    # Remove comments
    svg_text = _COMMENT_RE.sub("", svg_text)
    # Remove metadata tags
    svg_text = _META_TAG_RE.sub("", svg_text)
    svg_text = _META_SELFCLOSE_RE.sub("", svg_text)
    # Remove identity attributes
    svg_text = _IDENTITY_ATTRS_RE.sub(" ", svg_text)
    # Clean up extra whitespace in tags
    svg_text = re.sub(r"  +", " ", svg_text)
    svg_text = re.sub(r" /?>", "/>", svg_text)
    svg_text = re.sub(r" >", ">", svg_text)
    return svg_text.strip()


def sanitize_tag(tag_text: str) -> str:
    """Strip identity attributes from a single SVG tag (for source_tag in enrichment)."""
    result = _IDENTITY_ATTRS_RE.sub(" ", tag_text)
    result = re.sub(r"  +", " ", result)
    result = re.sub(r" /?>", "/>", result)
    result = re.sub(r" >", ">", result)
    return result.strip()


# ── Heavy anonymization (for research/testing) ───────────────────────────

REMOVE_TAGS = {
    "title", "desc", "metadata", "style", "defs",
    "{http://sodipodi.sourceforge.net/DTD/sodipodi-0.0.dtd}namedview",
    "{http://www.inkscape.org/namespaces/inkscape}perspective",
}

SHAPE_TAGS = {"path", "circle", "ellipse", "rect", "line", "polyline", "polygon"}

KEEP_ATTRS = {
    "path": ["d", "transform"],
    "circle": ["cx", "cy", "r", "transform"],
    "ellipse": ["cx", "cy", "rx", "ry", "transform"],
    "rect": ["x", "y", "width", "height", "rx", "ry", "transform"],
    "line": ["x1", "y1", "x2", "y2", "transform"],
    "polyline": ["points", "transform"],
    "polygon": ["points", "transform"],
}


def _strip_ns(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def _extract_shapes(element: ET.Element) -> list[dict[str, str]]:
    shapes: list[dict[str, str]] = []
    tag = _strip_ns(element.tag)

    if tag in REMOVE_TAGS or element.tag in REMOVE_TAGS:
        return shapes

    if tag in SHAPE_TAGS:
        kept = {"type": tag}
        for attr in KEEP_ATTRS.get(tag, []):
            val = element.get(attr)
            if val:
                kept[attr] = val
        if tag != "path" or "d" in kept:
            shapes.append(kept)

    for child in element:
        shapes.extend(_extract_shapes(child))

    return shapes


def anonymize(svg_text: str) -> tuple[str | None, str | None, str]:
    """Anonymize SVG. Returns (clean_svg, claude_text, summary)."""
    svg_text = re.sub(r"<!--.*?-->", "", svg_text, flags=re.DOTALL)
    svg_text = re.sub(r'xmlns:\w+="[^"]*"', "", svg_text)

    root = ET.fromstring(svg_text)
    if _strip_ns(root.tag) != "svg":
        return None, None, "No <svg> root element found."

    vb = root.get("viewBox", "0 0 100 100")
    shapes = _extract_shapes(root)

    if not shapes:
        return None, None, "No shape elements found."

    lines = [f'<svg viewBox="{vb}" xmlns="http://www.w3.org/2000/svg">']
    for s in shapes:
        tag = s["type"]
        attrs = " ".join(f'{k}="{v}"' for k, v in s.items() if k != "type")
        lines.append(f"  <{tag} {attrs} />")
    lines.append("</svg>")
    clean_svg = "\n".join(lines)

    claude_lines = [f"viewBox: {vb}", f"Elements: {len(shapes)}", ""]
    for i, s in enumerate(shapes):
        tag = s["type"]
        extra = ""
        if tag == "path" and "d" in s:
            sub_paths = len(re.findall(r"[Mm]", s["d"]))
            extra = f", {sub_paths} sub-path{'s' if sub_paths != 1 else ''}"
        claude_lines.append(f"--- Element {i} ({tag}{extra}) ---")
        if tag == "path":
            claude_lines.append(s.get("d", ""))
        else:
            attrs = {k: v for k, v in s.items() if k != "type"}
            claude_lines.append(str(attrs))
        claude_lines.append("")
    claude_text = "\n".join(claude_lines)

    type_counts: dict[str, int] = {}
    for s in shapes:
        type_counts[s["type"]] = type_counts.get(s["type"], 0) + 1
    summary = f"viewBox: {vb} | {len(shapes)} elements"
    summary += " (" + ", ".join(f"{v} {k}{'s' if v > 1 else ''}" for k, v in type_counts.items()) + ")"

    return clean_svg, claude_text, summary
