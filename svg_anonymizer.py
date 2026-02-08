"""
SVG Anonymizer — strips colors, IDs, classes, styles, metadata.
Keeps only geometry + transforms.

Usage:
  python svg_anonymizer.py input.svg                  # prints to terminal
  python svg_anonymizer.py input.svg -o clean.svg     # saves clean SVG
  python svg_anonymizer.py input.svg -o clean.svg -c  # also saves Claude-ready text
  python svg_anonymizer.py folder/ -o output_folder/   # batch process folder
"""

import xml.etree.ElementTree as ET
import argparse
import os
import sys
import re

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


def strip_ns(tag):
    """Remove namespace from tag name."""
    return tag.split("}")[-1] if "}" in tag else tag


def extract_shapes(element):
    """Recursively extract shape elements, keeping only geometry."""
    shapes = []
    tag = strip_ns(element.tag)

    if tag in REMOVE_TAGS or strip_ns(element.tag) in REMOVE_TAGS:
        return shapes

    if element.tag in REMOVE_TAGS:
        return shapes

    if tag in SHAPE_TAGS:
        kept = {"type": tag}
        for attr in KEEP_ATTRS.get(tag, []):
            val = element.get(attr)
            if val:
                kept[attr] = val
        if tag == "path" and "d" not in kept:
            pass  # skip paths with no d
        else:
            shapes.append(kept)

    for child in element:
        shapes.extend(extract_shapes(child))

    return shapes


def get_viewbox(svg):
    """Get or compute viewBox."""
    vb = svg.get("viewBox")
    if vb:
        return vb
    w = svg.get("width", "100").replace("px", "").replace("pt", "")
    h = svg.get("height", "100").replace("px", "").replace("pt", "")
    try:
        return f"0 0 {float(w)} {float(h)}"
    except ValueError:
        return "0 0 100 100"


def anonymize(svg_text):
    """Anonymize SVG, return (clean_svg, claude_text, summary)."""
    # Strip XML comments
    svg_text = re.sub(r"<!--.*?-->", "", svg_text, flags=re.DOTALL)

    # Remove namespace prefixes that trip up ElementTree
    svg_text = re.sub(r'xmlns:\w+="[^"]*"', "", svg_text)

    root = ET.fromstring(svg_text)
    if strip_ns(root.tag) != "svg":
        return None, None, "No <svg> root element found."

    viewbox = get_viewbox(root)
    shapes = extract_shapes(root)

    if not shapes:
        return None, None, "No shape elements found."

    # Build clean SVG
    lines = [f'<svg viewBox="{viewbox}" xmlns="http://www.w3.org/2000/svg">']
    for s in shapes:
        tag = s["type"]
        attrs = " ".join(f'{k}="{v}"' for k, v in s.items() if k != "type")
        lines.append(f"  <{tag} {attrs} />")
    lines.append("</svg>")
    clean_svg = "\n".join(lines)

    # Build Claude-ready text
    claude_lines = [f"viewBox: {viewbox}", f"Elements: {len(shapes)}", ""]
    for i, s in enumerate(shapes):
        tag = s["type"]
        extra = ""
        if tag == "path" and "d" in s:
            sub_paths = len(re.findall(r"[Mm]", s["d"]))
            extra = f", {sub_paths} sub-path{'s' if sub_paths != 1 else ''}"
        claude_lines.append(f"--- Element {i} ({tag}{extra}) ---")
        if tag == "path":
            claude_lines.append(s.get("d", ""))
            if "transform" in s:
                claude_lines.append(f"transform: {s['transform']}")
        else:
            attrs = {k: v for k, v in s.items() if k != "type"}
            claude_lines.append(str(attrs))
        claude_lines.append("")
    claude_text = "\n".join(claude_lines)

    summary = f"viewBox: {viewbox} | {len(shapes)} elements"
    type_counts = {}
    for s in shapes:
        type_counts[s["type"]] = type_counts.get(s["type"], 0) + 1
    summary += " (" + ", ".join(f"{v} {k}{'s' if v > 1 else ''}" for k, v in type_counts.items()) + ")"

    return clean_svg, claude_text, summary


def process_file(input_path, output_path=None, claude_output=False):
    """Process a single SVG file."""
    with open(input_path, "r", encoding="utf-8") as f:
        raw = f.read()

    clean_svg, claude_text, summary = anonymize(raw)

    if clean_svg is None:
        print(f"  ERROR: {summary}")
        return False

    print(f"  {summary}")

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(clean_svg)
        print(f"  → Saved: {output_path}")

        if claude_output:
            claude_path = output_path.rsplit(".", 1)[0] + "_claude.txt"
            with open(claude_path, "w", encoding="utf-8") as f:
                f.write(claude_text)
            print(f"  → Saved: {claude_path}")
    else:
        print("\n--- CLEAN SVG ---")
        print(clean_svg)
        print("\n--- CLAUDE-READY TEXT ---")
        print(claude_text)

    return True


def main():
    parser = argparse.ArgumentParser(description="SVG Anonymizer — geometry only")
    parser.add_argument("input", help="SVG file or folder of SVGs")
    parser.add_argument("-o", "--output", help="Output file or folder")
    parser.add_argument("-c", "--claude", action="store_true", help="Also save Claude-ready text file")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        # Batch mode
        svg_files = [f for f in os.listdir(args.input) if f.lower().endswith(".svg")]
        if not svg_files:
            print("No .svg files found in folder.")
            sys.exit(1)

        out_dir = args.output or args.input + "_anonymized"
        os.makedirs(out_dir, exist_ok=True)

        print(f"Processing {len(svg_files)} files...\n")
        success = 0
        for fname in sorted(svg_files):
            print(f"[{fname}]")
            out_path = os.path.join(out_dir, fname)
            if process_file(os.path.join(args.input, fname), out_path, args.claude):
                success += 1
            print()

        print(f"Done: {success}/{len(svg_files)} processed → {out_dir}")

    else:
        # Single file
        if not os.path.exists(args.input):
            print(f"File not found: {args.input}")
            sys.exit(1)

        print(f"[{os.path.basename(args.input)}]")
        process_file(args.input, args.output, args.claude)


if __name__ == "__main__":
    main()