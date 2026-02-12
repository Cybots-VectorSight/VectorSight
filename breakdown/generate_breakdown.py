"""Generate SVG path breakdown images.

Image 1: Original SVG
Image 2: All separate paths (each path rendered individually on same canvas)
Image 3: Closed paths only (paths whose `d` ends with Z)
"""

from __future__ import annotations

import io
import re
import copy
from pathlib import Path
from xml.etree import ElementTree as ET

import cairosvg
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
import matplotlib.patheffects as pe

# ── Config ──────────────────────────────────────────────────────────

SVG_PATH = Path(__file__).resolve().parent.parent / "samples" / "test" / "faker.svg"
OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(exist_ok=True)

CANVAS_W, CANVAS_H = 256, 308
RENDER_SCALE = 2  # render at 2x for clarity

BG = "#0f0f1a"
PANEL_BG = "#161625"
TEXT = "#eee"
SUBTLE = "#777"
ACCENT = "#e94560"

stroke = [pe.withStroke(linewidth=2.5, foreground=BG)]

# ── Parse the SVG ───────────────────────────────────────────────────

svg_text = SVG_PATH.read_text(encoding="utf-8")

# Register SVG namespace to avoid ns0: prefixes
ET.register_namespace("", "http://www.w3.org/2000/svg")
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")

tree = ET.parse(SVG_PATH)
root = tree.getroot()

NS = {"svg": "http://www.w3.org/2000/svg"}

# Find all shape elements (path, circle, ellipse, rect, line, polygon, polyline)
SHAPE_TAGS = {
    f"{{{NS['svg']}}}{tag}"
    for tag in ("path", "circle", "ellipse", "rect", "line", "polygon", "polyline")
}


def find_shapes(element, parent_chain=None):
    """Recursively find all shape elements with their parent group chain."""
    if parent_chain is None:
        parent_chain = []

    results = []
    tag_local = element.tag.split("}")[-1] if "}" in element.tag else element.tag

    if element.tag in SHAPE_TAGS:
        results.append({
            "element": element,
            "tag": tag_local,
            "parents": list(parent_chain),
            "attrib": dict(element.attrib),
        })

    # Recurse into children
    new_chain = parent_chain
    if tag_local == "g":
        new_chain = parent_chain + [element]

    for child in element:
        results.extend(find_shapes(child, new_chain))

    return results


shapes = find_shapes(root)
print(f"Found {len(shapes)} shape elements")

# Classify each path
for i, shape in enumerate(shapes):
    shape["index"] = i
    shape["id"] = f"E{i + 1}"

    d = shape["attrib"].get("d", "")
    shape["d"] = d
    shape["is_closed"] = d.rstrip().upper().endswith("Z") if d else False

    # For non-path elements (circle, rect, etc.), they're always closed
    if shape["tag"] in ("circle", "ellipse", "rect", "polygon"):
        shape["is_closed"] = True
    elif shape["tag"] in ("line", "polyline"):
        shape["is_closed"] = False

    # Get fill/stroke info
    fill = shape["attrib"].get("fill", "")
    stroke_attr = shape["attrib"].get("stroke", "")
    opacity = shape["attrib"].get("opacity", "1")

    # Check parent group for fill/opacity
    for parent in shape["parents"]:
        if not fill:
            fill = parent.attrib.get("fill", "")
        if opacity == "1":
            opacity = parent.attrib.get("opacity", "1")

    shape["fill"] = fill
    shape["opacity"] = opacity

closed_shapes = [s for s in shapes if s["is_closed"]]
open_shapes = [s for s in shapes if not s["is_closed"]]

print(f"  Closed paths: {len(closed_shapes)}")
print(f"  Open paths: {len(open_shapes)}")


# ── Render individual elements ──────────────────────────────────────

def make_solo_svg(shape_idx: int) -> str:
    """Create an SVG where only shape at shape_idx is visible.

    Strategy: clone the SVG tree, then hide all shapes except the target
    by setting fill=none, stroke=none, opacity=0.
    """
    new_tree = copy.deepcopy(tree)
    new_root = new_tree.getroot()

    all_shapes = []
    _collect_shapes(new_root, all_shapes)

    for si, el in enumerate(all_shapes):
        if si != shape_idx:
            el.set("fill", "none")
            el.set("stroke", "none")
            el.set("opacity", "0")
            el.set("fill-opacity", "0")
            el.set("stroke-opacity", "0")
        else:
            # Make sure target is visible
            if el.get("opacity", "1") == "0":
                el.set("opacity", "1")

    return ET.tostring(new_root, encoding="unicode")


def _collect_shapes(element, result):
    """Collect all shape elements in document order."""
    if element.tag in SHAPE_TAGS:
        result.append(element)
    for child in element:
        _collect_shapes(child, result)


def render_svg(svg_str: str, w: int, h: int) -> np.ndarray:
    """Render SVG string to RGBA numpy array."""
    png = cairosvg.svg2png(bytestring=svg_str.encode(), output_width=w, output_height=h)
    return np.array(Image.open(io.BytesIO(png)).convert("RGBA"))


# Render the original
print("\nRendering original...")
orig_arr = render_svg(svg_text, CANVAS_W * RENDER_SCALE, CANVAS_H * RENDER_SCALE)

# Render each shape individually
print(f"Rendering {len(shapes)} individual shapes...")
shape_renders = []
for i, shape in enumerate(shapes):
    try:
        solo = make_solo_svg(i)
        arr = render_svg(solo, CANVAS_W * RENDER_SCALE, CANVAS_H * RENDER_SCALE)
        shape_renders.append(arr)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(shapes)}")
    except Exception as e:
        print(f"  E{i+1} failed: {e}")
        # Create blank
        shape_renders.append(np.zeros((CANVAS_H * RENDER_SCALE, CANVAS_W * RENDER_SCALE, 4), dtype=np.uint8))

print(f"  Done: {len(shape_renders)} renders")


# ── Color palette for element grid ──────────────────────────────────

PALETTE = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#F0A500", "#98D8C8", "#FF8A5C", "#EA5455",
    "#786FA6", "#63CDDA", "#CF6A87", "#574B90", "#E77F67",
    "#F8A5C2", "#778BEB", "#E15F41", "#4B7BEC", "#FD79A8",
]


def hide_ax(ax):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


# ════════════════════════════════════════════════════════════════════
# IMAGE 1: Original SVG
# ════════════════════════════════════════════════════════════════════

print("\nGenerating Image 1: Original...")
fig1, ax1 = plt.subplots(1, 1, figsize=(8, 8), facecolor=BG, dpi=150)
ax1.imshow(orig_arr, extent=[0, CANVAS_W, CANVAS_H, 0])
ax1.set_xlim(0, CANVAS_W)
ax1.set_ylim(CANVAS_H, 0)
ax1.set_aspect("equal")
ax1.set_facecolor(PANEL_BG)
hide_ax(ax1)
ax1.set_title("Original SVG — Apache Flink Squirrel", color=TEXT,
              fontsize=14, fontweight="bold", pad=10)
ax1.text(CANVAS_W / 2, CANVAS_H + 5, f"{len(shapes)} total shape elements  |  canvas {CANVAS_W}×{CANVAS_H}",
         color=SUBTLE, fontsize=9, ha="center")

fig1.savefig(OUT_DIR / "01_original.png", dpi=150, facecolor=BG,
             bbox_inches="tight", pad_inches=0.3)
plt.close(fig1)
print(f"  Saved: {OUT_DIR / '01_original.png'}")


# ════════════════════════════════════════════════════════════════════
# IMAGE 2: All separate paths (grid of individual renders)
# ════════════════════════════════════════════════════════════════════

print("\nGenerating Image 2: All paths grid...")
n = len(shapes)
# Calculate grid size
cols = 10
rows = (n + cols - 1) // cols

fig2, axes2 = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4),
                            facecolor=BG, dpi=120)
fig2.suptitle(f"All {n} Paths — Each Element Rendered Individually",
              color=TEXT, fontsize=16, fontweight="bold", y=0.98)
fig2.text(0.5, 0.96, "Every <path> tag in the SVG shown separately on the same canvas position",
          color=SUBTLE, fontsize=9, ha="center")

for i in range(rows * cols):
    row, col = divmod(i, cols)
    ax = axes2[row, col] if rows > 1 else axes2[col]
    ax.set_facecolor(PANEL_BG)
    hide_ax(ax)

    if i < n:
        shape = shapes[i]
        arr = shape_renders[i]

        # Check if this render has any visible pixels
        alpha_sum = arr[:, :, 3].sum()
        if alpha_sum > 0:
            ax.imshow(arr, extent=[0, CANVAS_W, CANVAS_H, 0])
        ax.set_xlim(0, CANVAS_W)
        ax.set_ylim(CANVAS_H, 0)
        ax.set_aspect("equal")

        # Label
        color = "#4ECDC4" if shape["is_closed"] else "#FF6B6B"
        status = "closed" if shape["is_closed"] else "open"
        ax.set_title(f"E{i + 1}", color=color, fontsize=7, fontweight="bold", pad=2)
        ax.text(CANVAS_W / 2, CANVAS_H - 3, status, color=color,
                fontsize=5, ha="center", va="bottom",
                path_effects=[pe.withStroke(linewidth=1.5, foreground=PANEL_BG)])
    else:
        ax.set_visible(False)

fig2.subplots_adjust(hspace=0.35, wspace=0.15, left=0.02, right=0.98,
                      top=0.93, bottom=0.02)
fig2.savefig(OUT_DIR / "02_all_paths.png", dpi=120, facecolor=BG,
             bbox_inches="tight", pad_inches=0.3)
plt.close(fig2)
print(f"  Saved: {OUT_DIR / '02_all_paths.png'}")


# ════════════════════════════════════════════════════════════════════
# IMAGE 3: Closed paths only (highlighted)
# ════════════════════════════════════════════════════════════════════

print("\nGenerating Image 3: Closed paths comparison...")

n_closed = len(closed_shapes)
n_open = len(open_shapes)

# Two-panel layout: closed paths grid + composite overlay
fig3 = plt.figure(figsize=(20, 12), facecolor=BG, dpi=120)
fig3.suptitle(f"Closed vs Open Paths — {n_closed} closed, {n_open} open",
              color=TEXT, fontsize=18, fontweight="bold", y=0.97)

gs3 = fig3.add_gridspec(1, 2, wspace=0.08, left=0.03, right=0.97, top=0.91, bottom=0.04)

# Left: composite of all closed paths stacked on same canvas
ax3a = fig3.add_subplot(gs3[0, 0])
ax3a.set_facecolor(PANEL_BG)
hide_ax(ax3a)
ax3a.set_title(f"Closed Paths ({n_closed}) — composited", color="#4ECDC4",
               fontsize=13, fontweight="bold", pad=10, loc="left")

# Stack all closed path renders
composite_closed = np.zeros_like(orig_arr, dtype=np.float64)
for shape in closed_shapes:
    arr = shape_renders[shape["index"]]
    alpha = arr[:, :, 3:4].astype(np.float64) / 255.0
    rgb = arr[:, :, :3].astype(np.float64)
    composite_closed[:, :, :3] = composite_closed[:, :, :3] * (1 - alpha) + rgb * alpha
    composite_closed[:, :, 3] = np.clip(composite_closed[:, :, 3] + alpha[:, :, 0] * 255, 0, 255)

composite_closed = np.clip(composite_closed, 0, 255).astype(np.uint8)
ax3a.imshow(composite_closed, extent=[0, CANVAS_W, CANVAS_H, 0])
ax3a.set_xlim(0, CANVAS_W)
ax3a.set_ylim(CANVAS_H, 0)
ax3a.set_aspect("equal")

# List some closed path IDs
ids_text = ", ".join(f"E{s['index']+1}" for s in closed_shapes[:15])
if len(closed_shapes) > 15:
    ids_text += f" ... (+{len(closed_shapes) - 15} more)"
ax3a.text(CANVAS_W / 2, CANVAS_H + 5, ids_text, color=SUBTLE, fontsize=6, ha="center")

# Right: composite of open paths
ax3b = fig3.add_subplot(gs3[0, 1])
ax3b.set_facecolor(PANEL_BG)
hide_ax(ax3b)
ax3b.set_title(f"Open Paths ({n_open}) — composited", color="#FF6B6B",
               fontsize=13, fontweight="bold", pad=10, loc="left")

composite_open = np.zeros_like(orig_arr, dtype=np.float64)
for shape in open_shapes:
    arr = shape_renders[shape["index"]]
    alpha = arr[:, :, 3:4].astype(np.float64) / 255.0
    rgb = arr[:, :, :3].astype(np.float64)
    composite_open[:, :, :3] = composite_open[:, :, :3] * (1 - alpha) + rgb * alpha
    composite_open[:, :, 3] = np.clip(composite_open[:, :, 3] + alpha[:, :, 0] * 255, 0, 255)

composite_open = np.clip(composite_open, 0, 255).astype(np.uint8)
ax3b.imshow(composite_open, extent=[0, CANVAS_W, CANVAS_H, 0])
ax3b.set_xlim(0, CANVAS_W)
ax3b.set_ylim(CANVAS_H, 0)
ax3b.set_aspect("equal")

ids_text_open = ", ".join(f"E{s['index']+1}" for s in open_shapes[:15])
if len(open_shapes) > 15:
    ids_text_open += f" ... (+{len(open_shapes) - 15} more)"
ax3b.text(CANVAS_W / 2, CANVAS_H + 5, ids_text_open, color=SUBTLE, fontsize=6, ha="center")

fig3.text(0.5, 0.01,
          "Closed paths (d ends with Z) form filled shapes — these are the building blocks of the illustration.\n"
          "Open paths are strokes, arcs, and decorative lines.",
          color=SUBTLE, fontsize=9, ha="center")

fig3.savefig(OUT_DIR / "03_closed_vs_open.png", dpi=120, facecolor=BG,
             bbox_inches="tight", pad_inches=0.3)
plt.close(fig3)
print(f"  Saved: {OUT_DIR / '03_closed_vs_open.png'}")

print(f"\nAll done! Files in: {OUT_DIR}")
