"""Generate a multi-panel PNG poster showing what VectorSight's pipeline sees.

V2 — Redesigned for clarity: each panel isolates one concept instead of
overlaying everything on the same busy base image.
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import cairosvg
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
import matplotlib.patheffects as pe

# ── Render the Flink SVG ────────────────────────────────────────────

SVG_PATH = Path(__file__).resolve().parent.parent / "samples" / "test" / "apache-flink-icon (1).svg"
svg_text = SVG_PATH.read_text(encoding="utf-8")

CANVAS_W, CANVAS_H = 256, 259
png_bytes = cairosvg.svg2png(bytestring=svg_text.encode(), output_width=CANVAS_W * 2, output_height=CANVAS_H * 2)
base_img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
base_arr = np.array(base_img)

# Also render at 1x for smaller insets
png_1x = cairosvg.svg2png(bytestring=svg_text.encode(), output_width=CANVAS_W, output_height=CANVAS_H)
base_1x = np.array(Image.open(io.BytesIO(png_1x)).convert("RGBA"))

# ── Style ───────────────────────────────────────────────────────────

BG = "#0f0f1a"
PANEL_BG = "#161625"
ACCENT = "#e94560"
TEXT = "#eee"
SUBTLE = "#777"
TEAL = "#4ECDC4"
RED = "#FF6B6B"
BLUE = "#45B7D1"
GREEN = "#96CEB4"
YELLOW = "#FFEAA7"
PLUM = "#DDA0DD"
ORANGE = "#F0A500"

stroke = [pe.withStroke(linewidth=3, foreground=BG)]
stroke_thin = [pe.withStroke(linewidth=2, foreground=BG)]


def hide(ax):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


# ── Enrichment data ─────────────────────────────────────────────────

# Key elements with their bboxes (from real enrichment output)
# (id, x1, y1, x2, y2, cx, cy, shape_class, circ, area_desc, color_name)
KEY_ELEMENTS = [
    ("E62", 0, 0, 256, 259, 128, 130, "organic", 0.03, "primary boundary\n(overall outline)", "#0D0D0D"),
    ("E1", 4, 107, 253, 258, 130, 168, "organic", 0.16, "body mass\n(red region)", "#E65270"),
    ("E12", 52, 62, 204, 197, 128, 130, "organic", 0.09, "inner structure\n(body detail)", "#666"),
    ("E10", 218, 134, 237, 160, 229, 147, "elliptical", 0.86, "eye (white)\ncirc=0.86", "#F9E0E7"),
    ("E55", 222, 140, 236, 154, 229, 147, "circular", 1.00, "pupil\ncirc=1.00", "#333"),
    ("E22", 138, 7, 182, 67, 159, 33, "organic", 0.73, "ear\ncirc=0.73", "#888"),
    ("E7", 24, 135, 90, 201, 57, 168, "organic", 0.23, "hind leg\ncirc=0.23", "#666"),
    ("E40", 216, 198, 252, 229, 235, 215, "circular", 0.86, "front paw\ncirc=0.86", "#888"),
]

# Concentric groups (from enrichment)
CONCENTRIC_GROUPS = [
    {
        "label": "Eye",
        "members": ["E10", "E55", "E11"],
        "center": (229, 147),
        "radii": [15, 10, 6],  # approximate radii of each ring
        "color": RED,
        "crop": (205, 125, 255, 170),  # crop region for zoom
    },
    {
        "label": "Paw",
        "members": ["E40", "E41", "E50", "E51"],
        "center": (235, 215),
        "radii": [20, 14, 22, 22],
        "color": ORANGE,
        "crop": (205, 190, 259, 240),
    },
    {
        "label": "Ear",
        "members": ["E22", "E21"],
        "center": (159, 33),
        "radii": [32, 35],
        "color": BLUE,
        "crop": (125, 0, 195, 80),
    },
]

# Containment tree (from enrichment)
TREE = """E62 [primary boundary] ── the squirrel outline
├── E12 [inner structure]
│   ├── E17, E18, E19 (body layers)
│   ├── E32, E34, E35 (detail shapes)
│   └── E27, E38, E39 (accents)
├── E30 [white stripe]
├── E15, E16 [body stripes]
├── E33, E45, E59 [leg/tail details]
└── E3, E13, E28 (small features)

E1 [body mass] ── behind E62
├── E29 [large body shape]
├── E40, E41, E50, E51 (front paw)
├── E2, E5, E46 (accent shapes)
└── E61, E67 (bottom details)

E22 [ear] ── drawn on top
└── E20, E21 (ear layers)

E23 [head-top] ── drawn on top
└── E24, E26 (forehead layers)

E7 [hind leg]
└── E8, E9, E58, E60 (leg layers)"""

# Simplified outline
OUTLINE = [
    (192, 229), (168, 221), (164, 208), (140, 208),
    (60, 168), (14, 113), (14, 73), (200, 42),
    (216, 30), (240, 38), (246, 45), (170, 117),
    (174, 138), (198, 170), (198, 210),
]

# Radial profile (36 values at 10° intervals)
RADIAL = [
    11, 14, 19, 30, 32, 24, 20, 18, 17, 16, 16, 16,
    17, 18, 21, 24, 29, 28, 26, 22, 20, 20, 18, 18,
    18, 18, 19, 19, 22, 23, 20, 28, 24, 18, 13, 11,
]

# Symmetry pairs
SYM_PAIRS = [
    ("E2", "E59", (42, 235), (70, 178)),
    ("E3", "E7", (51, 115), (57, 168)),
    ("E10", "E44", (229, 147), (188, 142)),
    ("E15", "E32", (103, 112), (177, 146)),
    ("E16", "E34", (111, 121), (173, 146)),
    ("E27", "E38", (153, 138), (108, 99)),
    ("E28", "E35", (99, 80), (177, 144)),
]

# ASCII grids
ASCII_INTERIOR = [
    "..###....#....",
    "..######.##...",
    ".#.##.#####...",
    ".##########...",
    ".###########..",
    ".###########..",
    "###########...",
    "############..",
    "#########.....",
    ".####....##...",
    "...####..##...",
    ".........#....",
    "........##....",
    "..............",
]


# ════════════════════════════════════════════════════════════════════
# BUILD THE FIGURE — 2 rows x 3 columns
# ════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(22, 15), facecolor=BG, dpi=150)

fig.suptitle(
    "VectorSight — What the Pipeline Sees",
    color=TEXT, fontsize=24, fontweight="bold", y=0.975,
)
fig.text(
    0.5, 0.955,
    "62 geometric transforms turn raw SVG code into structured spatial text for an LLM — demonstrated on the Apache Flink squirrel (67 elements)",
    color=SUBTLE, fontsize=10, ha="center",
)

gs = fig.add_gridspec(
    2, 3, hspace=0.35, wspace=0.22,
    left=0.03, right=0.97, top=0.93, bottom=0.04,
)


# ═══════════════════════════════════════════════════════════════════
# PANEL 1 — Original SVG + a few isolated elements
# ═══════════════════════════════════════════════════════════════════

ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(PANEL_BG)
hide(ax1)
ax1.set_title("1. Input → Element Extraction", color=TEXT, fontsize=12,
              fontweight="bold", pad=10, loc="left")

# Show original SVG on the left half
ax1.set_xlim(0, 100)
ax1.set_ylim(100, 0)

# Original SVG (takes left ~40%)
ax1.imshow(base_arr, extent=[2, 40, 95, 2])

ax1.text(21, 98, "67 <path> tags", color=SUBTLE, fontsize=7, ha="center")

# Arrow from SVG to extracted elements
ax1.annotate("", xy=(45, 20), xytext=(40, 20),
             arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2))
ax1.annotate("", xy=(45, 50), xytext=(40, 50),
             arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2))
ax1.annotate("", xy=(45, 80), xytext=(40, 80),
             arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2))

ax1.text(42, 10, "parse", color=ACCENT, fontsize=7, ha="center", fontweight="bold", rotation=90)

# Show 3 key extracted elements as isolated shapes with their bboxes
# We'll crop from the rendered image to show individual elements
elements_to_show = [
    ("E55 — pupil", "circular, circ=1.00", TEAL,
     222, 140, 236, 154, 47, 5, 72, 30),
    ("E22 — ear", "organic, circ=0.73", BLUE,
     138, 7, 182, 67, 47, 35, 72, 65),
    ("E1 — body", "organic, circ=0.16", RED,
     4, 107, 253, 258, 47, 68, 72, 95),
]

for label, desc, color, sx1, sy1, sx2, sy2, dx1, dy1, dx2, dy2 in elements_to_show:
    # Draw bbox rectangle in the display area
    rect = mpatches.FancyBboxPatch(
        (dx1, dy1), dx2 - dx1, dy2 - dy1,
        boxstyle="round,pad=0.5", linewidth=1.5,
        edgecolor=color, facecolor=to_rgba(color, 0.06),
    )
    ax1.add_patch(rect)

    # Crop the element from the 2x rendered image
    # Map SVG coords to pixel coords (2x scale)
    px1 = max(0, int(sx1 * 2))
    py1 = max(0, int(sy1 * 2))
    px2 = min(base_arr.shape[1], int(sx2 * 2))
    py2 = min(base_arr.shape[0], int(sy2 * 2))
    crop = base_arr[py1:py2, px1:px2]
    if crop.size > 0:
        ax1.imshow(crop, extent=[dx1 + 0.5, dx2 - 0.5, dy2 - 0.5, dy1 + 0.5], zorder=2)

    # Label
    ax1.text(dx2 + 1, dy1 + 2, label, color=color, fontsize=6.5,
             fontweight="bold", path_effects=stroke_thin)
    ax1.text(dx2 + 1, dy1 + 6, desc, color=SUBTLE, fontsize=5.5)


# ═══════════════════════════════════════════════════════════════════
# PANEL 2 — Bounding Boxes on CLEAN view (fewer, labeled clearly)
# ═══════════════════════════════════════════════════════════════════

ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(base_arr, extent=[0, CANVAS_W, CANVAS_H, 0], alpha=0.25)
ax2.set_xlim(0, CANVAS_W)
ax2.set_ylim(CANVAS_H, 0)
ax2.set_aspect("equal")
ax2.set_facecolor(PANEL_BG)
hide(ax2)
ax2.set_title("2. Per-Element: Bounding Boxes + Centroids", color=TEXT,
              fontsize=12, fontweight="bold", pad=10, loc="left")

# Show only 5 NON-overlapping key elements with clear bboxes
show_elements = [
    ("E10\neye", 218, 134, 237, 160, 229, 147, RED, 0.86),
    ("E55\npupil", 222, 140, 236, 154, 229, 147, ORANGE, 1.00),
    ("E22\near", 138, 7, 182, 67, 159, 33, BLUE, 0.73),
    ("E40\npaw", 216, 198, 252, 229, 235, 215, GREEN, 0.86),
    ("E7\nhind leg", 24, 135, 90, 201, 57, 168, PLUM, 0.23),
    ("E14\nstripe", 102, 24, 165, 40, 135, 33, YELLOW, 0.11),
]

for label, x1, y1, x2, y2, cx, cy, color, circ in show_elements:
    w, h = x2 - x1, y2 - y1
    # Bbox
    rect = mpatches.Rectangle(
        (x1, y1), w, h, linewidth=2, edgecolor=color,
        facecolor=to_rgba(color, 0.12), linestyle="-",
    )
    ax2.add_patch(rect)
    # Centroid crosshair
    ax2.plot(cx, cy, "+", color=color, markersize=8, markeredgewidth=1.5)
    # Label (positioned outside bbox where possible)
    lx = x2 + 3 if x2 < 200 else x1 - 3
    ha = "left" if x2 < 200 else "right"
    ly = y1
    ax2.text(lx, ly, label, color=color, fontsize=7,
             fontweight="bold", ha=ha, va="top", path_effects=stroke)
    # Circularity badge
    ax2.text(lx, ly + 22 if ha == "left" else ly + 22, f"circ={circ:.2f}",
             color=SUBTLE, fontsize=5.5, ha=ha, path_effects=stroke_thin)

# Dimension lines for one element to show bbox concept
ax2.annotate("", xy=(237, 132), xytext=(218, 132),
             arrowprops=dict(arrowstyle="<->", color=TEXT, lw=1))
ax2.text(228, 128, "19px", color=TEXT, fontsize=5.5, ha="center", path_effects=stroke_thin)

ax2.text(128, 250, "Each element gets: bbox, centroid, area, circularity, convexity, aspect ratio,\n"
         "shape class, symmetry, corners, Fourier descriptors, Zernike moments...",
         color=SUBTLE, fontsize=6, ha="center", va="bottom")


# ═══════════════════════════════════════════════════════════════════
# PANEL 3 — Concentric Groups (ZOOMED IN)
# ═══════════════════════════════════════════════════════════════════

ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor(PANEL_BG)
hide(ax3)
ax3.set_title("3. Concentric Groups (zoomed)", color=TEXT, fontsize=12,
              fontweight="bold", pad=10, loc="left")
ax3.set_xlim(0, 100)
ax3.set_ylim(100, 0)

# Show 3 concentric groups as zoomed crops with ring annotations
group_positions = [
    (5, 5, 45, 50),    # eye — top-left
    (55, 5, 95, 50),   # paw — top-right
    (20, 55, 80, 95),  # ear — bottom-center
]

for gi, group in enumerate(CONCENTRIC_GROUPS):
    dx1, dy1, dx2, dy2 = group_positions[gi]
    sx1, sy1, sx2, sy2 = group["crop"]

    # Crop from 2x rendered image
    px1 = max(0, int(sx1 * 2))
    py1 = max(0, int(sy1 * 2))
    px2 = min(base_arr.shape[1], int(sx2 * 2))
    py2 = min(base_arr.shape[0], int(sy2 * 2))
    crop = base_arr[py1:py2, px1:px2]

    if crop.size > 0:
        ax3.imshow(crop, extent=[dx1, dx2, dy2, dy1], zorder=1)

    # Border
    rect = mpatches.Rectangle(
        (dx1, dy1), dx2 - dx1, dy2 - dy1,
        linewidth=2, edgecolor=group["color"],
        facecolor="none", zorder=3,
    )
    ax3.add_patch(rect)

    # Draw concentric rings overlaid
    mid_x = (dx1 + dx2) / 2
    mid_y = (dy1 + dy2) / 2
    scale_x = (dx2 - dx1) / (sx2 - sx1)
    scale_y = (dy2 - dy1) / (sy2 - sy1)
    scale = min(scale_x, scale_y)

    for ri, radius in enumerate(group["radii"]):
        r_scaled = radius * scale * 0.9
        circle = plt.Circle(
            (mid_x, mid_y), r_scaled,
            fill=False, edgecolor=group["color"],
            linewidth=1, linestyle="--", alpha=0.7, zorder=4,
        )
        ax3.add_patch(circle)

    # Labels
    ax3.text(dx1 + 1, dy1 + 3, group["label"], color=group["color"],
             fontsize=9, fontweight="bold", path_effects=stroke, zorder=5)
    members_str = " → ".join(group["members"])
    ax3.text(dx1 + 1, dy1 + 7.5, members_str, color=SUBTLE,
             fontsize=5.5, path_effects=stroke_thin, zorder=5)
    ax3.text(dx1 + 1, dy1 + 11, f"{len(group['members'])} layers, shared center",
             color=SUBTLE, fontsize=5, path_effects=stroke_thin, zorder=5)

# Explanation
ax3.text(50, 52, '"Elements sharing the same center = concentric group.\nThe pipeline detects these as multi-ring features\n(eyes, buttons, dials, paws...)"',
         color=GREEN, fontsize=6.5, ha="center", va="top", style="italic",
         path_effects=stroke_thin)


# ═══════════════════════════════════════════════════════════════════
# PANEL 4 — Simplified Outline + Radial Profile (CLEAN background)
# ═══════════════════════════════════════════════════════════════════

# Split into two sub-panels
gs_inner = gs[1, 0].subgridspec(1, 2, wspace=0.08)

# Left: Simplified outline on clean background
ax4a = fig.add_subplot(gs_inner[0, 0])
ax4a.set_facecolor(PANEL_BG)
hide(ax4a)
ax4a.set_title("4a. Simplified Outline", color=TEXT, fontsize=10,
               fontweight="bold", pad=8, loc="left")
ax4a.set_xlim(-10, CANVAS_W + 10)
ax4a.set_ylim(CANVAS_H + 10, -30)
ax4a.set_aspect("equal")

# Draw outline polygon
ox = [v[0] for v in OUTLINE] + [OUTLINE[0][0]]
oy = [v[1] for v in OUTLINE] + [OUTLINE[0][1]]
ax4a.fill(ox, oy, alpha=0.2, color=TEAL)
ax4a.plot(ox, oy, color=TEAL, linewidth=2.5, alpha=0.9)

# Numbered vertices
for i, (vx, vy) in enumerate(OUTLINE):
    ax4a.plot(vx, vy, "o", color=TEAL, markersize=5,
              markeredgecolor="white", markeredgewidth=0.5)
    # Offset labels to avoid overlap
    offsets = [
        (5, 8), (5, -5), (-8, -5), (-8, 5), (-10, 5), (-10, -5),
        (-8, -8), (5, -5), (5, -8), (8, 5), (8, 5), (5, 8),
        (8, 5), (8, -5), (8, 5),
    ]
    ox_off, oy_off = offsets[i % len(offsets)]
    ax4a.text(vx + ox_off, vy + oy_off, f"v{i+1}", color=TEAL,
              fontsize=5, ha="center", path_effects=stroke_thin)

# Small reference image in corner
ax4a.imshow(base_1x, extent=[CANVAS_W - 65, CANVAS_W + 5, 70, 0], alpha=0.6, zorder=5)
ax4a.text(CANVAS_W - 30, 75, "original", color=SUBTLE, fontsize=5.5, ha="center")

ax4a.text(CANVAS_W / 2, CANVAS_H + 5,
          "15 vertices capture the overall shape\n→ LLM traces this path mentally",
          color=SUBTLE, fontsize=6.5, ha="center")

# Right: Radial profile as polar plot
ax4b = fig.add_subplot(gs_inner[0, 1], projection="polar")
ax4b.set_facecolor(PANEL_BG)
ax4b.set_title("4b. Radial Profile", color=TEXT, fontsize=10,
               fontweight="bold", pad=15, loc="center")

angles_rad = [math.radians(i * 10) for i in range(36)]
angles_rad_closed = angles_rad + [angles_rad[0]]
radial_closed = RADIAL + [RADIAL[0]]

ax4b.fill(angles_rad_closed, radial_closed, alpha=0.2, color=TEAL)
ax4b.plot(angles_rad_closed, radial_closed, color=TEAL, linewidth=2)

# Highlight peaks
peaks = [(40, 32, "ear"), (160, 29, "tail"), (310, 28, "foot")]
for angle_deg, dist, label in peaks:
    ar = math.radians(angle_deg)
    ax4b.plot(ar, dist, "o", color=RED, markersize=6, zorder=5)
    ax4b.annotate(
        f"{label}\n{angle_deg}°",
        xy=(ar, dist), xytext=(ar + 0.3, dist + 5),
        color=RED, fontsize=6, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=RED, lw=0.8),
        path_effects=stroke_thin,
    )

ax4b.set_rticks([10, 20, 30])
ax4b.set_rlabel_position(270)
ax4b.tick_params(colors=SUBTLE, labelsize=6)
ax4b.grid(color=SUBTLE, alpha=0.3)
ax4b.spines["polar"].set_color(SUBTLE)
ax4b.spines["polar"].set_alpha(0.3)


# ═══════════════════════════════════════════════════════════════════
# PANEL 5 — Containment Tree (text diagram)
# ═══════════════════════════════════════════════════════════════════

ax5 = fig.add_subplot(gs[1, 1])
ax5.set_facecolor(PANEL_BG)
hide(ax5)
ax5.set_title("5. Containment Hierarchy", color=TEXT, fontsize=12,
              fontweight="bold", pad=10, loc="left")
ax5.set_xlim(0, 100)
ax5.set_ylim(100, 0)

# Show the SVG small on the left with containment layers highlighted
ax5.imshow(base_arr, extent=[2, 35, 72, 2], alpha=0.5)

# Draw containment layers on the small SVG
layers = [
    ("E62", 0, 0, 256, 259, TEAL, "boundary"),
    ("E1", 4, 107, 253, 258, RED, "body"),
    ("E12", 52, 62, 204, 197, GREEN, "inner"),
]
svgw = 35 - 2
svgh = 72 - 2
for lid, x1, y1, x2, y2, color, label in layers:
    # Scale to display coords
    dx1 = 2 + x1 / CANVAS_W * svgw
    dy1 = 2 + y1 / CANVAS_H * svgh
    dx2 = 2 + x2 / CANVAS_W * svgw
    dy2 = 2 + y2 / CANVAS_H * svgh
    rect = mpatches.Rectangle(
        (dx1, dy1), dx2 - dx1, dy2 - dy1,
        linewidth=1.5, edgecolor=color,
        facecolor=to_rgba(color, 0.08), linestyle="--",
    )
    ax5.add_patch(rect)

# Arrow to tree
ax5.annotate("", xy=(40, 12), xytext=(36, 12),
             arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2))

# Render the tree text
tree_lines = TREE.split("\n")
for i, line in enumerate(tree_lines):
    y = 5 + i * 4.2
    if y > 95:
        break

    # Color parent elements differently
    color = TEXT
    if "E62" in line and "primary" in line:
        color = TEAL
    elif "E1 [" in line:
        color = RED
    elif "E12" in line and "inner" in line:
        color = GREEN
    elif "E22" in line:
        color = BLUE
    elif "E23" in line:
        color = ORANGE
    elif "E7" in line:
        color = PLUM
    elif line.startswith("├") or line.startswith("│") or line.startswith("└"):
        color = SUBTLE

    ax5.text(40, y, line, color=color, fontsize=5.8,
             fontfamily="monospace", path_effects=stroke_thin)

ax5.text(50, 97, '"Who contains who" — the nesting hierarchy defines structural layers',
         color=GREEN, fontsize=6.5, ha="center", style="italic",
         path_effects=stroke_thin)


# ═══════════════════════════════════════════════════════════════════
# PANEL 6 — Final Output: ASCII Grid + Enrichment Stats
# ═══════════════════════════════════════════════════════════════════

ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor(PANEL_BG)
hide(ax6)
ax6.set_title("6. Output → What the LLM Actually Reads", color=TEXT,
              fontsize=12, fontweight="bold", pad=10, loc="left")
ax6.set_xlim(0, 100)
ax6.set_ylim(100, 0)

# ASCII grid on the left
ax6.text(2, 3, "ASCII art of interior features (14×14):", color=TEAL,
         fontsize=7, fontweight="bold", fontfamily="monospace")

cell = 2.8
grid_x0 = 4
grid_y0 = 7
for r, row in enumerate(ASCII_INTERIOR):
    for c, ch in enumerate(row):
        x = grid_x0 + c * cell
        y = grid_y0 + r * cell
        if ch == "#":
            ax6.add_patch(mpatches.Rectangle(
                (x, y), cell * 0.88, cell * 0.88,
                facecolor=TEAL, alpha=0.85, edgecolor="none",
            ))
        else:
            ax6.add_patch(mpatches.Rectangle(
                (x, y), cell * 0.88, cell * 0.88,
                facecolor="#1a1a2e", alpha=0.4, edgecolor="none",
            ))

# Small original for comparison
grid_bottom = grid_y0 + len(ASCII_INTERIOR) * cell + 2
ax6.text(grid_x0, grid_bottom, "Compare:", color=SUBTLE, fontsize=6)
ax6.imshow(base_1x, extent=[grid_x0 + 14, grid_x0 + 38, grid_bottom + 24, grid_bottom + 2], alpha=0.7)
ax6.annotate("", xy=(grid_x0 + 14, grid_bottom + 13), xytext=(grid_x0 + 10, grid_bottom + 13),
             arrowprops=dict(arrowstyle="<->", color=SUBTLE, lw=1))

# Stats on the right
mx = 58
sections = [
    (ACCENT, "bold", "ENRICHMENT SECTIONS:"),
    (TEXT, None, ""),
    (TEAL, "bold", "Spatial Interpretation"),
    (SUBTLE, None, "  silhouette, mass, skeleton,"),
    (SUBTLE, None, "  orientation, focal elements"),
    (TEXT, None, ""),
    (GREEN, "bold", "Shape Narrative"),
    (SUBTLE, None, "  contour walk, paired features"),
    (TEXT, None, ""),
    (BLUE, "bold", "Visual Pyramid"),
    (SUBTLE, None, "  ASCII grids at 4 zoom levels"),
    (TEXT, None, ""),
    (ORANGE, "bold", "Reconstruction Guide"),
    (SUBTLE, None, "  step-by-step mental image"),
    (TEXT, None, ""),
    (RED, "bold", "Key Elements"),
    (SUBTLE, None, "  top elements with measurements"),
    (TEXT, None, ""),
    (PLUM, "bold", "Structure"),
    (SUBTLE, None, "  containment, symmetry, overlaps"),
    (TEXT, None, ""),
    (YELLOW, "bold", "Learned Patterns"),
    (SUBTLE, None, "  accumulated analysis wisdom"),
]

for si, (color, weight, text) in enumerate(sections):
    y = 5 + si * 3.5
    ax6.text(mx, y, text, color=color, fontsize=6.5,
             fontweight=weight or "normal", fontfamily="monospace")

# Token budget box
ax6.add_patch(mpatches.FancyBboxPatch(
    (mx - 2, 87), 42, 10,
    boxstyle="round,pad=1", facecolor=to_rgba(ACCENT, 0.15),
    edgecolor=ACCENT, linewidth=1,
))
ax6.text(mx + 19, 90, "~1900 words total", color=ACCENT, fontsize=9,
         ha="center", fontweight="bold")
ax6.text(mx + 19, 94, "structured text → injected into LLM prompt",
         color=SUBTLE, fontsize=6.5, ha="center")


# ═══ Flow arrows ════════════════════════════════════════════════════

arrow_kw = dict(arrowstyle="-|>", color=ACCENT, lw=2.5, connectionstyle="arc3,rad=0")

# Top row: 1→2→3
fig.patches.append(FancyArrowPatch((0.315, 0.72), (0.345, 0.72), transform=fig.transFigure, **arrow_kw, zorder=10))
fig.patches.append(FancyArrowPatch((0.645, 0.72), (0.675, 0.72), transform=fig.transFigure, **arrow_kw, zorder=10))
# Vertical
fig.patches.append(FancyArrowPatch((0.50, 0.475), (0.50, 0.445), transform=fig.transFigure, **arrow_kw, zorder=10))
# Bottom row: 4→5→6
fig.patches.append(FancyArrowPatch((0.315, 0.26), (0.345, 0.26), transform=fig.transFigure, **arrow_kw, zorder=10))
fig.patches.append(FancyArrowPatch((0.645, 0.26), (0.675, 0.26), transform=fig.transFigure, **arrow_kw, zorder=10))

for fx, fy, label in [
    (0.330, 0.74, "Layer 0+1"),
    (0.660, 0.74, "Layer 3"),
    (0.525, 0.46, "Interpreter"),
    (0.330, 0.28, "Analysis"),
    (0.660, 0.28, "Formatter"),
]:
    fig.text(fx, fy, label, color=ACCENT, fontsize=7, ha="center", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.2", facecolor=BG, edgecolor=ACCENT, alpha=0.9))


# ═══ Footer ═════════════════════════════════════════════════════════

fig.text(
    0.5, 0.008,
    "VectorSight by Cybots  │  The Strange Data Project  │  No fine-tuning, no RAG — pure data transformation",
    color=SUBTLE, fontsize=8, ha="center", style="italic",
)


# ═══ Save ═══════════════════════════════════════════════════════════

out_path = Path(__file__).resolve().parent.parent / "pipeline-poster.png"
fig.savefig(out_path, dpi=150, facecolor=BG, bbox_inches="tight", pad_inches=0.3)
plt.close(fig)
print(f"\nSaved: {out_path}")
print(f"Size: {out_path.stat().st_size / 1024:.0f} KB")
