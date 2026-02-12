"""Extract the single silhouette path from the composite SVG.

Algorithm:
1. Render SVG to high-res raster
2. Alpha-threshold to binary mask (any visible pixel = 1)
3. Extract outer contour using marching squares (scikit-image)
4. Simplify with Douglas-Peucker to get a clean polygon
5. Convert pixel coords back to SVG viewBox coords
6. Output as SVG path + comparison image
"""

from __future__ import annotations

import io
from pathlib import Path

import cairosvg
import numpy as np
from PIL import Image
from skimage import measure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

# ── Config ──────────────────────────────────────────────────────────

SVG_PATH = Path(__file__).resolve().parent.parent / "samples" / "test" / "faker.svg"
OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(exist_ok=True)

CANVAS_W, CANVAS_H = 256, 308

# Render at high res for accurate contour extraction
RENDER_MULT = 4
RENDER_W = CANVAS_W * RENDER_MULT
RENDER_H = CANVAS_H * RENDER_MULT

BG = "#0f0f1a"
PANEL_BG = "#161625"
TEXT = "#eee"
SUBTLE = "#777"
TEAL = "#4ECDC4"
RED = "#FF6B6B"
ACCENT = "#e94560"

stroke = [pe.withStroke(linewidth=2.5, foreground=BG)]

# ── Render SVG ──────────────────────────────────────────────────────

svg_text = SVG_PATH.read_text(encoding="utf-8")

print("Rendering SVG at high resolution...")
png_bytes = cairosvg.svg2png(
    bytestring=svg_text.encode(),
    output_width=RENDER_W, output_height=RENDER_H,
)
img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
img_arr = np.array(img)

# Also render at 2x for display
png_2x = cairosvg.svg2png(bytestring=svg_text.encode(),
                           output_width=CANVAS_W * 2, output_height=CANVAS_H * 2)
display_arr = np.array(Image.open(io.BytesIO(png_2x)).convert("RGBA"))

# ── Extract binary mask ─────────────────────────────────────────────

print("Extracting binary silhouette...")

# Any pixel with alpha > 0 is part of the silhouette
alpha = img_arr[:, :, 3]
binary = (alpha > 10).astype(np.uint8)  # threshold at alpha=10 to ignore anti-aliasing noise

print(f"  Image size: {RENDER_W}x{RENDER_H}")
print(f"  Filled pixels: {binary.sum()} / {binary.size} ({100 * binary.sum() / binary.size:.1f}%)")

# ── Pad the mask so contours don't split at canvas edges ────────────

PAD = 4  # pixels of padding
binary_padded = np.pad(binary, PAD, mode="constant", constant_values=0)
print(f"  Padded to {binary_padded.shape[1]}x{binary_padded.shape[0]} (+{PAD}px border)")

# ── Extract contours ────────────────────────────────────────────────

print("Finding contours (marching squares)...")

# find_contours returns contours at sub-pixel accuracy
contours = measure.find_contours(binary_padded, level=0.5)

# Offset contour coords back by PAD to undo padding
contours = [c - PAD for c in contours]

print(f"  Found {len(contours)} contours")

# Sort by length (number of points) — longest = outer boundary
contours.sort(key=len, reverse=True)

for i, c in enumerate(contours[:5]):
    print(f"  Contour {i}: {len(c)} points")

# The longest contour is the outer silhouette
outer_contour = contours[0]
print(f"\nOuter silhouette: {len(outer_contour)} raw points")

# ── Simplify with Douglas-Peucker ──────────────────────────────────

from skimage.measure import approximate_polygon

# Try multiple tolerance levels
tolerances = [1.0, 2.0, 3.0, 5.0, 8.0]
simplified = {}
for tol in tolerances:
    approx = approximate_polygon(outer_contour, tolerance=tol)
    simplified[tol] = approx
    print(f"  Douglas-Peucker (tol={tol}): {len(approx)} vertices")

# Pick a good balance — ~80-150 vertices gives a clean but accurate silhouette
best_tol = 3.0
for tol in tolerances:
    if 50 <= len(simplified[tol]) <= 200:
        best_tol = tol
        break

silhouette = simplified[best_tol]
print(f"\nUsing tolerance={best_tol}: {len(silhouette)} vertices")

# ── Convert to SVG coordinates ──────────────────────────────────────

# Contour points are in (row, col) = (y_pixel, x_pixel)
# Convert to SVG viewBox coordinates
svg_points = []
for row, col in silhouette:
    svg_x = col / RENDER_MULT  # pixel → SVG x
    svg_y = row / RENDER_MULT  # pixel → SVG y
    svg_points.append((round(svg_x, 1), round(svg_y, 1)))

# Build SVG path d attribute
d_parts = [f"M {svg_points[0][0]},{svg_points[0][1]}"]
for x, y in svg_points[1:]:
    d_parts.append(f"L {x},{y}")
d_parts.append("Z")
svg_d = " ".join(d_parts)

print(f"\nSVG path: {len(svg_d)} chars, {len(svg_points)} vertices")

# ── Save the silhouette as standalone SVG ───────────────────────────

silhouette_svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {CANVAS_W} {CANVAS_H}" width="{CANVAS_W}" height="{CANVAS_H}">
  <path d="{svg_d}" fill="{TEAL}" fill-opacity="0.3" stroke="{TEAL}" stroke-width="1.5"/>
</svg>'''

svg_out = OUT_DIR / "04_silhouette.svg"
svg_out.write_text(silhouette_svg)
print(f"Saved SVG: {svg_out}")

# ── Generate comparison image ───────────────────────────────────────

print("\nGenerating comparison image...")


def hide(ax):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


fig = plt.figure(figsize=(22, 10), facecolor=BG, dpi=150)
fig.suptitle("Silhouette Extraction — One Path to Rule Them All",
             color=TEXT, fontsize=20, fontweight="bold", y=0.97)
fig.text(0.5, 0.94,
         f"67 paths → alpha threshold → marching squares contour → Douglas-Peucker simplification → {len(svg_points)} vertices",
         color=SUBTLE, fontsize=10, ha="center")

gs = fig.add_gridspec(1, 4, wspace=0.12, left=0.03, right=0.97, top=0.88, bottom=0.06)

# Panel 1: Original
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(display_arr, extent=[0, CANVAS_W, CANVAS_H, 0])
ax1.set_xlim(0, CANVAS_W)
ax1.set_ylim(CANVAS_H, 0)
ax1.set_aspect("equal")
ax1.set_facecolor(PANEL_BG)
hide(ax1)
ax1.set_title("Original (67 paths)", color=TEXT, fontsize=11, fontweight="bold", pad=8)

# Panel 2: Binary mask
ax2 = fig.add_subplot(gs[0, 1])
# Downsample binary for display
from skimage.transform import resize
binary_display = resize(binary, (CANVAS_H * 2, CANVAS_W * 2), order=0)
ax2.imshow(binary_display, extent=[0, CANVAS_W, CANVAS_H, 0],
           cmap="Greens", alpha=0.9)
ax2.set_xlim(0, CANVAS_W)
ax2.set_ylim(CANVAS_H, 0)
ax2.set_aspect("equal")
ax2.set_facecolor(PANEL_BG)
hide(ax2)
ax2.set_title("Binary Mask (alpha > 0)", color=TEXT, fontsize=11, fontweight="bold", pad=8)
ax2.text(CANVAS_W / 2, CANVAS_H + 5,
         f"{binary.sum():,} filled pixels",
         color=SUBTLE, fontsize=8, ha="center")

# Panel 3: Silhouette path on dark background
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor(PANEL_BG)
hide(ax3)
ax3.set_xlim(0, CANVAS_W)
ax3.set_ylim(CANVAS_H, 0)
ax3.set_aspect("equal")
ax3.set_title(f"Extracted Silhouette ({len(svg_points)} pts)", color=TEAL,
              fontsize=11, fontweight="bold", pad=8)

# Draw the silhouette polygon
xs = [p[0] for p in svg_points]
ys = [p[1] for p in svg_points]
ax3.fill(xs, ys, alpha=0.25, color=TEAL)
ax3.plot(xs + [xs[0]], ys + [ys[0]], color=TEAL, linewidth=1.5, alpha=0.9)

# Mark every Nth vertex
step = max(1, len(svg_points) // 20)
for i in range(0, len(svg_points), step):
    ax3.plot(svg_points[i][0], svg_points[i][1], "o", color=TEAL,
             markersize=3, markeredgecolor="white", markeredgewidth=0.3)

# Panel 4: Overlay — silhouette on original
ax4 = fig.add_subplot(gs[0, 3])
ax4.imshow(display_arr, extent=[0, CANVAS_W, CANVAS_H, 0], alpha=0.5)
ax4.fill(xs, ys, alpha=0.15, color=TEAL)
ax4.plot(xs + [xs[0]], ys + [ys[0]], color=TEAL, linewidth=2, alpha=0.9)
ax4.set_xlim(0, CANVAS_W)
ax4.set_ylim(CANVAS_H, 0)
ax4.set_aspect("equal")
ax4.set_facecolor(PANEL_BG)
hide(ax4)
ax4.set_title("Overlay — Silhouette on Original", color=TEXT,
              fontsize=11, fontweight="bold", pad=8)

# Footer with the actual SVG path (truncated)
d_preview = svg_d[:120] + "..." if len(svg_d) > 120 else svg_d
fig.text(0.5, 0.02,
         f'<path d="{d_preview}" />',
         color=SUBTLE, fontsize=7, ha="center", fontfamily="monospace")

fig.savefig(OUT_DIR / "04_silhouette.png", dpi=150, facecolor=BG,
            bbox_inches="tight", pad_inches=0.3)
plt.close(fig)
print(f"Saved PNG: {OUT_DIR / '04_silhouette.png'}")

# ── Also generate at multiple simplification levels ─────────────────

print("\nGenerating simplification comparison...")

fig2, axes = plt.subplots(1, len(tolerances) + 1, figsize=(24, 5),
                           facecolor=BG, dpi=150)
fig2.suptitle("Douglas-Peucker Simplification at Different Tolerances",
              color=TEXT, fontsize=16, fontweight="bold", y=0.97)

# First panel: raw contour
ax = axes[0]
raw_xs = outer_contour[:, 1] / RENDER_MULT
raw_ys = outer_contour[:, 0] / RENDER_MULT
ax.fill(raw_xs, raw_ys, alpha=0.2, color=TEAL)
ax.plot(raw_xs, raw_ys, color=TEAL, linewidth=0.5, alpha=0.7)
ax.set_xlim(0, CANVAS_W)
ax.set_ylim(CANVAS_H, 0)
ax.set_aspect("equal")
ax.set_facecolor(PANEL_BG)
hide(ax)
ax.set_title(f"Raw ({len(outer_contour)} pts)", color=TEAL,
             fontsize=9, fontweight="bold", pad=6)

# Remaining panels: each tolerance level
for ti, tol in enumerate(tolerances):
    ax = axes[ti + 1]
    pts = simplified[tol]
    pxs = pts[:, 1] / RENDER_MULT
    pys = pts[:, 0] / RENDER_MULT

    ax.fill(pxs, pys, alpha=0.2, color=TEAL)
    ax.plot(list(pxs) + [pxs[0]], list(pys) + [pys[0]], color=TEAL, linewidth=1.2, alpha=0.9)

    # Show vertices
    ax.plot(pxs, pys, "o", color=TEAL, markersize=2.5,
            markeredgecolor="white", markeredgewidth=0.3)

    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(CANVAS_H, 0)
    ax.set_aspect("equal")
    ax.set_facecolor(PANEL_BG)
    hide(ax)

    highlight = " ←" if tol == best_tol else ""
    color = ACCENT if tol == best_tol else TEAL
    ax.set_title(f"tol={tol} ({len(pts)} pts){highlight}", color=color,
                 fontsize=9, fontweight="bold", pad=6)

fig2.subplots_adjust(wspace=0.08)
fig2.savefig(OUT_DIR / "04b_simplification_levels.png", dpi=150, facecolor=BG,
             bbox_inches="tight", pad_inches=0.3)
plt.close(fig2)
print(f"Saved: {OUT_DIR / '04b_simplification_levels.png'}")

# Print the path
print(f"\n{'='*60}")
print(f"SILHOUETTE SVG PATH ({len(svg_points)} vertices):")
print(f"{'='*60}")
print(f'd="{svg_d}"')
print(f"{'='*60}")
