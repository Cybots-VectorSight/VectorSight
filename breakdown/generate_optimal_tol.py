"""Find the mathematically optimal Douglas-Peucker tolerance.

Three metrics at each tolerance level:
1. Area loss — how much area the simplified polygon lost (%)
2. Vertex count — how many vertices remain (the L-curve)
3. Mean edge-to-contour distance — average deviation from raw shape (px)

The optimal tolerance is the ELBOW POINT in the vertex count curve:
- Vertex count drops steeply at first (removing redundant collinear points)
- Then flattens (each vertex removed costs real shape detail)
- The elbow = transition from free compression to shape destruction

Elbow detection: Kneedle algorithm (max perpendicular distance from baseline).
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import cairosvg
import numpy as np
from PIL import Image
from skimage import measure
from skimage.measure import approximate_polygon
from scipy.spatial.distance import directed_hausdorff
from shapely.geometry import Polygon, LineString
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# ── Config ──────────────────────────────────────────────────────────

SVG_PATH = Path(__file__).resolve().parent.parent / "samples" / "test" / "faker.svg"
OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(exist_ok=True)

CANVAS_W, CANVAS_H = 256, 308
RENDER_MULT = 4
PAD = 4

BG = "#0f0f1a"
PANEL_BG = "#161625"
TEXT = "#eee"
SUBTLE = "#777"
TEAL = "#4ECDC4"
RED = "#FF6B6B"
ACCENT = "#e94560"
GREEN = "#96CEB4"
BLUE = "#45B7D1"

stroke_thin = [pe.withStroke(linewidth=1.5, foreground=BG)]


def hide(ax):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


def style_chart(ax):
    ax.tick_params(colors=SUBTLE, labelsize=7)
    ax.spines["bottom"].set_color(SUBTLE)
    ax.spines["left"].set_color(SUBTLE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color=SUBTLE, alpha=0.15)


# ── Extract the raw contour ─────────────────────────────────────────

svg_text = SVG_PATH.read_text(encoding="utf-8")

print("Rendering and extracting contour...")
png = cairosvg.svg2png(bytestring=svg_text.encode(),
                        output_width=CANVAS_W * RENDER_MULT,
                        output_height=CANVAS_H * RENDER_MULT)
img_arr = np.array(Image.open(io.BytesIO(png)).convert("RGBA"))
alpha = img_arr[:, :, 3]
binary = (alpha > 10).astype(np.uint8)
binary_padded = np.pad(binary, PAD, mode="constant", constant_values=0)

contours = measure.find_contours(binary_padded, level=0.5)
contours = [c - PAD for c in contours]
contours.sort(key=len, reverse=True)
raw_contour = contours[0]

print(f"Raw contour: {len(raw_contour)} points")

# Convert raw contour to SVG coordinates for area calculation
raw_svg = np.column_stack([
    raw_contour[:, 1] / RENDER_MULT,  # col → x
    raw_contour[:, 0] / RENDER_MULT,  # row → y
])
raw_poly = Polygon(raw_svg)
raw_area = raw_poly.area
raw_ring = LineString(list(raw_svg) + [raw_svg[0]])  # closed ring for distance calc
print(f"Raw area: {raw_area:.1f} sq px")

# ── Sweep tolerance values ──────────────────────────────────────────

# Fine-grained sweep from 0.25 to 20
tolerances = np.concatenate([
    np.arange(0.25, 3.0, 0.25),
    np.arange(3.0, 8.0, 0.5),
    np.arange(8.0, 21.0, 1.0),
])

results = []

print(f"\nSweeping {len(tolerances)} tolerance values...")
for tol in tolerances:
    simplified = approximate_polygon(raw_contour, tolerance=tol)
    n_verts = len(simplified)

    # Convert to SVG coords
    svg_pts = np.column_stack([
        simplified[:, 1] / RENDER_MULT,
        simplified[:, 0] / RENDER_MULT,
    ])

    # 1. Mean edge-to-contour distance (how far simplified edges deviate from raw)
    simp_ring = LineString(list(svg_pts) + [svg_pts[0]])
    # Sample points along raw contour and measure distance to simplified ring
    sample_step = max(1, len(raw_svg) // 500)
    dists = [simp_ring.distance(
        __import__("shapely.geometry", fromlist=["Point"]).Point(p))
        for p in raw_svg[::sample_step]]
    mean_dev = float(np.mean(dists))
    max_dev = float(np.max(dists))

    # 2. Area loss
    try:
        simp_poly = Polygon(svg_pts)
        simp_area = simp_poly.area
        area_loss_pct = abs(raw_area - simp_area) / raw_area * 100
    except Exception:
        simp_area = 0
        area_loss_pct = 100

    results.append({
        "tol": tol,
        "n_verts": n_verts,
        "mean_dev": mean_dev,
        "max_dev": max_dev,
        "area_loss_pct": area_loss_pct,
        "svg_pts": svg_pts,
    })

    if tol in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]:
        print(f"  tol={tol:5.1f}  verts={n_verts:4d}  "
              f"mean_dev={mean_dev:.2f}px  max_dev={max_dev:.2f}px  "
              f"area_loss={area_loss_pct:.2f}%")

# ── Find the elbow point (Kneedle on vertex count) ──────────────────

print("\nFinding optimal tolerance (elbow on vertex count)...")

tols = np.array([r["tol"] for r in results])
n_verts_arr = np.array([r["n_verts"] for r in results])

# Normalize both axes to [0, 1]
t_norm = (tols - tols.min()) / (tols.max() - tols.min())
v_norm = (n_verts_arr - n_verts_arr.min()) / (n_verts_arr.max() - n_verts_arr.min())

# Line from first to last point
x0, y0 = t_norm[0], v_norm[0]
x1, y1 = t_norm[-1], v_norm[-1]

# Perpendicular distance of each point from this line
a = y1 - y0
b = -(x1 - x0)
c = x1 * y0 - y1 * x0
denom = math.sqrt(a * a + b * b)

distances = np.abs(a * t_norm + b * v_norm + c) / denom

# The elbow is the point with maximum distance from the baseline
elbow_idx = np.argmax(distances)
elbow_tol = tols[elbow_idx]
elbow_result = results[elbow_idx]

print(f"\n{'='*60}")
print(f"OPTIMAL TOLERANCE: {elbow_tol}")
print(f"  Vertices: {elbow_result['n_verts']}")
print(f"  Mean deviation: {elbow_result['mean_dev']:.2f} px")
print(f"  Max deviation: {elbow_result['max_dev']:.2f} px")
print(f"  Area loss: {elbow_result['area_loss_pct']:.2f}%")
print(f"{'='*60}")

# Also show a few neighbors around the elbow for context
print("\nNeighborhood around elbow:")
for i in range(max(0, elbow_idx - 2), min(len(results), elbow_idx + 3)):
    r = results[i]
    marker = " << OPTIMAL" if i == elbow_idx else ""
    print(f"  tol={r['tol']:5.2f}  verts={r['n_verts']:4d}  "
          f"mean_dev={r['mean_dev']:.2f}px  area_loss={r['area_loss_pct']:.2f}%{marker}")

# ── Generate the visualization ──────────────────────────────────────

print("\nGenerating visualization...")

fig = plt.figure(figsize=(20, 11), facecolor=BG, dpi=150)
fig.suptitle("Finding the Optimal Simplification — Zero Magic Numbers",
             color=TEXT, fontsize=18, fontweight="bold", y=0.97)
fig.text(0.5, 0.935,
         "Sweep Douglas-Peucker tolerance, measure degradation at each level, "
         "find the elbow automatically",
         color=SUBTLE, fontsize=9, ha="center")

gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.28,
                      left=0.06, right=0.97, top=0.89, bottom=0.07)

# ── Panel 1: Vertex count L-curve (THE key curve) ───────────────────

ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(PANEL_BG)
n_verts_list = [r["n_verts"] for r in results]
ax1.plot(tols, n_verts_list, color=BLUE, linewidth=2.5, marker=".", markersize=5)
ax1.axvline(x=elbow_tol, color=ACCENT, linewidth=2, linestyle="--", alpha=0.8)
ax1.plot(elbow_tol, elbow_result["n_verts"], "o", color=ACCENT,
         markersize=12, markeredgecolor="white", markeredgewidth=1.5, zorder=5)
ax1.text(elbow_tol + 0.8, elbow_result["n_verts"] + 30,
         f"ELBOW\ntol={elbow_tol}\n{elbow_result['n_verts']} vertices",
         color=ACCENT, fontsize=9, fontweight="bold", path_effects=stroke_thin)

ax1.set_xlabel("Tolerance", color=SUBTLE, fontsize=9)
ax1.set_ylabel("Vertex Count", color=SUBTLE, fontsize=9)
ax1.set_title("Vertex Count vs Tolerance (L-curve)", color=BLUE,
              fontsize=11, fontweight="bold", pad=8)
style_chart(ax1)

# ── Panel 2: Mean deviation curve ───────────────────────────────────

ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(PANEL_BG)
mean_devs = [r["mean_dev"] for r in results]
ax2.plot(tols, mean_devs, color=TEAL, linewidth=2, marker=".", markersize=4)
ax2.axvline(x=elbow_tol, color=ACCENT, linewidth=2, linestyle="--", alpha=0.8)
ax2.plot(elbow_tol, elbow_result["mean_dev"], "o", color=ACCENT,
         markersize=12, markeredgecolor="white", markeredgewidth=1.5, zorder=5)
ax2.text(elbow_tol + 0.8, elbow_result["mean_dev"] + 0.1,
         f"{elbow_result['mean_dev']:.2f}px avg",
         color=ACCENT, fontsize=9, fontweight="bold", path_effects=stroke_thin)

ax2.set_xlabel("Tolerance", color=SUBTLE, fontsize=9)
ax2.set_ylabel("Mean Deviation (px)", color=SUBTLE, fontsize=9)
ax2.set_title("Mean Shape Deviation vs Tolerance", color=TEAL,
              fontsize=11, fontweight="bold", pad=8)
style_chart(ax2)

# ── Panel 3: Area loss curve ────────────────────────────────────────

ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor(PANEL_BG)
area_losses = [r["area_loss_pct"] for r in results]
ax3.plot(tols, area_losses, color=GREEN, linewidth=2, marker=".", markersize=4)
ax3.axvline(x=elbow_tol, color=ACCENT, linewidth=2, linestyle="--", alpha=0.8)
elbow_area = elbow_result["area_loss_pct"]
ax3.plot(elbow_tol, elbow_area, "o", color=ACCENT,
         markersize=12, markeredgecolor="white", markeredgewidth=1.5, zorder=5)
ax3.text(elbow_tol + 0.8, elbow_area + 0.05,
         f"{elbow_area:.2f}% lost",
         color=ACCENT, fontsize=9, fontweight="bold", path_effects=stroke_thin)

ax3.set_xlabel("Tolerance", color=SUBTLE, fontsize=9)
ax3.set_ylabel("Area Loss (%)", color=SUBTLE, fontsize=9)
ax3.set_title("Area Preservation vs Tolerance", color=GREEN,
              fontsize=11, fontweight="bold", pad=8)
style_chart(ax3)

# ── Panel 4: Kneedle visualization ──────────────────────────────────

ax4 = fig.add_subplot(gs[1, 0])
ax4.set_facecolor(PANEL_BG)
ax4.set_title("Kneedle Algorithm — Elbow Detection", color=TEXT,
              fontsize=11, fontweight="bold", pad=8)

# Plot normalized vertex count curve (inverted = it goes down)
ax4.plot(t_norm, v_norm, color=BLUE, linewidth=2.5, label="Normalized vertex count")
# Plot the baseline (first→last)
ax4.plot([x0, x1], [y0, y1], color=SUBTLE, linewidth=1.5, linestyle="--",
         label="Baseline", alpha=0.7)
# Draw perpendicular at elbow
# Project elbow point onto the baseline to get the foot
t_e = t_norm[elbow_idx]
v_e = v_norm[elbow_idx]
# Foot of perpendicular from (t_e, v_e) to baseline
dx, dy = x1 - x0, y1 - y0
t_param = ((t_e - x0) * dx + (v_e - y0) * dy) / (dx * dx + dy * dy)
foot_x = x0 + t_param * dx
foot_y = y0 + t_param * dy
ax4.plot([t_e, foot_x], [v_e, foot_y],
         color=ACCENT, linewidth=2.5, linestyle="-", zorder=4)
ax4.plot(t_e, v_e, "o", color=ACCENT,
         markersize=12, markeredgecolor="white", markeredgewidth=1.5, zorder=5)
ax4.text(t_e + 0.04, v_e + 0.05,
         f"max distance\n= elbow point",
         color=ACCENT, fontsize=8, fontweight="bold", path_effects=stroke_thin)

# Shade regions
ax4.axvspan(0, t_norm[elbow_idx], alpha=0.08, color=GREEN)
ax4.axvspan(t_norm[elbow_idx], 1, alpha=0.08, color=RED)
ax4.text(t_norm[elbow_idx] / 2, 0.9, "FREE\ncompression",
         color=GREEN, fontsize=9, ha="center", fontweight="bold", alpha=0.8)
ax4.text((t_norm[elbow_idx] + 1) / 2, 0.9, "SHAPE\ndestruction",
         color=RED, fontsize=9, ha="center", fontweight="bold", alpha=0.8)

ax4.set_xlabel("Normalized tolerance", color=SUBTLE, fontsize=9)
ax4.set_ylabel("Normalized vertex count", color=SUBTLE, fontsize=9)
style_chart(ax4)
ax4.legend(fontsize=7, facecolor=PANEL_BG, edgecolor=SUBTLE, labelcolor=TEXT,
           loc="center right")

# ── Panel 5: Optimal silhouette on clean background ─────────────────

ax5 = fig.add_subplot(gs[1, 1])
ax5.set_facecolor(PANEL_BG)
hide(ax5)
ax5.set_xlim(-5, CANVAS_W + 5)
ax5.set_ylim(CANVAS_H + 5, -5)
ax5.set_aspect("equal")
ax5.set_title(f"Optimal — tol={elbow_tol}, {elbow_result['n_verts']} vertices",
              color=ACCENT, fontsize=11, fontweight="bold", pad=8)

pts = elbow_result["svg_pts"]
xs = pts[:, 0]
ys = pts[:, 1]
ax5.fill(xs, ys, alpha=0.25, color=TEAL)
ax5.plot(list(xs) + [xs[0]], list(ys) + [ys[0]], color=TEAL, linewidth=2, alpha=0.9)

# Mark vertices
step = max(1, len(xs) // 30)
for i in range(0, len(xs), step):
    ax5.plot(xs[i], ys[i], "o", color=TEAL, markersize=3,
             markeredgecolor="white", markeredgewidth=0.4)

ax5.text(CANVAS_W / 2, CANVAS_H + 3,
         f"mean dev: {elbow_result['mean_dev']:.2f}px  |  "
         f"area loss: {elbow_result['area_loss_pct']:.2f}%",
         color=SUBTLE, fontsize=8, ha="center")

# ── Panel 6: Overlay on original ────────────────────────────────────

ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor(PANEL_BG)
hide(ax6)

# Render original for display
png_2x = cairosvg.svg2png(bytestring=svg_text.encode(),
                           output_width=CANVAS_W * 2, output_height=CANVAS_H * 2)
display_arr = np.array(Image.open(io.BytesIO(png_2x)).convert("RGBA"))

ax6.imshow(display_arr, extent=[0, CANVAS_W, CANVAS_H, 0], alpha=0.5)
ax6.fill(xs, ys, alpha=0.12, color=TEAL)
ax6.plot(list(xs) + [xs[0]], list(ys) + [ys[0]], color=TEAL, linewidth=2, alpha=0.9)
ax6.set_xlim(0, CANVAS_W)
ax6.set_ylim(CANVAS_H, 0)
ax6.set_aspect("equal")
ax6.set_title("Optimal Silhouette on Original", color=TEXT,
              fontsize=11, fontweight="bold", pad=8)

# ── Footer ──────────────────────────────────────────────────────────

fig.text(0.5, 0.015,
         f"Kneedle on vertex-count L-curve: sweep tolerance → count vertices → "
         f"normalize → find max perpendicular distance from baseline → "
         f"tol={elbow_tol} ({elbow_result['n_verts']} vertices, "
         f"{elbow_result['mean_dev']:.2f}px mean deviation, "
         f"{elbow_result['area_loss_pct']:.2f}% area loss)",
         color=SUBTLE, fontsize=8, ha="center")

fig.savefig(OUT_DIR / "05_optimal_tolerance.png", dpi=150, facecolor=BG,
            bbox_inches="tight", pad_inches=0.3)
plt.close(fig)
print(f"\nSaved: {OUT_DIR / '05_optimal_tolerance.png'}")
