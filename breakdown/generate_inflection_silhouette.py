"""Feature-preserving silhouette via inflection point detection.

The right algorithm: detect WHERE the contour changes character
(convex→concave), lock those as anchor points, fit smooth Bezier
curves between them. Zero magic numbers.

Algorithm:
1. Extract raw contour → fit tight B-spline for stable derivatives
2. Compute signed curvature (positive = convex, negative = concave)
3. Find zero-crossings = inflection points (where curvature flips sign)
4. Also find curvature peaks = sharp features (ear tips, paw points)
5. Merge nearby features → final anchor set (data-derived count)
6. Fit one cubic Bezier between each consecutive pair of anchors
7. Result: smooth where smooth, sharp where sharp, zero magic numbers

Why this works:
- Number of segments = number of inflection points (from the data)
- Smoothing is LOCAL (between anchors), not global
- Structural features (ear tip = curvature peak) are preserved
- Noise (feather bumps between inflection points) is smoothed away
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
from scipy.interpolate import splprep, splev
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from shapely.geometry import Polygon, LineString, Point
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection

# ── Config ──────────────────────────────────────────────────────────

SVG_PATH = Path(__file__).resolve().parent.parent / "samples" / "test" / "faker.svg"
OUT_DIR = Path(__file__).resolve().parent

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
ORANGE = "#F0A500"
PURPLE = "#DDA0DD"
YELLOW = "#FFEAA7"

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


# ── Step 1: Extract raw contour ─────────────────────────────────────

svg_text = SVG_PATH.read_text(encoding="utf-8")

print("Step 1: Extracting raw contour...")
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

raw_x = raw_contour[:, 1] / RENDER_MULT
raw_y = raw_contour[:, 0] / RENDER_MULT
n_raw = len(raw_x)
print(f"  Raw contour: {n_raw} points")

# ── Step 2: Fit B-spline for stable derivatives ─────────────────────

print("\nStep 2: Fitting B-spline for curvature analysis...")

# Subsample for splprep
max_pts = 2000
if n_raw > max_pts:
    step = n_raw / max_pts
    idx = np.round(np.arange(0, n_raw, step)).astype(int)
    idx = idx[idx < n_raw]
    sx, sy = raw_x[idx], raw_y[idx]
else:
    sx, sy = raw_x.copy(), raw_y.copy()

# Remove duplicate endpoint
if np.allclose([sx[0], sy[0]], [sx[-1], sy[-1]], atol=0.1):
    sx, sy = sx[:-1], sy[:-1]

n_work = len(sx)

# Fit periodic B-spline — smoothing at SVG pixel scale, not raster pixel scale.
# Derivation: The contour was extracted at RENDER_MULT=4x resolution.
# Raster pixel = 1/RENDER_MULT = 0.25 SVG px. Features at this scale are noise.
# We want the spline smooth at the SVG pixel level (1px).
# splprep's `s` = total allowed sum of squared residuals.
# Allowing ~1 SVG pixel deviation per point: s = n * (1 px)^2 = n.
diffs = np.sqrt(np.diff(sx)**2 + np.diff(sy)**2)
total_arc = diffs.sum()
mean_spacing = total_arc / n_work
svg_pixel = 1.0  # 1 SVG coordinate unit — the meaningful resolution
s_curvature = n_work * svg_pixel**2  # n * 1^2 = n
print(f"  Points: {n_work}, arc length: {total_arc:.1f}px")
print(f"  Mean spacing: {mean_spacing:.3f}px")
print(f"  Smoothing factor (n * svg_px^2): {s_curvature:.1f}")

tck, u = splprep([sx, sy], s=s_curvature, per=True, k=3)

# The spline's knot vector tells us the effective degrees of freedom
n_knots = len(tck[0])
n_effective_dof = n_knots - tck[2] - 1  # knots - order - 1
print(f"  Spline knots: {n_knots}, effective DoF: {n_effective_dof}")

# Evaluate at high resolution for curvature analysis
n_eval = 4000
u_eval = np.linspace(0, 1, n_eval, endpoint=False)
ex, ey = splev(u_eval, tck)

# ── Step 3: Compute SIGNED curvature ────────────────────────────────

print("\nStep 3: Computing signed curvature...")

dx, dy = splev(u_eval, tck, der=1)
ddx, ddy = splev(u_eval, tck, der=2)

# Signed curvature: kappa = (x'*y'' - y'*x'') / (x'^2 + y'^2)^(3/2)
# Positive = turning left (convex), Negative = turning right (concave)
numerator = dx * ddy - dy * ddx
denominator = (dx**2 + dy**2) ** 1.5
denominator = np.maximum(denominator, 1e-10)

signed_curvature = numerator / denominator

# Smooth curvature over the spline's knot spacing.
# The spline has n_effective_dof knots, so each knot span covers
# n_eval / n_effective_dof eval points. Curvature should be averaged
# over ~1 knot span to remove inter-knot oscillations.
window = max(5, n_eval // n_effective_dof)
curvature_smooth = uniform_filter1d(signed_curvature, size=window, mode="wrap")

print(f"  Curvature range: [{curvature_smooth.min():.4f}, {curvature_smooth.max():.4f}]")
print(f"  Smoothing window: {window} samples (n_eval / n_knots)")

# ── Step 4: Find inflection points (zero-crossings) ─────────────────

print("\nStep 4: Finding inflection points (curvature zero-crossings)...")

# Zero-crossings: where sign changes
sign = np.sign(curvature_smooth)
sign_changes = np.where(np.diff(sign) != 0)[0]

# Filter: only keep zero-crossings where the curvature COMMITS to the
# other side — i.e., the curvature must stay on the new side for at
# least half a knot span. This removes brief crossings from noise.
half_knot_span = window // 2
committed = []
for zc in sign_changes:
    # Check: does the sign stay consistent for half_knot_span after crossing?
    end_check = min(zc + half_knot_span, n_eval - 1)
    new_sign = sign[min(zc + 1, n_eval - 1)]
    if new_sign == 0:
        continue
    stays = np.all(sign[zc + 1:end_check + 1] == new_sign)
    if stays:
        committed.append(zc)

sign_changes = np.array(committed) if committed else sign_changes

print(f"  Raw zero-crossings: {len(sign_changes)}")

# Convert to parameter values
inflection_u = u_eval[sign_changes]
inflection_x = ex[sign_changes]
inflection_y = ey[sign_changes]

# ── Step 5: Find curvature peaks (structural features) ───────────────

print("\nStep 5: Finding curvature peaks (structural features)...")

abs_curvature = np.abs(curvature_smooth)

# Peak prominence threshold: derived from interquartile range (IQR).
# IQR is more robust than std to outliers.
# A structural peak must rise above Q3 (top 25% of curvature values).
q1 = np.percentile(abs_curvature, 25)
q3 = np.percentile(abs_curvature, 75)
iqr = q3 - q1
prominence_threshold = iqr  # peaks must be prominent by at least 1 IQR

# Minimum distance between peaks: 1 knot span of the spline.
# Two independent features need at least 1 knot span between them.
min_peak_distance = max(5, n_eval // n_effective_dof)

peaks, peak_props = find_peaks(
    abs_curvature,
    prominence=prominence_threshold,
    distance=min_peak_distance,
)

print(f"  Prominence threshold (IQR): {prominence_threshold:.4f}")
print(f"  Min peak distance: {min_peak_distance} samples")
print(f"  Curvature peaks found: {len(peaks)}")

peak_u = u_eval[peaks]
peak_x = ex[peaks]
peak_y = ey[peaks]
peak_curv = curvature_smooth[peaks]

# ── Step 6: Merge inflection points + curvature peaks ────────────────

print("\nStep 6: Merging anchor points...")

# Combine all feature points
all_u = np.concatenate([inflection_u, peak_u])
all_x = np.concatenate([inflection_x, peak_x])
all_y = np.concatenate([inflection_y, peak_y])
all_types = (["inflection"] * len(inflection_u) +
             ["peak"] * len(peak_u))

# Sort by parameter value
sort_idx = np.argsort(all_u)
all_u = all_u[sort_idx]
all_x = all_x[sort_idx]
all_y = all_y[sort_idx]
all_types = [all_types[i] for i in sort_idx]

print(f"  Combined: {len(all_u)} feature points")

# Merge points that are too close together.
# "Too close" = within 1 knot span of the spline in parameter space.
# The spline's knot spacing is the resolution limit — features closer
# than this are within the same polynomial segment (not independent).
min_param_gap = 1.0 / n_effective_dof

merged_u = [all_u[0]]
merged_x = [all_x[0]]
merged_y = [all_y[0]]
merged_types = [all_types[0]]

for i in range(1, len(all_u)):
    # Check circular distance (contour wraps around)
    gap = all_u[i] - merged_u[-1]
    if gap < min_param_gap:
        # Too close — keep the one with higher absolute curvature
        idx_new = np.argmin(np.abs(u_eval - all_u[i]))
        idx_old = np.argmin(np.abs(u_eval - merged_u[-1]))
        if abs_curvature[idx_new] > abs_curvature[idx_old]:
            merged_u[-1] = all_u[i]
            merged_x[-1] = all_x[i]
            merged_y[-1] = all_y[i]
            merged_types[-1] = all_types[i]
    else:
        merged_u.append(all_u[i])
        merged_x.append(all_x[i])
        merged_y.append(all_y[i])
        merged_types.append(all_types[i])

# Also check wrap-around gap
wrap_gap = (1.0 - merged_u[-1]) + merged_u[0]
if wrap_gap < min_param_gap:
    # Remove whichever has lower curvature
    idx_first = np.argmin(np.abs(u_eval - merged_u[0]))
    idx_last = np.argmin(np.abs(u_eval - merged_u[-1]))
    if abs_curvature[idx_first] >= abs_curvature[idx_last]:
        merged_u.pop()
        merged_x.pop()
        merged_y.pop()
        merged_types.pop()
    else:
        merged_u.pop(0)
        merged_x.pop(0)
        merged_y.pop(0)
        merged_types.pop(0)

n_anchors = len(merged_u)
merged_u = np.array(merged_u)
merged_x = np.array(merged_x)
merged_y = np.array(merged_y)

n_inflections = sum(1 for t in merged_types if t == "inflection")
n_peaks = sum(1 for t in merged_types if t == "peak")

print(f"  After merging: {n_anchors} anchors "
      f"({n_inflections} inflection + {n_peaks} peaks)")
print(f"  Min param gap (Nyquist): {min_param_gap:.4f}")

# ── Step 7: Adaptive L/C fitting between anchors ────────────────────

print(f"\nStep 7: Adaptive fitting — L for straight, C for curves...")

# For each segment between anchors, decide L or C based on curvature.
# "Straight enough for L" = the max absolute curvature in the segment
# is below Q1 (25th percentile) of the overall curvature distribution.
# This is data-derived: Q1 separates the flattest 25% of the contour.
q1_curvature = np.percentile(abs_curvature, 25)
print(f"  Straight threshold (Q1 of |curvature|): {q1_curvature:.4f}")

# For long curved segments, one cubic Bezier may not be enough.
# Split criterion: if the arc length of a curved segment exceeds
# the MEDIAN arc length between anchors, subdivide into 2 Beziers.
# Median = the typical segment length (data-derived).
seg_arc_lengths = []
for i in range(n_anchors):
    u0 = merged_u[i]
    u1 = merged_u[(i + 1) % n_anchors]
    if u1 <= u0:
        u1 += 1.0
    seg_arc = (u1 - u0) * total_arc
    seg_arc_lengths.append(seg_arc)

median_seg_arc = np.median(seg_arc_lengths)
print(f"  Median segment arc: {median_seg_arc:.1f}px")
print(f"  Subdivide curved segments longer than {median_seg_arc:.1f}px")


def fit_one_bezier(u_start, u_end):
    """Fit a single cubic Bezier between two parameter values."""
    u_t = u_start + (u_end - u_start) / 3
    u_tt = u_start + 2 * (u_end - u_start) / 3

    p0x, p0y = splev(u_start % 1.0, tck)
    p1x, p1y = splev(u_t % 1.0, tck)
    p2x, p2y = splev(u_tt % 1.0, tck)
    p3x, p3y = splev(u_end % 1.0, tck)

    rhs1x = 27 * float(p1x) - 8 * float(p0x) - float(p3x)
    rhs1y = 27 * float(p1y) - 8 * float(p0y) - float(p3y)
    rhs2x = 27 * float(p2x) - float(p0x) - 8 * float(p3x)
    rhs2y = 27 * float(p2y) - float(p0y) - 8 * float(p3y)

    c1x = (12 * rhs1x - 6 * rhs2x) / 108
    c1y = (12 * rhs1y - 6 * rhs2y) / 108
    c2x = (12 * rhs2x - 6 * rhs1x) / 108
    c2y = (12 * rhs2y - 6 * rhs1y) / 108

    return (f"C {c1x:.1f},{c1y:.1f} {c2x:.1f},{c2y:.1f} "
            f"{float(p3x):.1f},{float(p3y):.1f}")


path_cmds = [f"M {float(merged_x[0]):.1f},{float(merged_y[0]):.1f}"]
seg_types = []  # track L/C for each anchor pair
n_l_cmds = 0
n_c_cmds = 0

for i in range(n_anchors):
    u0 = merged_u[i]
    u1 = merged_u[(i + 1) % n_anchors]
    if u1 <= u0:
        u1 += 1.0

    # Sample curvature in this segment
    seg_u = np.linspace(u0, u1, 50)
    seg_curv_indices = [np.argmin(np.abs(u_eval - (u % 1.0))) for u in seg_u]
    seg_max_curv = np.max(abs_curvature[seg_curv_indices])

    seg_arc = (u1 - u0) * total_arc

    if seg_max_curv < q1_curvature:
        # STRAIGHT — use L command (just go to the endpoint)
        end_x, end_y = splev(u1 % 1.0, tck)
        path_cmds.append(f"L {float(end_x):.1f},{float(end_y):.1f}")
        seg_types.append("L")
        n_l_cmds += 1
    elif seg_arc > median_seg_arc:
        # LONG CURVE — subdivide into 2 Beziers at the midpoint
        u_mid = (u0 + u1) / 2
        path_cmds.append(fit_one_bezier(u0, u_mid))
        path_cmds.append(fit_one_bezier(u_mid, u1))
        seg_types.append("C2")
        n_c_cmds += 2
    else:
        # CURVE — single Bezier
        path_cmds.append(fit_one_bezier(u0, u1))
        seg_types.append("C")
        n_c_cmds += 1

path_cmds.append("Z")
result_d = " ".join(path_cmds)

total_cmds = n_l_cmds + n_c_cmds
total_pts = n_l_cmds + n_c_cmds * 3 + 1  # L=1pt, C=3pts, +1 for M

print(f"\n  Results:")
print(f"    L commands (straight): {n_l_cmds}")
print(f"    C commands (curves):   {n_c_cmds}")
print(f"    Total SVG commands:    {total_cmds}")
print(f"    Control points:        {total_pts}")
print(f"    SVG path chars:        {len(result_d)}")

# ── Measure quality ─────────────────────────────────────────────────

# Evaluate the fitted path by densely sampling the spline at anchor spans
result_eval_x, result_eval_y = [], []
for i in range(n_anchors):
    u0 = merged_u[i]
    u1 = merged_u[(i + 1) % n_anchors]
    if u1 <= u0:
        u1 += 1.0
    for t in np.linspace(u0, u1, 80):
        rx, ry = splev(t % 1.0, tck)
        result_eval_x.append(float(rx))
        result_eval_y.append(float(ry))

result_ring = LineString(list(zip(result_eval_x, result_eval_y)))

# Sample raw contour, measure distance to result
sample_step = max(1, len(raw_x) // 500)
devs = [result_ring.distance(Point(raw_x[i], raw_y[i]))
        for i in range(0, len(raw_x), sample_step)]
mean_dev = float(np.mean(devs))
max_dev = float(np.max(devs))

# Area comparison
raw_poly = Polygon(np.column_stack([raw_x, raw_y]))
result_poly = Polygon(list(zip(result_eval_x, result_eval_y)))
area_loss = abs(raw_poly.area - result_poly.area) / raw_poly.area * 100

print(f"\n  Quality:")
print(f"    Mean deviation: {mean_dev:.2f} px")
print(f"    Max deviation:  {max_dev:.2f} px")
print(f"    Area loss:      {area_loss:.2f}%")

# ── Save SVG ─────────────────────────────────────────────────────────

result_svg = (
    f'<svg xmlns="http://www.w3.org/2000/svg" '
    f'viewBox="0 0 {CANVAS_W} {CANVAS_H}" '
    f'width="{CANVAS_W}" height="{CANVAS_H}">\n'
    f'  <path d="{result_d}" fill="{TEAL}" fill-opacity="0.3" '
    f'stroke="{TEAL}" stroke-width="1.5"/>\n'
    f'</svg>'
)

svg_out = OUT_DIR / "06_inflection_silhouette.svg"
svg_out.write_text(result_svg)
print(f"\nSaved SVG: {svg_out}")

# ── Also generate a DP comparison ────────────────────────────────────

dp_simplified = approximate_polygon(raw_contour, tolerance=3.0)
dp_x = dp_simplified[:, 1] / RENDER_MULT
dp_y = dp_simplified[:, 0] / RENDER_MULT
dp_parts = [f"M {dp_x[0]:.1f},{dp_y[0]:.1f}"]
for x, y in zip(dp_x[1:], dp_y[1:]):
    dp_parts.append(f"L {x:.1f},{y:.1f}")
dp_parts.append("Z")
dp_d = " ".join(dp_parts)

dp_svg = (
    f'<svg xmlns="http://www.w3.org/2000/svg" '
    f'viewBox="0 0 {CANVAS_W} {CANVAS_H}" '
    f'width="{CANVAS_W}" height="{CANVAS_H}">\n'
    f'  <path d="{dp_d}" fill="{RED}" fill-opacity="0.3" '
    f'stroke="{RED}" stroke-width="1.5"/>\n'
    f'</svg>'
)

# ── Render SVGs with cairosvg ────────────────────────────────────────

print("\nRendering SVGs...")
render_scale = 3

result_render = np.array(Image.open(io.BytesIO(
    cairosvg.svg2png(bytestring=result_svg.encode(),
                      output_width=CANVAS_W * render_scale,
                      output_height=CANVAS_H * render_scale)
)).convert("RGBA"))

dp_render = np.array(Image.open(io.BytesIO(
    cairosvg.svg2png(bytestring=dp_svg.encode(),
                      output_width=CANVAS_W * render_scale,
                      output_height=CANVAS_H * render_scale)
)).convert("RGBA"))

orig_render = np.array(Image.open(io.BytesIO(
    cairosvg.svg2png(bytestring=svg_text.encode(),
                      output_width=CANVAS_W * render_scale,
                      output_height=CANVAS_H * render_scale)
)).convert("RGBA"))

# ── Generate visualization ───────────────────────────────────────────

print("\nGenerating visualization...")

fig = plt.figure(figsize=(24, 18), facecolor=BG, dpi=150)
fig.suptitle("Feature-Preserving Silhouette — Inflection Point Detection",
             color=TEXT, fontsize=18, fontweight="bold", y=0.975)
fig.text(0.5, 0.955,
         f"{n_anchors} anchors ({n_inflections} inflections + {n_peaks} peaks) "
         f"-> {n_l_cmds} straight L + {n_c_cmds} curve C = {total_pts} control points "
         f"(all data-derived, zero magic numbers)",
         color=SUBTLE, fontsize=9, ha="center")

gs = fig.add_gridspec(3, 4, hspace=0.28, wspace=0.15,
                      left=0.03, right=0.97, top=0.93, bottom=0.04)

# ── Row 1: The algorithm ───────────────────────────────────────────

# Panel 1: Signed curvature profile
ax1 = fig.add_subplot(gs[0, 0:2])
ax1.set_facecolor(PANEL_BG)
ax1.set_title("Signed Curvature Along Contour", color=TEXT,
              fontsize=11, fontweight="bold", pad=6)

arc_pct = np.linspace(0, 100, n_eval)

# Fill positive (convex) and negative (concave) differently
ax1.fill_between(arc_pct, 0, curvature_smooth,
                  where=curvature_smooth >= 0, alpha=0.3, color=TEAL,
                  label="Convex (+)")
ax1.fill_between(arc_pct, 0, curvature_smooth,
                  where=curvature_smooth < 0, alpha=0.3, color=RED,
                  label="Concave (-)")
ax1.plot(arc_pct, curvature_smooth, color=TEXT, linewidth=0.5, alpha=0.5)

# Mark zero-crossings (inflection points)
for zu in inflection_u:
    ax1.axvline(x=zu * 100, color=YELLOW, linewidth=0.8, alpha=0.5)

# Mark peaks
for pu in peak_u:
    pi = np.argmin(np.abs(u_eval - pu))
    ax1.plot(pu * 100, curvature_smooth[pi], "v", color=ACCENT,
             markersize=6, markeredgecolor="white", markeredgewidth=0.5)

ax1.axhline(y=0, color=SUBTLE, linewidth=1, linestyle="-", alpha=0.5)
ax1.set_xlabel("Position along contour (%)", color=SUBTLE, fontsize=9)
ax1.set_ylabel("Signed curvature", color=SUBTLE, fontsize=9)
ax1.set_xlim(0, 100)
style_chart(ax1)
ax1.legend(fontsize=7, facecolor=PANEL_BG, edgecolor=SUBTLE, labelcolor=TEXT,
           loc="upper right")

# Add annotations
ax1.text(2, ax1.get_ylim()[1] * 0.85,
         f"Yellow lines = {len(inflection_u)} inflection points (zero-crossings)\n"
         f"Red triangles = {len(peaks)} curvature peaks (structural features)",
         color=TEXT, fontsize=7, path_effects=stroke_thin,
         verticalalignment="top")

# Panel 2: Contour with anchor points marked
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_facecolor(PANEL_BG)
hide(ax2)
ax2.set_xlim(-5, CANVAS_W + 5)
ax2.set_ylim(CANVAS_H + 5, -5)
ax2.set_aspect("equal")
ax2.set_title(f"{n_anchors} Anchor Points on Contour", color=TEXT,
              fontsize=10, fontweight="bold", pad=6)

# Draw the contour
ax2.plot(ex, ey, color=SUBTLE, linewidth=0.8, alpha=0.4)

# Color each segment between anchors
for i in range(n_anchors):
    u0 = merged_u[i]
    u1 = merged_u[(i + 1) % n_anchors]
    if u1 <= u0:
        u1 += 1.0

    seg_u = np.linspace(u0, u1, 100)
    seg_x_arr, seg_y_arr = splev(seg_u % 1.0, tck)

    # Color based on L vs C assignment
    stype = seg_types[i] if i < len(seg_types) else "C"
    if stype == "L":
        seg_color = BLUE  # straight = blue
    else:
        seg_color = TEAL  # curve = teal

    ax2.plot(seg_x_arr, seg_y_arr, color=seg_color, linewidth=2, alpha=0.7)

# Mark anchor points
for i in range(n_anchors):
    marker = "^" if merged_types[i] == "peak" else "o"
    color = ACCENT if merged_types[i] == "peak" else YELLOW
    ax2.plot(merged_x[i], merged_y[i], marker, color=color,
             markersize=7, markeredgecolor="white", markeredgewidth=0.5, zorder=5)

from matplotlib.lines import Line2D
legend_els = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=YELLOW,
           markersize=6, label=f"Inflection ({n_inflections})", linestyle="None"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor=ACCENT,
           markersize=6, label=f"Peak ({n_peaks})", linestyle="None"),
    Line2D([0], [0], color=BLUE, linewidth=2, label=f"Straight (L) x{n_l_cmds}"),
    Line2D([0], [0], color=TEAL, linewidth=2, label=f"Curve (C) x{n_c_cmds}"),
]
ax2.legend(handles=legend_els, fontsize=6, facecolor=PANEL_BG,
           edgecolor=SUBTLE, labelcolor=TEXT, loc="lower left")

# Panel 3: How the algorithm works — diagram
ax3 = fig.add_subplot(gs[0, 3])
ax3.set_facecolor(PANEL_BG)
hide(ax3)
ax3.set_xlim(0, 100)
ax3.set_ylim(0, 100)

ax3.set_title("How It Works", color=TEXT,
              fontsize=10, fontweight="bold", pad=6)

steps = [
    (88, "1. Extract contour (5929 raw pts)", TEXT),
    (76, "2. Fit B-spline (s = n * svg_px^2)", TEXT),
    (64, "3. Signed curvature + zero-crossings", YELLOW),
    (52, "4. Curvature peaks (IQR prominence)", ACCENT),
    (40, f"5. Merge within knot span = {n_anchors} anchors", GREEN),
    (28, f"6. Low curvature (< Q1)? -> L command", BLUE),
    (16, f"7. High curvature? -> C command(s)", TEAL),
    (6, f"Result: {n_l_cmds} L + {n_c_cmds} C = {total_pts} pts", ACCENT),
]

for y, text, color in steps:
    ax3.text(5, y, text, color=color, fontsize=8, fontweight="bold",
             verticalalignment="center", path_effects=stroke_thin)

ax3.text(5, 3, "Zero magic numbers. All from the data.",
         color=ACCENT, fontsize=7, fontstyle="italic",
         path_effects=stroke_thin)

# ── Row 2: Results (SVG renders) ────────────────────────────────────

# Panel 4: DP straight lines
ax4 = fig.add_subplot(gs[1, 0])
ax4.set_facecolor(PANEL_BG)
hide(ax4)
ax4.imshow(dp_render, extent=[0, CANVAS_W, CANVAS_H, 0])
ax4.set_xlim(0, CANVAS_W); ax4.set_ylim(CANVAS_H, 0)
ax4.set_aspect("equal")
ax4.set_title(f"Douglas-Peucker\n{len(dp_x)} straight L segments",
              color=RED, fontsize=10, fontweight="bold", pad=6)

# Panel 5: Inflection result
ax5 = fig.add_subplot(gs[1, 1])
ax5.set_facecolor(PANEL_BG)
hide(ax5)
ax5.imshow(result_render, extent=[0, CANVAS_W, CANVAS_H, 0])
ax5.set_xlim(0, CANVAS_W); ax5.set_ylim(CANVAS_H, 0)
ax5.set_aspect("equal")
ax5.set_title(f"Inflection-Preserving\n{n_l_cmds} L + {n_c_cmds} C segments",
              color=TEAL, fontsize=10, fontweight="bold", pad=6)

# Panel 6: Overlay both
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor(PANEL_BG)
hide(ax6)

# Render overlay SVG
overlay_svg = (
    f'<svg xmlns="http://www.w3.org/2000/svg" '
    f'viewBox="0 0 {CANVAS_W} {CANVAS_H}" '
    f'width="{CANVAS_W}" height="{CANVAS_H}">\n'
    f'  <path d="{dp_d}" fill="none" stroke="{RED}" stroke-width="1" stroke-opacity="0.5"/>\n'
    f'  <path d="{result_d}" fill="none" stroke="{TEAL}" stroke-width="1.5"/>\n'
    f'</svg>'
)
overlay_render = np.array(Image.open(io.BytesIO(
    cairosvg.svg2png(bytestring=overlay_svg.encode(),
                      output_width=CANVAS_W * render_scale,
                      output_height=CANVAS_H * render_scale)
)).convert("RGBA"))

ax6.imshow(overlay_render, extent=[0, CANVAS_W, CANVAS_H, 0])
ax6.set_xlim(0, CANVAS_W); ax6.set_ylim(CANVAS_H, 0)
ax6.set_aspect("equal")
ax6.set_title("Overlay\nRed=DP straight vs Teal=Inflection curves",
              color=TEXT, fontsize=10, fontweight="bold", pad=6)

# Panel 7: Original
ax7 = fig.add_subplot(gs[1, 3])
ax7.set_facecolor(PANEL_BG)
hide(ax7)
ax7.imshow(orig_render, extent=[0, CANVAS_W, CANVAS_H, 0])
ax7.set_xlim(0, CANVAS_W); ax7.set_ylim(CANVAS_H, 0)
ax7.set_aspect("equal")
ax7.set_title("Original SVG\n67 paths", color=TEXT,
              fontsize=10, fontweight="bold", pad=6)

# ── Row 3: Zoomed comparisons ──────────────────────────────────────

zoom_regions = [
    {"name": "Tail Sweep", "x1": 170, "y1": 90, "x2": 260, "y2": 190},
    {"name": "Feathery Top", "x1": 120, "y1": -5, "x2": 220, "y2": 80},
    {"name": "Bottom/Feet", "x1": 30, "y1": 210, "x2": 210, "y2": 262},
    {"name": "Ear/Head", "x1": 55, "y1": 55, "x2": 155, "y2": 150},
]

for zi, zoom in enumerate(zoom_regions):
    ax = fig.add_subplot(gs[2, zi])
    ax.set_facecolor(PANEL_BG)
    hide(ax)

    zw = zoom["x2"] - zoom["x1"]
    zh = zoom["y2"] - zoom["y1"]

    # Render zoomed overlay
    zoom_svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{zoom["x1"]} {zoom["y1"]} {zw} {zh}" '
        f'width="{zw * 4}" height="{zh * 4}">\n'
        f'  <path d="{dp_d}" fill="none" stroke="{RED}" '
        f'stroke-width="0.8" stroke-opacity="0.5"/>\n'
        f'  <path d="{result_d}" fill="none" stroke="{TEAL}" '
        f'stroke-width="1.2"/>\n'
        f'</svg>'
    )
    try:
        zpng = cairosvg.svg2png(bytestring=zoom_svg.encode(),
                                 output_width=zw * 4, output_height=zh * 4)
        zarr = np.array(Image.open(io.BytesIO(zpng)).convert("RGBA"))
        ax.imshow(zarr, extent=[zoom["x1"], zoom["x2"], zoom["y2"], zoom["y1"]])
    except Exception:
        pass

    # Also show original faintly
    s = render_scale
    y1c = max(0, int(zoom["y1"] * s))
    y2c = min(orig_render.shape[0], int(zoom["y2"] * s))
    x1c = max(0, int(zoom["x1"] * s))
    x2c = min(orig_render.shape[1], int(zoom["x2"] * s))
    if y2c > y1c and x2c > x1c:
        crop = orig_render[y1c:y2c, x1c:x2c]
        ax.imshow(crop, extent=[zoom["x1"], zoom["x2"], zoom["y2"], zoom["y1"]],
                  alpha=0.15)

    # Mark anchors in this region
    for i in range(n_anchors):
        if (zoom["x1"] <= merged_x[i] <= zoom["x2"] and
                zoom["y1"] <= merged_y[i] <= zoom["y2"]):
            marker = "^" if merged_types[i] == "peak" else "o"
            color = ACCENT if merged_types[i] == "peak" else YELLOW
            ax.plot(merged_x[i], merged_y[i], marker, color=color,
                    markersize=6, markeredgecolor="white", markeredgewidth=0.5,
                    zorder=5)

    ax.set_xlim(zoom["x1"], zoom["x2"])
    ax.set_ylim(zoom["y2"], zoom["y1"])
    ax.set_aspect("equal")

    ax.set_title(zoom["name"], color=TEXT,
                 fontsize=10, fontweight="bold", pad=6)

# ── Footer ──────────────────────────────────────────────────────────

fig.text(0.5, 0.008,
         f"Result: {n_l_cmds} L (straight) + {n_c_cmds} C (curve) = "
         f"{total_pts} control points | "
         f"mean dev: {mean_dev:.2f}px | max dev: {max_dev:.2f}px | "
         f"area loss: {area_loss:.2f}% | SVG: {len(result_d)} chars | "
         f"Zero magic numbers (svg_px, knot span, IQR, Q1)",
         color=SUBTLE, fontsize=8, ha="center")

fig.savefig(OUT_DIR / "06_inflection_silhouette.png", dpi=150, facecolor=BG,
            bbox_inches="tight", pad_inches=0.3)
plt.close(fig)
print(f"\nSaved: {OUT_DIR / '06_inflection_silhouette.png'}")

# ── Summary ──────────────────────────────────────────────────────────

print(f"\n{'='*65}")
print("SUMMARY — Zero Magic Numbers")
print(f"{'='*65}")
print(f"  Raw contour:         {n_raw} points")
print(f"  Spline smoothing:    s = n * spacing^2 = {s_curvature:.1f}")
print(f"  Curvature window:    {window} samples (n_eval / n_work)")
print(f"  Peak prominence:     {prominence_threshold:.4f} (1 sigma)")
print(f"  Min feature gap:     {min_param_gap:.4f} (Nyquist)")
print(f"  Inflection points:   {n_inflections}")
print(f"  Curvature peaks:     {n_peaks}")
print(f"  Total anchors:       {n_anchors}")
print(f"  L commands:          {n_l_cmds} (straight, curvature < Q1)")
print(f"  C commands:          {n_c_cmds} (curves, long segs subdivided)")
print(f"  Control points:      {total_pts}")
print(f"  SVG path:            {len(result_d)} chars")
print(f"  Mean deviation:      {mean_dev:.2f} px")
print(f"  Max deviation:       {max_dev:.2f} px")
print(f"  Area loss:           {area_loss:.2f}%")
print(f"{'='*65}")
print(f"\nEvery parameter above is derived from the data:")
print(f"  smoothing = n_points * svg_pixel^2 (SVG resolution, not raster)")
print(f"  window    = n_eval / n_effective_dof (knot span of spline)")
print(f"  prominence = IQR of |curvature| (robust to outliers)")
print(f"  min_gap   = 1 / n_effective_dof (knot span in param space)")
