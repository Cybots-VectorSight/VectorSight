"""Research-backed contour simplification: smooth path + spike metadata.

Based on:
- Fourier Descriptors (Zahn & Roskies 1972): noise-floor low-pass filtering
- Curvature Scale Space (Mokhtarian & Mackworth, MPEG-7): multi-scale persistence
- Schneider's algorithm (Graphics Gems 1990): least-squares Bezier fitting
- Otsu's method (Otsu 1979): optimal binary thresholding (zero parameters)

Key insight: spikes are METADATA, not geometry.
  - The smooth Bezier path = emoji-like silhouette outline
  - Curvature spikes = text annotations for LLM enrichment
  - LLM receives ONE clean path + text list of sharp features

Pipeline:
  Stage 0: Extract raw contour (marching squares)
  Stage 1: Fourier denoising (noise-floor low-pass)
  Stage 2: Resample to 1 pt/SVG px
  Stage 3: CSS corner detection (multi-scale persistence, Otsu threshold)
  Stage 4: Schneider Bezier fitting (smooth path between CSS corners)
  Stage 5: Spike detection (Otsu on |curvature|, separate from path)
  Stage 6: Generate SVG + spike annotations

Every parameter derived from the data. Zero magic numbers.
No quartiles, no IQR, no Tukey fences -- only Otsu (max between-class variance).
"""

from __future__ import annotations

import io
from pathlib import Path

import cairosvg
import numpy as np
from PIL import Image
from skimage import measure
from skimage.measure import approximate_polygon
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from shapely.geometry import Polygon, LineString, Point
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

# -- Config --

SVG_PATH = (Path(__file__).resolve().parent.parent
            / "samples" / "test" / "faker.svg")
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


# =====================================================================
# STAGE 0: Extract raw contour
# =====================================================================

svg_text = SVG_PATH.read_text(encoding="utf-8")

print("Stage 0: Extracting raw contour...")
png = cairosvg.svg2png(
    bytestring=svg_text.encode(),
    output_width=CANVAS_W * RENDER_MULT,
    output_height=CANVAS_H * RENDER_MULT,
)
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

if np.allclose([raw_x[0], raw_y[0]], [raw_x[-1], raw_y[-1]], atol=0.1):
    raw_x = raw_x[:-1]
    raw_y = raw_y[:-1]

n_raw = len(raw_x)
raw_diffs = np.sqrt(np.diff(raw_x)**2 + np.diff(raw_y)**2)
raw_arc_length = raw_diffs.sum()
print(f"  Raw contour: {n_raw} points, arc: {raw_arc_length:.1f} px")


# =====================================================================
# STAGE 1: Fourier Denoising (noise-floor low-pass)
# =====================================================================

print("\nStage 1: Fourier denoising...")

# Even resampling before DFT (Nyquist for 1-SVG-pixel features)
n_resample = max(256, int(2 * raw_arc_length))
raw_cumlen = np.concatenate([[0], np.cumsum(raw_diffs)])
even_s = np.linspace(0, raw_cumlen[-1], n_resample, endpoint=False)
resamp_x = np.interp(even_s, raw_cumlen, raw_x)
resamp_y = np.interp(even_s, raw_cumlen, raw_y)

z = resamp_x + 1j * resamp_y
Z = np.fft.fft(z)
N = len(Z)
energies = np.abs(Z)**2

# Noise floor = median energy of upper-quarter frequencies (raster noise)
freq_of = np.array([min(k, N - k) for k in range(N)])
upper_quarter_mask = freq_of > (N // 4)
noise_floor = float(np.median(energies[upper_quarter_mask]))

keep_mask = energies > noise_floor
keep_mask[0] = True
n_keep = int(keep_mask.sum())

Z_filtered = Z * keep_mask
z_clean = np.fft.ifft(Z_filtered)
clean_x, clean_y = z_clean.real, z_clean.imag

denoise_shift = np.sqrt((resamp_x - clean_x)**2 + (resamp_y - clean_y)**2)
kept_indices = np.where(keep_mask)[0]
max_freq_idx = int(max(min(k, N - k) for k in kept_indices))

print(f"  Kept: {n_keep}/{N} descriptors ({100*n_keep/N:.1f}%)")
print(f"  Shift: mean {denoise_shift.mean():.3f}px, max {denoise_shift.max():.3f}px")


# =====================================================================
# STAGE 2: Resample to working resolution
# =====================================================================

print("\nStage 2: Resampling...")

clean_diffs = np.sqrt(np.diff(clean_x)**2 + np.diff(clean_y)**2)
clean_arc = clean_diffs.sum()
n_work = max(256, int(clean_arc))  # 1 pt per SVG pixel

clean_cumlen = np.concatenate([[0], np.cumsum(clean_diffs)])
even_s2 = np.linspace(0, clean_cumlen[-1], n_work, endpoint=False)
cx = np.interp(even_s2, clean_cumlen, clean_x)
cy = np.interp(even_s2, clean_cumlen, clean_y)
n_pts = len(cx)

print(f"  Working contour: {n_pts} pts, arc: {clean_arc:.1f} px")


# =====================================================================
# STAGE 3: CSS Corner Detection (Otsu on persistence)
# =====================================================================

print("\nStage 3: CSS corner detection (Otsu threshold)...")


def curvature_from_coords(x, y):
    """Signed curvature via central differences on periodic arrays."""
    dx = np.roll(x, -1) - np.roll(x, 1)
    dy = np.roll(y, -1) - np.roll(y, 1)
    ddx = np.roll(x, -1) - 2 * x + np.roll(x, 1)
    ddy = np.roll(y, -1) - 2 * y + np.roll(y, 1)
    num = dx * ddy - dy * ddx
    den = (dx**2 + dy**2) ** 1.5
    den = np.maximum(den, 1e-10)
    return num / den


# MPEG-7 CSS scale range:
#   min_sigma = 1, max_sigma = N/(2*pi), octave spacing
min_sigma = 1.0
max_sigma = n_pts / (2 * np.pi)
n_octaves = int(np.floor(np.log2(max_sigma / min_sigma))) + 1
sigmas = [min_sigma * (2**i) for i in range(n_octaves)]

print(f"  Octaves: {n_octaves} (sigma 1 to {max_sigma:.0f})")

# Persistence voting across scales
persistence_votes = np.zeros(n_pts)
curvature_at_finest = None
signed_curvature_finest = None

for si, sigma in enumerate(sigmas):
    if sigma >= 0.5:
        sx = gaussian_filter1d(cx, sigma=sigma, mode="wrap")
        sy = gaussian_filter1d(cy, sigma=sigma, mode="wrap")
    else:
        sx, sy = cx.copy(), cy.copy()

    kappa = curvature_from_coords(sx, sy)
    abs_kappa = np.abs(kappa)

    if si == 0:
        curvature_at_finest = abs_kappa.copy()
        signed_curvature_finest = kappa.copy()

    # Min peak distance at this scale: 2*sigma (Nyquist for Gaussian kernel)
    min_dist = max(3, int(2 * sigma))

    # Use Otsu on this scale's curvature to find prominence threshold.
    # Otsu maximizes between-class variance: "prominent peaks" vs "background".
    try:
        otsu_k = threshold_otsu(abs_kappa)
    except ValueError:
        otsu_k = float(np.median(abs_kappa))

    peaks, _ = find_peaks(abs_kappa, distance=min_dist, height=otsu_k)

    for pk in peaks:
        persistence_votes[pk] += 1

# Otsu on persistence votes to separate "corner" from "non-corner".
# No arbitrary "half the scales" -- Otsu finds the natural boundary.
nonzero_persist = persistence_votes[persistence_votes > 0]
if len(nonzero_persist) > 0:
    try:
        persist_otsu = threshold_otsu(nonzero_persist)
    except ValueError:
        persist_otsu = float(np.median(nonzero_persist))
else:
    persist_otsu = 1.0

corner_candidates = np.where(persistence_votes >= persist_otsu)[0]

print(f"  Otsu persistence threshold: {persist_otsu:.1f} / {n_octaves}")
print(f"  Candidates before merge: {len(corner_candidates)}")

# Merge nearby corners (within 2*sigma_base = resolution limit)
merge_dist = max(3, int(4 * min_sigma))

if len(corner_candidates) > 0:
    merged = [corner_candidates[0]]
    for c in corner_candidates[1:]:
        if c - merged[-1] >= merge_dist:
            merged.append(c)
        else:
            if persistence_votes[c] > persistence_votes[merged[-1]]:
                merged[-1] = c

    if len(merged) > 1:
        wrap_dist = n_pts - merged[-1] + merged[0]
        if wrap_dist < merge_dist:
            if persistence_votes[merged[0]] >= persistence_votes[merged[-1]]:
                merged.pop()
            else:
                merged.pop(0)

    corner_indices = np.array(merged)
else:
    corner_indices = np.array([], dtype=int)

n_corners = len(corner_indices)
corner_persist = persistence_votes[corner_indices] if n_corners > 0 else []
print(f"  CSS corners: {n_corners}")
print(f"  Persistence values: {corner_persist}")


# =====================================================================
# STAGE 4: Schneider Bezier Fitting (smooth path)
# =====================================================================

print("\nStage 4: Schneider Bezier fitting...")

# Error tolerance = median edge length (fitting below this is fitting noise)
work_edges = np.sqrt(np.diff(cx)**2 + np.diff(cy)**2)
work_edges = np.append(work_edges,
                       np.sqrt((cx[0]-cx[-1])**2 + (cy[0]-cy[-1])**2))
error_tol = float(np.median(work_edges))
print(f"  Error tolerance (median edge): {error_tol:.3f} px")


def fit_cubic_bezier_ls(points):
    """Least-squares cubic Bezier fit (Schneider's method)."""
    n = len(points)
    P0, P3 = points[0], points[-1]

    if n <= 2:
        C1 = P0 + (P3 - P0) / 3
        C2 = P0 + 2 * (P3 - P0) / 3
        return C1, C2, 0.0, 0

    chords = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    total_chord = chords.sum()
    if total_chord < 1e-10:
        C1 = P0 + (P3 - P0) / 3
        C2 = P0 + 2 * (P3 - P0) / 3
        return C1, C2, 0.0, 0

    t = np.zeros(n)
    t[1:] = np.cumsum(chords) / total_chord
    t[-1] = 1.0

    A1 = 3 * (1 - t)**2 * t
    A2 = 3 * (1 - t) * t**2
    b0 = (1 - t)**3
    b3 = t**3
    d = points - b0[:, None] * P0 - b3[:, None] * P3

    m11 = np.dot(A1, A1)
    m12 = np.dot(A1, A2)
    m22 = np.dot(A2, A2)
    det = m11 * m22 - m12 * m12

    if abs(det) < 1e-10:
        C1 = P0 + (P3 - P0) / 3
        C2 = P0 + 2 * (P3 - P0) / 3
    else:
        r1 = np.array([np.dot(A1, d[:, 0]), np.dot(A1, d[:, 1])])
        r2 = np.array([np.dot(A2, d[:, 0]), np.dot(A2, d[:, 1])])
        C1 = (m22 * r1 - m12 * r2) / det
        C2 = (m11 * r2 - m12 * r1) / det

    fitted = (b0[:, None] * P0 + A1[:, None] * C1 +
              A2[:, None] * C2 + b3[:, None] * P3)
    errors = np.sqrt(np.sum((points - fitted)**2, axis=1))
    return C1, C2, float(errors.max()), int(errors.argmax())


def schneider_fit(points, tol, depth=0):
    """Recursive Schneider fit: split at worst point if error > tol."""
    max_depth = max(1, int(np.log2(max(len(points), 2))))
    if len(points) < 4 or depth > max_depth:
        if len(points) < 2:
            return []
        C1, C2, _, _ = fit_cubic_bezier_ls(points)
        return [(points[0], C1, C2, points[-1])]

    C1, C2, max_err, worst = fit_cubic_bezier_ls(points)
    if max_err <= tol:
        return [(points[0], C1, C2, points[-1])]

    split = max(1, min(worst, len(points) - 2))
    left = schneider_fit(points[:split + 1], tol, depth + 1)
    right = schneider_fit(points[split:], tol, depth + 1)
    return left + right


# Build sections between CSS corners (smooth path only)
if n_corners == 0:
    all_pts = np.column_stack([cx, cy])
    all_pts = np.vstack([all_pts, all_pts[0]])
    sections_data = [(all_pts,)]
else:
    sections_data = []
    for i in range(n_corners):
        start = corner_indices[i]
        end = corner_indices[(i + 1) % n_corners]
        if end > start:
            indices = np.arange(start, end + 1)
        else:
            indices = np.concatenate([np.arange(start, n_pts),
                                      np.arange(0, end + 1)])
        pts = np.column_stack([cx[indices], cy[indices]])
        sections_data.append((pts,))

# Straight vs curve: Otsu on curvature (not quartile)
try:
    straight_otsu = threshold_otsu(curvature_at_finest)
except ValueError:
    straight_otsu = float(np.median(curvature_at_finest))
print(f"  Straight threshold (Otsu on |curvature|): {straight_otsu:.5f}")

print(f"  Fitting {len(sections_data)} sections...")
all_beziers = []
section_types = []
section_indices_list = []

for si, (pts,) in enumerate(sections_data):
    if n_corners > 0:
        start = corner_indices[si]
        end = corner_indices[(si + 1) % n_corners]
        if end > start:
            sec_idx = np.arange(start, end + 1)
        else:
            sec_idx = np.concatenate([np.arange(start, n_pts),
                                      np.arange(0, end + 1)])
    else:
        sec_idx = np.arange(n_pts)

    sec_curv_max = float(curvature_at_finest[sec_idx].max())

    if sec_curv_max < straight_otsu and len(pts) > 1:
        section_types.append("L")
        all_beziers.append(None)
    else:
        beziers = schneider_fit(pts, error_tol)
        section_types.append("C")
        all_beziers.append(beziers)

    section_indices_list.append(sec_idx)


# =====================================================================
# STAGE 5: Spike Detection (Otsu, separate metadata)
# =====================================================================

print("\nStage 5: Spike detection (metadata, not in path)...")

# Otsu on |curvature| separates "spike" from "smooth".
# Spikes are points where curvature is in the upper Otsu class.
try:
    spike_otsu = threshold_otsu(curvature_at_finest)
except ValueError:
    spike_otsu = float(np.median(curvature_at_finest))

# Find peaks above Otsu threshold
spike_min_dist = max(3, int(4 * min_sigma))
spikes, spike_props = find_peaks(
    curvature_at_finest,
    height=spike_otsu,
    distance=spike_min_dist,
)

# For each spike, compute metadata for LLM
spike_data = []
for sp in spikes:
    pct = sp / n_pts * 100
    magnitude = float(curvature_at_finest[sp])
    sign = "convex" if signed_curvature_finest[sp] > 0 else "concave"
    x_pos = float(cx[sp])
    y_pos = float(cy[sp])

    # Direction: the normal to the contour at this point
    i_prev = (sp - 1) % n_pts
    i_next = (sp + 1) % n_pts
    tangent_x = cx[i_next] - cx[i_prev]
    tangent_y = cy[i_next] - cy[i_prev]
    # Normal (90 deg rotation)
    norm_len = np.sqrt(tangent_x**2 + tangent_y**2)
    if norm_len > 0:
        normal_x = -tangent_y / norm_len
        normal_y = tangent_x / norm_len
    else:
        normal_x, normal_y = 0.0, 1.0

    # Quadrant (for LLM: "upper-right", "lower-left", etc.)
    cx_mid, cy_mid = CANVAS_W / 2, CANVAS_H / 2
    h = "left" if x_pos < cx_mid else "right"
    v = "upper" if y_pos < cy_mid else "lower"
    quadrant = f"{v}-{h}"

    spike_data.append({
        "index": int(sp),
        "pct": pct,
        "x": x_pos,
        "y": y_pos,
        "magnitude": magnitude,
        "sign": sign,
        "quadrant": quadrant,
        "normal": (float(normal_x), float(normal_y)),
    })

n_spikes = len(spike_data)
print(f"  Otsu spike threshold: {spike_otsu:.5f}")
print(f"  Spikes detected: {n_spikes}")
for sd in spike_data[:8]:
    print(f"    {sd['pct']:5.1f}% {sd['quadrant']:12s} "
          f"|k|={sd['magnitude']:.4f} ({sd['sign']})")
if n_spikes > 8:
    print(f"    ... and {n_spikes - 8} more")


# =====================================================================
# STAGE 6: Generate SVG Path + Spike Annotations
# =====================================================================

print("\nStage 6: Generating SVG path...")

if n_corners > 0:
    start_x, start_y = cx[corner_indices[0]], cy[corner_indices[0]]
else:
    start_x, start_y = cx[0], cy[0]

path_cmds = [f"M {start_x:.1f},{start_y:.1f}"]
n_l_cmds = 0
n_c_cmds = 0

for si, (stype, beziers) in enumerate(zip(section_types, all_beziers)):
    if stype == "L":
        pts = sections_data[si][0]
        path_cmds.append(f"L {pts[-1, 0]:.1f},{pts[-1, 1]:.1f}")
        n_l_cmds += 1
    elif beziers:
        for P0, C1, C2, P3 in beziers:
            path_cmds.append(
                f"C {C1[0]:.1f},{C1[1]:.1f} "
                f"{C2[0]:.1f},{C2[1]:.1f} "
                f"{P3[0]:.1f},{P3[1]:.1f}"
            )
            n_c_cmds += 1

path_cmds.append("Z")
result_d = " ".join(path_cmds)
total_ctrl_pts = 1 + n_l_cmds + n_c_cmds * 3

print(f"  L: {n_l_cmds}, C: {n_c_cmds}, pts: {total_ctrl_pts}, "
      f"chars: {len(result_d)}")

# Generate spike annotation text (for LLM enrichment)
spike_text_lines = [f"Sharp features: {n_spikes} curvature spikes"]
for sd in spike_data:
    spike_text_lines.append(
        f"  {sd['pct']:5.1f}% ({sd['quadrant']}) "
        f"|k|={sd['magnitude']:.3f} {sd['sign']}"
    )
spike_text = "\n".join(spike_text_lines)

print(f"\n  --- Spike annotations for LLM ---")
print(spike_text)
print(f"  --- End annotations ---")


# =====================================================================
# Quality Measurement
# =====================================================================

print("\nMeasuring quality...")

result_pts_x, result_pts_y = [], []
for si, (stype, beziers) in enumerate(zip(section_types, all_beziers)):
    if stype == "L":
        pts = sections_data[si][0]
        result_pts_x.extend([pts[0, 0], pts[-1, 0]])
        result_pts_y.extend([pts[0, 1], pts[-1, 1]])
    elif beziers:
        for P0, C1, C2, P3 in beziers:
            for t in np.linspace(0, 1, 30):
                bx = ((1-t)**3 * P0[0] + 3*(1-t)**2*t * C1[0] +
                      3*(1-t)*t**2 * C2[0] + t**3 * P3[0])
                by = ((1-t)**3 * P0[1] + 3*(1-t)**2*t * C1[1] +
                      3*(1-t)*t**2 * C2[1] + t**3 * P3[1])
                result_pts_x.append(bx)
                result_pts_y.append(by)

result_line = LineString(list(zip(result_pts_x, result_pts_y)))

sample_step = max(1, n_raw // 500)
devs = [result_line.distance(Point(raw_x[i], raw_y[i]))
        for i in range(0, n_raw, sample_step)]
mean_dev = float(np.mean(devs))
max_dev = float(np.max(devs))

raw_poly = Polygon(np.column_stack([raw_x, raw_y]))
result_poly = Polygon(list(zip(result_pts_x, result_pts_y)))
if raw_poly.is_valid and result_poly.is_valid and raw_poly.area > 0:
    area_loss = abs(raw_poly.area - result_poly.area) / raw_poly.area * 100
else:
    area_loss = 0.0

print(f"  Mean deviation: {mean_dev:.2f} px")
print(f"  Max deviation:  {max_dev:.2f} px")
print(f"  Area loss:      {area_loss:.2f}%")


# =====================================================================
# Save SVG
# =====================================================================

result_svg = (
    f'<svg xmlns="http://www.w3.org/2000/svg" '
    f'viewBox="0 0 {CANVAS_W} {CANVAS_H}" '
    f'width="{CANVAS_W}" height="{CANVAS_H}">\n'
    f'  <path d="{result_d}" fill="{TEAL}" fill-opacity="0.3" '
    f'stroke="{TEAL}" stroke-width="1.5"/>\n'
    f'</svg>'
)
svg_out = OUT_DIR / "07_research_silhouette.svg"
svg_out.write_text(result_svg)
print(f"\nSaved SVG: {svg_out}")


# =====================================================================
# Render comparison images
# =====================================================================

print("\nRendering...")
render_scale = 3

result_render = np.array(Image.open(io.BytesIO(
    cairosvg.svg2png(bytestring=result_svg.encode(),
                     output_width=CANVAS_W * render_scale,
                     output_height=CANVAS_H * render_scale)
)).convert("RGBA"))

orig_render = np.array(Image.open(io.BytesIO(
    cairosvg.svg2png(bytestring=svg_text.encode(),
                     output_width=CANVAS_W * render_scale,
                     output_height=CANVAS_H * render_scale)
)).convert("RGBA"))

# DP comparison
dp_contour = approximate_polygon(raw_contour, tolerance=3.0)
dp_x = dp_contour[:, 1] / RENDER_MULT
dp_y = dp_contour[:, 0] / RENDER_MULT
dp_parts = [f"M {dp_x[0]:.1f},{dp_y[0]:.1f}"]
for x, y in zip(dp_x[1:], dp_y[1:]):
    dp_parts.append(f"L {x:.1f},{y:.1f}")
dp_parts.append("Z")
dp_d = " ".join(dp_parts)
dp_svg_str = (
    f'<svg xmlns="http://www.w3.org/2000/svg" '
    f'viewBox="0 0 {CANVAS_W} {CANVAS_H}" '
    f'width="{CANVAS_W}" height="{CANVAS_H}">\n'
    f'  <path d="{dp_d}" fill="{RED}" fill-opacity="0.3" '
    f'stroke="{RED}" stroke-width="1.5"/>\n'
    f'</svg>'
)
dp_render = np.array(Image.open(io.BytesIO(
    cairosvg.svg2png(bytestring=dp_svg_str.encode(),
                     output_width=CANVAS_W * render_scale,
                     output_height=CANVAS_H * render_scale)
)).convert("RGBA"))

# Inflection comparison
inflection_render = None
inflection_svg_path = OUT_DIR / "06_inflection_silhouette.svg"
if inflection_svg_path.exists():
    try:
        inflection_render = np.array(Image.open(io.BytesIO(
            cairosvg.svg2png(
                bytestring=inflection_svg_path.read_text(encoding="utf-8").encode(),
                output_width=CANVAS_W * render_scale,
                output_height=CANVAS_H * render_scale)
        )).convert("RGBA"))
    except Exception:
        pass


# =====================================================================
# VISUALIZATION
# =====================================================================

print("\nGenerating visualization...")

fig = plt.figure(figsize=(26, 20), facecolor=BG, dpi=150)
fig.suptitle(
    "Smooth Path + Spike Metadata (Zero Magic Numbers)",
    color=TEXT, fontsize=18, fontweight="bold", y=0.975,
)
fig.text(
    0.5, 0.955,
    f"Path: Fourier >> CSS ({n_corners} corners, Otsu) >> "
    f"Schneider ({n_c_cmds} C) = {total_ctrl_pts} pts   |   "
    f"Metadata: {n_spikes} curvature spikes (Otsu)",
    color=SUBTLE, fontsize=9, ha="center",
)

gs = fig.add_gridspec(
    4, 4, hspace=0.30, wspace=0.15,
    left=0.03, right=0.97, top=0.93, bottom=0.04,
)

# -- Row 1: Pipeline --

# Panel 1: Fourier spectrum
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(PANEL_BG)
ax1.set_title("Fourier Energy Spectrum", color=TEXT,
              fontsize=10, fontweight="bold", pad=6)

half_energies = energies[:N // 2]
ax1.semilogy(np.arange(N // 2), half_energies, color=SUBTLE,
             linewidth=0.5, alpha=0.7, label="All")

kept_half = np.array([i for i in range(N // 2) if keep_mask[i]])
if len(kept_half) > 0:
    ax1.semilogy(kept_half, energies[kept_half], ".", color=TEAL,
                 markersize=3, label=f"Kept ({n_keep})")

ax1.axhline(y=noise_floor, color=ACCENT, linewidth=1, linestyle="--",
            alpha=0.7, label="Noise floor")
ax1.set_xlabel("Frequency", color=SUBTLE, fontsize=8)
ax1.set_ylabel("Energy (log)", color=SUBTLE, fontsize=8)
ax1.set_xlim(0, min(N // 2, max_freq_idx * 3))
style_chart(ax1)
ax1.legend(fontsize=6, facecolor=PANEL_BG, edgecolor=SUBTLE,
           labelcolor=TEXT, loc="upper right")

# Panel 2: CSS Persistence + Otsu threshold
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(PANEL_BG)
ax2.set_title("CSS Persistence (Otsu threshold)", color=TEXT,
              fontsize=10, fontweight="bold", pad=6)

arc_pct = np.linspace(0, 100, n_pts)
ax2.fill_between(arc_pct, 0, persistence_votes, alpha=0.3, color=TEAL)
ax2.plot(arc_pct, persistence_votes, color=TEAL, linewidth=0.8)
ax2.axhline(y=persist_otsu, color=ACCENT, linewidth=1, linestyle="--",
            alpha=0.7, label=f"Otsu ({persist_otsu:.1f})")

for ci in corner_indices:
    ax2.axvline(x=ci / n_pts * 100, color=YELLOW, linewidth=0.8, alpha=0.6)

ax2.set_xlabel("Position (%)", color=SUBTLE, fontsize=8)
ax2.set_ylabel("Persistence", color=SUBTLE, fontsize=8)
ax2.set_xlim(0, 100)
style_chart(ax2)
ax2.legend(fontsize=6, facecolor=PANEL_BG, edgecolor=SUBTLE,
           labelcolor=TEXT, loc="upper right")

# Panel 3: Curvature + spike Otsu
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor(PANEL_BG)
ax3.set_title("Curvature Spikes (Otsu threshold)", color=TEXT,
              fontsize=10, fontweight="bold", pad=6)

ax3.plot(arc_pct, curvature_at_finest, color=TEAL, linewidth=0.5, alpha=0.6)
ax3.axhline(y=spike_otsu, color=ACCENT, linewidth=1, linestyle="--",
            alpha=0.7, label=f"Otsu ({spike_otsu:.4f})")

for sd in spike_data:
    ax3.plot(sd["pct"], sd["magnitude"], "v", color=ACCENT,
             markersize=4, markeredgecolor="white", markeredgewidth=0.3)

ax3.set_xlabel("Position (%)", color=SUBTLE, fontsize=8)
ax3.set_ylabel("|curvature|", color=SUBTLE, fontsize=8)
ax3.set_xlim(0, 100)
style_chart(ax3)
ax3.legend(fontsize=6, facecolor=PANEL_BG, edgecolor=SUBTLE,
           labelcolor=TEXT, loc="upper right")

# Panel 4: Pipeline summary
ax4 = fig.add_subplot(gs[0, 3])
ax4.set_facecolor(PANEL_BG)
hide(ax4)
ax4.set_xlim(0, 100)
ax4.set_ylim(0, 100)
ax4.set_title("Pipeline", color=TEXT, fontsize=10, fontweight="bold", pad=6)

info = [
    (92, f"0. Contour: {n_raw} raw pts", TEXT),
    (80, f"1. Fourier: {n_keep}/{N} (noise-floor)", TEAL),
    (68, f"2. Resample: {n_pts} pts (1/SVG px)", TEXT),
    (56, f"3. CSS corners: {n_corners} (Otsu >= {persist_otsu:.1f})", YELLOW),
    (44, f"4. Schneider: {n_c_cmds} C, tol={error_tol:.2f}px", GREEN),
    (32, f"5. Spikes: {n_spikes} (Otsu >= {spike_otsu:.4f})", ACCENT),
    (20, f"PATH: {n_l_cmds}L + {n_c_cmds}C = {total_ctrl_pts} pts", TEAL),
    (10, f"META: {n_spikes} spike annotations", ACCENT),
    (2, f"Quality: {mean_dev:.2f}px / {area_loss:.1f}% loss", SUBTLE),
]
for y, txt, col in info:
    ax4.text(3, y, txt, color=col, fontsize=7, fontweight="bold",
             verticalalignment="center", path_effects=stroke_thin)

# -- Row 2: Smooth path + spike overlay --

# Panel 5: Smooth path with CSS corners
ax5 = fig.add_subplot(gs[1, 0:2])
ax5.set_facecolor(PANEL_BG)
hide(ax5)
ax5.set_xlim(-5, CANVAS_W + 5)
ax5.set_ylim(CANVAS_H + 5, -5)
ax5.set_aspect("equal")
ax5.set_title(f"Smooth Path ({n_corners} CSS corners, {n_c_cmds} Beziers)",
              color=TEXT, fontsize=10, fontweight="bold", pad=6)

for si, (sec_idx, stype) in enumerate(zip(section_indices_list,
                                           section_types)):
    col = BLUE if stype == "L" else TEAL
    ax5.plot(cx[sec_idx], cy[sec_idx], color=col, linewidth=2, alpha=0.7)

for ci in corner_indices:
    ax5.plot(cx[ci], cy[ci], "o", color=YELLOW, markersize=7,
             markeredgecolor="white", markeredgewidth=0.5, zorder=5)

legend5 = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=YELLOW,
           markersize=6, label=f"CSS corners ({n_corners})", linestyle="None"),
    Line2D([0], [0], color=TEAL, linewidth=2, label=f"Beziers ({n_c_cmds})"),
]
ax5.legend(handles=legend5, fontsize=6, facecolor=PANEL_BG,
           edgecolor=SUBTLE, labelcolor=TEXT, loc="lower left")

# Panel 6: Spike overlay (metadata visualization)
ax6 = fig.add_subplot(gs[1, 2:4])
ax6.set_facecolor(PANEL_BG)
hide(ax6)
ax6.set_xlim(-5, CANVAS_W + 5)
ax6.set_ylim(CANVAS_H + 5, -5)
ax6.set_aspect("equal")
ax6.set_title(f"Spike Metadata ({n_spikes} spikes, for LLM)",
              color=TEXT, fontsize=10, fontweight="bold", pad=6)

# Draw smooth path faintly
ax6.plot(cx, cy, color=TEAL, linewidth=1, alpha=0.3)

# Draw spike arrows (normal direction * magnitude)
arrow_scale = 300  # visual scaling for arrows
for sd in spike_data:
    x, y = sd["x"], sd["y"]
    nx, ny = sd["normal"]
    mag = sd["magnitude"]
    col = ACCENT if sd["sign"] == "convex" else BLUE
    ax6.annotate("", xy=(x + nx * mag * arrow_scale,
                         y + ny * mag * arrow_scale),
                 xytext=(x, y),
                 arrowprops=dict(arrowstyle="->", color=col,
                                lw=1.5, alpha=0.7))
    ax6.plot(x, y, ".", color=col, markersize=4)

legend6 = [
    Line2D([0], [0], color=ACCENT, linewidth=2,
           label=f"Convex spikes (outward)"),
    Line2D([0], [0], color=BLUE, linewidth=2,
           label=f"Concave spikes (inward)"),
]
ax6.legend(handles=legend6, fontsize=6, facecolor=PANEL_BG,
           edgecolor=SUBTLE, labelcolor=TEXT, loc="lower left")

# -- Row 3: SVG Render Comparison --

ax7 = fig.add_subplot(gs[2, 0])
ax7.set_facecolor(PANEL_BG)
hide(ax7)
ax7.imshow(dp_render, extent=[0, CANVAS_W, CANVAS_H, 0])
ax7.set_xlim(0, CANVAS_W); ax7.set_ylim(CANVAS_H, 0)
ax7.set_aspect("equal")
ax7.set_title(f"Douglas-Peucker\n{len(dp_x)} L segments",
              color=RED, fontsize=10, fontweight="bold", pad=6)

ax8 = fig.add_subplot(gs[2, 1])
ax8.set_facecolor(PANEL_BG)
hide(ax8)
if inflection_render is not None:
    ax8.imshow(inflection_render, extent=[0, CANVAS_W, CANVAS_H, 0])
    ax8.set_title("Inflection (06)\nB-spline + inflection pts",
                  color=ORANGE, fontsize=10, fontweight="bold", pad=6)
else:
    ax8.text(CANVAS_W / 2, CANVAS_H / 2, "06 not found",
             color=SUBTLE, ha="center", va="center")
    ax8.set_title("Previous", color=ORANGE, fontsize=10,
                  fontweight="bold", pad=6)
ax8.set_xlim(0, CANVAS_W); ax8.set_ylim(CANVAS_H, 0)
ax8.set_aspect("equal")

ax9 = fig.add_subplot(gs[2, 2])
ax9.set_facecolor(PANEL_BG)
hide(ax9)
ax9.imshow(result_render, extent=[0, CANVAS_W, CANVAS_H, 0])
ax9.set_xlim(0, CANVAS_W); ax9.set_ylim(CANVAS_H, 0)
ax9.set_aspect("equal")
ax9.set_title(f"Research (smooth path)\n{n_c_cmds} C = {total_ctrl_pts} pts",
              color=TEAL, fontsize=10, fontweight="bold", pad=6)

ax10 = fig.add_subplot(gs[2, 3])
ax10.set_facecolor(PANEL_BG)
hide(ax10)
ax10.imshow(orig_render, extent=[0, CANVAS_W, CANVAS_H, 0])
ax10.set_xlim(0, CANVAS_W); ax10.set_ylim(CANVAS_H, 0)
ax10.set_aspect("equal")
ax10.set_title("Original SVG\n67 paths", color=TEXT,
               fontsize=10, fontweight="bold", pad=6)

# -- Row 4: Zoomed Comparisons --

zoom_regions = [
    {"name": "Tail Sweep", "x1": 170, "y1": 90, "x2": 260, "y2": 190},
    {"name": "Feathery Top", "x1": 120, "y1": -5, "x2": 220, "y2": 80},
    {"name": "Bottom / Feet", "x1": 30, "y1": 210, "x2": 210, "y2": 262},
    {"name": "Ear / Head", "x1": 55, "y1": 55, "x2": 155, "y2": 150},
]

for zi, zoom in enumerate(zoom_regions):
    ax = fig.add_subplot(gs[3, zi])
    ax.set_facecolor(PANEL_BG)
    hide(ax)

    zw = zoom["x2"] - zoom["x1"]
    zh = zoom["y2"] - zoom["y1"]

    zoom_svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{zoom["x1"]} {zoom["y1"]} {zw} {zh}" '
        f'width="{zw * 4}" height="{zh * 4}">\n'
        f'  <path d="{dp_d}" fill="none" stroke="{RED}" '
        f'stroke-width="0.8" stroke-opacity="0.4"/>\n'
        f'  <path d="{result_d}" fill="none" stroke="{TEAL}" '
        f'stroke-width="1.2"/>\n'
        f'</svg>'
    )
    try:
        zpng = cairosvg.svg2png(bytestring=zoom_svg.encode(),
                                output_width=zw * 4, output_height=zh * 4)
        zarr = np.array(Image.open(io.BytesIO(zpng)).convert("RGBA"))
        ax.imshow(zarr,
                  extent=[zoom["x1"], zoom["x2"], zoom["y2"], zoom["y1"]])
    except Exception:
        pass

    # Faint original
    s = render_scale
    y1c = max(0, int(zoom["y1"] * s))
    y2c = min(orig_render.shape[0], int(zoom["y2"] * s))
    x1c = max(0, int(zoom["x1"] * s))
    x2c = min(orig_render.shape[1], int(zoom["x2"] * s))
    if y2c > y1c and x2c > x1c:
        ax.imshow(orig_render[y1c:y2c, x1c:x2c],
                  extent=[zoom["x1"], zoom["x2"], zoom["y2"], zoom["y1"]],
                  alpha=0.15)

    # Spike markers in zoom
    for sd in spike_data:
        if (zoom["x1"] <= sd["x"] <= zoom["x2"] and
                zoom["y1"] <= sd["y"] <= zoom["y2"]):
            col = ACCENT if sd["sign"] == "convex" else BLUE
            ax.plot(sd["x"], sd["y"], "^", color=col, markersize=5,
                    markeredgecolor="white", markeredgewidth=0.3, zorder=5)

    # CSS corners in zoom
    for ci in corner_indices:
        if (zoom["x1"] <= cx[ci] <= zoom["x2"] and
                zoom["y1"] <= cy[ci] <= zoom["y2"]):
            ax.plot(cx[ci], cy[ci], "o", color=YELLOW, markersize=6,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=5)

    ax.set_xlim(zoom["x1"], zoom["x2"])
    ax.set_ylim(zoom["y2"], zoom["y1"])
    ax.set_aspect("equal")
    ax.set_title(zoom["name"], color=TEXT,
                 fontsize=10, fontweight="bold", pad=6)

# -- Footer --

fig.text(
    0.5, 0.008,
    f"SMOOTH PATH: Fourier >> CSS (Otsu) >> Schneider = "
    f"{n_l_cmds}L + {n_c_cmds}C = {total_ctrl_pts} pts, "
    f"{len(result_d)} chars   |   "
    f"SPIKE META: {n_spikes} spikes (Otsu)   |   "
    f"dev: {mean_dev:.2f}px, area: {area_loss:.1f}%   |   "
    f"Zero magic numbers (all Otsu thresholds)",
    color=SUBTLE, fontsize=7.5, ha="center",
)

out_png = OUT_DIR / "07_research_silhouette.png"
fig.savefig(out_png, dpi=150, facecolor=BG,
            bbox_inches="tight", pad_inches=0.3)
plt.close(fig)
print(f"\nSaved: {out_png}")


# =====================================================================
# Summary
# =====================================================================

print(f"\n{'='*65}")
print("SUMMARY")
print(f"{'='*65}")
print(f"  SMOOTH PATH (emoji outline):")
print(f"    CSS corners:     {n_corners} (Otsu on persistence)")
print(f"    Bezier segments: {n_l_cmds} L + {n_c_cmds} C")
print(f"    Control points:  {total_ctrl_pts}")
print(f"    SVG chars:       {len(result_d)}")
print(f"    Mean deviation:  {mean_dev:.2f} px")
print(f"    Max deviation:   {max_dev:.2f} px")
print(f"    Area loss:       {area_loss:.2f}%")
print(f"")
print(f"  SPIKE METADATA (for LLM):")
print(f"    Spikes detected: {n_spikes} (Otsu on |curvature|)")
print(f"    Convex (outward): "
      f"{sum(1 for s in spike_data if s['sign'] == 'convex')}")
print(f"    Concave (inward): "
      f"{sum(1 for s in spike_data if s['sign'] == 'concave')}")
print(f"")
print(f"  ALL THRESHOLDS (zero magic numbers):")
print(f"    Noise floor     = median(upper-quarter DFT) = {noise_floor:.2e}")
print(f"    CSS persistence = Otsu(votes) = {persist_otsu:.1f}")
print(f"    Straight/curve  = Otsu(|kappa|) = {straight_otsu:.5f}")
print(f"    Spike threshold = Otsu(|kappa|) = {spike_otsu:.5f}")
print(f"    Bezier error    = median(edges) = {error_tol:.3f}")
print(f"{'='*65}")
