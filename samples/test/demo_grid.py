"""
VectorSight Grid Generator
Renders SVG to pixels, downsamples to text grid.
No AI. Just rendering + downsampling.
"""
from svgpathtools import svg2paths2
import numpy as np

# ============================================================
# STEP 1: Parse SVG and sample points from paths
# ============================================================
paths, attributes, svg_attrs = svg2paths2('anthropic.svg')

# Canvas size from viewBox
canvas_w, canvas_h = 24, 24

# Sample many points along each path to create a filled representation
all_points = []
for path in paths:
    # Sample the outline
    for t in np.linspace(0, 1, 500):
        try:
            pt = path.point(t)
            all_points.append((pt.real, pt.imag))
        except:
            pass

    # Fill interior by sampling between path segments
    for seg in path:
        start = seg.start
        end = seg.end
        for t in np.linspace(0, 1, 50):
            pt = seg.point(t)
            all_points.append((pt.real, pt.imag))

# ============================================================
# STEP 2: Create pixel grid (rasterize to NxN)
# ============================================================
def make_grid(points, canvas_w, canvas_h, resolution):
    grid = np.zeros((resolution, resolution), dtype=int)

    for x, y in points:
        # Map canvas coordinates to grid cell
        col = int((x / canvas_w) * resolution)
        row = int((y / canvas_h) * resolution)

        # Clamp to grid bounds
        col = max(0, min(resolution - 1, col))
        row = max(0, min(resolution - 1, row))

        grid[row][col] = 1

    return grid


def grid_to_text(grid, filled="X", empty="."):
    rows = []
    for row in grid:
        rows.append(" ".join(filled if cell else empty for cell in row))
    return "\n".join(rows)


# ============================================================
# STEP 3: Generate grids at different resolutions
# ============================================================

print("=" * 60)
print("ANTHROPIC LOGO - TEXT GRIDS")
print("Computed from SVG paths. Zero AI.")
print("=" * 60)

for res in [8, 16, 24]:
    grid = make_grid(all_points, canvas_w, canvas_h, res)
    text = grid_to_text(grid)

    filled_count = np.sum(grid)
    total_cells = res * res
    fill_pct = (filled_count / total_cells) * 100

    print(f"\n--- {res}x{res} Grid ---")
    print(f"Filled cells: {filled_count}/{total_cells} ({fill_pct:.1f}%)")
    print()
    print(text)


# ============================================================
# STEP 4: Per-shape grids
# ============================================================
print("\n" + "=" * 60)
print("PER-SHAPE GRIDS (16x16)")
print("=" * 60)

for i, path in enumerate(paths):
    shape_points = []
    for t in np.linspace(0, 1, 500):
        try:
            pt = path.point(t)
            shape_points.append((pt.real, pt.imag))
        except:
            pass
    for seg in path:
        for t in np.linspace(0, 1, 50):
            pt = seg.point(t)
            shape_points.append((pt.real, pt.imag))

    grid = make_grid(shape_points, canvas_w, canvas_h, 16)
    text = grid_to_text(grid)
    print(f"\n--- path-{i+1} ---")
    print(text)


# ============================================================
# STEP 5: Show what goes into the LLM prompt
# ============================================================
print("\n" + "=" * 60)
print("COMBINED PROMPT (what Claude would receive)")
print("=" * 60)

grid_16 = make_grid(all_points, canvas_w, canvas_h, 16)

print("""
SPATIAL DATA:
- path-1: filled shape, center (18.77, 12.00), area 56.88, 4 segments
- path-2: filled shape, center (8.53, 12.00), area 123.61, 11 segments
- Distance: 3.17 units
- Alignment: horizontal (y diff 0.86)
- Overlap: none

VISUAL GRID (16x16):
""")
print(grid_to_text(grid_16))
print("""
Each X is a filled cell. Each . is empty.
The grid represents the SVG rendered on a 16x16 canvas.

USER QUESTION: What does this look like? What are the two shapes?
""")
