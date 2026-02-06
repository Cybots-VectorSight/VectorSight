"""
VectorSight proof of concept
Shows what geometry libraries extract from raw SVG — no AI involved.
"""
from svgpathtools import svg2paths2
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import numpy as np
import json

# ============================================================
# STEP 1: Parse SVG (svgpathtools — XML parser, not AI)
# ============================================================
paths, attributes, svg_attributes = svg2paths2('anthropic.svg')

print("=" * 60)
print("STEP 1: PARSE SVG (svgpathtools)")
print("=" * 60)

for i, (path, attr) in enumerate(zip(paths, attributes)):
    print(f"\n--- Shape {i+1} ---")
    print(f"  Raw 'd' attribute: {attr.get('d', 'N/A')[:80]}...")
    print(f"  Fill: {attr.get('fill', 'none')}")
    print(f"  Number of segments: {len(path)}")

    # Bounding box — pure math
    xmin, xmax, ymin, ymax = path.bbox()
    print(f"  Bounding box: x=[{xmin:.2f}, {xmax:.2f}] y=[{ymin:.2f}, {ymax:.2f}]")
    print(f"  Width: {xmax - xmin:.2f}")
    print(f"  Height: {ymax - ymin:.2f}")

    # Center point — pure math
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    print(f"  Center: ({cx:.2f}, {cy:.2f})")

    # Path length — pure math
    print(f"  Path length: {path.length():.2f}")

    # Extract all points along the path
    points = []
    for seg in path:
        points.append((seg.start.real, seg.start.imag))
        points.append((seg.end.real, seg.end.imag))
    print(f"  Points extracted: {len(points)}")


# ============================================================
# STEP 2: Build Shapely geometries (computational geometry, not AI)
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: BUILD GEOMETRY (Shapely)")
print("=" * 60)

shapes = []
for i, path in enumerate(paths):
    # Sample points along the path to create polygon
    num_samples = 100
    points = []
    for t in np.linspace(0, 1, num_samples):
        try:
            pt = path.point(t)
            points.append((pt.real, pt.imag))
        except:
            pass

    if len(points) >= 3:
        try:
            poly = Polygon(points)
            if not poly.is_valid:
                poly = poly.buffer(0)  # Fix invalid geometry
            shapes.append(poly)
            print(f"\n--- Shape {i+1} as Polygon ---")
            print(f"  Valid: {poly.is_valid}")
            print(f"  Area: {poly.area:.2f}")
            centroid = poly.centroid
            print(f"  Centroid: ({centroid.x:.2f}, {centroid.y:.2f})")
        except Exception as e:
            print(f"\n--- Shape {i+1}: Could not create polygon: {e}")


# ============================================================
# STEP 3: Compute relationships (geometry math, not AI)
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: COMPUTE RELATIONSHIPS (Shapely + numpy)")
print("=" * 60)

if len(shapes) >= 2:
    s1, s2 = shapes[0], shapes[1]

    print(f"\n  Distance between shapes: {s1.distance(s2):.2f}")
    print(f"  Do they overlap: {s1.intersects(s2)}")
    print(f"  Does shape 1 contain shape 2: {s1.contains(s2)}")
    print(f"  Does shape 2 contain shape 1: {s2.contains(s1)}")

    # Bounding box gap
    b1 = s1.bounds  # (minx, miny, maxx, maxy)
    b2 = s2.bounds

    print(f"\n  Shape 1 bounds: x=[{b1[0]:.2f}, {b1[2]:.2f}] y=[{b1[1]:.2f}, {b1[3]:.2f}]")
    print(f"  Shape 2 bounds: x=[{b2[0]:.2f}, {b2[2]:.2f}] y=[{b2[1]:.2f}, {b2[3]:.2f}]")

    # Horizontal gap
    gap = b1[0] - b2[2]  # shape1 left edge - shape2 right edge
    if gap < 0:
        gap = b2[0] - b1[2]  # try other direction
    print(f"  Horizontal gap: {abs(gap):.2f}")

    # Vertical alignment
    c1 = s1.centroid
    c2 = s2.centroid
    y_diff = abs(c1.y - c2.y)
    print(f"  Vertical alignment (y difference): {y_diff:.2f}")
    if y_diff < 1:
        print(f"  >> Shapes are vertically aligned (same row)")

    # Symmetry check
    canvas_center_x = 12  # 24/2
    d1 = abs(c1.x - canvas_center_x)
    d2 = abs(c2.x - canvas_center_x)
    print(f"\n  Shape 1 distance from center: {d1:.2f}")
    print(f"  Shape 2 distance from center: {d2:.2f}")

    # Relative sizes
    a1 = s1.area
    a2 = s2.area
    print(f"\n  Shape 1 area: {a1:.2f}")
    print(f"  Shape 2 area: {a2:.2f}")
    ratio = max(a1, a2) / min(a1, a2) if min(a1, a2) > 0 else 0
    print(f"  Size ratio: {ratio:.2f}x")


# ============================================================
# STEP 4: Generate Spatial JSON (our output format)
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: SPATIAL JSON OUTPUT")
print("=" * 60)

output = {
    "source": "anthropic-logo.svg",
    "canvas": {"width": 24, "height": 24},
    "shapes": [],
    "relationships": []
}

for i, (path, attr) in enumerate(zip(paths, attributes)):
    xmin, xmax, ymin, ymax = path.bbox()
    shape_data = {
        "id": f"path-{i+1}",
        "type": "filled-path",
        "bounds": {
            "x_min": round(xmin, 2),
            "x_max": round(xmax, 2),
            "y_min": round(ymin, 2),
            "y_max": round(ymax, 2)
        },
        "center": {
            "x": round((xmin + xmax) / 2, 2),
            "y": round((ymin + ymax) / 2, 2)
        },
        "width": round(xmax - xmin, 2),
        "height": round(ymax - ymin, 2),
        "fill": attr.get("fill", "none"),
        "segments": len(path)
    }
    if i < len(shapes):
        shape_data["area"] = round(shapes[i].area, 2)
    output["shapes"].append(shape_data)

if len(shapes) >= 2:
    output["relationships"] = [
        {
            "type": "adjacent",
            "shapes": ["path-1", "path-2"],
            "gap": round(abs(gap), 2),
            "direction": "horizontal"
        },
        {
            "type": "alignment",
            "shapes": ["path-1", "path-2"],
            "axis": "horizontal",
            "y_difference": round(y_diff, 2)
        }
    ]

print(json.dumps(output, indent=2))

print("\n" + "=" * 60)
print("DONE — All of this was math. Zero AI.")
print("=" * 60)
