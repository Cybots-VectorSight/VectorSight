"""PipelineContext — the single mutable state object flowing through all transforms.

Per-element results → SubPathData.features
Cross-element results → PipelineContext.* (containment_matrix, clusters, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Polygon


@dataclass
class SubPathData:
    """Data for a single sub-path / element extracted from the SVG."""

    id: str
    # Raw sampled boundary points: Nx2 array of (x, y)
    points: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 2)))
    # Shapely polygon built from points
    polygon: Polygon | None = None
    # Original SVG path segments (from svgpathtools)
    segments: list[Any] = field(default_factory=list)
    # Original SVG attributes
    attributes: dict[str, str] = field(default_factory=dict)
    # Is this sub-path closed?
    closed: bool = False
    # Winding direction: 1 = CCW, -1 = CW, 0 = unknown
    winding: int = 0
    # Z-order index (SVG document order)
    z_order: int = 0
    # Bounding box: (xmin, ymin, xmax, ymax)
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    # All computed features go here (keyed by transform ID or feature name)
    features: dict[str, Any] = field(default_factory=dict)

    @property
    def centroid(self) -> tuple[float, float]:
        if self.polygon is not None and not self.polygon.is_empty:
            c = self.polygon.centroid
            return (c.x, c.y)
        if len(self.points) > 0:
            return (float(np.mean(self.points[:, 0])), float(np.mean(self.points[:, 1])))
        return (0.0, 0.0)

    @property
    def area(self) -> float:
        if self.polygon is not None and not self.polygon.is_empty:
            return float(self.polygon.area)
        return 0.0

    @property
    def perimeter(self) -> float:
        if self.polygon is not None and not self.polygon.is_empty:
            return float(self.polygon.length)
        return 0.0

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


@dataclass
class PipelineContext:
    """Shared state flowing through the entire pipeline."""

    # Raw SVG code
    svg_raw: str = ""
    # Canvas dimensions from viewBox
    canvas_width: float = 24.0
    canvas_height: float = 24.0
    # Parsed sub-paths / elements
    subpaths: list[SubPathData] = field(default_factory=list)
    # Is this SVG stroke-based or fill-based?
    is_stroke_based: bool = False
    # Fill rule
    fill_rule: str = "nonzero"

    # --- Cross-element computed state (populated by Layer 3) ---
    # Containment matrix: containment_matrix[i][j] = True means i contains j
    containment_matrix: NDArray[np.bool_] | None = None
    # Distance matrix: pairwise minimum distances
    distance_matrix: NDArray[np.float64] | None = None
    # DBSCAN cluster labels per subpath
    cluster_labels: NDArray[np.int64] | None = None
    # Connected component labels
    component_labels: list[int] = field(default_factory=list)
    # Visual stacking tree: list of (parent_idx, child_idx) pairs
    stacking_tree: list[tuple[int, int]] = field(default_factory=list)
    # Symmetry info
    symmetry_axis: str | None = None
    symmetry_score: float = 0.0
    symmetry_pairs: list[tuple[int, int]] = field(default_factory=list)

    # --- Composite / silhouette ---
    # Composite silhouette as a rasterized grid
    composite_grid: NDArray[np.int8] | None = None
    # Composite silhouette polygon
    composite_silhouette: Polygon | None = None
    # Negative space regions
    negative_space_regions: list[Polygon] = field(default_factory=list)

    # --- Layer 2 visualizations ---
    ascii_grid_positive: str = ""
    ascii_grid_negative: str = ""
    ascii_grid_halfblock: str = ""
    region_map: dict[str, Any] = field(default_factory=dict)

    # --- Enrichment output ---
    enrichment_text: str = ""

    # --- Pipeline metadata ---
    completed_transforms: set[str] = field(default_factory=set)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def num_elements(self) -> int:
        return len(self.subpaths)

    @property
    def viewbox_diagonal(self) -> float:
        return float(np.sqrt(self.canvas_width**2 + self.canvas_height**2))

    def get_subpath(self, element_id: str) -> SubPathData | None:
        for sp in self.subpaths:
            if sp.id == element_id:
                return sp
        return None
