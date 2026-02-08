"""Pipeline configuration — controls adaptive behavior."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """Controls which transforms run based on SVG complexity."""

    # Bezier sampling
    samples_per_segment: int = 12
    min_total_samples: int = 500
    max_total_samples: int = 2000
    adaptive_sampling: bool = True

    # ASCII grid resolution
    grid_resolution: int = 32

    # DBSCAN clustering
    dbscan_eps_pct: float = 0.10  # 10% of viewBox diagonal
    dbscan_min_samples: int = 2

    # Complexity thresholds for adaptive pipeline
    simple_threshold: int = 5  # ≤5 elements: skip multi-element transforms
    complex_threshold: int = 15  # ≥15 elements: trigger full enrichment

    # Corner detection (VTracer method)
    corner_threshold: float = 0.5  # radians

    # Symmetry detection
    symmetry_match_eps: float = 2.0  # max distance for centroid matching

    # Morphological close gap for composite silhouette
    morph_close_gap_pct: float = 0.02  # 2% of viewBox diagonal

    # RDP simplification epsilon
    rdp_epsilon_pct: float = 0.01  # 1% of diagonal

    # Shape classification thresholds
    circularity_threshold: float = 0.85
    rectangularity_threshold: float = 0.85
    linear_aspect_threshold: float = 5.0
    convexity_threshold: float = 0.90
