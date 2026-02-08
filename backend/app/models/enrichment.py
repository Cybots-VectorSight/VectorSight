"""Core enrichment data model — the structured output of the pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ElementSummary(BaseModel):
    id: str
    shape_class: str = "organic"
    area: float = 0.0
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    centroid: tuple[float, float] = (0.0, 0.0)
    circularity: float = 0.0
    convexity: float = 0.0
    aspect_ratio: float = 1.0
    size_tier: str = "MEDIUM"  # LARGE, MEDIUM, SMALL


class ContainmentRelation(BaseModel):
    parent: str
    children: list[str] = Field(default_factory=list)


class SymmetryInfo(BaseModel):
    axis_type: str = "none"  # vertical, horizontal, diagonal, rotational
    score: float = 0.0
    pairs: list[tuple[str, str]] = Field(default_factory=list)
    on_axis: list[str] = Field(default_factory=list)
    rotational_order: int | None = None


class ClusterInfo(BaseModel):
    cluster_id: int
    members: list[str] = Field(default_factory=list)
    centroid: tuple[float, float] = (0.0, 0.0)
    description: str = ""


class ComponentInfo(BaseModel):
    component_id: int
    members: list[str] = Field(default_factory=list)
    description: str = ""


class StackingNode(BaseModel):
    element_id: str
    shape_class: str = "organic"
    size_tier: str = "MEDIUM"
    z_order: int = 0
    position: str = ""
    children: list[StackingNode] = Field(default_factory=list)


class SilhouetteInfo(BaseModel):
    shape_class: str = "organic"
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    aspect_ratio: float = 1.0
    circularity: float = 0.0
    convexity: float = 0.0


class EnrichmentOutput(BaseModel):
    """Complete enrichment output from the pipeline."""

    # Metadata
    source: str = ""
    canvas: tuple[float, float] = (24.0, 24.0)
    element_count: int = 0
    subpath_count: int = 0
    is_stroke_based: bool = False

    # Per-element summaries
    elements: list[ElementSummary] = Field(default_factory=list)

    # Relationships
    containment: list[ContainmentRelation] = Field(default_factory=list)
    clusters: list[ClusterInfo] = Field(default_factory=list)
    components: list[ComponentInfo] = Field(default_factory=list)

    # Symmetry
    symmetry: SymmetryInfo = Field(default_factory=SymmetryInfo)

    # Size tiers
    size_tiers: dict[str, list[str]] = Field(default_factory=dict)

    # Shape similarity groups
    similarity_groups: dict[str, list[str]] = Field(default_factory=dict)

    # Stacking tree
    stacking_tree: list[StackingNode] = Field(default_factory=list)

    # Silhouette
    silhouette: SilhouetteInfo | None = None

    # Visualizations
    ascii_grid_positive: str = ""
    ascii_grid_negative: str = ""

    # Enrichment text (formatted for LLM injection)
    enrichment_text: str = ""
