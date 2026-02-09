# VectorSight Visual Knowledge

> This document is read and edited by the self-reflection system.
> Everything below is learned from actual SVG analysis sessions.
> Add new learnings, reorganize, or correct entries as experience grows.

## Learned

- Stroke-based circles with perfect metrics (1.00 circularity/convexity/aspect) are a reliable UI icon marker. The low fill percentage in such cases reflects rendering technique (stroke vs. fill), not compositional sparsity. Perfect symmetry across all three dimensions, combined with single-element structure and centered placement, is a strong indicator of a geometric UI element rather than an illustration or logo. This pattern can help distinguish simple icons from other SVG types early.

- Stroke-based geometric primitives (circles, squares) with perfect metric symmetry (1.00 circularity, convexity, aspect ratio) across all three dimensions, combined with single-element structure, centered placement, and low fill percentage are highly reliable indicators of functional UI elements. The corner count in such cases reflects Bézier control point density (typically 12 for a 4-curve circle approximation), not visual angularity. This pattern should be recognized as a distinct category—'functional geometric UI element'—separate from logos, illustrations, and ornamental designs. The absence of fill is a rendering technique marker, not a compositional property.

- Confirmed and reinforced: stroke-based circles with perfect metrics (1.00 across circularity, convexity, aspect ratio), single-element structure, centered placement, and 8% fill percentage are a reliable marker for functional geometric UI icons. The grid corner count of 12 for a smooth circle is a standard artifact of cubic Bézier approximation (4 curves × 3 control points per curve). This should be recognized as a distinct visual category: 'functional geometric UI icon'—clean, symmetric, mathematically idealized, and minimally rendered. No new learning; this analysis reinforces the prior knowledge pattern.

- Reinforced: The combination of perfect circularity (1.00), convexity (1.00), aspect ratio (1.00), and 12 corners, paired with centered placement and stroke-based rendering with 8% fill, is a highly reliable signature of a minimal geometric UI icon. This pattern should be recognized immediately as belonging to the 'canonical UI element' category. No new learning; this confirms the prior pattern that was already documented.

- Reinforced: The combination of perfect circularity (1.00), convexity (1.00), aspect ratio (1.00), 12 corners, single-element structure, centered placement, and stroke-based rendering with 8% fill is a highly reliable signature of a canonical UI icon circle. This pattern appears across checkbox unchecked states, radio buttons, selection rings, and decorative circular borders. The 12-corner count is a standard Bézier approximation artifact and should not be misinterpreted as visual angularity. No new learning; this reinforces the established prior knowledge that stroke-based geometric primitives with perfect metrics are functional UI elements, distinct from illustrations or logos.

- No new learning. This SVG confirms the established prior knowledge pattern: perfect circularity (1.00), convexity (1.00), aspect ratio (1.00), 12 Bézier corners, single-element structure, centered placement, and stroke-only rendering with 8% fill are a highly reliable signature of a canonical geometric UI icon. This pattern should be recognized immediately as a functional UI element distinct from illustrations or logos.
