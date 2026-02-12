"""Shared spatial constants for Layer 1-3 relationship transforms.

All distance fractions are based on Weber's Law for spatial perception:
Just-Noticeable Difference (JND) for position discrimination is 1-3%
of the viewing field. In SVG analysis, the viewBox diagonal serves as
the reference field size.
"""

# Weber's Law: 2% JND for spatial position discrimination.
# Standard in GIS software (ESRI "near" default tolerance).
SPATIAL_JND_FRACTION = 0.02

# 1.5x JND for shared-center detection (slightly relaxed because
# center matching involves two independent position estimates).
SHARED_CENTER_FRACTION = SPATIAL_JND_FRACTION * 1.5  # = 0.03

# 2.5x JND for symmetry mirror matching (reflects across axis,
# so positional error doubles: eps_point + eps_reflection).
SYMMETRY_MIRROR_FRACTION = SPATIAL_JND_FRACTION * 2.5  # = 0.05

# Sub-JND: half the threshold for sub-pixel edge detection (tiling).
TILING_EDGE_FRACTION = SPATIAL_JND_FRACTION / 2  # = 0.01
