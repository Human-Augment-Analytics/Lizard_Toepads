"""
WFLW-specific graph topology and flip pair definitions.

Re-exports make_wflw_edge_index from the common topology registry
and defines WFLW_FLIP_PAIRS for horizontal flip augmentation.

WFLW_FLIP_PAIRS maps each landmark to its mirror counterpart across
the vertical midline of the face. Midline landmarks map to themselves.
Applying this permutation twice returns the original landmark order.

Landmark index reference (98-point WFLW scheme):
  0–32:   Jaw contour (0 = right side, 16 = chin midline, 32 = left side)
  33–41:  Right eyebrow
  42–50:  Left eyebrow
  51–54:  Nose bridge (midline)
  55–59:  Nose base (55/59, 56/58 are pairs; 57 is midline)
  60–67:  Right eye
  68–75:  Left eye
  76–87:  Outer mouth
  88–95:  Inner mouth
  96:     Right pupil
  97:     Left pupil
"""
import sys
from pathlib import Path

# Re-export from common registry
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.graph_topologies import make_wflw_edge_index  # noqa: F401


# Horizontal flip pairs for WFLW 98-point landmarks.
# Each tuple (i, j) means landmark i maps to j and j maps to i on horizontal flip.
# Midline landmarks are listed as (i, i) — they map to themselves.
WFLW_FLIP_PAIRS = [
    # Jaw contour: (0,32), (1,31), ..., (15,17); 16 is chin midline
    (0, 32), (1, 31), (2, 30), (3, 29), (4, 28), (5, 27), (6, 26),
    (7, 25), (8, 24), (9, 23), (10, 22), (11, 21), (12, 20), (13, 19),
    (14, 18), (15, 17), (16, 16),

    # Eyebrows: right (33-41) ↔ left (42-50)
    # Right brow: 33=outer_r, 34,35,36,37=arch upper, 38,39,40,41=arch lower
    # Left brow:  42=inner_l, 43,44,45,46=arch upper, 47,48,49,50=arch lower
    # After flip, outer_r (33) maps to outer_l (46), inner_r (37) maps to inner_l (42)
    (33, 46), (34, 45), (35, 44), (36, 43), (37, 42),
    (38, 50), (39, 49), (40, 48), (41, 47),

    # Nose bridge: all midline
    (51, 51), (52, 52), (53, 53), (54, 54),

    # Nose base: (55,59), (56,58); 57 is midline
    (55, 59), (56, 58), (57, 57),

    # Eyes: right (60-67) ↔ left (68-75)
    # Both eyes traverse counter-clockwise from the inner corner:
    # Right eye: 60=inner_corner_left, 61,62,63=upper, 64=outer_corner_right, 65,66,67=lower
    # Left eye:  68=inner_corner_right, 69,70,71=upper, 72=outer_corner_left, 73,74,75=lower
    # After horizontal flip, right-eye inner (60) maps to left-eye inner (68), etc.
    (60, 68), (61, 69), (62, 70), (63, 71),
    (64, 72), (65, 73), (66, 74), (67, 75),

    # Outer mouth: (76,82), (77,81), (78,80); 79 is midline top
    #              (83,87), (84,86); 85 is midline bottom
    (76, 82), (77, 81), (78, 80), (79, 79),
    (83, 87), (84, 86), (85, 85),

    # Inner mouth: (88,92), (89,91); 90 is midline top
    #              (93,95); 94 is midline bottom
    (88, 92), (89, 91), (90, 90),
    (93, 95), (94, 94),

    # Pupils: right (96) ↔ left (97)
    (96, 97),
]
