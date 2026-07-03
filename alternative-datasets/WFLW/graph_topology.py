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
    (33, 42), (34, 43), (35, 44), (36, 45), (37, 46),
    (38, 47), (39, 48), (40, 49), (41, 50),

    # Nose bridge: all midline
    (51, 51), (52, 52), (53, 53), (54, 54),

    # Nose base: (55,59), (56,58); 57 is midline
    (55, 59), (56, 58), (57, 57),

    # Eyes: right (60-67) ↔ left (68-75)
    # Right eye order: 60=outer_r, 61, 62, 63, 64=center_r, 65, 66, 67=inner_r
    # Left eye order:  68=inner_l, 69, 70, 71, 72=center_l, 73, 74, 75=outer_l
    (60, 68), (61, 75), (62, 74), (63, 73),
    (64, 72), (65, 71), (66, 70), (67, 69),

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
