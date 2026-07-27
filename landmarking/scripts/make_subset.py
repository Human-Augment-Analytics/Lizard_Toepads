"""CLI utility for generating landmark subset indices.

Usage:
    # By step size
    python -m landmarking.scripts.make_subset --step 4

    # By target fraction
    python -m landmarking.scripts.make_subset --fraction 0.25

Prints a JSON array of selected landmark indices and the count.
"""

import argparse
import json
import sys

from ..common.sparsity import make_subset_indices


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate landmark subset indices for sparsity experiments."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--step", type=int,
        help="Step size for uniform sampling (integer in [1, 97])."
    )
    group.add_argument(
        "--fraction", type=float,
        help="Target fraction of landmarks to keep (float in (0, 1))."
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.fraction is not None:
        if args.fraction <= 0.0 or args.fraction >= 1.0:
            print(f"Error: --fraction must be in (0, 1), got {args.fraction}", file=sys.stderr)
            sys.exit(1)
        step = round(1.0 / args.fraction)
    else:
        step = args.step

    if step < 1 or step > 97:
        print(f"Error: computed step must be in [1, 97], got {step}", file=sys.stderr)
        sys.exit(1)

    indices = make_subset_indices(total=98, step=step)
    print(json.dumps(indices))
    print(f"Count: {len(indices)}")


if __name__ == "__main__":
    main()
