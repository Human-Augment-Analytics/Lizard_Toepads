#!/usr/bin/env python3
"""Verify a package's native dlib example using its explicit BGR contract."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    args = parser.parse_args()
    import cv2
    import dlib
    import numpy as np
    expected = json.loads((args.package / 'example/expected.json').read_text())
    if expected['color_order'] != 'BGR':
        raise ValueError('Unsupported example color contract')
    image = cv2.imread(str(args.package / expected['input']), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Cannot read example input')
    predictor = dlib.shape_predictor(str(args.package / expected['predictor']))
    shape = predictor(image, dlib.rectangle(*expected['rectangle']))
    actual = np.asarray([[p.x, p.y] for p in shape.parts()], dtype=float)
    target = np.asarray(expected['points'], dtype=float)
    if actual.shape != target.shape:
        raise ValueError('Example landmark count differs')
    error = float(np.max(np.abs(actual - target)))
    if error > expected['absolute_tolerance_px']:
        raise ValueError(f'Example inference mismatch: {error} pixels')
    print(json.dumps({'verified': True, 'landmarks': len(actual), 'max_abs_error_px': error}))


if __name__ == '__main__':
    main()
