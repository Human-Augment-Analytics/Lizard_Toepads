#!/usr/bin/env python3
"""Build two detector datasets from an existing prepared ID-card dataset."""
import argparse
import csv
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

from pipeline import rotate_eval_image

SEED = 42
TRAIN_ORIGINALS = 612
ROTATED_ORIGINALS = 527
ANGLES = (-15, 15)
FULL_ROTATION_DEGREES = 180.0
JPEG_QUALITY = 95
DEFAULT_TRAIN_CONFIG = Path(__file__).resolve().parents[2] / 'configs/id_card_affine15.yaml'
DEFAULT_SOURCE = '~/scratch/id_cards/v4_affine15'
DEFAULT_OUTPUT = '~/scratch/id_cards/v4_rotation_ab'


def load_inputs(source):
    if not (source / 'READY').is_file():
        raise ValueError(f'No prepared dataset at {source}')
    with (source / 'splits.csv').open(newline='') as file:
        rows = list(csv.DictReader(file))
    if len({r['image_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate image IDs in source manifest')
    groups = {}
    selected = [r for r in rows if r['split'] in ('train', 'val', 'test')]
    for row in selected:
        prior = groups.setdefault(row['group_id'], row['split'])
        if prior != row['split']:
            raise ValueError(f"Group crosses splits: {row['group_id']}")
        for directory, extension in (('images', 'jpg'), ('labels', 'txt')):
            path = source / directory / row['split'] / f"{row['image_id']}.{extension}"
            if not path.is_file():
                raise FileNotFoundError(path)
    return selected, yaml.safe_load((source / 'config.yaml').read_text())


def select_rows(rows):
    train = sorted((r for r in rows if r['split'] == 'train'), key=lambda r: r['image_id'])
    counts = Counter(r['split'] for r in rows)
    if len(train) < TRAIN_ORIGINALS or not counts['val'] or not counts['test']:
        raise ValueError(f'Need at least {TRAIN_ORIGINALS} train rows and nonempty val/test: {dict(counts)}')
    rng = random.Random(SEED)
    originals = rng.sample(train, TRAIN_ORIGINALS)
    rotate = rng.sample(originals, ROTATED_ORIGINALS)
    return originals + [r for r in rows if r['split'] != 'train'], rotate


def write_rotations(source, target, row):
    stem = row['image_id']
    image_path = source / 'images/train' / f'{stem}.jpg'
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f'Cannot decode {image_path}')
    label = (source / 'labels/train' / f'{stem}.txt').read_text().split()
    if len(label) != 9 or label[0] != '0':
        raise ValueError(f'Expected one ID card box for {stem}')
    points = np.array(label[1:], dtype=float).reshape(4, 2)
    if not np.isfinite(points).all() or np.any((points < 0) | (points > 1)):
        raise ValueError(f'Invalid normalized corners for {stem}')
    points *= np.array(image.shape[:2][::-1])
    for angle in ANGLES:
        pixels, corners, size = rotate_eval_image(image, points, angle)
        variant = f'{stem}__rot_{angle:+d}'
        image_out = target / 'images/train' / f'{variant}.jpg'
        label_out = target / 'labels/train' / f'{variant}.txt'
        if image_out.exists() or label_out.exists():
            raise FileExistsError(variant)
        if not cv2.imwrite(str(image_out), pixels, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
            raise OSError(f'Cannot write {image_out}')
        normalized = corners / size
        label_out.write_text('0 ' + ' '.join(f'{v:.8f}' for v in normalized.ravel()) + '\n')
        yield dict(row, image_id=variant, source_image_id=stem, angle_deg=angle)


def build_arm(source, target, rows, selected, config):
    target.mkdir(parents=True)
    for directory, extension in (('images', 'jpg'), ('labels', 'txt')):
        for split in ('train', 'val', 'test'):
            (target / directory / split).mkdir(parents=True)
        for row in rows:
            path = Path(directory) / row['split'] / f"{row['image_id']}.{extension}"
            (target / path).symlink_to(source / path)
    variants = [variant for row in selected for variant in write_rotations(source, target, row)]
    with (target / 'splits.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (target / 'rotated_training.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) + ['source_image_id', 'angle_deg'])
        writer.writeheader()
        writer.writerows(variants)
    dataset = yaml.safe_load((source / 'dataset.yaml').read_text())
    dataset.update(path=str(target), train='images/train', val='images/val', test='images/test')
    (target / 'dataset.yaml').write_text(yaml.safe_dump(dataset, sort_keys=False))
    cfg = dict(config, output=str(target), manifest=str(target / 'splits.csv'),
               quarter_turn_fraction=0.0)
    cfg['train'] = dict(config['train'])
    expected = TRAIN_ORIGINALS + 2 * len(selected)
    for d, ext in (('images', 'jpg'), ('labels', 'txt')):
        if len(list((target / d / 'train').glob(f'*.{ext}'))) != expected:
            raise ValueError(f'Unexpected training file count in {target / d}')
    (target / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    (target / 'READY').write_text('Prepared for detector training and evaluation.\n')
    print(f'{target.name}: {expected} training files; config: {target / "config.yaml"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=DEFAULT_SOURCE)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--train-config', type=Path, default=DEFAULT_TRAIN_CONFIG,
                        help='Repository YAML whose train settings apply to both arms')
    parser.add_argument('--b-degrees', type=float, default=FULL_ROTATION_DEGREES,
                        help='Maximum random rotation in degrees for B (default: 180)')
    parser.add_argument('--variant', choices=('A', 'B', 'both'), default='both')
    args = parser.parse_args()
    if not 0 <= args.b_degrees <= FULL_ROTATION_DEGREES:
        parser.error('--b-degrees must be between 0 and 180')
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if source == output or source in output.parents:
        parser.error('Use a separate output folder outside the source dataset')
    arms = ('A', 'B') if args.variant == 'both' else (args.variant,)
    for arm in arms:
        if (output / arm).exists():
            parser.error(f'Refusing to overwrite {output / arm}')
    rows, config = load_inputs(source)
    training = yaml.safe_load(args.train_config.expanduser().read_text())['train']
    originals, rotated = select_rows(rows)
    for arm in arms:
        arm_training = dict(training)
        if arm == 'B':
            arm_training['degrees'] = args.b_degrees
        arm_config = dict(config, train=arm_training)
        build_arm(source, output / arm, originals, rotated if arm == 'B' else [], arm_config)


if __name__ == '__main__':
    main()
