#!/usr/bin/env python3
"""One-class ID-card OBB preparation and detector-independent field OCR baseline."""
import argparse
from collections import Counter
import csv
import hashlib
from importlib.metadata import distributions
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unicodedata
import cv2
import numpy as np
from PIL import Image, ImageDraw
import yaml
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
Image.MAX_IMAGE_PIXELS = None  # Known, very large research scans; decode with draft.


def resolve(value):
    p = Path(value).expanduser()
    return (ROOT / p).resolve() if not p.is_absolute() else p.resolve()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def scan(path, max_side):
    """Read source dimensions before downsampling; never apply EXIF rotation silently."""
    if max_side < 32:
        raise ValueError('scan_max_side must be at least 32')
    with Image.open(path) as im:
        size = im.size
        im.draft('RGB', (max_side, max_side))
        im = im.convert('RGB')
        im.thumbnail((max_side, max_side))
        return im.copy(), size


def corners(record, size):
    """Stored corners already include rotation. Apply inverse affine exactly once."""
    pts = np.asarray(record['obb_corners'], dtype=np.float64)
    inverse = np.asarray(record['transform_inverse'], dtype=np.float64)
    forward = np.asarray(record['transform'], dtype=np.float64)
    if pts.shape != (4, 2) or inverse.shape != (2, 3) or forward.shape != (2, 3):
        raise ValueError('Invalid annotation matrix/corner shape')
    if not all(np.isfinite(v).all() for v in (pts, inverse, forward)):
        raise ValueError('Non-finite annotation geometry')
    a = np.vstack([forward, [0, 0, 1]])
    b = np.vstack([inverse, [0, 0, 1]])
    if not np.allclose(a @ b, np.eye(3), atol=1e-5):
        raise ValueError('Forward and inverse transforms disagree')
    original = pts @ inverse[:, :2].T + inverse[:, 2]
    w, h = size
    # Allow only subpixel roundoff at a boundary; do not silently clip bad boxes.
    if (original < -1e-3).any() or (original > np.array([w, h]) + 1e-3).any():
        raise ValueError('Box lies outside the source image; check image pairing')
    normalized = np.clip(original / np.array([w, h]), 0, 1)
    contour = normalized.astype(np.float32)
    if not cv2.isContourConvex(contour) or abs(cv2.contourArea(contour)) < 1e-8:
        raise ValueError('Card box is degenerate or non-convex')
    return original


def inventory(cfg):
    directory, annotations = resolve(cfg['images']), resolve(cfg['annotations'])
    if not directory.is_dir():
        raise FileNotFoundError(f'Source scans missing: {directory}. Locate matching scans; do not use TPS images by assumption.')
    images = {}
    for p in sorted(directory.iterdir()):
        if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            if p.stem in images:
                raise ValueError(f'Duplicate image ID: {p.stem}')
            images[p.stem] = p
    if not annotations.is_dir():
        raise FileNotFoundError(f'Annotation folder missing: {annotations}')
    records = {}
    missing = []
    for p in sorted(annotations.glob('*.json')):
        rec = json.loads(p.read_text())
        image_id = str(rec['image_id'])
        if p.stem != image_id or image_id in records:
            raise ValueError(f'Annotation ID/filename mismatch or duplicate: {p}')
        if image_id not in images:
            missing.append(image_id)
            continue
        with Image.open(images[image_id]) as im:
            corners(rec, im.size)
        records[image_id] = rec
    if missing:
        raise ValueError('Annotations lack matching images: ' + ', '.join(missing[:20]))
    if not records:
        raise ValueError('No matched annotations')
    return images, records


def manifest(cfg, images, records):
    path = resolve(cfg['manifest'])
    if not path.is_file():
        raise FileNotFoundError(f'Create reviewed split manifest first: {path}')
    rows, seen, group_splits = [], set(), {}
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            row = {k: str(v or '').strip() for k, v in row.items()}
            image_id, group, split = row['image_id'], row['group_id'], row['split']
            if image_id in seen or image_id not in records or image_id not in images:
                raise ValueError(f'Duplicate or unmatched manifest image: {image_id}')
            if not group or split not in ('train', 'val', 'test', 'field'):
                raise ValueError(f'Invalid group/split for {image_id}')
            if group in group_splits and group_splits[group] != split:
                raise ValueError(f'Group {group} leaks across splits')
            turns = int(row.get('rotation_quarters', '0') or '0')
            if turns not in (0, 1, 2, 3):
                raise ValueError('rotation_quarters must be 0, 1, 2, or 3')
            row['rotation_quarters'] = turns
            seen.add(image_id)
            group_splits[group] = split
            rows.append(row)
    counts = Counter(r['split'] for r in rows)
    if not 50 <= counts['train'] <= 100 or not 20 <= counts['test'] <= 50 or counts['val'] < 1:
        raise ValueError(f'Need 50–100 train, 20–50 test, and a separate nonempty validation set; got {dict(counts)}')
    return rows


def rectify(image, points, card_size, turns=0):
    w, h = map(int, card_size)
    if min(w, h) < 2:
        raise ValueError('Invalid rectified card dimensions')
    pts = np.roll(np.asarray(points, np.float32), -turns, axis=0)
    dest = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], np.float32)
    transform = cv2.getPerspectiveTransform(pts, dest)
    rgb = cv2.warpPerspective(np.asarray(image), transform, (w, h))
    return Image.fromarray(rgb)


def fields(cfg):
    definitions = json.loads(resolve(cfg['template']).read_text())['fields']
    if not definitions:
        raise ValueError('Field template is empty. Define and review normalized rectangles first.')
    for name, box in definitions.items():
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
            raise ValueError(f'Invalid field name: {name}')
        if len(box) != 4 or not all(np.isfinite(x) and 0 <= x <= 1 for x in box):
            raise ValueError(f'Invalid normalized field rectangle: {name}')
        x1, y1, x2, y2 = box
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f'Empty field: {name}')
    return definitions


def prepare(cfg):
    images, records = inventory(cfg)
    rows = manifest(cfg, images, records)
    out = resolve(cfg['output'])
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}; choose a new output path')
    for source in (resolve(cfg['images']), resolve(cfg['annotations'])):
        if out == source or source in out.parents or out in source.parents:
            raise ValueError('Output must not overlap source data')
    out.mkdir(parents=True)
    for row in rows:
        image_id, split = row['image_id'], row['split']
        im, size = scan(images[image_id], int(cfg['scan_max_side']))
        original = corners(records[image_id], size)
        pts = original * np.array(im.size) / np.array(size)
        # Persist image at a bounded resolution; normalized labels stay aligned.
        image_path = out / 'images' / split / f'{image_id}.jpg'
        label_path = out / 'labels' / split / f'{image_id}.txt'
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(image_path, quality=95)
        normalized = original / np.array(size)
        label_path.write_text('0 ' + ' '.join(f'{x:.8f}' for x in normalized.ravel()) + '\n')
        annotated = im.copy()
        draw = ImageDraw.Draw(annotated)
        draw.line([tuple(p) for p in pts] + [tuple(pts[0])], fill='red', width=4)
        review = out / 'review' / split
        review.mkdir(parents=True, exist_ok=True)
        annotated.save(review / f'{image_id}_box.jpg')
        card = rectify(im, pts, cfg['card_size'], row['rotation_quarters'])
        cards = out / 'cards' / split
        cards.mkdir(parents=True, exist_ok=True)
        card.save(cards / f'{image_id}.png')
    data = {'path': str(out), 'train': 'images/train', 'val': 'images/val',
            'test': 'images/test', 'names': {0: 'id_card'}, 'nc': 1}
    (out / 'dataset.yaml').write_text(yaml.safe_dump(data, sort_keys=False))
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    shutil.copyfile(resolve(cfg['manifest']), out / 'splits.csv')
    dump(out / 'records.json', records)
    dump(out / 'summary.json', {'split_counts': dict(Counter(r['split'] for r in rows)),
                               'source_images': {r['image_id']: str(images[r['image_id']]) for r in rows},
                               'source_file_stats': {r['image_id']: {'bytes': images[r['image_id']].stat().st_size,
                                   'mtime_ns': images[r['image_id']].stat().st_mtime_ns} for r in rows},
                               'manifest_sha256': hashlib.sha256(resolve(cfg['manifest']).read_bytes()).hexdigest()})
    (out / 'READY').write_text('Review boxes AND upright rectified cards before training or OCR.\n')
    print(f'Prepared dataset: {out}')


def prepared(cfg):
    out = resolve(cfg['output'])
    if not (out / 'READY').is_file():
        raise ValueError('Run prepare successfully first')
    saved = yaml.safe_load((out / 'config.yaml').read_text())
    for key in ('images', 'annotations', 'manifest', 'card_size', 'scan_max_side'):
        if saved[key] != cfg[key]:
            raise ValueError(f'Preparation setting changed ({key}); prepare a new output')
    return out


def model_action(args, cfg):
    out = prepared(cfg)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable; use an allocated GPU session')
    options = dict(cfg['train'])
    model_name = options.pop('model')
    if args.command == 'test':
        if not args.weights or not resolve(args.weights).is_file():
            raise ValueError('test requires --weights pointing to an existing best.pt')
        model = YOLO(str(resolve(args.weights)), task='obb')
        if model.task != 'obb' or model.names != {0: 'id_card'}:
            raise ValueError('Expected a trained one-class id_card OBB checkpoint')
        results = model.val(data=str(out/'dataset.yaml'), split='test', device=options['device'],
                            imgsz=options['imgsz'], batch=options['batch'],
                            project=str(out/'runs'), name='yolo_test', plots=True)
        dump(Path(results.save_dir)/'metrics.json', results.results_dict)
        return
    options.update(data=str(out/'dataset.yaml'), project=str(out/'runs'),
                   name=args.command, exist_ok=False)
    if args.command == 'smoke':
        options.update(epochs=2, batch=2, imgsz=640, workers=2)
    if args.resume:
        if args.command != 'train' or not resolve(args.resume).is_file():
            raise ValueError('--resume requires train and an existing last.pt')
        model = YOLO(str(resolve(args.resume)), task='obb')
        if model.task != 'obb' or model.names != {0: 'id_card'}:
            raise ValueError('Resume requires a trained one-class id_card OBB checkpoint')
        if resolve(model.ckpt['train_args']['data']) != out/'dataset.yaml':
            raise ValueError('Checkpoint belongs to a different dataset')
        results = model.train(resume=True, device=options['device'])
    else:
        model = YOLO(str(resolve(model_name)) if '/' in model_name else model_name, task='obb')
        if model.task != 'obb':
            raise ValueError('Expected OBB pretrained model')
        results = model.train(**options)
    run = Path(results.save_dir)
    (run/'packages.txt').write_text('\n'.join(sorted(f'{d.metadata["Name"]}=={d.version}' for d in distributions()))+'\n')
    dump(run/'pipeline_config.json', cfg)


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def distance(a, b):
    prev = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        curr = [i]
        for j, y in enumerate(b, 1):
            curr.append(min(curr[-1]+1, prev[j]+1, prev[j-1]+(x != y)))
        prev = curr
    return prev[-1]


def baseline(args, cfg):
    definitions = fields(cfg)
    manual_ids = getattr(args, 'image_id', None)
    if manual_ids:
        out = resolve(cfg['output'])
        target = out.parent/(out.name + '_manual_baseline')
        samples = []
        for image_id in manual_ids:
            if not image_id.isdigit():
                raise ValueError('Manual image IDs must be numeric, preserving leading zeros')
            candidates = [p for p in resolve(cfg['images']).iterdir()
                          if p.is_file() and p.stem == image_id and p.suffix.lower() in ('.jpg','.jpeg','.png')]
            if len(candidates) != 1:
                raise ValueError(f'Need exactly one matching source image for {image_id}')
            rec = json.loads((resolve(cfg['annotations'])/f'{image_id}.json').read_text())
            if str(rec['image_id']) != image_id:
                raise ValueError('Annotation image ID does not match requested scan')
            samples.append((image_id, candidates[0], rec))
        def card_images():
            for image_id, image_path, rec in samples:
                im, size = scan(image_path, int(cfg['scan_max_side']))
                pts = corners(rec, size)*np.array(im.size)/np.array(size)
                yield image_id, rectify(im, pts, cfg['card_size'], args.rotation_quarters)
    else:
        out = prepared(cfg)
        cards = out/'cards'/args.split
        if not cards.is_dir():
            raise ValueError(f'No annotated cards in split {args.split}')
        target = out/'baseline'/args.split
        def card_images():
            for path in sorted(cards.glob('*.png')):
                with Image.open(path) as im:
                    yield path.stem, im.convert('RGB')
    if target.exists():
        raise FileExistsError('Baseline output exists; preserve it or move it before repeating')
    if args.ocr and not shutil.which(cfg['ocr']['executable']):
        raise ValueError('Tesseract executable unavailable; run crops without --ocr first')
    target.mkdir(parents=True)
    shutil.copyfile(resolve(cfg['template']), target/'template.json')
    predictions = []
    for image_id, card in card_images():
        card.save(target/f'{image_id}_card.png')
        for name, box in definitions.items():
            rect = tuple(round(v * card.size[i % 2]) for i, v in enumerate(box))
            if rect[0] >= rect[2] or rect[1] >= rect[3]:
                raise ValueError(f'Field {name} rounds to an empty crop')
            cropped = target/f'{image_id}_{name}.png'
            card.crop(rect).save(cropped)
            if args.ocr:
                result = subprocess.run([cfg['ocr']['executable'], str(cropped), 'stdout',
                    '-l', cfg['ocr']['language'], '--psm', str(cfg['ocr']['psm'])],
                    capture_output=True, text=True, timeout=120)
                predictions.append({'image_id': image_id, 'field': name, 'text': result.stdout.strip(),
                                    'error': result.stderr.strip() if result.returncode else ''})
    if args.ocr:
        dump(target/'predictions.json', predictions)
        if cfg['ocr'].get('truth'):
            score(predictions, resolve(cfg['ocr']['truth']), target)
    print(f'Baseline field crops saved to {target}; no YOLO detector used')


def score(predictions, truth_path, out, scope="annotated-card OCR baseline; excludes detector failures"):
    truth = {}
    with truth_path.open(newline='') as f:
        for row in csv.DictReader(f):aft

def detection_order(points):
    """Order a detector quadrilateral clockwise starting at image-space top-left.

    This geometric convention cannot infer text orientation; review field crops.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1]-center[1], pts[:, 0]-center[0]))]
    return np.roll(pts, -int(np.argmin(pts.sum(axis=1))), axis=0)


def field_evaluate(args, cfg):
    out = prepared(cfg)
    definitions = fields(cfg)
    if not args.weights or not resolve(args.weights).is_file():
        raise ValueError('field-evaluate requires an existing --weights best.pt')
    if not torch.cuda.is_available():
        raise RuntimeError('Use a GPU session for field evaluation')
    if not shutil.which(cfg['ocr']['executable']):
        raise ValueError('Tesseract executable unavailable')
    with (out/'splits.csv').open(newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['split'] == 'field']
    if not rows:
        raise ValueError('No independent annotated field samples in the manifest')
    target = out/'field_evaluation'
    if target.exists():
        raise FileExistsError('Field evaluation output already exists; preserve it before repeating')
    model = YOLO(str(resolve(args.weights)), task='obb')
    if model.task != 'obb' or model.names != {0: 'id_card'}:
        raise ValueError('Use the trained one-class id_card OBB model')
    target.mkdir(parents=True)
    records = json.loads((out/'records.json').read_text())
    summary, predictions = [], []
    source_images, _ = inventory(cfg)
    for row in rows:
        image_id = row['image_id']
        path = out/'images/field'/f'{image_id}.jpg'
        result = model.predict(source=str(path), conf=cfg['ocr']['detection_conf'],
                               device=cfg['train']['device'], imgsz=cfg['train']['imgsz'], verbose=False)[0]
        boxes = result.obb
        with Image.open(path) as src:
            im = src.convert('RGB')
        if boxes is None or len(boxes) == 0:
            summary.append({'image_id': image_id, 'detected': False, 'iou': 0.0})
            predictions.extend({'image_id': image_id, 'field': name, 'text': '',
                                'error': 'card detection missed'} for name in definitions)
            continue
        index = int(boxes.conf.argmax().item())
        points = detection_order(boxes.xyxyxyxy[index].cpu().numpy())
        # Annotation inverse transforms refer to original scan dimensions, not resized images.
        with Image.open(source_images[image_id]) as original:
            original_size = original.size
        expected = corners(records[image_id], original_size) * np.array(im.size) / np.array(original_size)
        intersection, _ = cv2.intersectConvexConvex(points.astype(np.float32), expected.astype(np.float32))
        union = abs(cv2.contourArea(points)) + abs(cv2.contourArea(expected.astype(np.float32))) - intersection
        iou = float(intersection/union) if union > 0 else 0.0
        summary.append({'image_id': image_id, 'detected': True, 'iou': iou,
                        'confidence': float(boxes.conf[index].item())})
        card = rectify(im, points, cfg['card_size'], int(row.get('rotation_quarters') or 0))
        card.save(target/f'{image_id}_card.png')
        for name, box in definitions.items():
            rectangle = tuple(round(v*card.size[i % 2]) for i, v in enumerate(box))
            cropped = target/f'{image_id}_{name}.png'
            card.crop(rectangle).save(cropped)
            ocr = subprocess.run([cfg['ocr']['executable'], str(cropped), 'stdout', '-l',
                                  cfg['ocr']['language'], '--psm', str(cfg['ocr']['psm'])],
                                 capture_output=True, text=True, timeout=120)
            predictions.append({'image_id': image_id, 'field': name, 'text': ocr.stdout.strip(),
                                'error': ocr.stderr.strip() if ocr.returncode else ''})
    dump(target/'detections.json', summary)
    dump(target/'predictions.json', predictions)
    dump(target/'detection_metrics.json', {'images': len(summary),
         'missed_cards': sum(not r['detected'] for r in summary),
         'localization_success_iou50': sum(r['iou'] >= .5 for r in summary)/len(summary)})
    if cfg['ocr'].get('truth'):
        score(predictions, resolve(cfg['ocr']['truth']), target,
              scope='full pipeline: detector misses included as field failures')
    print(f'Field evaluation saved to {target}. Review card orientation before interpreting OCR.')


def retire():
    """Archive the locally edited old config only; never delete datasets or checkpoints."""
    old = ROOT/'configs/oneal_obb.yaml'
    if old.exists():
        target = ROOT/'data/obb/retired-configs'
        target.mkdir(parents=True, exist_ok=True)
        dst = target/'oneal_obb.yaml'
        if dst.exists():
            raise FileExistsError('Retired config already exists; no overwrite performed')
        shutil.copyfile(old, dst)
        old.unlink()
        print(f'Archived old config to {dst}')
    print('Existing TPS datasets, annotations, and checkpoints were preserved.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['check','prepare','smoke','train','test','baseline','field-evaluate','retire-tps'])
    p.add_argument('--config', default='configs/id_card_pipeline.yaml')
    p.add_argument('--weights')
    p.add_argument('--resume')
    p.add_argument('--split', choices=['train','val','test','field'], default='val')
    p.add_argument('--ocr', action='store_true')
    p.add_argument('--image-id', action='append', help='Manual OCR baseline from a single annotated source scan; repeat for several')
    p.add_argument('--rotation-quarters', type=int, choices=range(4), default=0)
    args = p.parse_args()
    if args.image_id and args.command != 'baseline':
        p.error('--image-id is only supported with baseline')
    if args.resume and args.command != 'train':
        p.error('--resume is only supported with train')
    if args.command == 'retire-tps':
        retire()
        return
    cfg = yaml.safe_load(resolve(args.config).read_text())
    if args.command == 'check':
        images, records = inventory(cfg)
        print(f'Matched images and annotations: {len(records)}')
        if resolve(cfg['manifest']).is_file():
            rows = manifest(cfg, images, records)
            print('Split counts:', dict(Counter(r['split'] for r in rows)))
        else:
            print(f'Next: create and review {resolve(cfg["manifest"])}')
    elif args.command == 'prepare':
        prepare(cfg)
    elif args.command == 'baseline':
        baseline(args, cfg)
    elif args.command == 'field-evaluate':
        field_evaluate(args, cfg)
    else:
        model_action(args, cfg)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        sys.exit(str(exc))
