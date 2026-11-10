#!/usr/bin/env python3
"""Train a bounded, specimen-grouped landmark baseline and build a pilot bundle.

Run heavy image preparation/training in a scheduled PACE CPU job. The source
XMLs and OBB labels are read-only. This is an oracle-box landmark evaluation,
not an independently held-out evaluation of the frozen legacy H7 detector.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import re
import shutil
import sys
import time
import xml.etree.ElementTree as ET


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def collect_samples(source):
    records, source_hashes = {}, {}
    for digit in ('finger', 'toe'):
        for old_partition in ('train', 'test'):
            xml = source / 'ml-morph' / f'{digit}_{old_partition}_yolo_obb_aligned.xml'
            source_hashes[str(xml)] = sha256(xml)
            for element in ET.parse(xml).getroot().findall('./images/image'):
                image = Path(element.attrib['file'])
                match = re.fullmatch(r'(\d+)', image.stem)
                if not match:
                    raise ValueError(f'Cannot infer specimen ID from {image}')
                specimen = match.group(1)
                label = source / 'data/bot_obb_from_tps' / f'{image.stem}.txt'
                if not image.is_file() or not label.is_file():
                    continue
                boxes = element.findall('box')
                if len(boxes) != 1:
                    raise ValueError(f'Expected one {digit} annotation for {image}')
                parts = sorted(boxes[0].findall('part'), key=lambda p: int(p.attrib['name']))
                if [p.attrib['name'] for p in parts] != [str(i) for i in range(9)]:
                    raise ValueError(f'Expected landmark order 0..8 for {image}')
                class_id = 2 if digit == 'finger' else 3
                matches = [line.split() for line in label.read_text().splitlines()
                           if line.split() and int(line.split()[0]) == class_id]
                if len(matches) != 1 or len(matches[0]) != 9:
                    continue
                row = {'specimen': specimen, 'digit': digit, 'image': str(image),
                       'label': str(label), 'label_sha256': sha256(label),
                       'corners_normalized': [float(v) for v in matches[0][1:]],
                       'landmarks': [[int(p.attrib['x']), int(p.attrib['y'])] for p in parts]}
                key = (specimen, digit)
                if key in records and records[key] != row:
                    raise ValueError(f'Conflicting annotations for {key}')
                records[key] = row
    return list(records.values()), source_hashes


def split_specimens(records, seed, train_count, val_count, test_count):
    ids = sorted({r['specimen'] for r in records}, key=int)
    random.Random(seed).shuffle(ids)
    needed = train_count + val_count + test_count
    if len(ids) < needed:
        raise ValueError(f'Need {needed} specimens; found {len(ids)}')
    forced_test = ['1004'] if '1004' in ids else []
    ids = [i for i in ids if i not in forced_test]
    test = forced_test + ids[:test_count - len(forced_test)]
    ids = ids[test_count - len(forced_test):]
    val, train = ids[:val_count], ids[val_count:val_count + train_count]
    partitions = {'train': train, 'validation': val, 'test': test}
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
    for digit in ('finger', 'toe'):
        for part, members in partitions.items():
            if not any(r['digit'] == digit and r['specimen'] in members for r in records):
                raise ValueError(f'Empty {digit} {part} partition')
    return partitions


def rectify(image, row, cv2, np):
    """Match AutoMorph _predict_toepad_crop(rectify_512=True) arithmetic."""
    height, width = image.shape[:2]
    corners = np.asarray(row['corners_normalized'], np.float32).reshape(4, 2)
    corners *= [width, height]
    by_y = corners[np.argsort(corners[:, 1])]
    top = by_y[:2][np.argsort(by_y[:2, 0])]
    bottom = by_y[2:][np.argsort(by_y[2:, 0])]
    box = np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)
    w = int(round(np.linalg.norm(box[1] - box[0])))
    h = int(round(np.linalg.norm(box[2] - box[1])))
    if w < 2 or h < 2 or abs(cv2.contourArea(box)) < 1:
        raise ValueError('Degenerate source OBB')
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    transform = cv2.getPerspectiveTransform(box, dst)
    crop = cv2.warpPerspective(image, transform, (w, h))
    scale = min(512.0 / h, 512.0 / w)
    nw, nh = int(w * scale), int(h * scale)
    if nw < 1 or nh < 1:
        raise ValueError('Degenerate resized crop')
    px, py = (512 - nw) // 2, (512 - nh) // 2
    canvas = np.zeros((512, 512, 3), dtype=np.uint8)
    canvas[py:py + nh, px:px + nw] = cv2.resize(crop, (nw, nh))
    points = np.asarray(row['landmarks'], np.float64)
    projected = np.column_stack([points, np.ones(9)]) @ transform.T
    points = projected[:, :2] / projected[:, 2:3]
    points = points * scale + [px, py]
    if not np.isfinite(points).all() or (points < 0).any() or (points > 512).any():
        raise ValueError(f'Landmarks outside prepared canvas: {row["specimen"]} {row["digit"]}')
    return canvas, points


def evaluate(predictor, samples, dlib, np):
    errors, per_sample = [], []
    for row, image, target in samples:
        shape = predictor(image, dlib.rectangle(0, 0, 512, 512))
        prediction = np.asarray([[p.x, p.y] for p in shape.parts()], float)
        distances = np.linalg.norm(prediction - target, axis=1)
        errors.extend(distances.tolist())
        per_sample.append({'specimen': row['specimen'], 'digit': row['digit'],
                           'mean_error_px': float(distances.mean())})
    return {'samples': len(samples), 'landmarks': len(errors),
            'mean_error_px': float(np.mean(errors)), 'median_error_px': float(np.median(errors)),
            'p95_error_px': float(np.percentile(errors, 95)), 'per_sample': per_sample}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--detector', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--seed', type=int, default=20261109)
    parser.add_argument('--train-specimens', type=int, default=80)
    parser.add_argument('--validation-specimens', type=int, default=20)
    parser.add_argument('--test-specimens', type=int, default=20)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if min(args.train_specimens, args.validation_specimens, args.test_specimens, args.threads) < 1:
        parser.error('Partition sizes and threads must be positive')
    if not re.fullmatch(r'[0-9a-f]{40}', args.source_commit):
        parser.error('source-commit must identify the exact training source')
    records, xml_hashes = collect_samples(args.source_root)
    splits = split_specimens(records, args.seed, args.train_specimens,
                            args.validation_specimens, args.test_specimens)
    config = {'seed': args.seed, 'threads': args.threads, 'tree_depth': 3,
              'cascade_depth': 6, 'num_trees_per_cascade_level': 80,
              'nu': 0.1, 'oversampling_amount': 5, 'feature_pool_size': 250,
              'num_test_splits': 20, 'selection': 'fixed configuration; no test-driven tuning'}
    contract = {'id': 'toepad-rectify-512', 'color_order': 'BGR', 'canvas': [512, 512],
                'padding': 'centered black letterbox', 'landmark_order': [str(i) for i in range(9)],
                'dlib_rectangle': [0, 0, 512, 512],
                'training_boxes': 'existing ground-truth TPS-derived OBB labels',
                'serving_boxes': 'frozen H7 six-class dual-pass detector outputs',
                'metric': 'Euclidean distance per landmark in 512-canvas pixels, float target',
                'claim': 'landmark-only oracle-box evaluation; E2E smoke test; no accuracy promotion'}
    selected_ids = {i for values in splits.values() for i in values}
    selected = [r for r in records if r['specimen'] in selected_ids]
    plan = {'run_id': args.run_id, 'splits': splits, 'config': config, 'contract': contract,
            'source_xml_sha256': xml_hashes, 'samples': len(selected),
            'source_commit': args.source_commit, 'training_code_sha256': sha256(__file__)}
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'plan.json', plan)
    write_json(output / 'splits.json', splits)
    write_json(output / 'config.json', config)
    write_json(output / 'preprocessing.json', contract)
    (output / 'training-command.txt').write_text(' '.join(sys.argv) + '\n')
    import cv2
    import dlib
    import numpy as np
    versions = {'python': platform.python_version(), 'dlib': dlib.__version__,
                'opencv': cv2.__version__, 'numpy': np.__version__}
    write_json(output / 'environment.json', versions)
    (output / 'environment.txt').write_text('\n'.join(f'{k}=={v}' for k, v in versions.items()) + '\n')
    samples = {digit: {part: [] for part in splits} for digit in ('finger', 'toe')}
    by_specimen = {}
    for row in selected:
        by_specimen.setdefault(row['specimen'], []).append(row)
    manifest = []
    for specimen, rows in sorted(by_specimen.items(), key=lambda pair: int(pair[0])):
        path = Path(rows[0]['image'])
        digest = sha256(path)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Could not decode {path}')
        partition = next(part for part, ids in splits.items() if specimen in ids)
        for row in rows:
            if row['image'] != str(path):
                raise ValueError('Conflicting image for specimen')
            canvas, target = rectify(image, row, cv2, np)
            relative = Path('crops') / partition / f'{specimen}_{row["digit"]}.png'
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(destination), canvas):
                raise IOError(f'Could not write {destination}')
            saved = {**row, 'partition': partition, 'image_sha256': digest,
                     'crop': str(relative), 'crop_sha256': sha256(destination),
                     'target_512': target.tolist()}
            manifest.append(saved)
            samples[row['digit']][partition].append((saved, canvas, target))
        del image
        print('prepared', specimen, partition, flush=True)
    write_json(output / 'dataset-manifest.json', manifest)
    weights = output / 'weights'
    weights.mkdir()
    results = {'metric': contract['metric'], 'evaluation_scope': contract['claim'], 'models': {}}
    for digit in ('finger', 'toe'):
        options = dlib.shape_predictor_training_options()
        for key, value in config.items():
            if hasattr(options, key):
                setattr(options, key, value)
        options.num_threads = args.threads
        options.random_seed = str(args.seed)
        options.be_verbose = True
        train = samples[digit]['train']
        images = [sample[1] for sample in train]
        objects = []
        for _, _, target in train:
            points = [dlib.point(int(round(x)), int(round(y))) for x, y in target]
            objects.append([dlib.full_object_detection(dlib.rectangle(0, 0, 512, 512), points)])
        started = time.monotonic()
        predictor = dlib.train_shape_predictor(images, objects, options)
        predictor.save(str(weights / f'{digit}.dat'))
        results['models'][digit] = {'training_seconds': time.monotonic() - started,
            **{part: evaluate(predictor, values, dlib, np) for part, values in samples[digit].items()}}
        print('trained', digit, results['models'][digit]['test']['mean_error_px'], flush=True)
    write_json(output / 'evaluation.json', results)
    bundle = output / 'bundle'
    bundle.mkdir()
    shutil.copyfile(args.detector, bundle / 'detector.onnx')
    for digit in ('finger', 'toe'):
        shutil.copyfile(weights / f'{digit}.dat', bundle / f'{digit}.dat')
    provenance = {'source_commit': args.source_commit, 'training_code_sha256': sha256(__file__),
                  'run_id': args.run_id, 'evaluation_status': 'e2e-smoke-oracle-box-heldout-landmarks',
                  'training_config': config, 'environment': versions, 'preprocessing': contract,
                  'split_sha256': sha256(output / 'splits.json'),
                  'dataset_manifest_sha256': sha256(output / 'dataset-manifest.json'),
                  'evaluation_sha256': sha256(output / 'evaluation.json'),
                  'detector_sha256': sha256(args.detector),
                  'limitations': 'Small landmark-only baseline; detector training exposure unknown; no default promotion',
                  'metrics': {digit: {part: {k: v for k, v in metrics.items() if k != 'per_sample'}
                                      for part, metrics in results['models'][digit].items()
                                      if isinstance(metrics, dict)} for digit in ('finger', 'toe')}}
    spec = {'format': 'lizard-pipeline-v1', 'provenance': provenance,
            'pipeline': {'schema_version': 1, 'id': args.run_id,
                'name': 'Own PACE-trained ML-Morph E2E candidate',
                'description': 'New finger/toe predictors; small research pipeline verification',
                'detector': {'artifact': 'detector.onnx', 'geometry': 'obb', 'confidence': .25,
                             'iou': .45, 'inference_protocol': 'toepad-dual-pass'},
                'classes': [{'id': i, 'name': 'bot_' + digit, 'predictor': digit + '.dat',
                             'landmark_schema': 'digit', 'crop_padding': 0.0,
                             'preprocessing': 'toepad-rectify-512', 'predictor_type': 'dlib'}
                            for i, digit in enumerate(('finger', 'toe'))],
                'landmark_schemas': {'digit': {'points': [str(i) for i in range(9)]}}}}
    write_json(bundle / 'bundle.json', spec)
    example = output / 'example'
    example.mkdir()
    test_specimen = splits['test'][0]
    row = next(r for r in manifest if r['specimen'] == test_specimen)
    image = cv2.imread(row['image'], cv2.IMREAD_COLOR)
    scale = min(1.0, 4096 / max(image.shape[:2]))
    derivative = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image
    if not cv2.imwrite(str(example / f'{test_specimen}.jpg'), derivative):
        raise IOError('Could not write example full image')
    write_json(example / 'input.json', {'specimen': test_specimen, 'partition': 'test',
               'source_sha256': row['image_sha256'], 'scale': scale,
               'input_sha256': sha256(example / f'{test_specimen}.jpg')})
    shutil.copyfile(output / row['crop'], example / 'crop.png')
    predictor = dlib.shape_predictor(str(weights / f'{row["digit"]}.dat'))
    shape = predictor(cv2.imread(str(example / 'crop.png')), dlib.rectangle(0, 0, 512, 512))
    write_json(example / 'expected.json', {'input': 'example/crop.png',
               'predictor': f'weights/{row["digit"]}.dat', 'color_order': 'BGR',
               'rectangle': [0, 0, 512, 512], 'coordinate_space': '512-canvas-pixels',
               'points': [[p.x, p.y] for p in shape.parts()], 'absolute_tolerance_px': 0})
    write_json(output / 'metadata.json', {
        'schema_version': 1, 'run_id': args.run_id, 'owner': 'yloh30',
        'model_family': 'dlib-shape-predictor-nine-landmarks',
        'source_commit': args.source_commit, 'training_command': ' '.join(sys.argv),
        'training_job_id': __import__('os').environ.get('SLURM_JOB_ID'),
        'checkpoint': ['weights/finger.dat', 'weights/toe.dat'],
        'resolved_config': 'config.json', 'environment': 'environment.txt',
        'preprocessing': 'preprocessing.json', 'dataset_manifest': 'dataset-manifest.json',
        'splits': 'splits.json', 'evaluation': 'evaluation.json',
        'evaluation_status': provenance['evaluation_status'],
        'example_input': 'example/crop.png', 'example_expected_output': 'example/expected.json',
        'example_command': 'python scripts/deployment/verify_dlib_example.py --package PACKAGE',
        'full_image_e2e_input': f'example/{test_specimen}.jpg',
        'auxiliary_artifacts': ['bundle/detector.onnx'],
        'known_limitations': [provenance['limitations']],
        'original_run_directory': str(output.resolve())})
    artifact_hashes = {str(p.relative_to(output)): sha256(p) for p in output.rglob('*') if p.is_file()}
    write_json(output / 'artifact-hashes.json', artifact_hashes)
    (output / 'READY').write_text('Training and landmark-only evaluation complete. Candidate; no promotion.\n')
    print(json.dumps({'output': str(output), 'models': results['models'], 'example': test_specimen}, indent=2))


if __name__ == '__main__':
    main()
