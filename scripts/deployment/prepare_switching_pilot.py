#!/usr/bin/env python3
"""Prepare two explicit research bundles using existing, locally supplied weights.

Required files: yolo_obb_6class_h7_int8.onnx, ml_morph_best.dat, albert_vit.onnx,
albert_vit.onnx.json (export receipt). No downloads or model promotion occur.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = '493ea405e3214a992fcf79c406e6fe1a3d7f6d97'
    for family in ['ml-morph', 'albert-vit']:
        root = args.output / family
        root.mkdir(parents=True, exist_ok=True)
        predictor_source = 'ml_morph_best.dat' if family == 'ml-morph' else 'albert_vit.onnx'
        predictor = 'landmarks.dat' if family == 'ml-morph' else 'landmarks.onnx'
        shutil.copyfile(args.fixtures / 'yolo_obb_6class_h7_int8.onnx', root / 'detector.onnx')
        shutil.copyfile(args.fixtures / predictor_source, root / predictor)
        classes = [{'id': index, 'name': name, 'predictor': predictor, 'landmark_schema': 'digit',
                    'crop_padding': 0.0, 'preprocessing': 'toepad-rectify-512' if family == 'ml-morph' else 'affine',
                    'predictor_type': 'dlib' if family == 'ml-morph' else 'onnx-coordinates'}
                   for index, name in enumerate(['bot_finger', 'bot_toe'])]
        provenance = {'source_commit': source, 'evaluation_status': 'research-pilot-no-heldout-evaluation',
                      'supported_classes': ['bot_finger','bot_toe'],
                      'detector_contract': 'H7 six-class dual-pass; only bottom finger/toe exposed',
                      'landmark_contract': 'rectify-512' if family == 'ml-morph' else 'albert-vit-aabb-224-bgr-v1',
                      'source_commit_role': 'Source/adapter contract reference, not checkpoint training-run provenance',
                      'training_provenance': 'Historical checkpoint; original training run/split not supplied',
                      'predictor_sha256': hashlib.sha256((root / predictor).read_bytes()).hexdigest(),
                      'limitations': 'Research integration only; H7 AABB envelope differs from original ViT H5 detector crops. No accuracy claim.'}
        if family == 'albert-vit':
            provenance['export'] = json.loads((args.fixtures / 'albert_vit.onnx.json').read_text())
        spec = {'format': 'lizard-pipeline-v1', 'provenance': provenance,
                'pipeline': {'schema_version': 1, 'id': family, 'name': f'{family} bottom-digit pilot',
                    'description': 'Experimental bottom finger/toe pipeline; accuracy not evaluated',
                    'detector': {'artifact': 'detector.onnx', 'geometry': 'obb', 'confidence': .25,
                                 'iou': .45, 'inference_protocol': 'toepad-dual-pass'},
                    'classes': classes, 'landmark_schemas': {'digit': {'points': [str(i) for i in range(9)]}}}}
        (root / 'bundle.json').write_text(json.dumps(spec, indent=2))
        print(root)


if __name__ == '__main__':
    main()
