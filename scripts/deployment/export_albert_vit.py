#!/usr/bin/env python3
"""Export Albert's original trained ViT checkpoint without pretrained downloads.

Architecture matches landmarking-model-study 493ea405's ViT notebook. This is
an older research checkpoint, not the October DINO work. Export alone does not
establish accuracy; parity is checked on representative normalized inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    import numpy as np
    import onnx
    import onnxruntime as ort
    import timm
    import torch
    from torch import nn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)

    class ViTLandmark(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = timm.create_model('vit_small_patch16_224', pretrained=False)
            features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
            self.head = nn.Sequential(nn.Linear(features, 2048), nn.LayerNorm(2048), nn.ReLU(),
                                      nn.Linear(2048, 1024), nn.LayerNorm(1024), nn.ReLU(), nn.Linear(1024, 18))

        def forward(self, image):
            return torch.sigmoid(self.head(self.backbone(image)))

    model = ViTLandmark().eval()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    # Avoid timm's fused attention operator in the portable ONNX graph.
    for module in model.modules():
        if hasattr(module, 'fused_attn'):
            module.fused_attn = False
    sample = torch.zeros(1, 3, 224, 224)
    torch.onnx.export(model, sample, args.output, input_names=['image'], output_names=['coordinates'], opset_version=17)
    graph = onnx.load(args.output)
    metadata = graph.metadata_props.add()
    metadata.key, metadata.value = 'lizard_contract', 'albert-vit-aabb-224-bgr-v1'
    onnx.checker.check_model(graph)
    onnx.save(graph, args.output)
    options = ort.SessionOptions(); options.intra_op_num_threads = 2
    session = ort.InferenceSession(args.output, sess_options=options, providers=['CPUExecutionProvider'])
    errors = []
    rng = np.random.default_rng(42)
    for image in [np.zeros((1,3,224,224), dtype=np.float32), rng.uniform(-2, 2, (1,3,224,224)).astype(np.float32)]:
        with torch.no_grad():
            expected = model(torch.from_numpy(image)).numpy()
        actual = session.run(None, {'image': image})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
        errors.append(float(np.max(np.abs(actual - expected))))
    receipt = {'checkpoint_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
               'onnx_sha256': hashlib.sha256(Path(args.output).read_bytes()).hexdigest(),
               'max_abs_error': max(errors), 'torch': torch.__version__, 'timm': timm.__version__,
               'contract': 'albert-vit-aabb-224-bgr-v1', 'evaluation_status': 'research-pilot-no-heldout-evaluation'}
    Path(args.output + '.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
