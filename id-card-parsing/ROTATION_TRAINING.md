# Rotation training experiment

The training configuration now uses degrees: 180.0 instead of 15.0. Ultralytics samples a continuous angle between -180 and +180 degrees and transforms the image and its OBB labels together. This includes orientations near 90, 180 and 270 degrees; it is not a discrete quarter-turn-only augmentation. Mirror flips remain disabled.

Rotations are applied online during training; source scans, JSON annotations, manifests, validation images, and test images are unchanged. No preparation is needed solely for this training setting. Start a new training run rather than resume an old checkpoint: resume restores saved training settings. Keep the old run for comparison and select settings using validation data, not the held-out 0375 test failure.

Large rotations combined with translation and scaling can clip cards near image edges. Inspect the new train_batch preview images and labels before committing to a full run. Review rotated validation examples separately to assess orientation robustness without replacing the original validation images. This change does not make OCR crops upright automatically.

On PACE, preserve the scratch output path when updating the configuration. Set cfg['train']['degrees'] = 180.0 in the active YAML, then run smoke and train. Evaluate the resulting best checkpoint on the same fixed 130 test images; use the run path printed on completion.

Implementation reference: https://github.com/ultralytics/ultralytics/blob/v8.4.174/ultralytics/data/augment.py (RandomPerspective and v8_transforms).
