# Lizard ID-card detection, fixed fields, and OCR

This workflow follows the annotation setup in id-card-parsing/README.md. It uses
whole-card JSON annotations, not the older finger/toe/ruler/ID TPS dataset.
One YOLO class: id_card. The shared TPS files are never modified.

## Scope and team decisions

- Agree on the new card layout, whole-card boundary, upright orientation, field
  names, transcription conventions, source image identity, and independent groups.
- Use 50–100 original scans for YOLO training with online augmentation; reserve
  20–50 distinct original scans for the YOLO test set and separate validation scans.
- Select and annotate an independent field test set from roughly 2,000 new images.
  Define its size and sampling policy with the team; do not invent a field-test count.
- Establish a fixed-template OCR baseline without upstream YOLO, using annotated
  card boxes for rectification. This can be done before detector training.
- Measure full-pipeline performance on field samples, including missed detections.

The code enforces training/test count ranges and group separation. It does not
assume that JSON image 0001 is the old shared TPS image 1001. Source scans are
expected at id-card-parsing/data/{image_id}.jpg, as the annotator README states.
They are git-ignored. Find the correct originals before preparing actual data.

## Transition from the TPS experiment

Apply id-card-transition.patch in the repository root after the previous two
pipeline patches. It removes only our earlier pipeline script, tests, docs, and
batch job; repository-owned preexisting TPS utilities remain intact. Because the
old config was edited on PACE, it is archived separately rather than removed by
a fragile text patch:

```bash
git apply --check ~/id-card-transition.patch
git apply ~/id-card-transition.patch
.venv/bin/python scripts/id_cards/pipeline.py retire-tps
```

Retirement copies configs/oneal_obb.yaml into data/obb/retired-configs before
removing that config. The local annotation copies, generated TPS dataset,
checkpoints, and previous trial results remain untouched. A Git commit does not
carry ignored source images between machines.

## Dependencies and configuration

Use the already working Python 3.11/CUDA environment. Required preparation
packages: NumPy, Pillow, OpenCV, PyYAML. Ultralytics and GPU PyTorch are needed for
detector training/evaluation. OCR calls the separately installed Tesseract CLI;
`command -v tesseract` checks availability. No OCR engine is installed automatically.
Tesseract provides a baseline, not a claim of accurate handwriting recognition.

configs/id_card_pipeline.yaml controls paths, crop size, training, and OCR.
Default canonical card dimensions are a provisional 800x1200 portrait canvas;
agree on aspect ratio and orientation using the reference card before preparation.
Default decoding bounds scans to 2048 pixels on the longest side for memory use.
Increase scan_max_side after testing if small handwriting loses detail; compare
OCR at several resolutions using validation only. Source scans are not changed.

## Explicit reviewed split manifest

Create id-card-parsing/splits.csv with:

```csv
image_id,group_id,split,rotation_quarters
```

Rows use actual source IDs and independent group IDs, never fabricated filenames.
Valid split values: train, val, test, field. Every row requires a matching source
scan and JSON annotation. Multiple scans of the same animal, session, or duplicated
image should use the same group_id and cannot cross splits. Group identity is
provided by the team, not inferred from numbers or image hashes.

rotation_quarters is 0, 1, 2, or 3 and cycles the four card corners before warping;
review resulting orientation. This is an explicit orientation convention and does
not automatically read text direction. Annotation corners are TL/TR/BR/BL in the
box frame, not necessarily the image axes. For field detection, corners are ordered
geometrically from image-space top-left; calibrate orientation and review outputs.
Rotation metadata and orientation handling must be locked before field evaluation;
it must not be adjusted after inspecting test transcriptions.

With 100 usable records, a possible allocation is 60 train, 20 validation, 20 test,
subject to group sizes. Field samples must be separate new scans with reviewed
card JSONs and transcribed fields. Add them to the same configured source and
annotation directories and mark their rows field. Source scans without JSONs are
not included automatically. A separate dataset version can later include the
new field samples without overwriting the earlier experiment.

## Check and prepare

```bash
.venv/bin/python scripts/id_cards/pipeline.py check
.venv/bin/python scripts/id_cards/pipeline.py prepare
```

Preparation reconstructs original corners using transform_inverse, checks forward
and inverse transforms, bounds, and convexity, then normalizes original x/y by
image width/height. Stored corners already include rotation; no second rotation
from obb_xywhr is applied. Images are resized with aspect ratio preserved and
labels stay normalized. No TPS annotations are used.

Output under data/id_cards/v1:

- images/{train,val,test,field} and matching one-class labels
- dataset.yaml: absolute root with train/val/test only; field excluded from training
- cards/{split}: annotation-rectified upright cards for the independent OCR baseline
- review/{split}: whole-card annotation overlays for every selected sample
- config.yaml, splits.csv, records.json, summary.json, READY

Preparation refuses existing output directories. Review overlays, card orientation,
and field alignment before training. A changed input/layout/split requires a new
output version; do not manually reuse READY from a partial preparation.

## Fixed-template baseline without YOLO

For a manual test on one available annotated scan, no YOLO preparation or split
manifest is required. Define the field template first, then run:

```bash
.venv/bin/python scripts/id_cards/pipeline.py baseline --image-id 0001
# Or include OCR in the same run when Tesseract is available:
.venv/bin/python scripts/id_cards/pipeline.py baseline --image-id 0001 --ocr
```

These are alternative first runs; the manual output folder is not overwritten.
Repeat --image-id to include several scans. --rotation-quarters adjusts orientation
for these manual tests. Crops and rectified cards are saved in a sibling output folder named v1_manual_baseline, so they do not block later
YOLO dataset preparation.
With an empty field template, the command explains that rectangles must be defined
before it can run. The source image ID is illustrative; use an actually available ID.


Open rectified cards from the validation split. Define field rectangles in
configs/id_card_fields.json using normalized [left,top,right,bottom] coordinates.
The delivered template is deliberately empty: the README does not define field
positions. For example, a field name should map to four reviewed numbers between
0 and 1, with left<right and top<bottom. Do not copy arbitrary example rectangles
onto a real card. Different card layouts require separate versioned templates.

```bash
.venv/bin/python scripts/id_cards/pipeline.py baseline --split val
# After confirming Tesseract availability:
.venv/bin/python scripts/id_cards/pipeline.py baseline --split test --ocr
```

Run only one baseline per split/output version; it refuses to overwrite prior
results. If doing crops first and OCR later, preserve/move the previous baseline
folder first. Choose template and OCR settings using validation samples, then freeze
them before test or field evaluation.

For metrics set ocr.truth to a CSV with image_id,field,value. Values containing
commas must be CSV-quoted. Predictions without corresponding truth are explicitly
unscored. Metrics report normalized exact-field accuracy (casefold, Unicode NFKC,
whitespace collapsed) and character error rate. They do not report numeric accuracy,
biological validity, or automatically establish correct interpretation of units.
Detector-independent baseline metrics exclude detection failures by definition.

## YOLO smoke, full training, and held-out test

```bash
.venv/bin/python scripts/id_cards/pipeline.py smoke
.venv/bin/python scripts/id_cards/pipeline.py train
mkdir -p logs
sbatch sbatch/train_id_card_obb.sbatch train
```

Smoke is 2 epochs at 640px/batch2; full settings default to YOLO11m-OBB, 1280px,
batch4, up to 200 epochs with validation early stopping. Pretrained weights download
on first use; prefetch/set a local path for an offline job. Online photometric and
geometric augmentations apply only during training. Test/field scans stay independent.
Use actual run directory names; Ultralytics adds suffixes on repeated runs.

```bash
sbatch sbatch/train_id_card_obb.sbatch train --resume data/id_cards/v1/runs/train/weights/last.pt
sbatch sbatch/train_id_card_obb.sbatch test --weights data/id_cards/v1/runs/train/weights/best.pt
```

Batch defaults request coc/coc-ice, coc-gpu, one A40, 4 CPUs, 32GB, 4 hours.
Adapt to your allocation. Running the direct Python command inside an existing
GPU session uses whichever GPU was allocated. New-model downloads and real GPU
training have not been executed by the patch creator.

## Independent full-pipeline field evaluation

After selecting the model/template/OCR settings on validation:

```bash
.venv/bin/python scripts/id_cards/pipeline.py field-evaluate \
  --weights data/id_cards/v1/runs/train/weights/best.pt
```

Requires field manifest rows, corresponding card annotations, a trained one-class
model, and Tesseract. It detects the whole card, chooses the highest-confidence
box, rectifies it, crops the fixed fields, and applies OCR. It records confidence,
reference-box IoU, missing cards, localization success at IoU >= .5, per-field
predictions, and transcription metrics. Missed cards become field failures rather
than being omitted. This assumes one target card per scan; multiple-card scenarios
need a separate matching policy. Orientation still needs calibration; a YOLO OBB
rectangle alone does not identify which end of a card is its text top.

No field test images/transcriptions are bundled, and no actual OCR scores are claimed.

## Verification

```bash
python -m unittest discover -s tests -p test_id_card_pipeline.py
```

Synthetic tests cover inverse-transform conversion, geometry validation, missing
images, grouped/count-constrained splits, card/field crops, OCR scoring, and GPU
command dispatch. GPU dispatch is mocked; real source-data preparation, CUDA
training, Tesseract recognition, and detector-driven field evaluation still require
PACE validation. Existing annotator code and its tests remain unchanged.
