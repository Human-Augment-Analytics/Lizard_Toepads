# ID Card Parsing

A desktop annotator for locating the whole ID card in lizard scans. It supports
manual oriented bounding boxes (OBBs), automatic template detection, and JSON
annotations for later preprocessing and model training.
## Setup

Use an existing Python environment with Tkinter available. The application uses
Pillow, NumPy, and OpenCV; no GPU or deep-learning framework is required.

Activate your existing Python environment, then install dependencies from
`id-card-parsing/`:

```bash
python -m pip install -r requirements.txt
```

Prepare the inputs:

- Put source images in `data/` as `{id}.jpg`, for example `0001.jpg`.
- Put a reference card at `template/template.jpg`, tightly cropped to the outer
  card boundary.
- Start with a few scans rather than copying the entire dataset; source images
  can be very large.

Launch from `id-card-parsing/`:

```bash
python -m annotator
```

The main window shows centered **Loading…** with an animated bar while libraries
and images load in the background. Template features are prepared when an image
first needs detection. Image changes use the same centered loading indicator;
saved annotations can load without template preparation.

## Annotation

Search for an image number and click **Search** or press **Enter**. `68`, `0068`,
and `0068.jpg` all select `0068`. Search errors appear below the field. The header
shows the current image ID and its position in the available image list. This
count is the number of available files, not the highest numeric ID.

Use Prev/Next or the image-list scrollbar to navigate. The sidebar also scrolls
so its controls remain reachable in short windows.

### Mouse and view controls

- **Left-click and drag:** draw a new box.
- **Drag a corner handle:** resize the box while keeping the opposite corner fixed.
- **Angle slider:** rotate the box; the displayed value is in degrees.
- **Middle-click and drag:** pan the image.
- **Mouse wheel:** zoom around the cursor.
- **Canvas scrollbars:** move horizontally or vertically through a zoomed image.
- **Fit image:** center the complete preview in the viewport.

Images initially fit the viewport. Zooming, panning, scrolling, and fitting
change only the view, not the stored annotation coordinates. Box editing, angle
adjustment, and saving are disabled while another image loads.

### Template detection and review

The matcher uses printed card features to predict the card boundary, then refines
nearby edges where contrast is clear. It prefers nearby card edges over stronger
background objects and avoids unrelated black padding boundaries. Weak edges
retain their estimate; automatic boxes are constrained to the visible scan area
while preserving rotation.

Existing annotations take precedence over automatic detections. If the template
is unavailable or no reliable match is found, draw the box manually. Unreadable
saved annotations are reported without preventing manual repair; clicking Save
explicitly replaces the damaged annotation.

Review every automatic box before accepting it. Rounded corners can leave small
background areas inside a rectangular box. Clipped cards, unusual layouts, and
low contrast may need manual adjustment.

### Save and reset

- **Save annotation** saves only the currently displayed image, including edits.
- **Reset box** restores the box and angle initially loaded for that image: its
  template detection or saved annotation. If no initial box existed, Reset clears
  a manually drawn box. Reset does not change the file until you click Save.

Unsaved edits survive image switches during the session. Closing warns before
those edits are discarded; they are not recovered after restarting the app.
Failed image loads retain the previous image, and failed saves retain your edits
and any previous JSON. Completed files are published atomically so a failed write
does not expose a partially written annotation.

## Bulk save

**Save range…** opens a separate dialog for inclusive numeric start/end IDs, such
as `0001` through `0100`. Each available image uses its own template detection.
The current editor box and manual edits are not copied to other images; save
manual edits first if you want to keep them on disk.

Bulk saving skips:

- Existing annotation files, including files created by another writer.
- Missing image IDs.
- Images without a reliable detection.

Per-image processing or write failures are reported, and remaining images are
still attempted. The dialog shows measured progress and saved/skipped/failed
counts. After completion, the progress bar resets and a scrollable results list
shows IDs and reasons. Contiguous missing IDs appear as inclusive ranges.

**Cancel** stops further work when cancellation is observed and keeps files
already saved. Closing the app also cancels the batch. A write already committed
remains saved; the results list identifies images left unprocessed by cancellation.
Review bulk-generated boxes before using them for training.

## Files and coordinate format

- `data/{id}.jpg`: source scans, excluded from Git by the repository's ignore rules.
- `template/template.jpg`: cropped reference card used for matching.
- `annotations/{id}.json`: annotation records containing metadata and geometry,
  without image pixels.

The default preview is **1024 × 1024 pixels**, with the source image resized and
letterboxed. Each JSON record stores:

- The image ID and target resolution.
- Forward and inverse affine transforms between original and preview coordinates.
- The oriented box as `[cx, cy, width, height, angle]` in preview pixels, with the
  angle in **radians**.
- Four box corners in box-relative **TL, TR, BR, BL** order, defined in the box's
  unrotated frame.

Use the inverse transform to reconstruct geometry in original-image coordinates.
These JSON records are the annotator's own format, not YOLO label files. A
separate conversion step is needed to produce the training labels and normalized
coordinates required by your chosen model.

## PaddleOCR handwriting extraction

`card_ocr` extracts writing from template-aligned card fields using the English
PP-OCRv5 recognizer (`en_PP-OCRv5_mobile_rec`) on CPU. The annotator stays
independent of the OCR framework. The pipeline manages its generated output
under `results/ocr/current/`. The notebook and CLI use only PaddleOCR.

### Install in the existing environment

Activate your project virtual environment, then run this command from
`id-card-parsing`:

```bash
python -m pip install -r requirements.txt
```

This includes PaddleOCR and the CPU Paddle runtime. Keep one OpenCV
package installed: the base requirements pin `opencv-contrib-python==4.10.0.84`,
which is also required by PaddleX. Do not install another OpenCV distribution
alongside it. No additional Python installation or environment is needed.

### Use the notebook

Open `notebooks/paddle_ocr.ipynb` with your project environment selected as the
kernel (for example, `omscs`). OCR uses that kernel’s Python. Restart the kernel
after code updates.

For the first run, set `RUN_MODE='run'` and choose **Run All**. To inspect saved
results without inference, use `RUN_MODE='review'` (the notebook default). If
you change the image IDs, fields, or preprocessing settings, use run mode with
`START_NEW_RUN=True` to rebuild the current output.

- `RUN_MODE='review'` opens saved Paddle predictions, displays crops and generated
  text. It does not run OCR or install packages.
- `RUN_MODE='run'` prepares missing crops and runs Paddle once in a separate CPU
  process. Successful recognition replaces the saved JSON/CSV pair in
  `results/ocr/current/predictions/paddle/`. Interrupting the cell stops recognition.

The default cards are `0010`, `0011`, `0014`, `0015`, and `0043`, with 14 standard
fields each. Update `IMAGE_IDS` or `FIELDS` to select other inputs. Optional
corrected-date, header-note, other-species and bottom-note regions remain in
`template/fields.json` for cards that need them.

`START_NEW_RUN=False` reuses crops only when IDs, fields and preprocessing recipe
match. Set it to `True` in run mode to rebuild `results/ocr/current/`. A failed
crop rebuild preserves the existing run. A successful rebuild replaces the run
and removes predictions for the previous crops. Review mode only reads matching
saved crops and predictions. The pipeline creates all output directories itself.

Install dependencies with `python -m pip install -r requirements.txt` before
opening the notebook. The notebook uses its selected kernel and does not install
packages or create environments.

### Compare crops and OCR text

Run **Review predictions** to inspect each input crop beside
its raw Paddle text and extracted value. `SHOW_IMAGE_IDS` limits the displayed cards;
`SHOW_RAW_CROPS=True` also displays aligned raw crops.

Missing predictions and recognition errors are distinct from genuinely empty
model outputs. There are no human-label or crop-review output lines.

### Prepare crops with the CLI

```bash
python -m card_ocr prepare --ids 0010 0011 \
  --fields time mass perch initials flag
```

With no IDs, preparation selects the first three available scans. Numeric IDs
accept omitted leading zeros. Large JPEGs use draft decoding followed by a card
warp at a default width of 1,000 pixels. Saved outer-card annotations are read
first; template matching is the fallback. Internal SIFT/RANSAC registration
rejects unreliable fits before extracting writing regions.

Field boxes and margins are normalized to the upright template. Margins sample
source pixels before cropping. Cleanup removes known printing while protecting
added handwriting. Header cleanup conservatively removes separate printed ID
fragments without shrinking genuine notes; ambiguous connected strokes stay
intact. Keep raw crops for inspecting overlaps and unusual writing.

Output includes:

- `manifest.json`: crop coordinates, original-scan polygons, preprocessing recipe,
  registration diagnostics and crop warnings.
- `{id}/card.png`, `card_clean.png`, `fields.png`, and `scan_fields.jpg`: aligned
  card images and coordinate overlays.
- `{id}/{field}.png` and `{field}_raw.png`: exact cleaned OCR inputs and raw crops.
- Checkbox images and measurements in `manifest.json`: heuristic checkbox
  suggestions, separate from text recognition.

The default output is `results/ocr/current/`. Use `--replace` to rebuild an
existing run. Crops are prepared in a temporary directory and published only
after every selected image succeeds. Source scans and annotations remain intact.

### Run Paddle with the CLI

```bash
python -m card_ocr run --manifest results/ocr/current/manifest.json
```

Paddle runs on CPU. `--model-id` overrides the checkpoint. Predictions default to
`predictions/paddle/results.json` and `results.csv` inside the run. Use `--replace`
to rerun; failed recognition preserves the previous pair. The notebook uses this
replacement behavior automatically. `--output` selects an alternative filename.

The model cache defaults to `.cache/card-ocr` inside the active environment and
can be changed with `--cache-dir`. First use may download checkpoint files.

The JSON retains unchanged OCR text in `predictions` and extracted card records
in `data`, along with the exact extraction rules used. CSV contains one row per
card with the extracted values. Crop warnings remain in the manifest.

### Configure extracted values

`template/ocr_rules.json` defines each field's `type` and extraction `pattern`.
Integer and decimal fields can specify `min`/`max`; text fields can specify
`case`. Time fields define accepted `input_formats` and their output `format`:
`9.45` becomes `09:45`, and `1333` becomes `13:33`. Decimal measurements
such as mass `2.37` stay decimal. Dotted initials such as `x . y.` become
`X.Y.`. Day, month, and year are one- or two-digit integers.
Their patterns accept a stray dot between digits and `remove_characters: "."`
removes it: `1.3` becomes day `13`, `0.5` becomes month `5`, and `2.6` becomes
year `26`. Four-digit years are not accepted by the default rules. Decimal
measurement fields retain their decimal points.

The `date` definition combines those components into `DD/MM/YY`. Its explicit
`century: 2000` is used only for calendar checks such as leap days; the extracted
year remains `26`, and the combined date is `13/05/26`.

A pattern must find exactly one value. Empty, ambiguous, out-of-range, or invalid
values become JSON `null` and empty CSV cells. Rules do not substitute guessed
characters or establish that a plausible OCR value is correct. When nonblank
numeric or time OCR cannot be parsed, the pipeline retries the aligned raw crop
once, followed by a binary ink mask if needed. JSON retains `retry_text`,
`retry_crop`, and `retry_processing` when masking recovered a usable reading.
Numeric `retry_decimal_aliases` can interpret a middle dash as a decimal point
when a changed retry preserves or recovers digits: `30-1` becomes `30.1`.
An unchanged ambiguous reading such as `3-1` stays empty. Extraction uses a retry
only if the primary value cannot be parsed. The notebook shows both readings.
No missing digits are inserted by the rules. Fields without
a rule retain trimmed text. Recognition failures remain separate from parsing.

CLI recognition accepts `--rules path/to/ocr_rules.json`; the notebook uses
`RULES_PATH`. Review applies the current rules to saved raw text without running
Paddle or rewriting saved predictions. The next successful OCR run saves the
new structured JSON and CSV. Rule changes do not require rebuilding crops.

Cold imports and native-library startup can take longer than subsequent runs.
