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

## Tests

Install pytest into the active environment if needed:

```bash
python -m pip install pytest
```

Run the complete suite from `id-card-parsing/`:

```bash
python -m pytest tests -q
```

The suite covers backend geometry, records, matching, bulk saves, and GUI safety.
GUI checks run in isolated processes and skip when no Tk display is available.
To run only tests that do not create GUI windows:

```bash
python -m pytest tests -q --ignore=tests/test_ui_safety.py
```
