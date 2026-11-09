# PACE model handoff

Team handoff root:

`/storage/ice-shared/cs8903onl/Lizard_Toepads/model-handoff`

Use this location for sharing complete model packages with the integration team.
Keep experimental working directories in personal scratch. Accepted packages
must remain accessible independently of the author's home-directory permissions.
The project group is `coc-pace-ice-access`; both `axu39` and `yloh30` were verified
members on 2026-11-09. The integration coordinator maintains this handoff area.

## Layout

```text
model-handoff/
  README.md                         # this document
  templates/checkpoint-package/     # package checklist and metadata example
  incoming/axu39/<run-id>/           # Albert writes a new complete package here
  incoming/yloh30/<run-id>/          # our own training runs
  ready/<run-id>/                   # coordinator-validated immutable snapshot
  receipts/<run-id>/                # registry/import/evaluation/E2E evidence
```

Use unique IDs such as `2026-11-09-dino-heatmap-run123456`. Never overwrite an
earlier package. A `ready` snapshot means packaging/compatibility was checked;
scientific evaluation status is recorded separately. It does not approve a
default model or move an MLflow alias.

## What Albert should supply

Create `incoming/axu39/<run-id>/` and include:

1. `weights/`: the checkpoint and every auxiliary weight, backbone, shape basis,
   or first-stage model required to load it without undocumented downloads.
2. `config.json` (or the resolved YAML equivalent), architecture/variant, exact
   source commit, training command, and any uncommitted source patch.
3. `environment.txt` or a lock file with Python/framework/library versions.
4. `preprocessing.json`: crop/detector geometry, orientation/flips, color order,
   resize/padding/normalization, landmark order, output decoding and coordinate
   backprojection. Document detector changes from training to serving.
5. `splits.json` and data-manifest references: dataset snapshot, specimen IDs and
   train/validation/test membership. Related source images and crops stay in
   the same specimen partition. Do not invent missing historical provenance.
6. `evaluation.json`: command, metric definitions/units, results, limitations and
   weak slices. Pixel/crop error and full-image or calibrated millimeter error
   must be distinguished.
7. `example/`: one permitted input and expected native inference output, command
   and numerical tolerance. State any preprocessing already applied to the input.
8. `metadata.json`: run owner, run ID, source locations and entries from the
   supplied metadata example. Supply checksums for all package files.

Place a plain-text `READY` file last, after the upload finishes and checksums are
verified. The coordinator only consumes completed packages. A researcher should
confirm which evaluation data may be copied or reused; large datasets may remain
in curated shared storage with immutable manifests.

## Ownership and permissions

Only this new handoff area uses the shared project group. Existing parent
permissions and personal directories remain unchanged. `incoming` inherits
group-readable/group-writable permissions through setgid and default ACLs, so
normal uploads do not silently become owner-only. Verify the destination after
tools that preserve source permissions, such as `rsync -a` or `cp -p`.

The coordinator writes `ready` snapshots; project members can read them but not
edit them. `receipts` follows the same coordinator-write/team-read policy. The
coordinator verifies actual file readability before registration. If account
membership changes, ask the project storage owner for access to this directory.

## Integration record

Record the package digest, exact research/app source revisions, MLflow model name,
numeric version, run ID, evaluation status, app model ID, and E2E test receipt.
The primary published model record is MLflow and its backed-up artifact store;
this directory is a project handoff/archive, not a second mutable registry.

Retain source metadata and receipts. Model/data retention must follow project
storage policy; no automatic cleanup policy is configured by this setup.
Promotion/default changes remain a separate research-owner decision.

## Our first full-pipeline verification

`scripts/deployment/train_dlib_e2e.py` trains new nine-point finger/toe dlib
predictors in a bounded PACE CPU job. It uses the existing H7 detector, explicit
BGR rectify-512 preprocessing, specimen-grouped splits, source/data hashes,
landmark-only validation/test metrics, and a withheld real-image example.
Its output includes a complete `bundle/`, provenance and `READY` marker.
This is a pipeline smoke test; detector independence and final scientific
accuracy are not claimed.
