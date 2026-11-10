# LizardMorph / MLflow team guide

Last verified: **November 9, 2026**.

This is the team entry point for development organization, PACE model handoff,
MLflow registration, application verification, and the completed training-to-app
experiment. It covers both [Lizard_Toepads](https://github.com/Human-Augment-Analytics/Lizard_Toepads)
and [LizardMorph](https://github.com/Human-Augment-Analytics/LizardMorph).

## Start here

- **Albert / model contributors:** use [the PACE handoff steps](#share-a-model-on-pace).
- **Integration engineers:** follow [registration and app verification](#register-and-verify-a-model).
- **Developers:** use [the repository and development workflow](#organize-development).
- **Reviewers:** see [the completed E2E experiment](#completed-e2e-experiment).
- **Runtime maintainers:** see [private access and operations](#private-access-and-operations).

### Current availability

The repositories include the complete-bundle MLflow integration: packaging and
registration in `Lizard_Toepads`, and model discovery, validated import, version
selection, inference, rollback and export provenance in `LizardMorph`.
The private research runtime has exercised ML-Morph, an older Albert ViT checkpoint,
and the newly trained dlib candidate below. Start with current `main` checkouts;
the historical source archive below is for reproducing the recorded experiment.

The runtime remains private and coordinator-operated. Shared runtime access,
permanent operating ownership and backup/restore verification remain team work.
No scientific default-model promotion has been made.

### What happens when a new model version is registered?

**A new model version does not create a new app or deployment.** One LizardMorph
app can install several model versions side by side:

1. Training/evaluation produces a complete compatible package.
2. `register_bundle.py` explicitly creates an MLflow run and a numeric version
   under a named model. Saving a checkpoint alone does not register it.
3. Open **Research models** in LizardMorph and refresh its registry list. The new
   version becomes available for inspection; it is not automatically installed.
4. Choose **Import this version** to download and validate its full bundle.
5. Choose **Use this version** to open an analysis with that installed model.

Existing analyses retain their model identity. Registration/import does not move
an alias, change a default, rebuild the app or migrate existing analyses. Compatible
weights use the existing adapter; a new architecture or preprocessing/output
contract may require app code, tests and a deployment. A fresh deployment also
needs its configured registry and persistent data; merging code does not transfer
the private runtime's models or experiment database.

Read the [interactive HTML guide](mlflow-guide.html) for the system diagram and
searchable document map. GitHub displays HTML source; download the file and open
it in a browser. The [PACE handoff checklist](MODEL_HANDOFF.md) and
[metadata template](model-handoff-metadata.example.json) are the upload references.

## What each system does

```text
GitHub code + versioned data references
                |
                v
PACE scheduled training / evaluation
                |
                v
Complete model package in shared PACE storage
                |
                v
MLflow run + named model version + checksummed bundle
                |
                v
LizardMorph catalog import -> select version -> infer -> export provenance
```

MLflow records experiment settings, metrics, output artifacts, and model versions.
It connects a model to the run and evidence that produced it. These records must
be explicitly logged; missing historical training provenance cannot be recovered
by registering a checkpoint alone. See the official [tracking](https://mlflow.org/docs/latest/tracking/)
and [registry](https://mlflow.org/docs/latest/registry/) documentation.

The integration serves bundle artifacts from MLflow's persistent artifact
store. The older [MLflow deployment plan](PLAN_MLFLOW_DEPLOYMENT.md) describes an
earlier GitHub Releases distribution design. Treat that file as historical design
context; use this guide for the verified bundle workflow.

## Organize development

The recommended organization reuses the two repositories and the working integration.
The project board and permanent role assignments described here are proposals,
not claims that those arrangements are already configured.

| Item | Home | Working rule |
| --- | --- | --- |
| Research code | `Lizard_Toepads` | Models, preprocessing, splits, training, evaluation, exporters, and bundle production. Record the source revision and any uncommitted patch. |
| App/runtime code | `LizardMorph` | Catalog, adapters, API/UI, import validation, exports, Compose, and runtime operations. |
| Published model identity | One team MLflow registry | Use named numeric versions, complete bundles, hashes, and explicit evaluation status. |
| Images, labels, splits | Curated PACE project storage | Retain original specimens; record dataset and label snapshots. Keep related images/crops in one specimen partition. |
| Work and decisions | One proposed GitHub project board across both repos | One issue owns each outcome; link related PRs, source revisions, decisions, and verification receipts to it. |

Each issue should name a research/evaluation owner, an integration owner, and an
operating owner as relevant. Albert or the contributing researcher supplies the
model contract; the evaluator owns benchmark definitions and scientific approval;
the engineer owns compatibility and app behavior; the runtime maintainer owns
access, recovery, and rollback. Confirm people rather than assuming availability.

Use short task branches/worktrees from current `main`, linked PRs for changes
across both repos, and a tested compatible pair of revisions. Keep a release
receipt containing code/image IDs, model versions/hashes, evaluation status,
test evidence, and rollback target. Evidence applies to the reviewed revisions;
recheck affected behavior after later changes.

Environment roles:

- Laptop: edit, coordinate, review, and run lightweight checks.
- PACE: research data and scheduled training/evaluation. Run heavy work through
  Slurm; save job IDs, configurations, checkpoints, and resume instructions.
- Integration test/staging host: isolated Docker tests and private staging. Run
  database-backed tests in the project's isolated containers against the exact
  checkout being reviewed.
- Additional GPU hosts: use only after verifying their role, data access, and
  resource availability for the specific task. The same model contract applies.

Start with one bounded job and one complete model delivery. Increase training
scope after measuring resource use and checking the small end-to-end slice.

## Share a model on PACE

### Stable location

```text
/storage/ice-shared/cs8903onl/Lizard_Toepads/model-handoff
```

Albert's upload directory is:

```text
/storage/ice-shared/cs8903onl/Lizard_Toepads/model-handoff/incoming/axu39/<run-id>/
```

The layout is already created:

```text
model-handoff/
  README.md
  templates/checkpoint-package/
    README.md
    metadata.example.json
  incoming/axu39/<run-id>/       # Albert's completed packages
  incoming/yloh30/<run-id>/      # our own training packages
  ready/<run-id>/               # coordinator-validated snapshots
  receipts/<run-id>/            # registration, evaluation, and E2E evidence
```

Choose a unique run ID, for example `2026-11-09-dino-heatmap-job123456`. Keep
experimental working files in scratch and copy a complete package into the
handoff area. Do not overwrite an earlier run. Treat `ready` snapshots as
immutable. A package being ready means its packaging/compatibility was checked;
scientific evaluation status and default approval are separate.

### Required package

| File or directory | Required information |
| --- | --- |
| `weights/` | Checkpoint plus all auxiliary/backbone/shape-basis/first-stage weights needed to load it. Copy files rather than leaving dependencies as personal-directory symlinks. |
| Resolved configuration | Architecture, variant, exact source commit, training command, Slurm job ID, and any uncommitted source patch. |
| `environment.txt` or lock file | Python, framework, and library versions. |
| `preprocessing.json` | Detector/crop geometry, orientation and flips, color order, resize/padding/normalization, landmark order, output decoding, and full-image backprojection. Identify changes from training to serving. |
| `splits.json` and data manifest | Dataset/label snapshot references, specimen IDs, partition membership, and hashes. State missing provenance explicitly. |
| `evaluation.json` | Evaluation command, metric definitions/units, results, failures and weak slices. Distinguish crop pixels, full-image pixels, and calibrated millimeters. |
| `example/` | One permitted input, expected native inference output, exact command and numerical tolerance. State preprocessing already applied to the input. |
| `metadata.json` | Run owner/ID, source locations, and references to the files above. Start from the shared metadata example and replace its placeholders. |
| Checksum inventory | Hashes of the files actually supplied. The completed example uses `artifact-hashes.json`; `SHA256SUMS` is also suitable for a handoff. |
| `READY` | Completion marker written last, after transfer and checksum verification. |

Large datasets can stay in curated shared storage with immutable manifests and
usable access permissions. Confirm which example/evaluation images may be reused.
Preserve metadata and receipts; no automatic retention/cleanup policy has been
configured for this handoff area.

### Upload procedure

On PACE, create the new directory and copy the metadata template:

```sh
handoff_root=/storage/ice-shared/cs8903onl/Lizard_Toepads/model-handoff
handoff_run=2026-11-09-dino-heatmap-job123456  # replace for this run
package_dir="$handoff_root/incoming/$USER/$handoff_run"
mkdir -p "$package_dir"
cp "$handoff_root/templates/checkpoint-package/metadata.example.json" \
  "$package_dir/metadata.json"
```

Populate the package and edit `metadata.json`. After the copy has finished,
generate and verify checksums. For large weights, perform this work within the
scheduled job rather than adding heavy I/O to a login node:

```sh
set -eu
cd "$package_dir"
find . -type f ! -name SHA256SUMS ! -name READY -print0 \
  | sort -z | xargs -0 sha256sum > SHA256SUMS
sha256sum -c SHA256SUMS
printf 'Package complete; candidate for integration review.\n' > READY
```

The coordinator verifies files, dependencies, native inference, and the serving
contract before making a new `ready/<run-id>` snapshot and recording its receipt.

### Access and permissions

The handoff uses the project group `coc-pace-ice-access`. Both `axu39` and `yloh30`
were verified members on November 9. The incoming directories use setgid and
default ACLs for inherited group read/write access. `ready` and `receipts` are
coordinator-write/team-read. Existing parent and private-home permissions were
not changed.

Check the destination after using tools that preserve source modes, such as
`rsync -a` or `cp -p`; they can preserve owner-only files. If access fails, send
the coordinator the affected project path and ask the storage owner to verify
membership/ACLs. Do not require a teammate to depend on access to a personal
training directory.

## Register and verify a model

### 1. Freeze the contract and evaluate

Record source revision, data/label snapshot, specimen-disjoint splits, benchmark,
serving preprocessing, one hand-checked example, and known weak cases. Agree on
metric definitions and acceptance thresholds before deciding whether to promote
a candidate. Check export/native parity and intended-host latency/memory.

New model families need an adapter if their input/output contract differs. The
integration supports dlib rectify-512 and the specific older ViT coordinate contract
`albert-vit-aabb-224-bgr-v1`. Its dual-pass detector decoder expects the legacy
H7 six-class output. A four-class H10/H11 detector or a newer DINO/HRNet model is
not automatically compatible with that decoder/adapter.

### 2. Build the complete pipeline bundle

The producer accepts a directory containing `bundle.json` plus all referenced
detector/predictor files. The `lizard-pipeline-v1` bundle records the pipeline
manifest, preprocessing, landmark schema, checksums, source, and evaluation
provenance. The importer validates compatibility before exposing a model.

Use a current `Lizard_Toepads` main checkout. With
MLflow **2.22.4** installed and an explicit `BUNDLE_DIR` and `MODEL_NAME`, its CLI is:

```sh
python scripts/deployment/register_bundle.py \
  --bundle-dir "$BUNDLE_DIR" \
  --name "$MODEL_NAME" \
  --tracking-uri http://127.0.0.1:18600
```

The loopback address requires the runtime or the SSH forwarding below. Inside
the Compose network, use `http://mlflow:5000`. Each invocation creates a
new run/version, so do not re-register a model just to restart a service.

The November 9 orchestration additionally attached training parameters, 32
metrics, source/config/environment records, data/split/evaluation manifests,
the Slurm log, and the native example to the same MLflow run. That orchestration
is preserved with the experiment receipts; it is not a generic main-branch CLI.

### 3. Import an explicit version in the app

1. Open **Research models** on the private app.
2. Find the named model and numeric version, then choose **Import this version**.
3. Confirm validation succeeds, then choose **Use this version**.
4. Upload a permitted held-out full image and inspect the rendered landmarks.
5. Choose **Export All Data** and inspect the downloaded ZIP.

An analysis retains its selected model identity. The ZIP includes TPS coordinates,
an annotated image, measurements, and image-specific `.model.json` provenance.
Verify the registry name/version, bundle hash, and input-image hash; where present,
also verify the source/data/split/evaluation hashes.

Registration/import does not move an alias or default. To roll back an analysis
workflow, open the previously installed version. Scientific default promotion
remains an explicit research-owner decision.

### 4. Record acceptance evidence

Use isolated Docker test containers on the integration host against the reviewed
checkout for database-backed checks. Include bundle rejection/compatibility, native parity,
real registration/download/import, repeated import identity, held-out full-image
inference, browser rendering/export, version switching, restart persistence,
and unchanged existing models/aliases. Record exact code/image/model hashes and
obtain independent review for substantive code or model changes.

## Completed E2E experiment

### Identity and preserved artifacts

The bounded November 9 experiment trained new nine-point finger and toe dlib
predictors from scratch and reused the frozen H7 detector. It used a scheduled
PACE CPU job, not a GPU training campaign.

| Record | Value |
| --- | --- |
| Run directory ID | `2026-11-09-own-dlib-e2e-r1` |
| Slurm job | `6908609`, completed with exit `0:0` in 54 seconds |
| Resource request | 4 CPUs, 12 GB, 20-minute limit; peak RSS about 1.37 GiB |
| Training source | `ba434823da4274156f5831ab9eacbd1af7dc20f0` (local reviewed commit; archived below) |
| App source | `1ca95f9a0ba60aaffd4088b7201f3c724fb2c66f` (reviewed private pilot checkout) |
| MLflow model/version | `toepad-landmarks-own-pace-dlib-20261109`, version `1` |
| MLflow run | `4975c4fe04eb413ba0d1766936284012` |
| App model ID | `registry-5a8af92de70c7304611cdc51` |
| Bundle SHA256 | `8e5d3576adb12726876f4e197f51ac7d89fdd016deb12075184a754829db1df2` |
| Withheld app specimen | `1484`, downsampled full-image derivative with maximum dimension 4096 |

On PACE, under the handoff root:

```text
ready/2026-11-09-own-dlib-e2e-r1/
  source.tar                   # complete producer/training source archive
  train.sbatch                 # exact historical job; contains original paths
  source-record.json
  plan.json, config.json, preprocessing.json, environment.txt
  dataset-manifest.json, splits.json, evaluation.json
  weights/, bundle/, crops/, example/, artifact-hashes.json, READY

receipts/2026-11-09-own-dlib-e2e-r1/
  e2e-receipt.json
  independent-review.json
  training-verification.json
  registry-receipt.json
  browser-export-verification.json
  browser-analysis-1484.zip
  register_training_run.executed.py
```

The source archive SHA256 is
`959d742f411caa5c8a686978a65231663765b2761df3cac34bfe54c7960ca1b9`.
Check it before extraction. The checkpoint package and example are available to
the project group without entering the author's home directory.

### Reproduce the native example or prepare another training run

The preserved environment was Python 3.12.2, dlib 19.24.2, OpenCV 4.13.0, and
NumPy 1.26.4. The same native example was independently checked in the application
container with dlib 19.24.6 and produced identical coordinates.

On PACE, with a compatible environment, extract the source into a new scratch
directory and verify the preserved example:

```sh
handoff_root=/storage/ice-shared/cs8903onl/Lizard_Toepads/model-handoff
saved_package="$handoff_root/ready/2026-11-09-own-dlib-e2e-r1"
workdir=$(mktemp -d "$HOME/scratch/lizard-example.XXXXXX")
tar -xf "$saved_package/source.tar" -C "$workdir"
python "$workdir/scripts/deployment/verify_dlib_example.py" \
  --package "$saved_package"
```

Expected: `verified: true`, nine landmarks, and `max_abs_error_px: 0.0`.

For a new training run, use `python scripts/deployment/train_dlib_e2e.py --help`
in a current main checkout. Use the archived revision for an exact historical rerun.
The script expects a verified research-data worktree supplied through
`--source-root`, with the four original-coordinate finger/toe train/test XMLs
under `ml-morph/`, OBB label files under `data/bot_obb_from_tps/`, and readable
raw images referenced by the XMLs. Verify their hashes against the saved plan
when attempting an exact historical rerun. The source archive alone does not
contain those original raw-data inputs; coordinate their shared access first.
Prepared crops and targets are preserved in the package for inspection.

Use a new output directory/run ID and submit the work through Slurm. The saved
`train.sbatch` documents the tested resource request and exact original command;
update its source/data/environment/output paths before using it for another run.
Record the new source revision, freeze the split before looking at results, and
retain the new plan, metrics, source archive, job log, and hashes.

### Coverage, results, and limits

Specimens were split once across both digit classes: **80 train / 20 validation /
20 test**, totaling 120 specimens and 129 eligible digit crops. Specimen IDs and
identical raw-image hashes did not cross partitions.

| Digit | Train crops | Validation crops | Test crops | Test mean landmark error |
| --- | ---: | ---: | ---: | ---: |
| Finger | 49 | 12 | 14 | 11.29 pixels |
| Toe | 36 | 11 | 7 | 16.92 pixels |

These are mean Euclidean landmark errors on **512-pixel canvases constructed from
ground-truth-derived boxes**. They do not establish independently held-out
detector-plus-landmark accuracy. H7's training exposure is unknown. Toe validation
has major failures on specimens 1055 and 1540; this small candidate is not approved
as a scientific default.

Verified outcomes:

- All 153 inventoried package files matched their hashes; archived code and
  installed model weights matched the reviewed/trained artifacts.
- Training pixels matched the app's BGR rectify-512 contract, coordinates
  round-tripped, and the native example agreed across hosts.
- **19 targeted producer/app bundle tests passed with no skips** in isolated
  Docker test containers, including SQLite-backed import checks.
- Browser import, upload of withheld specimen 1484, 18 rendered landmarks, and
  the real downloaded export ZIP passed. Its 18 TPS coordinates matched the API;
  model/input/data/split/evaluation hashes matched the package.
- Candidate -> previous ML-Morph v1 -> candidate switching preserved the
  candidate's predictions; old and new models produced different predictions.
- Repeated import retained identity. App restart and browser reload retained
  the candidate/history. Six prior catalog manifests and existing aliases were
  unchanged; the candidate remained unaliased.
- Final independent review passed with no blocking findings. Training and
  test/registration containers stopped; the four private services remained
  available. No public deployment or scientific promotion occurred.

## Private access and operations

The private runtime binds services to loopback on the Docker host:

| Service | Host port |
| --- | ---: |
| MLflow | 18600 |
| Model API | 18601 |
| App backend | 18602 |
| Frontend | 18603 |

Ask the coordinator for an approved runtime host/account. With SSH access,
forward the private endpoints from your laptop:

```sh
ssh -N \
  -L 127.0.0.1:18600:127.0.0.1:18600 \
  -L 127.0.0.1:18601:127.0.0.1:18601 \
  -L 127.0.0.1:18603:127.0.0.1:18603 \
  approved-user@approved-runtime-host
```

Then open the app at `http://127.0.0.1:18603/#/models`, or MLflow at
`http://127.0.0.1:18600`. The experiment above is MLflow experiment `1`, run
`4975c4fe04eb413ba0d1766936284012`. These addresses are private tunnel endpoints,
not team-wide public URLs. The private runtime has no public-serving authentication layer.

From a **current LizardMorph main checkout** on the runtime host, the
coordinator can use:

```sh
docker compose -f deploy/mlflow-pilot/compose.yaml ps
docker compose -f deploy/mlflow-pilot/compose.yaml logs --tail=100
docker compose -f deploy/mlflow-pilot/compose.yaml up -d --build
docker compose -f deploy/mlflow-pilot/compose.yaml stop
```

Starting the stack does not require registering its models again. Persistent
volumes retain registry/artifacts, models, work data, and sessions. Restart
policies and health checks are configured; the runtime uses bounded inference
caches and a single worker.

For a consistent backup, the documented procedure is to stop this Compose
project, archive all four named volumes (`registry`, `models`, `work`, `sessions`)
together, retain source/image IDs, and restart. Restore into a separate project
and verify model hashes and inference. **Backup/restore was not exercised by
the November 9 E2E run**; it remains an operating acceptance task. Avoid
`down -v` when preserving the runtime's data.

## Next team milestones

1. Albert shares one complete newer checkpoint package through `incoming/axu39`.
2. Use the integrated code from both main branches and record the exact compatible
   pair of source revisions in every release receipt. The older
   [LizardMorph PR #18](https://github.com/Human-Augment-Analytics/LizardMorph/pull/18)
   targeted an earlier architecture; the current complete-bundle implementation
   supersedes that integration path.
3. Name the research/evaluation and runtime owners, agree on an independently
   held-out benchmark/thresholds, and qualify the next model through the same
   package/import/export path.
4. Establish shared runtime access and test backup/restore before expanding use.

The coordinator should keep the PACE handoff README
and template instructions aligned with it. Preserve experiment receipts as
historical evidence rather than updating them to describe later code or models.
