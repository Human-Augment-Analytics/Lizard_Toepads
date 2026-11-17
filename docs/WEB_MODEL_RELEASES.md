# Release a model to the production website

Training, release publication, and successful deployment are different states.
The production website keeps its URL. A new release changes the pinned default
model used by new toepad analyses only after Haag independently validates and
smoke-tests it. Existing analyses retain their original model identity.

## Eligibility

The web release channel is `web-production`. It accepts the existing H7
six-class ONNX detector's dual-pass protocol with a **full four-digit dlib
pipeline**: `up_finger`, `up_toe`, `bot_finger`, and `bot_toe`, nine uniquely
named landmarks each, using rectify-512 preprocessing. Detectors must be ONNX;
predictors must be `.dat` dlib files. ViT/ONNX coordinate predictors and partial
bottom-digit research bundles are rejected, even if somebody adds a web alias.

Before packaging/registering the bundle, its provenance must explicitly contain:

```json
{
  "deployment_targets": ["web"],
  "evaluation_status": "heldout-evaluated",
  "source_commit": "<full 40-character source commit SHA>"
}
```

These are assertions that must be backed by the actual evaluation. Relabeling an
unevaluated bundle does not qualify it scientifically. The earlier research demo
bundles intentionally do not qualify and are not promoted by this workflow.

The operator supplies `evaluation.json`, format `lizard-web-evaluation-v1`, with:

- `approved: true` and an accountable, nonempty `reviewer`.
- `benchmark_id`, `source_commit`, and `dataset_sha256` identifying the frozen
  held-out benchmark. Source commit must match bundle provenance.
- `bundle_sha256` identifying the **exact registered archive** that was evaluated.
- Nonempty `metrics`, mapping metric names to measured finite numbers.
- Nonempty `criteria`, each containing `metric`, `operator` (`>=` or `<=`), and
  numeric `threshold`. Every criterion must pass.

Choose thresholds and benchmark coverage before comparing candidates. This
workflow verifies the recorded checks; it does not invent thresholds, infer
scientific approval from training completion, or authenticate the reviewer's
name. Access to GitHub release/workflow publication controls who can release.

## Prepare a draft from MLflow

Use MLflow 2.22.4 and an already authenticated GitHub CLI session. The helper
prefers `gh-axi` for release creation when installed and uses `gh` for the raw
read-only tag lookup, which the wrapper does not expose. It does not read `.env`
files, print tokens, or accept credentials as command-line arguments. Configure
MLflow authentication using the usual client environment outside the command.

Register the complete bundle with `register_bundle.py`. Select an explicit
numeric version or assign its `web-candidate` alias in MLflow, then run:

```sh
python scripts/deployment/stage_web_release.py \
  --tracking-uri https://haag-1.cc.gatech.edu/mlflow \
  --name YOUR_REGISTERED_MODEL --alias web-candidate \
  --tag model/toepad-web/v1.3.0 \
  --evaluation /path/to/actual-evaluation.json \
  --output /path/to/new-empty-release-directory \
  --target FULL_EXISTING_COMMIT_SHA_IN_THIS_REPOSITORY
```

Replace `--alias web-candidate` with `--version 7` to pin an explicit version.
The alias is resolved once; the archive, metadata, and evaluation then pin the
numeric version and digest. The registry source must be exactly that run's
`runs:/<run_id>/bundle.zip`, preventing a registered arbitrary URL from becoming
an implicit download source. A matching registry `bundle_sha256` tag is required.

`--prepare-only` writes and validates the three assets without touching GitHub.
The final draft command refuses existing tags and never replaces release assets.
An interrupted upload may leave a partial draft: inspect it manually and choose
a fresh version rather than assuming a partially uploaded release is deployable.

The staged draft contains **only** `bundle.zip`, `web-release.json`, and
`evaluation.json`. Do not upload raw specimen images. Haag's smoke image stays
on its local filesystem. Evaluation metadata will become visible with the release;
review it for sensitive dataset identifiers before publication.

## Release to web

After this code is present on the repository's default branch, open GitHub
Actions → **Release model to web** → **Run workflow**, and enter the staged tag.
This is the explicit release action; changing an MLflow alias by itself does not
publish a release.

The workflow uses trusted default-branch validation code, requires a draft with
exactly the three assets, validates hashes, archive paths/inventory, the full web
runtime contract, and every approved evaluation criterion, then checks that the
draft did not change during validation. Only then does it publish. It does not
mark the release as GitHub's global `latest`, since model and software channels
must not be confused. Enable GitHub immutable releases in repository settings to
protect published tags and assets against later changes.

The manifest is `lizard-web-release-v1`, schema version 1, with:

- Fixed `release_tag` under `model/toepad-web/vMAJOR.MINOR.PATCH` and channel
  `web-production`.
- Archive filename, size and SHA256; exact MLflow model name, version and run ID.
- Compatibility `{target: web, runtime: onnx-dlib, contract: toepad-full-v1}`.
- Approved evaluation filename and SHA256.

Haag discovers only this channel, independently validates it, loads the actual
models, runs a configured local smoke image, installs into an immutable model
directory, and atomically changes its default pointer. A failed deployment keeps
the previous default and records the failure. Only successful activation should
update MLflow's `web-production` alias/deployment receipt; publication alone must
never claim deployment. The deployment consumer lives in the LizardMorph repo.

## Rollback and validation

Rollback selects a previous installed bundle. Never overwrite an old release's
assets, move its version tag, or redownload a mutable `latest` URL as rollback.
Keep the previous deployment receipt and model directory.

Pure contract tests (no database or model execution):

```sh
python -m pytest tests/test_web_release.py tests/test_pipeline_bundles.py
python scripts/deployment/web_release.py --directory RELEASE_DIR \
  --expected-tag model/toepad-web/v1.3.0
```

Database-backed integration tests run in isolated Docker on the configured test
host against the exact reviewed checkout. Real loader/smoke acceptance must cover
the deployment consumer separately. Passing archive validation alone does not
prove a model's accuracy, loader compatibility, or successful activation on Haag.
