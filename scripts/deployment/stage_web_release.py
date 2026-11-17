#!/usr/bin/env python3
"""Stage an approved, explicit MLflow version as a NEW GitHub draft release.

Requires an existing authenticated GitHub CLI session and mlflow==2.22.4.
Never publishes, promotes an alias, replaces assets, or invents evaluation data.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from urllib.parse import quote

from web_release import (file_sha256, load_json, validate_evaluation, validate_release,
                         validate_release_tag, validate_web_bundle)

DEFAULT_REPO = "Human-Augment-Analytics/Lizard_Toepads"


def github_cli():
    # The ergonomic wrapper does not expose raw API requests, so those use gh.
    gh = shutil.which("gh")
    if not gh:
        raise RuntimeError("An already authenticated GitHub CLI (gh) is required")
    return gh, shutil.which("gh-axi") or gh


def ensure_new_tag(repo, tag, gh):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("Expected owner/repository")
    endpoint = f"repos/{repo}/git/ref/tags/{quote(tag, safe='')}"
    response = subprocess.run([gh, "api", "--include", endpoint], capture_output=True, text=True)
    statuses = re.findall(r"^HTTP/\S+\s+(\d+)", response.stdout, re.M)
    status = int(statuses[-1]) if statuses else None
    if status == 404:
        return
    if status == 200:
        raise ValueError("Release tag already exists; choose a new version")
    raise RuntimeError("Cannot establish that the release tag is new; check existing CLI authentication/network")


def prepare_assets(*, tracking_uri, name, version, alias, tag, evaluation_path, destination):
    import mlflow

    validate_release_tag(tag)
    if bool(version) == bool(alias):
        raise ValueError("Choose exactly one explicit numeric version or web-candidate alias")
    if version and re.fullmatch(r"[1-9][0-9]*", version) is None:
        raise ValueError("A positive numeric version is required")
    if alias and alias != "web-candidate":
        raise ValueError("Only the explicit web-candidate alias can be staged")
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    selected = (client.get_model_version(name, version) if version else
                client.get_model_version_by_alias(name, alias))
    # Resolve the mutable alias ONCE; all following operations pin this number.
    selected_version = str(selected.version)
    source = getattr(selected, "source", "")
    run_id = getattr(selected, "run_id", "")
    if (re.fullmatch(r"[0-9a-f]{32}", run_id or "") is None or
            source != f"runs:/{run_id}/bundle.zip"):
        raise ValueError("Expected the registered full bundle from its exact MLflow run")
    expected_sha = (getattr(selected, "tags", {}) or {}).get("bundle_sha256")
    if not isinstance(expected_sha, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None:
        raise ValueError("The registered version must pin its bundle SHA256")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError("Output directory must be empty")
    with tempfile.TemporaryDirectory(prefix="web-release-download-") as temporary:
        previous = mlflow.get_tracking_uri()
        try:
            mlflow.set_tracking_uri(tracking_uri)
            downloaded = Path(mlflow.artifacts.download_artifacts(artifact_uri=source, dst_path=temporary))
        finally:
            mlflow.set_tracking_uri(previous)
        if downloaded.is_symlink() or not downloaded.is_file() or file_sha256(downloaded) != expected_sha:
            raise ValueError("Registered bundle download checksum mismatch")
        bundle = validate_web_bundle(downloaded)
        report = load_json(evaluation_path)
        validate_evaluation(report, bundle, expected_sha)
        shutil.copyfile(downloaded, destination / "bundle.zip")
        shutil.copyfile(evaluation_path, destination / "evaluation.json")
    metadata = {
        "format": "lizard-web-release-v1", "schema_version": 1,
        "channel": "web-production", "release_tag": tag,
        "bundle": {"asset": "bundle.zip", "sha256": expected_sha,
                   "size": (destination / "bundle.zip").stat().st_size},
        "registry": {"name": name, "version": selected_version, "run_id": run_id},
        "compatibility": {"target": "web", "runtime": "onnx-dlib", "contract": "toepad-full-v1"},
        "evaluation": {"asset": "evaluation.json", "sha256": file_sha256(destination / "evaluation.json"),
                       "status": "approved"},
    }
    (destination / "web-release.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    validate_release(destination, expected_tag=tag)
    return metadata


def stage_draft(directory, metadata, repo, target):
    if re.fullmatch(r"[0-9a-f]{40}", target) is None:
        raise ValueError("--target must be an existing full commit SHA in the release repository")
    gh, release_cli = github_cli()
    tag = validate_release_tag(metadata["release_tag"])
    ensure_new_tag(repo, tag, gh)
    # Creating a release fails if another operator wins the same tag race. Never
    # use release upload --clobber or edit an existing release in this command.
    notes = (f"Approved web candidate: {metadata['registry']['name']} "
             f"v{metadata['registry']['version']}\n\n"
             f"Bundle SHA256: {metadata['bundle']['sha256']}\n\n"
             "Draft only. Run the Release model to web GitHub workflow to validate and publish. "
             "Publication does not establish that Haag deployed successfully.\n")
    with tempfile.TemporaryDirectory(prefix="web-release-notes-") as temporary:
        body = Path(temporary) / "notes.md"
        body.write_text(notes)
        subprocess.run([release_cli, "release", "create", tag, "--repo", repo,
                        "--draft", "--target", target, "--title", tag,
                        "--notes-file", str(body),
                        *[str(Path(directory) / name) for name in
                          ("bundle.zip", "web-release.json", "evaluation.json")]], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--name", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--version")
    selection.add_argument("--alias", choices=["web-candidate"])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--target", help="Existing full repository commit SHA; required to stage GitHub draft")
    parser.add_argument("--prepare-only", action="store_true", help="Validate/write assets without creating a GitHub draft")
    args = parser.parse_args()
    if not args.prepare_only and not args.target:
        parser.error("--target is required unless --prepare-only is selected")
    metadata = prepare_assets(tracking_uri=args.tracking_uri, name=args.name,
                              version=args.version, alias=args.alias, tag=args.tag,
                              evaluation_path=args.evaluation, destination=args.output)
    if not args.prepare_only:
        stage_draft(args.output, metadata, args.repo, args.target)
    print(json.dumps({"status": "prepared" if args.prepare_only else "draft-staged",
                      "release_tag": args.tag, "registry": metadata["registry"],
                      "published": False, "deployed": False}, indent=2))


if __name__ == "__main__":
    main()
