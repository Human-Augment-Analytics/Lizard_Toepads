#!/usr/bin/env python3
"""Package a research pipeline directory and register an explicit MLflow version.

Input bundle.json: format, pipeline (AutoMorph Manifest), provenance. Artifacts
are the pipeline's relative detector/predictor paths; no implicit dependencies.
No alias/default promotion occurs. Run with mlflow==2.22.4.
"""
import argparse
from contextlib import redirect_stdout
import sys
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import tempfile
import unicodedata
import zipfile

WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {f"{prefix}{n}" for prefix in ("COM", "LPT") for n in range(1, 10)}


def package_bundle(directory, output):
    root = Path(directory).resolve()
    spec = json.loads((root / 'bundle.json').read_text())
    if spec.get('format') != 'lizard-pipeline-v1':
        raise ValueError('Unsupported bundle format')
    if not spec.get('provenance', {}).get('source_commit') or not spec['provenance'].get('evaluation_status'):
        raise ValueError('Source commit and evaluation status are required')
    pipeline = spec['pipeline']
    references = {c['predictor'] for c in pipeline['classes'] if c.get('predictor')}
    if pipeline['detector'].get('artifact'):
        references.add(pipeline['detector']['artifact'])
    if not references:
        raise ValueError('Pipeline has no artifacts')
    normalized = [unicodedata.normalize('NFC', ref).casefold() for ref in references]
    if len(set(normalized)) != len(normalized) or 'bundle.json' in normalized:
        raise ValueError('Artifact paths collide on a portable filesystem')
    spec['artifacts'] = {}
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        def write_member(name, content):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)
        for ref in sorted(references):
            path = PurePosixPath(ref)
            source = root / ref
            windows = PureWindowsPath(ref)
            reserved = any(part.split(".", 1)[0].upper() in WINDOWS_RESERVED
                           or part.endswith((" ", ".")) for part in path.parts)
            if (path.is_absolute() or windows.drive or windows.root or reserved
                    or ':' in ref or '..' in path.parts or '\\' in ref or str(path) != ref
                    or not source.is_file() or source.resolve() != source.absolute()):
                raise ValueError(f'Invalid artifact: {ref}')
            content = source.read_bytes()
            spec['artifacts'][ref] = {'sha256': hashlib.sha256(content).hexdigest(), 'size': len(content)}
            write_member(ref, content)
        write_member('bundle.json', json.dumps(spec, sort_keys=True, indent=2))
    return spec


def register_bundle(directory, name, tracking_uri):
    import mlflow
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment('lizard-pipeline-bundles')
    with tempfile.TemporaryDirectory() as work:
        archive_path = Path(work) / 'bundle.zip'
        spec = package_bundle(directory, archive_path)
        digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tags({'source_commit': spec['provenance']['source_commit'],
                             'evaluation_status': spec['provenance']['evaluation_status'],
                             'bundle_sha256': digest})
            mlflow.log_dict(spec, 'bundle-manifest.json')
            mlflow.log_artifact(str(archive_path))
            client = mlflow.MlflowClient()
            try:
                client.get_registered_model(name)
            except mlflow.exceptions.MlflowException as exc:
                if exc.error_code != 'RESOURCE_DOES_NOT_EXIST':
                    raise
                client.create_registered_model(name)
            version = client.create_model_version(
                name, source=f'runs:/{run.info.run_id}/bundle.zip', run_id=run.info.run_id,
                tags={'bundle_format': spec['format'], 'bundle_sha256': digest,
                      'evaluation_status': spec['provenance']['evaluation_status']})
            return {'name': name, 'version': str(version.version), 'run_id': run.info.run_id,
                    'bundle_sha256': digest, 'promotion': 'none'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-dir', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--tracking-uri', required=True)
    args = parser.parse_args()
    with redirect_stdout(sys.stderr):
        result = register_bundle(args.bundle_dir, args.name, args.tracking_uri)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
