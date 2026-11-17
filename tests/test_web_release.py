import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

SCRIPTS = Path(__file__).parents[1] / 'scripts' / 'deployment'
sys.path.insert(0, str(SCRIPTS))
import web_release as release
import stage_web_release as stage


def write_candidate(root, *, mutate_bundle=None, mutate_report=None):
    root.mkdir(exist_ok=True)
    artifacts = {'detector.onnx': b'nonexecuted detector fixture', 'landmarks.dat': b'nonexecuted dlib fixture'}
    bundle = {
        'format': 'lizard-pipeline-v1',
        'provenance': {'source_commit': 'a' * 40, 'deployment_targets': ['web'],
                       'evaluation_status': 'heldout-evaluated'},
        'pipeline': {
            'schema_version': 1, 'id': 'test', 'name': 'test', 'description': 'contract fixture only',
            'detector': {'artifact': 'detector.onnx', 'geometry': 'obb', 'inference_protocol': 'toepad-dual-pass'},
            'classes': [{'id': i, 'name': name, 'landmark_schema': 'digit', 'predictor': 'landmarks.dat',
                         'predictor_type': 'dlib', 'preprocessing': 'toepad-rectify-512'}
                        for i, name in enumerate(['up_finger', 'up_toe', 'bot_finger', 'bot_toe'])],
            'landmark_schemas': {'digit': {'points': [str(i) for i in range(9)]}},
        },
        'artifacts': {name: {'size': len(value), 'sha256': __import__('hashlib').sha256(value).hexdigest()}
                      for name, value in artifacts.items()},
    }
    if mutate_bundle:
        mutate_bundle(bundle)
    with zipfile.ZipFile(root / 'bundle.zip', 'w') as archive:
        archive.writestr('bundle.json', json.dumps(bundle))
        for name, value in artifacts.items():
            archive.writestr(name, value)
    digest = release.file_sha256(root / 'bundle.zip')
    report = {'format': 'lizard-web-evaluation-v1', 'approved': True, 'reviewer': 'test reviewer',
              'benchmark_id': 'synthetic-contract-test-only', 'source_commit': 'a' * 40,
              'dataset_sha256': 'b' * 64, 'bundle_sha256': digest,
              'metrics': {'error': 0.04, 'recall': 0.98},
              'criteria': [{'metric': 'error', 'operator': '<=', 'threshold': 0.05},
                           {'metric': 'recall', 'operator': '>=', 'threshold': 0.95}]}
    if mutate_report:
        mutate_report(report)
    (root / 'evaluation.json').write_text(json.dumps(report))
    metadata = {'format': 'lizard-web-release-v1', 'schema_version': 1,
                'channel': 'web-production', 'release_tag': 'model/toepad-web/v1.2.3',
                'bundle': {'asset': 'bundle.zip', 'sha256': digest, 'size': (root / 'bundle.zip').stat().st_size},
                'registry': {'name': 'full-toepad', 'version': '2', 'run_id': 'c' * 32},
                'compatibility': {'target': 'web', 'runtime': 'onnx-dlib', 'contract': 'toepad-full-v1'},
                'evaluation': {'asset': 'evaluation.json', 'status': 'approved',
                               'sha256': release.file_sha256(root / 'evaluation.json')}}
    (root / 'web-release.json').write_text(json.dumps(metadata))
    return metadata


def test_valid_release_checks_both_metric_directions_and_exact_tag(tmp_path):
    metadata = write_candidate(tmp_path)
    assert release.validate_release(tmp_path, metadata['release_tag']) == metadata
    with pytest.raises(ValueError, match='tag mismatch'):
        release.validate_release(tmp_path, 'model/toepad-web/v1.2.4')


@pytest.mark.parametrize('mutate', [
    lambda b: b['provenance'].update(deployment_targets=['desktop']),
    lambda b: b['provenance'].update(evaluation_status='research-pilot-no-heldout-evaluation'),
    lambda b: b['pipeline']['classes'].pop(),
    lambda b: b['pipeline']['classes'][0].update(predictor_type='onnx-coordinates'),
    lambda b: b['pipeline']['detector'].update(inference_protocol='generic-obb'),
    lambda b: b['pipeline']['landmark_schemas']['digit'].update(points=['same'] * 9),
    lambda b: b['artifacts']['detector.onnx'].update(sha256='0' * 64),
])
def test_research_vit_partial_or_corrupt_bundle_is_rejected(tmp_path, mutate):
    write_candidate(tmp_path, mutate_bundle=mutate)
    with pytest.raises(ValueError):
        release.validate_release(tmp_path)


@pytest.mark.parametrize('mutate', [
    lambda r: r.update(approved=False),
    lambda r: r.update(bundle_sha256='0' * 64),
    lambda r: r.update(source_commit='d' * 40),
    lambda r: r.update(reviewer=''),
    lambda r: r.update(criteria=[]),
    lambda r: r['metrics'].update(error=0.2),
    lambda r: r['criteria'][0].update(operator='__import__("os")'),
    lambda r: r['metrics'].update(recall=True),
    lambda r: r['metrics'].update(recall=float('nan')),
])
def test_unapproved_wrong_bundle_or_failed_evaluation_is_rejected(tmp_path, mutate):
    write_candidate(tmp_path, mutate_report=mutate)
    with pytest.raises(ValueError):
        release.validate_release(tmp_path)


def test_tampered_evaluation_asset_is_rejected(tmp_path):
    write_candidate(tmp_path)
    (tmp_path / 'evaluation.json').write_text('{}')
    with pytest.raises(ValueError, match='Evaluation asset checksum'):
        release.validate_release(tmp_path)


@pytest.mark.parametrize('tag', ['v1.2.3', '../v1.2.3', '-x', 'model/toepad-web/v1.2.3;touch x',
                                 'model/toepad-web/v01.2.3', 'model/toepad-web/v1.2.3-beta'])
def test_only_versioned_web_tags_are_accepted(tag):
    with pytest.raises(ValueError):
        release.validate_release_tag(tag)


def test_case_colliding_or_path_traversal_zip_rejected(tmp_path):
    write_candidate(tmp_path)
    for extra in ['../outside.dat', 'LANDMARKS.DAT']:
        path = tmp_path / (extra.replace('/', '_') + '.zip')
        with zipfile.ZipFile(tmp_path / 'bundle.zip') as original, zipfile.ZipFile(path, 'w') as changed:
            for name in original.namelist():
                changed.writestr(name, original.read(name))
            changed.writestr(extra, b'bad')
        with pytest.raises(ValueError):
            release.validate_web_bundle(path)


def test_alias_staging_pins_version_and_creates_only_three_assets(tmp_path, monkeypatch):
    metadata = write_candidate(tmp_path / 'candidate')
    selected = SimpleNamespace(version='2', run_id='c' * 32, source=f"runs:/{'c' * 32}/bundle.zip",
                               tags={'bundle_sha256': metadata['bundle']['sha256']})
    calls = []
    client = SimpleNamespace(get_model_version_by_alias=lambda name, alias: (calls.append((name, alias)) or selected))
    fake = SimpleNamespace(MlflowClient=lambda **kwargs: client, get_tracking_uri=lambda: 'before',
                           set_tracking_uri=lambda uri: None,
                           artifacts=SimpleNamespace(download_artifacts=lambda **kwargs: str(tmp_path / 'candidate/bundle.zip')))
    monkeypatch.setitem(sys.modules, 'mlflow', fake)
    result = stage.prepare_assets(tracking_uri='http://private-registry', name='full-toepad', version=None,
                                  alias='web-candidate', tag=metadata['release_tag'],
                                  evaluation_path=tmp_path / 'candidate/evaluation.json', destination=tmp_path / 'out')
    assert calls == [('full-toepad', 'web-candidate')]
    assert result['registry']['version'] == '2'
    assert {p.name for p in (tmp_path / 'out').iterdir()} == release.ASSETS
    assert release.validate_release(tmp_path / 'out') == result


@pytest.mark.parametrize('status,exists', [(200, True), (404, False), (403, None), (None, None)])
def test_tag_probe_fails_closed_on_auth_or_network_errors(monkeypatch, status, exists):
    response = SimpleNamespace(stdout=f'HTTP/2.0 {status}\n' if status else '', returncode=0)
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: response)
    if exists is False:
        stage.ensure_new_tag('owner/repo', 'model/toepad-web/v1.2.3', 'gh')
    else:
        with pytest.raises(ValueError if exists else RuntimeError):
            stage.ensure_new_tag('owner/repo', 'model/toepad-web/v1.2.3', 'gh')


def test_staging_uses_wrapper_draft_and_exact_files_without_clobber(tmp_path, monkeypatch):
    metadata = write_candidate(tmp_path)
    calls = []
    monkeypatch.setattr(stage, 'github_cli', lambda: ('gh', 'gh-axi'))
    monkeypatch.setattr(stage, 'ensure_new_tag', lambda *args: None)
    monkeypatch.setattr(subprocess, 'run', lambda args, **kwargs: calls.append(args))
    stage.stage_draft(tmp_path, metadata, 'owner/repo', 'd' * 40)
    assert len(calls) == 1 and calls[0][:3] == ['gh-axi', 'release', 'create']
    assert '--draft' in calls[0] and '--clobber' not in calls[0]
    assert all(str(tmp_path / name) in calls[0] for name in release.ASSETS)
