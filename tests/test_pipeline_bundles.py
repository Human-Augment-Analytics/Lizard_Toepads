import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

module_path = Path(__file__).parents[1] / 'scripts/deployment/register_bundle.py'
spec = importlib.util.spec_from_file_location('register_bundle', module_path)
producer = importlib.util.module_from_spec(spec); spec.loader.exec_module(producer)


def write_spec(directory, predictor='landmarks.dat'):
    (directory / 'landmarks.dat').write_bytes(b'fixture')
    (directory / 'bundle.json').write_text(json.dumps({'format': 'lizard-pipeline-v1',
        'provenance': {'source_commit': 'abc', 'evaluation_status': 'not-evaluated'},
        'pipeline': {'detector': {'artifact': ''}, 'classes': [{'predictor': predictor}]}}))


def test_same_inputs_produce_identical_package_with_declared_hashes(tmp_path):
    write_spec(tmp_path)
    first, second = tmp_path / 'a.zip', tmp_path / 'b.zip'
    manifest = producer.package_bundle(tmp_path, first)
    producer.package_bundle(tmp_path, second)
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        content = archive.read('landmarks.dat')
        assert manifest['artifacts']['landmarks.dat']['sha256'] == hashlib.sha256(content).hexdigest()
        assert json.loads(archive.read('bundle.json')) == manifest


@pytest.mark.parametrize('path', ['../outside.dat', 'C:/outside.dat', 'C:outside.dat', 'NUL.dat', 'bad.'])
def test_nonportable_artifact_paths_are_rejected(tmp_path, path):
    write_spec(tmp_path, path)
    with pytest.raises(ValueError, match='Invalid artifact'):
        producer.package_bundle(tmp_path, tmp_path / 'bundle.zip')


def test_case_colliding_artifact_inventory_is_rejected(tmp_path):
    write_spec(tmp_path)
    path = tmp_path / 'bundle.json'
    spec = json.loads(path.read_text())
    spec['pipeline']['classes'].append({'predictor': 'LANDMARKS.dat'})
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='collide'):
        producer.package_bundle(tmp_path, tmp_path / 'bundle.zip')
