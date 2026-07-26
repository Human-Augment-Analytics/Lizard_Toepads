"""Unit tests for split utilities."""

import json
import tempfile
from pathlib import Path

import pytest

from landmarking.common.split_utils import (
    generate_split,
    sample_fraction,
    write_split,
)


class TestSampleFraction:
    def test_returns_correct_count(self):
        paths = [f"file_{i}.pt" for i in range(100)]
        result = sample_fraction(paths, 0.2, seed=42)
        assert len(result) == 20

    def test_deterministic(self):
        paths = [f"file_{i}.pt" for i in range(100)]
        r1 = sample_fraction(paths, 0.3, seed=123)
        r2 = sample_fraction(paths, 0.3, seed=123)
        assert r1 == r2

    def test_different_seeds_give_different_results(self):
        paths = [f"file_{i}.pt" for i in range(100)]
        r1 = sample_fraction(paths, 0.3, seed=1)
        r2 = sample_fraction(paths, 0.3, seed=2)
        assert r1 != r2

    def test_invalid_fraction_raises(self):
        with pytest.raises(ValueError):
            sample_fraction(["a", "b"], 0.0, seed=1)
        with pytest.raises(ValueError):
            sample_fraction(["a", "b"], 1.5, seed=1)


class TestWriteSplit:
    def test_writes_valid_json(self, tmp_path):
        out = str(tmp_path / "split.json")
        write_split(["a.pt", "b.pt"], ["c.pt"], ["d.pt"], out)
        with open(out) as f:
            data = json.load(f)
        assert "train" in data
        assert "val" in data
        assert "test" in data
        assert data["train"] == ["a.pt", "b.pt"]


class TestGenerateSplit:
    def test_with_directory(self, tmp_path):
        # Create fake .pt files
        for i in range(20):
            (tmp_path / f"sample_{i:03d}.pt").touch()

        split = generate_split(
            data_dir=str(tmp_path),
            fractions={"train": 0.7, "val": 0.15, "test": 0.15},
            seed=42,
        )
        assert len(split["train"]) == 14  # floor(20*0.7)
        assert len(split["val"]) == 3     # floor(20*0.15)
        assert len(split["test"]) == 3    # floor(20*0.15)

        # No overlap
        all_items = split["train"] + split["val"] + split["test"]
        assert len(set(all_items)) == len(all_items)

    def test_determinism(self, tmp_path):
        for i in range(10):
            (tmp_path / f"s_{i}.pt").touch()

        s1 = generate_split(str(tmp_path), {"train": 0.6, "val": 0.2, "test": 0.2}, seed=99)
        s2 = generate_split(str(tmp_path), {"train": 0.6, "val": 0.2, "test": 0.2}, seed=99)
        assert s1 == s2

    def test_writes_json_when_output_path_given(self, tmp_path):
        for i in range(10):
            (tmp_path / f"s_{i}.pt").touch()

        out_path = str(tmp_path / "out" / "split.json")
        generate_split(str(tmp_path), {"train": 0.8, "val": 0.1, "test": 0.1}, output_path=out_path)
        assert Path(out_path).exists()

        with open(out_path) as f:
            data = json.load(f)
        assert "train" in data
