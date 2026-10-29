"""Synthetic filesystem checks; no images, model dependencies, or databases are needed."""

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.preprocessing import create_train_val_split as splitter


class TrainValSplitTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.images = self.root / "source-images"
        self.labels = self.root / "source-labels"
        self.output = self.root / "dataset"
        self.images.mkdir()
        self.labels.mkdir()
        for name in ("z.jpg", "b_flipud.JPG", "a.jpg", "b.jpg", "a_flipud.png"):
            (self.images / name).write_text(f"image: {name}\n")
            (self.labels / f"{Path(name).stem}.txt").write_text(f"label: {name}\n")
        # Neither unmatched files nor unsupported image extensions enter a split.
        (self.images / "orphan.jpg").write_text("no label")
        (self.images / "ignored.gif").write_text("unsupported")
        (self.labels / "ignored.txt").write_text("unsupported")
        (self.labels / "label-only.txt").write_text("no image")

    def split(self, *, copy, train_ratio=0.6, group_by_suffix=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return splitter.split_and_materialize(
                self.images,
                self.labels,
                self.output,
                train_ratio,
                42,
                copy,
                (".jpg", ".png"),
                group_by_suffix,
            )

    def assert_split(self, split_name, names, *, copy):
        expected_paths = []
        for folder, source_dir in (("images", self.images), ("labels", self.labels)):
            destination = self.output / folder / split_name
            expected_names = names if folder == "images" else [f"{Path(name).stem}.txt" for name in names]
            if not names:
                self.assertFalse(destination.exists())
                continue
            self.assertEqual(sorted(path.name for path in destination.iterdir()), sorted(expected_names))
            for name in expected_names:
                source = source_dir / name
                materialized = destination / name
                self.assertEqual(materialized.read_bytes(), source.read_bytes())
                self.assertEqual(materialized.is_symlink(), not copy)
                if not copy:
                    self.assertFalse(Path(os.readlink(materialized)).is_absolute())
                    self.assertEqual(materialized.resolve(), source.resolve())
                expected_paths.append(materialized.relative_to(self.output))
        manifest = self.output / "splits" / f"{split_name}.txt"
        self.assertEqual(manifest.read_text(), "".join(f"{(self.images / name).resolve()}\n" for name in names))
        expected_paths.append(manifest.relative_to(self.output))
        return expected_paths

    def check_dataset(self, train, val, *, copy):
        expected = self.assert_split("train", train, copy=copy) + self.assert_split("val", val, copy=copy)
        actual = [path.relative_to(self.output) for path in self.output.rglob("*") if path.is_file()]
        self.assertEqual(sorted(actual), sorted(expected))

    def test_seeded_split_preserves_pair_and_manifest_order(self):
        train = ["b_flipud.JPG", "a_flipud.png", "b.jpg"]
        val = ["z.jpg", "a.jpg"]
        for copy in (True, False):
            with self.subTest(copy=copy):
                events = []
                materialize = splitter.make_link_or_copy
                write_lists = splitter.write_split_lists

                def record_file(source, destination, copy):
                    events.append(str(destination.relative_to(self.output)))
                    materialize(source, destination, copy)

                def record_lists(*args):
                    events.append("manifests")
                    write_lists(*args)

                with patch.object(splitter, "make_link_or_copy", side_effect=record_file):
                    with patch.object(splitter, "write_split_lists", side_effect=record_lists):
                        self.assertEqual(self.split(copy=copy), (3, 2))
                expected_events = []
                for split_name, names in (("train", train), ("val", val)):
                    for name in names:
                        expected_events.extend(
                            (f"images/{split_name}/{name}", f"labels/{split_name}/{Path(name).stem}.txt")
                        )
                self.assertEqual(events, expected_events + ["manifests"])
                self.check_dataset(train, val, copy=copy)

    def test_augmented_pairs_stay_together_in_seeded_group_order(self):
        for copy in (True, False):
            with self.subTest(copy=copy):
                self.assertEqual(self.split(copy=copy, train_ratio=0.5, group_by_suffix="_flipud"), (2, 3))
                self.check_dataset(["b.jpg", "b_flipud.JPG"], ["a.jpg", "a_flipud.png", "z.jpg"], copy=copy)

    def test_empty_split_has_an_empty_manifest_and_no_materialized_directory(self):
        all_names = ["b_flipud.JPG", "a_flipud.png", "b.jpg", "z.jpg", "a.jpg"]
        for ratio, train, val in ((0.0, [], all_names), (1.0, all_names, [])):
            with self.subTest(train_ratio=ratio):
                self.output = self.root / f"dataset-{ratio}"
                self.assertEqual(self.split(copy=True, train_ratio=ratio), (len(train), len(val)))
                self.check_dataset(train, val, copy=True)

    def test_rerun_replaces_files_directories_dangling_links_and_manifests(self):
        self.split(copy=True)
        image = self.output / "images/train/b_flipud.JPG"
        image.unlink()
        image.symlink_to("missing.jpg")
        label = self.output / "labels/train/b_flipud.txt"
        label.unlink()
        label.mkdir()
        (label / "stale.txt").write_text("old output")
        (self.output / "splits/train.txt").write_text("stale manifest\n")
        for copy in (False, True):
            with self.subTest(copy=copy):
                self.assertEqual(self.split(copy=copy), (3, 2))
                self.check_dataset(["b_flipud.JPG", "a_flipud.png", "b.jpg"], ["z.jpg", "a.jpg"], copy=copy)

    def test_relative_link_failure_falls_back_to_source_path(self):
        source = self.images / "a.jpg"
        destination = self.output / "fallback.jpg"
        original = Path.symlink_to
        targets = []

        def fail_relative_link(path, target, *args, **kwargs):
            targets.append(target)
            if not Path(target).is_absolute():
                raise OSError("relative link unavailable")
            return original(path, target, *args, **kwargs)

        with patch.object(Path, "symlink_to", new=fail_relative_link):
            splitter.make_link_or_copy(source, destination, copy=False)
        self.assertEqual(targets, [os.path.relpath(source, destination.parent), source])
        self.assertEqual(destination.resolve(), source.resolve())

    def test_no_pairs_fails_without_creating_output(self):
        empty_images = self.root / "empty-images"
        empty_images.mkdir()
        (empty_images / "unmatched.jpg").write_text("no label")
        self.images = empty_images
        with self.assertRaisesRegex(RuntimeError, r"No \(image, label\) pairs found"):
            self.split(copy=True)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
