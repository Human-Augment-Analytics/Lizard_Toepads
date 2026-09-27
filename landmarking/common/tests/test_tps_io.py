"""Unit tests for consolidated TPS reading."""

import numpy as np
import pytest

from landmarking.common.tps_io import read_consolidated_tps


def write_tps(tmp_path, text: str, name: str = "sample.tps") -> str:
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def block(points, image=None, declared=None) -> str:
    declared = len(points) if declared is None else declared
    lines = [f"LM={declared}"]
    lines += [f"{x:.5f} {y:.5f}" for x, y in points]
    if image is not None:
        lines.append(f"IMAGE={image}")
    return "\n".join(lines)


NINE = [(float(i), float(2 * i)) for i in range(9)]


class TestReadConsolidatedTps:
    def test_reads_single_specimen(self, tmp_path):
        path = write_tps(tmp_path, block(NINE, image="1003.jpg") + "\n")

        specimens = read_consolidated_tps(path, expected_landmarks=9)

        assert len(specimens) == 1
        assert specimens[0]["landmarks"].shape == (9, 2)
        assert specimens[0]["image"] == "1003.jpg"
        assert specimens[0]["ruler_px"] is None
        np.testing.assert_allclose(specimens[0]["landmarks"], np.array(NINE))

    def test_reads_multiple_specimens(self, tmp_path):
        text = "\n".join(
            [
                block(NINE, image="a.jpg"),
                block([(x + 100, y + 100) for x, y in NINE], image="b.jpg"),
                block([(x + 200, y + 200) for x, y in NINE], image="c.jpg"),
            ]
        )
        path = write_tps(tmp_path, text + "\n")

        specimens = read_consolidated_tps(path, expected_landmarks=9)

        assert len(specimens) == 3
        assert [s["image"] for s in specimens] == ["a.jpg", "b.jpg", "c.jpg"]

    def test_final_specimen_without_image_line_is_kept(self, tmp_path):
        # The last block in a file may not be followed by IMAGE=; it must still
        # be flushed rather than silently dropped.
        text = block(NINE, image="a.jpg") + "\n" + block(NINE)
        path = write_tps(tmp_path, text + "\n")

        specimens = read_consolidated_tps(path, expected_landmarks=9)

        assert len(specimens) == 2
        assert specimens[1]["image"] is None

    def test_strips_leading_ruler_points(self, tmp_path):
        # LM=11 means 2 leading ruler points followed by the 9 anatomical ones,
        # matching the convention get_tps_coords encodes as skip = 2.
        ruler = [(0.0, 0.0), (3.0, 4.0)]
        path = write_tps(tmp_path, block(ruler + NINE, image="x.jpg") + "\n")

        specimens = read_consolidated_tps(path, expected_landmarks=9)

        assert len(specimens) == 1
        assert specimens[0]["landmarks"].shape == (9, 2)
        np.testing.assert_allclose(specimens[0]["landmarks"], np.array(NINE))
        assert specimens[0]["ruler_px"] == pytest.approx(5.0)

    def test_mixed_landmark_counts_in_one_file(self, tmp_path):
        ruler = [(0.0, 0.0), (0.0, 10.0)]
        text = "\n".join(
            [
                block(NINE, image="plain.jpg"),
                block(ruler + NINE, image="withruler.jpg"),
            ]
        )
        path = write_tps(tmp_path, text + "\n")

        specimens = read_consolidated_tps(path, expected_landmarks=9)

        assert len(specimens) == 2
        assert specimens[0]["ruler_px"] is None
        assert specimens[1]["ruler_px"] == pytest.approx(10.0)

    def test_skips_block_with_too_few_landmarks(self, tmp_path):
        text = "\n".join(
            [
                block(NINE[:5], image="short.jpg"),
                block(NINE, image="good.jpg"),
            ]
        )
        path = write_tps(tmp_path, text + "\n")

        specimens = read_consolidated_tps(path, expected_landmarks=9)

        assert len(specimens) == 1
        assert specimens[0]["image"] == "good.jpg"

    def test_skips_block_whose_points_do_not_match_declaration(self, tmp_path):
        # Declares 9 but supplies 7: truncated record, must not be half-read.
        text = block(NINE[:7], image="truncated.jpg", declared=9)
        path = write_tps(tmp_path, text + "\n" + block(NINE, image="ok.jpg") + "\n")

        specimens = read_consolidated_tps(path, expected_landmarks=9)

        assert len(specimens) == 1
        assert specimens[0]["image"] == "ok.jpg"

    def test_ignores_other_header_lines(self, tmp_path):
        text = "\n".join(
            [
                "LM=9",
                *[f"{x:.5f} {y:.5f}" for x, y in NINE],
                "ID=17",
                "SCALE=0.0254",
                "COMMENT=whatever",
                "IMAGE=z.jpg",
            ]
        )
        path = write_tps(tmp_path, text + "\n")

        specimens = read_consolidated_tps(path, expected_landmarks=9)

        assert len(specimens) == 1
        assert specimens[0]["image"] == "z.jpg"

    def test_blank_lines_tolerated(self, tmp_path):
        text = "\n\n" + block(NINE, image="a.jpg").replace("\n", "\n\n") + "\n\n"
        path = write_tps(tmp_path, text)

        specimens = read_consolidated_tps(path, expected_landmarks=9)

        assert len(specimens) == 1

    def test_empty_file_returns_empty_list(self, tmp_path):
        path = write_tps(tmp_path, "")
        assert read_consolidated_tps(path, expected_landmarks=9) == []

    def test_expected_landmarks_must_be_positive(self, tmp_path):
        path = write_tps(tmp_path, block(NINE, image="a.jpg") + "\n")
        with pytest.raises(ValueError, match="expected_landmarks must be positive"):
            read_consolidated_tps(path, expected_landmarks=0)

    def test_dtype_is_float64(self, tmp_path):
        path = write_tps(tmp_path, block(NINE, image="a.jpg") + "\n")
        specimens = read_consolidated_tps(path, expected_landmarks=9)
        assert specimens[0]["landmarks"].dtype == np.float64


class TestReExport:
    def test_tps_utils_reexports_same_function(self):
        # datasets.lizard.tps_utils re-exports the reader so all TPS I/O is
        # discoverable from one place, while the implementation stays torch-free
        # in common/. Skipped when torch is unavailable, since importing
        # landmarking.datasets pulls it in.
        pytest.importorskip("torch")
        from landmarking.datasets.lizard.tps_utils import (
            read_consolidated_tps as reexported,
        )

        assert reexported is read_consolidated_tps
