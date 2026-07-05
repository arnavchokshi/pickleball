from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.racketsport.generate_synthetic_court_keypoints import (
    SyntheticCourtGenerationConfig,
    generate_synthetic_court_corpus,
    sha256_file,
)
from scripts.racketsport.train_court_keypoint_heatmap import load_real_court_keypoint_labels
from threed.racketsport.court_calibration import project_planar_points
from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINTS
from threed.racketsport.court_synth_scenes import SCENARIO_NAMES


def _load_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_generate_synthetic_corpus_writes_trainer_compatible_schema(tmp_path: Path) -> None:
    out = tmp_path / "court_synthetic"
    manifest = generate_synthetic_court_corpus(
        SyntheticCourtGenerationConfig(
            out_dir=out,
            count=3,
            seed=1234,
            image_size=(320, 180),
            spot_check_count=2,
            generated_at_utc="2026-07-01T00:00:00+00:00",
        )
    )

    assert manifest["artifact_type"] == "synthetic_court_keypoint_corpus_manifest"
    assert manifest["status"] == "synthetic_training_ammunition_not_gate_evidence"
    assert manifest["seed"] == 1234
    assert manifest["sample_count"] == 3
    assert len(manifest["spot_check_overlays"]) == 2
    assert all((out / ref["path"]).is_file() for ref in manifest["spot_check_overlays"])

    rows = load_real_court_keypoint_labels(out)
    assert len(rows) == 3
    # CAL-R2 provenance fix: synthetic rows carry their own 'synthetic' status, distinct from
    # 'reviewed_static_camera_copy' (which is reserved for owner-approved copies of REAL human
    # review), so a gate can never silently count synthetic renders as any form of human
    # verification.
    assert {row["label_status"] for row in rows} == {"synthetic"}
    assert all(set(row["keypoints"]) == {point.name for point in PICKLEBALL_KEYPOINTS} for row in rows)

    first_label = _load_payload(out / "synthetic_court_000000" / "labels" / "court_keypoints.json")
    assert first_label["review"]["status"] == "reviewed"
    assert first_label["review"]["human_reviewed"] is False
    assert first_label["review"]["independent_reviewed_count"] == 0
    assert first_label["review"]["static_camera_copy_count"] == 0
    assert first_label["review"]["synthetic_count"] == 1
    assert first_label["annotation"]["items"][0]["status"] == "synthetic"
    assert first_label["annotation"]["items"][0]["net_keypoint_height_convention"] == "regulation_net_top"
    assert first_label["generation"]["net_keypoint_height_convention"] == "regulation_net_top"
    assert first_label["generation"]["keypoint_world_xyz_m"]["net_center"][2] > 0.0
    assert first_label["provenance"]["synthetic"] is True
    assert first_label["provenance"]["human_labels"] is False
    assert first_label["annotation"]["items"][0]["provenance"]["synthetic"] is True


def test_keypoints_reproject_through_generation_homography(tmp_path: Path) -> None:
    out = tmp_path / "court_synthetic"
    manifest = generate_synthetic_court_corpus(
        SyntheticCourtGenerationConfig(
            out_dir=out,
            count=1,
            seed=9876,
            image_size=(320, 180),
            spot_check_count=0,
            distortion_k1_range=(0.0, 0.0),
            generated_at_utc="2026-07-01T00:00:00+00:00",
        )
    )
    sample = manifest["samples"][0]
    payload = _load_payload(out / sample["label_path"])
    item = payload["annotation"]["items"][0]
    homography = payload["generation"]["world_to_image_homography"]

    world_xy = [point.world_xyz_m[:2] for point in PICKLEBALL_KEYPOINTS]
    projected = project_planar_points(homography, world_xy)

    for point, expected_xy in zip(PICKLEBALL_KEYPOINTS, projected, strict=True):
        if point.name.startswith("net_") or point.name == "net_center":
            continue
        assert item["keypoints"][point.name] == pytest.approx(expected_xy, abs=1e-5)

    line_checks = [
        ("near_left_corner", "near_baseline_center", "near_right_corner"),
        ("far_left_corner", "far_baseline_center", "far_right_corner"),
        ("near_nvz_left", "near_nvz_center", "near_nvz_right"),
        ("net_left_sideline", "net_center", "net_right_sideline"),
        ("far_nvz_left", "far_nvz_center", "far_nvz_right"),
    ]
    for left, center, right in line_checks:
        x1, y1 = item["keypoints"][left]
        x2, y2 = item["keypoints"][right]
        xc, yc = item["keypoints"][center]
        distance_num = abs((y2 - y1) * xc - (x2 - x1) * yc + x2 * y1 - y2 * x1)
        distance_den = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
        assert distance_num / distance_den <= 1e-5


def test_generation_is_deterministic_under_seed(tmp_path: Path) -> None:
    config_a = SyntheticCourtGenerationConfig(
        out_dir=tmp_path / "a",
        count=2,
        seed=42,
        image_size=(320, 180),
        spot_check_count=1,
        generated_at_utc="2026-07-01T00:00:00+00:00",
    )
    config_b = SyntheticCourtGenerationConfig(
        out_dir=tmp_path / "b",
        count=2,
        seed=42,
        image_size=(320, 180),
        spot_check_count=1,
        generated_at_utc="2026-07-01T00:00:00+00:00",
    )

    manifest_a = generate_synthetic_court_corpus(config_a)
    manifest_b = generate_synthetic_court_corpus(config_b)

    assert [sample["image_sha256"] for sample in manifest_a["samples"]] == [
        sample["image_sha256"] for sample in manifest_b["samples"]
    ]
    assert [
        _load_payload(config_a.out_dir / sample["label_path"])["annotation"]["items"][0]["keypoints"]
        for sample in manifest_a["samples"]
    ] == [
        _load_payload(config_b.out_dir / sample["label_path"])["annotation"]["items"][0]["keypoints"]
        for sample in manifest_b["samples"]
    ]
    assert sha256_file(config_a.out_dir / manifest_a["spot_check_overlays"][0]["path"]) == sha256_file(
        config_b.out_dir / manifest_b["spot_check_overlays"][0]["path"]
    )


def test_generator_covers_at_least_seven_scenario_families() -> None:
    assert len(SCENARIO_NAMES) >= 7


def test_scenarios_option_restricts_which_families_are_rendered(tmp_path: Path) -> None:
    out = tmp_path / "court_synthetic"
    manifest = generate_synthetic_court_corpus(
        SyntheticCourtGenerationConfig(
            out_dir=out,
            count=12,
            seed=555,
            image_size=(320, 180),
            spot_check_count=0,
            scenarios=("tennis_overlay", "adjacent_multi_court"),
            generated_at_utc="2026-07-05T00:00:00+00:00",
        )
    )
    assert set(manifest["scenario_counts"]) == {"tennis_overlay", "adjacent_multi_court"}
    assert sum(manifest["scenario_counts"].values()) == 12
    for sample in manifest["samples"]:
        assert sample["scenario"] in {"tennis_overlay", "adjacent_multi_court"}
        label = _load_payload(out / sample["label_path"])
        assert label["generation"]["scenario"] == sample["scenario"]


def test_scenario_weights_option_biases_the_mixture(tmp_path: Path) -> None:
    out = tmp_path / "court_synthetic"
    manifest = generate_synthetic_court_corpus(
        SyntheticCourtGenerationConfig(
            out_dir=out,
            count=40,
            seed=777,
            image_size=(320, 180),
            spot_check_count=0,
            scenario_weights={"portrait_phone": 5.0, "harsh_shadow": 1.0},
            generated_at_utc="2026-07-05T00:00:00+00:00",
        )
    )
    counts = manifest["scenario_counts"]
    assert set(counts) <= {"portrait_phone", "harsh_shadow"}
    assert counts.get("portrait_phone", 0) > counts.get("harsh_shadow", 0)


def test_unknown_scenario_name_fails_closed(tmp_path: Path) -> None:
    out = tmp_path / "court_synthetic"
    with pytest.raises(ValueError):
        generate_synthetic_court_corpus(
            SyntheticCourtGenerationConfig(out_dir=out, count=1, scenarios=("not_a_real_scenario",))
        )
    assert not out.exists()


def test_every_scenario_writes_a_complete_labeled_sample(tmp_path: Path) -> None:
    """Full self-consistency (<0.5px, incl. elevated net keypoints + nonzero distortion) is proved
    directly against the render engine in test_court_synth_stream.py, which the disk CLI below
    delegates to; here we confirm every one of the >=7 families round-trips through the disk-corpus
    schema with the correct scenario tag, complete 15-keypoint set, and valid per-point visibility.
    """

    expected_names = {point.name for point in PICKLEBALL_KEYPOINTS}
    for scenario in SCENARIO_NAMES:
        out = tmp_path / f"court_synthetic_{scenario}"
        manifest = generate_synthetic_court_corpus(
            SyntheticCourtGenerationConfig(
                out_dir=out,
                count=2,
                seed=2026,
                image_size=(320, 180),
                spot_check_count=0,
                scenarios=(scenario,),
                generated_at_utc="2026-07-05T00:00:00+00:00",
            )
        )
        for sample in manifest["samples"]:
            label = _load_payload(out / sample["label_path"])
            item = label["annotation"]["items"][0]
            generation = label["generation"]
            assert generation["scenario"] == scenario
            assert set(item["keypoints"]) == expected_names
            assert set(generation["keypoints_vis"]) == expected_names
            assert set(generation["keypoints_vis"].values()) <= {0, 1, 2}


def test_run_generate_synthetic_court_keypoints_cli_help_runs_from_repo_root() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/racketsport/generate_synthetic_court_keypoints.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()
    assert "--count" in completed.stdout


def test_run_generate_synthetic_court_keypoints_cli_fails_closed_on_non_positive_count(tmp_path: Path) -> None:
    out_dir = tmp_path / "court_synthetic"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/generate_synthetic_court_keypoints.py",
            "--out",
            str(out_dir),
            "--count",
            "0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert not out_dir.exists()


def test_run_generate_synthetic_court_keypoints_cli_scenario_flags(tmp_path: Path) -> None:
    out_dir = tmp_path / "court_synthetic"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/generate_synthetic_court_keypoints.py",
            "--out",
            str(out_dir),
            "--count",
            "6",
            "--seed",
            "9",
            "--image-width",
            "320",
            "--image-height",
            "180",
            "--spot-check-count",
            "0",
            "--scenarios",
            "tennis_overlay,adjacent_multi_court",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["sample_count"] == 6
    assert set(payload["scenario_counts"]) == {"tennis_overlay", "adjacent_multi_court"}
    manifest = _load_payload(out_dir / "manifest.json")
    assert set(manifest["scenario_counts"]) == {"tennis_overlay", "adjacent_multi_court"}
