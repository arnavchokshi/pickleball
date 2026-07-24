from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.racketsport.measure_audio_event_alignment import (
    RuntimeAccessObserver,
    _binary_auroc,
    _effective_audio_origin,
    _read_text,
    _runtime_access_observation,
    _threshold_confusion,
    _validate_detector_source_identity,
    crosscheck_insights_timebase,
    derive_reference_mapping,
    extract_pbvision_events,
    extract_pbvision_rally_intervals,
    extract_tt_snippet_core_features,
    greedy_nearest_one_to_one,
    measure_rally_conditioned_null,
    measure_tolerance,
)
from threed.racketsport.audio_onsets_v2 import (
    DEFAULT_ADAPTIVE_WINDOW_S,
    DEFAULT_BANDPASS_HIGH_HZ,
    DEFAULT_BANDPASS_LOW_HZ,
    DEFAULT_FRAME_SIZE_S,
    DEFAULT_HOP_S,
    DEFAULT_MIN_HFC_EVIDENCE,
    DEFAULT_MIN_POP_BAND_RATIO,
    DEFAULT_MIN_SEPARATION_S,
    DEFAULT_MIN_SPECTRAL_EVIDENCE,
    DEFAULT_THRESHOLD_MAD,
    _detect_onsets,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/racketsport/measure_audio_event_alignment.py"


def test_aac_priming_skip_maps_first_effective_sample_to_zero() -> None:
    origin = _effective_audio_origin(
        {"sample_rate": "44100", "start_time": "0.000000"},
        [{
            "pts_time": "-0.023220",
            "side_data_list": [{"side_data_type": "Skip Samples", "skip_samples": 1024}],
        }],
    )

    assert origin == 0.0


def test_greedy_nearest_matching_is_one_to_one_and_reports_every_denominator() -> None:
    references = [1.000, 1.050]
    onsets = [1.020]

    matches = greedy_nearest_one_to_one(references, onsets, 0.033)
    metrics = measure_tolerance(
        references,
        onsets,
        tolerance_s=0.033,
        media_fps=60.0,
        duration_s=60.0,
    )

    assert len(matches) == 1
    assert matches[0]["reference_index"] == 0
    assert matches[0]["onset_index"] == 0
    assert metrics["tolerance_ms"] == pytest.approx(33.0)
    assert metrics["tolerance_frames_at_media_fps"] == pytest.approx(1.98)
    assert metrics["reference_event_count"] == 2
    assert metrics["onset_count"] == 1
    assert metrics["matched_count"] == 1
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["precision_proxy"] == pytest.approx(1.0)
    assert metrics["unmatched_onset_rate_per_min"] == pytest.approx(0.0)
    assert metrics["median_absolute_offset_ms"] == pytest.approx(20.0)


def test_pbvision_selected_events_and_insights_prove_export_timebase() -> None:
    frame = lambda selected=None: {  # noqa: E731
        "actions": {
            "shot": {"confidence": 0.9},
            "bounce": {"confidence": 0.8},
        },
        "balls": (
            {} if selected is None else {
                "selected": selected,
                selected: {"interpolated": False},
            }
        ),
    }
    cv_export = {
        "camera": {"fps": 30},
        "sessions": [{
            "rallies": [{
                "frame_index": 30,
                "frames": [frame(), frame("shot"), frame("bounce")],
            }],
        }],
    }
    insights = {
        "rallies": [{
            "start_ms": 1000,
            "shots": [{"start_ms": 1033}],
        }],
    }

    events, export_fps = extract_pbvision_events(cv_export)
    intervals = extract_pbvision_rally_intervals(cv_export, export_fps=export_fps)
    crosscheck = crosscheck_insights_timebase(
        cv_export, insights, events, export_fps=export_fps
    )

    assert [item["event_type"] for item in events] == ["hit", "bounce"]
    assert [item["frame_index_export"] for item in events] == [31, 32]
    assert events[0]["export_time_s"] == pytest.approx(31 / 30)
    assert intervals == [{
        "session_index": 0,
        "rally_index": 0,
        "start_export_s": 1.0,
        "end_export_s": 1.1,
        "duration_s": pytest.approx(0.1),
        "frame_count": 3,
    }]
    assert crosscheck["insights_shot_match_fraction"] == pytest.approx(1.0)
    assert crosscheck["first_rally_cv_minus_insights_s"] == pytest.approx(0.0)


def test_reference_mapping_fails_closed_without_audio_pts() -> None:
    mapping = derive_reference_mapping(
        {
            "video_effective_origin_pts_s": 0.0,
            "audio_effective_origin_pts_s": None,
            "media_fps": 60.0,
            "video_stream": {"duration": "2.0"},
        },
        export_fps=30.0,
        events=[{"export_time_s": 1.0}],
        insights_crosscheck={
            "insights_shot_match_fraction": 1.0,
            "first_rally_cv_minus_insights_s": 0.0,
        },
    )

    assert mapping["status"] == "AUDIO_REFERENCE_UNALIGNABLE"
    assert "audio_effective_origin_pts_unavailable" in mapping["blockers"]


def test_runtime_access_observer_records_forbidden_sentinel_touch(
    tmp_path: Path,
) -> None:
    sentinel = tmp_path / "forbidden-sentinel.txt"
    sentinel.write_text("must be observed", encoding="utf-8")
    observer = RuntimeAccessObserver(mode="sentinel_test")

    with _runtime_access_observation(observer):
        assert _read_text(sentinel) == "must be observed"

    audit = observer.snapshot(
        pre_access_validation={
            "mode": "sentinel_test",
            "all_declared_inputs_validated_before_access": True,
            "validated_inputs": [],
            "excluded_without_access": [],
        },
        output_roots=[],
    )
    assert audit["forbidden_input_access_count"] == 1
    assert audit["all_observed_input_accesses_allowed"] is False
    assert audit["observed_access_event_count"] == 1
    assert audit["observed_accesses"][0]["path"] == sentinel.as_posix()
    assert audit["observed_accesses"][0]["operation"] == "read_text"
    assert audit["observed_accesses"][0]["classification"] == (
        "FORBIDDEN_UNREGISTERED_INPUT"
    )


def test_measure_audio_event_alignment_direct_cli_help() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()
    assert "--reference" in completed.stdout
    assert "--viability" in completed.stdout
    assert "--excluded-reference" in completed.stdout
    assert "--tt-sounds-labels" in completed.stdout
    assert "--tt-sounds-snippets" in completed.stdout


@pytest.mark.parametrize("offered_id", ["83gyqyc10y8f", "83GYQY%63-10Y8F"])
def test_direct_cli_refuses_compare_only_id_alias_before_path_access(
    tmp_path: Path, offered_id: str
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--reference",
            offered_id,
            str(tmp_path / "never-open-video.mp4"),
            str(tmp_path / "never-open-cv.json"),
            str(tmp_path / "never-open-insights.json"),
            "UNKNOWN",
            "--out",
            str(tmp_path / "report.json"),
            "--raw-dir",
            str(tmp_path / "raw"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "registry refusal before input access: compare-only identity" in completed.stderr
    assert not (tmp_path / "report.json").exists()
    assert not (tmp_path / "raw").exists()


def test_tt_snippet_features_use_one_candidate_and_frozen_thresholds() -> None:
    sample_rate_hz = 44_100
    time_s = np.arange(661, dtype=np.float64) / sample_rate_hz
    samples = 0.05 * np.sin(2.0 * np.pi * 3_500.0 * time_s)
    samples[260:270] += np.hanning(10)

    features = extract_tt_snippet_core_features(samples, sample_rate_hz=sample_rate_hz)

    assert features["analysis_frame_count"] == 10
    assert 0 <= features["candidate_frame_index"] < 10
    assert set(features["threshold_checks"]) == {
        "onset_strength_gte_threshold_mad",
        "pop_band_ratio_gte_minimum",
        "spectral_plus_hfc_gte_minimum",
        "hfc_gte_minimum",
    }
    assert features["threshold_eligible"] == all(features["threshold_checks"].values())
    assert all(np.isfinite(features[name]) for name in (
        "onset_strength", "high_frequency_content", "spectral_flux", "pop_band_ratio"
    ))


def test_tt_feature_equations_match_committed_detector_implementation() -> None:
    sample_rate_hz = 24_000
    samples = np.random.default_rng(4).normal(0.0, 0.0001, sample_rate_hz)
    burst_time_s = np.arange(240, dtype=np.float64) / sample_rate_hz
    samples[12_000:12_240] += (
        0.8
        * np.sin(2.0 * np.pi * 3_500.0 * burst_time_s)
        * np.hanning(240)
    )

    snippet = extract_tt_snippet_core_features(
        samples, sample_rate_hz=sample_rate_hz
    )
    detector_onsets, _ = _detect_onsets(
        samples,
        sample_rate_hz=sample_rate_hz,
        bandpass_low_hz=DEFAULT_BANDPASS_LOW_HZ,
        bandpass_high_hz=DEFAULT_BANDPASS_HIGH_HZ,
        frame_size_s=DEFAULT_FRAME_SIZE_S,
        hop_s=DEFAULT_HOP_S,
        min_separation_s=DEFAULT_MIN_SEPARATION_S,
        threshold_mad=DEFAULT_THRESHOLD_MAD,
        adaptive_window_s=DEFAULT_ADAPTIVE_WINDOW_S,
        min_pop_band_ratio=DEFAULT_MIN_POP_BAND_RATIO,
        min_spectral_evidence=DEFAULT_MIN_SPECTRAL_EVIDENCE,
        min_hfc_evidence=DEFAULT_MIN_HFC_EVIDENCE,
        time_offset_s=0.0,
    )

    strongest = max(detector_onsets, key=lambda item: float(item["onset_strength"]))
    assert snippet["candidate_frame_index"] == pytest.approx(
        float(strongest["window_start_s"]) / DEFAULT_HOP_S
    )
    assert snippet["onset_strength"] == pytest.approx(
        strongest["onset_strength"], abs=1e-6
    )
    assert snippet["spectral_flux"] == pytest.approx(
        strongest["features"]["spectral_flux"], abs=1e-6
    )
    assert snippet["high_frequency_content"] == pytest.approx(
        strongest["features"]["high_frequency_content"], abs=1e-6
    )
    assert snippet["pop_band_ratio"] == pytest.approx(
        strongest["features"]["pop_band_ratio"], abs=1e-6
    )


def test_rally_conditioned_null_is_deterministic_and_preserves_occupancy() -> None:
    kwargs = {
        "rally_intervals_s": [(0.0, 3.0), (10.0, 12.0)],
        "tolerance_s": 0.066,
        "seed": 20260722,
        "draws": 50,
    }
    first = measure_rally_conditioned_null(
        [1.0, 2.0, 10.5], [0.98, 2.02, 5.0, 10.48], **kwargs
    )
    second = measure_rally_conditioned_null(
        [1.0, 2.0, 10.5], [0.98, 2.02, 5.0, 10.48], **kwargs
    )

    assert first == second
    assert first["rally_interval_count"] == 2
    assert first["rally_duration_s"] == pytest.approx(5.0)
    assert first["reference_event_count_in_rallies"] == 3
    assert first["onset_count_in_rallies"] == 3
    assert first["actual_matched_count"] == 3
    assert first["draw_count"] == 50


def test_tt_auroc_and_threshold_confusion_preserve_exact_denominators() -> None:
    assert _binary_auroc([2.0, 3.0], [0.0, 1.0]) == pytest.approx(1.0)
    assert _binary_auroc([0.0, 1.0], [2.0, 3.0]) == pytest.approx(0.0)
    assert _binary_auroc([1.0], [1.0]) == pytest.approx(0.5)

    confusion = _threshold_confusion(
        [{"threshold_eligible": True}, {"threshold_eligible": False}],
        [{"threshold_eligible": True}, {"threshold_eligible": False}, {"threshold_eligible": False}],
    )
    assert confusion == {
        "positive_n": 2,
        "background_n": 3,
        "true_positive": 1,
        "false_negative": 1,
        "false_positive": 1,
        "true_negative": 2,
        "precision_at_observed_class_mix": pytest.approx(0.5),
        "recall": pytest.approx(0.5),
        "specificity": pytest.approx(2 / 3),
        "false_positive_rate": pytest.approx(1 / 3),
    }


@pytest.mark.parametrize(
    ("clip_id", "offered_path", "message"),
    [
        (
            "83gyqyc10y8f",
            "never-open-compare-only.mp4",
            "compare-only identity",
        ),
        (
            "Ezz6HDNHlnk",
            "eval_clips/ball/indoor_doubles_fwuks_0500_long_mid_baseline/source.mp4",
            "protected/owner path",
        ),
        (
            "Ezz6HDNHlnk",
            "data/event_labels_owner_20260719/never-open-owner.mp4",
            "protected/owner path",
        ),
    ],
)
def test_direct_cli_refuses_forbidden_viability_before_path_access(
    tmp_path: Path, clip_id: str, offered_path: str, message: str
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--viability", clip_id, offered_path, "UNKNOWN",
            "--out", str(tmp_path / "report.json"),
            "--raw-dir", str(tmp_path / "raw"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert f"registry refusal before input access: {message}" in completed.stderr
    assert not (tmp_path / "report.json").exists()
    assert not (tmp_path / "raw").exists()


def test_direct_cli_refuses_protected_reference_path_alias_before_access(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--reference", "xkadsq9bli3h",
            "eval_clips/ball/indoor_doubles_fwuks_0500_long_mid_baseline/../indoor_doubles_fwuks_0500_long_mid_baseline/source.mp4",
            "data/pbvision_gallery_20260719/xkadsq9bli3h/cv_export.json",
            "data/pbvision_gallery_20260719/xkadsq9bli3h/insights.json",
            "OUTDOOR_DAY",
            "--out", str(tmp_path / "report.json"),
            "--raw-dir", str(tmp_path / "raw"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "registry refusal before input access: protected/owner path" in completed.stderr
    assert not (tmp_path / "report.json").exists()
    assert not (tmp_path / "raw").exists()


@pytest.mark.parametrize(
    "offered_labels",
    [
        "data/pbvision_11min_20260713/never-open-compare.csv",
        "eval_clips/ball/indoor_doubles_fwuks_0500_long_mid_baseline/labels/never-open.csv",
        "data/event_labels_owner_20260719/never-open.csv",
    ],
)
def test_direct_cli_refuses_forbidden_path_in_tt_mode_before_access(
    tmp_path: Path, offered_labels: str
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--tt-sounds-labels", offered_labels,
            "--tt-sounds-snippets", "data/event_public_20260713/tt_sounds_data/sounds_extracted/sounds",
            "--anchor-report", "runs/lanes/ball_audio_repair2_20260722/alignment_report_v3.json",
            "--trackd-findings", "runs/lanes/trackD_owner_queue_20260722/BATCH01_FINDINGS.md",
            "--out", str(tmp_path / "report.json"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "registry refusal before input access: protected/owner path" in completed.stderr
    assert not (tmp_path / "report.json").exists()


def test_direct_cli_refuses_noncanonical_detector_source_before_access(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--tt-sounds-labels", "data/event_public_20260713/tt_sounds_data/full.csv",
            "--tt-sounds-snippets", "data/event_public_20260713/tt_sounds_data/sounds_extracted/sounds",
            "--anchor-report", "runs/lanes/ball_audio_repair2_20260722/alignment_report_v3.json",
            "--trackd-findings", "runs/lanes/trackD_owner_queue_20260722/BATCH01_FINDINGS.md",
            "--detector-source", "scripts/racketsport/measure_audio_event_alignment.py",
            "--out", str(tmp_path / "report.json"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "unregistered canonical_detector_source path" in completed.stderr
    assert not (tmp_path / "report.json").exists()


def test_detector_source_identity_is_canonical_non_null_and_head_equal() -> None:
    identity = _validate_detector_source_identity(
        ROOT / "threed/racketsport/audio_onsets_v2.py"
    )

    assert identity["canonical_repo_path"] == "threed/racketsport/audio_onsets_v2.py"
    assert len(identity["working_sha256"]) == 64
    assert len(identity["working_git_blob"]) == 40
    assert identity["working_git_blob"] == identity["committed_head_git_blob"]


def test_alignment_mode_direct_cli_runs_registered_reference_end_to_end(
    tmp_path: Path,
) -> None:
    output = tmp_path / "alignment.json"
    raw_dir = tmp_path / "raw"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--reference", "xkadsq9bli3h",
            "data/pbv_replay_20260720/xkadsq9bli3h/max.mp4",
            "data/pbvision_gallery_20260719/xkadsq9bli3h/cv_export.json",
            "data/pbvision_gallery_20260719/xkadsq9bli3h/insights.json",
            "OUTDOOR_DAY",
            "--out", str(output),
            "--raw-dir", str(raw_dir),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["access_audit"]["forbidden_input_access_count"] == 0
    assert payload["access_audit"]["audit_kind"] == (
        "runtime_observed_filesystem_access"
    )
    assert payload["access_audit"]["all_observed_input_accesses_allowed"] is True
    assert payload["access_audit"]["observed_accesses"]
    assert payload["data_fences"]["derivation"] == (
        "derived_from_runtime_observed_access_events"
    )
    assert payload["data_fences"]["observed_accesses_sha256"] == (
        payload["access_audit"]["observed_accesses_sha256"]
    )
    assert payload["reference_clips"][0]["status"] == "MEASURED_TEACHER_ALIGNMENT"
    assert payload["reference_clips"][0]["rally_conditioned_null_by_tolerance"]
    assert (raw_dir / "xkadsq9bli3h.audio_onsets_v2.json").is_file()
