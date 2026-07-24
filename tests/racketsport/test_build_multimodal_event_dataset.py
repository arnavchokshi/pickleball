"""CLI tests for the multimodal event dataset builder (synthetic fixtures + real-artifact checks)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CLI_PATH = REPO / "scripts/racketsport/build_multimodal_event_dataset.py"

spec = importlib.util.spec_from_file_location("build_multimodal_event_dataset_under_test", CLI_PATH)
cli = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(cli)

from threed.racketsport.multimodal_event_dataset import load_records_jsonl  # noqa: E402

FPS = 30.0
CLIP_START_S = 50.0


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _owner_row(label_id: str, *, split: str, clip_id: str, family: str, start_frame: int,
               events: list[dict], sha: str) -> dict:
    return {
        "clip_id": clip_id,
        "events": events,
        "fps": FPS,
        "label_id": label_id,
        "num_frames": 64,
        "review": {"decision": "paddle" if events else "none", "dt": 0.0, "x": 0.5, "y": 0.5},
        "source_start_frame": start_frame,
        "source_video": family,
        "split": split,
        "stratum": "uniform_random",
        "video": label_id,
        "video_path": f"data/online_harvest_20260706/rallies/{family}/{clip_id}.mp4",
        "video_sha256": sha,
    }


def _teacher_row(family: str, event_id: str, *, start_frame: int, agreements: list[dict],
                 sha: str, split: str = "train") -> dict:
    return {
        "events": [{
            "agreement_count": len(agreements),
            "audio_weight_eligible": any(a["family"] == "audio_onset" for a in agreements),
            "class": "HIT",
            "filter_decision": "accepted_independent_agreement",
            "frame": 32,
            "independent_agreements": agreements,
            "teacher_confidence": 0.9,
        }],
        "focal_event_id": event_id,
        "fps": FPS,
        "num_frames": 64,
        "sample_weight": 0.25,
        "source_lineage_key": "lineage_" + event_id,
        "source_start_frame": start_frame,
        "source_video": family,
        "source_video_sha256": sha,
        "split": split,
        "video": f"{family}:{event_id}",
    }


def _onset(time_s: float, score: float = 0.8) -> dict:
    return {"corrected_time_s": time_s, "onset_strength": 12.5, "score": score, "time_s": time_s}


@pytest.fixture()
def fixture_repo(tmp_path: Path) -> dict:
    root = tmp_path / "repo"
    family = "famA"
    val_family = "famVal"
    teacher_fam = "pbv001"
    teacher_missing_fam = "pbv404"
    clip_id = f"{family}_rally_0001"
    val_clip = f"{val_family}_rally_0001"

    # Owner manifest: 2 train (1 HIT @32, 1 negative), 1 val BOUNCE @32.
    owner_rows = [
        _owner_row("lab_001", split="train", clip_id=clip_id, family=family, start_frame=1800,
                   events=[{"class": "HIT", "frame": 32}], sha="1" * 64),
        _owner_row("lab_002", split="train", clip_id=clip_id, family=family, start_frame=3000,
                   events=[], sha="1" * 64),
        _owner_row("lab_003", split="val", clip_id=val_clip, family=val_family, start_frame=900,
                   events=[{"class": "BOUNCE", "frame": 32}], sha="2" * 64),
    ]
    owner_manifest = {"config": {"window_frames": 64}, "rows": owner_rows}
    _write_json(root / "owner_manifest.json", owner_manifest)

    kink = [{"family": "ball_velocity_kink", "absolute_delta_s": 0.01}]
    teacher_rows = [
        _teacher_row(teacher_fam, "evt1", start_frame=600, agreements=kink, sha="3" * 64),
        _teacher_row(teacher_missing_fam, "evt2", start_frame=600, agreements=kink, sha="4" * 64),
    ]
    teacher_manifest = {"rows": teacher_rows}
    _write_json(root / "teacher_manifest.json", teacher_manifest)

    # Source-video audio onsets: famA has one onset inside lab_001's window
    # (window [60.0, 62.1333); onset at 61.0667 => frame_offset 32) and none in
    # lab_002's window. famVal and pbv001 each get an in-window onset.
    _write_json(root / "cues/famA.audio_onsets_v2.json", {
        "detector_version": "audio_onset_pop_v2", "media_sha256": "a" * 64,
        "onsets": [_onset(1800 / FPS + 32 / FPS), _onset(10.0)], "status": "review_only",
    })
    _write_json(root / "cues/famVal.audio_onsets_v2.json", {
        "detector_version": "audio_onset_pop_v2", "media_sha256": "b" * 64,
        "onsets": [_onset(900 / FPS + 32 / FPS)], "status": "review_only",
    })
    _write_json(root / "cues/pbv001.audio_onsets_v2.json", {
        "detector_version": "audio_onset_pop_v2", "media_sha256": "3" * 64,
        "onsets": [_onset(600 / FPS + 32 / FPS)], "status": "review_only",
    })

    # Ball inflections for the train clip; clip time = source time - CLIP_START_S.
    _write_json(root / "cues/ball" / f"{clip_id}.ball_inflections.json", {
        "candidates": [{"confidence": 0.6, "time_s": 1800 / FPS + 33 / FPS - CLIP_START_S,
                        "turn_angle_deg": 80.0}],
        "source": "ball_track_image_motion",
    })
    _write_json(root / "prelabels" / clip_id / "ball_track.json", {"fps": FPS, "frames": []})

    # Clip provenance + v0 clip onsets for the time-mapping measurement
    # (six matching onsets so the mapping verifies with residual 0).
    _write_json(root / f"data/online_harvest_20260706/rallies/{family}/{clip_id}.provenance.json",
                {"rally_segment": {"start_s": CLIP_START_S}})
    _write_json(root / f"data/online_harvest_20260706/rallies/{val_family}/{val_clip}.provenance.json",
                {"rally_segment": {"start_s": 0.0}})
    shared = [10.0, 20.0, 30.0, 40.0, 50.5, 55.25]
    _write_json(root / "cues" / f"{clip_id}.clip_onsets_v0.json",
                {"onsets": [_onset(t - CLIP_START_S) for t in shared] + [_onset(1800 / FPS + 32 / FPS - CLIP_START_S)]})
    root_famA = json.loads((root / "cues/famA.audio_onsets_v2.json").read_text())
    root_famA["onsets"] += [_onset(t) for t in shared]
    _write_json(root / "cues/famA.audio_onsets_v2.json", root_famA)

    cue_index = {
        "artifact_type": cli.CUE_INDEX_ARTIFACT_TYPE,
        "audio_onsets_v2": {
            family: {"detector_version": "audio_onset_pop_v2", "media_sha256": "a" * 64,
                     "path": "cues/famA.audio_onsets_v2.json", "sha256": "0" * 64, "status": "review_only"},
            val_family: {"detector_version": "audio_onset_pop_v2", "media_sha256": "b" * 64,
                         "path": "cues/famVal.audio_onsets_v2.json", "sha256": "0" * 64, "status": "review_only"},
            teacher_fam: {"detector_version": "audio_onset_pop_v2", "media_sha256": "3" * 64,
                          "path": "cues/pbv001.audio_onsets_v2.json", "sha256": "0" * 64, "status": "review_only"},
        },
        "ball_inflections": {
            clip_id: {"ball_track_path": f"prelabels/{clip_id}/ball_track.json",
                      "ball_track_sha256": "0" * 64,
                      "path": f"cues/ball/{clip_id}.ball_inflections.json", "sha256": "0" * 64},
        },
        "clip_audio_onsets_v0": {clip_id: {"path": f"cues/{clip_id}.clip_onsets_v0.json", "sha256": "0" * 64}},
        "media": {family: {"path": None, "present": False, "sha256": None},
                  val_family: {"path": None, "present": False, "sha256": None},
                  teacher_fam: {"path": None, "present": False, "sha256": None},
                  teacher_missing_fam: {"path": None, "present": False, "sha256": None}},
        "schema_version": 1,
        "wrist_velocity_peaks": {},
    }
    _write_json(root / "cue_index.json", cue_index)

    # Protected seed selector + inventory: one protected window on the TRAIN
    # clip far away from every fixture row (rows can be moved onto it per test).
    selector = {"labels": [{
        "anchor": {"frame": 4500, "pts_s": 150.0},
        "label_id": f"{clip_id}__a__000001",
        "window": {"end_pts_s": 150.1, "half_width_s": 0.1, "start_pts_s": 149.9},
    }], "seed": 1}
    _write_json(root / "selector.json", selector)
    _write_json(root / "inventory.json", {"clips": [
        {"clip_id": clip_id, "video_sha256": "1" * 64},
        {"clip_id": val_clip, "video_sha256": "2" * 64},
    ]})
    return {"root": root, "clip_id": clip_id, "family": family, "val_family": val_family,
            "teacher_fam": teacher_fam, "teacher_missing_fam": teacher_missing_fam}


def _build_args(fixture: dict, out_dir: Path, *, expect_split: str = "2:1") -> list[str]:
    return [
        "build",
        "--repo-root", str(fixture["root"]),
        "--owner-manifest", "owner_manifest.json",
        "--teacher-manifest", "teacher_manifest.json",
        "--cue-index", "cue_index.json",
        "--protected-selector", "selector.json",
        "--protected-inventory", "inventory.json",
        "--out-dir", str(out_dir),
        "--expect-owner-split", expect_split,
    ]


def test_build_emits_masks_offsets_and_labels(fixture_repo: dict, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    assert cli.main(_build_args(fixture_repo, out_dir)) == 0
    owner = load_records_jsonl(out_dir / "records/owner_records.jsonl")
    teacher = load_records_jsonl(out_dir / "records/teacher_records.jsonl")
    assert [record["record_id"] for record in owner] == ["owner:lab_001", "owner:lab_002", "owner:lab_003"]
    assert [record["record_id"] for record in teacher] == ["teacher:pbv001:evt1"]

    by_id = {record["record_id"]: record for record in owner}
    hit = by_id["owner:lab_001"]
    assert hit["label"]["class"] == "HIT" and hit["label"]["dt_s"] == 0.0
    audio = hit["modalities"]["audio_onsets_v2"]
    assert audio["available"] is True
    assert [event["frame_offset"] for event in audio["events"]] == [32]
    assert audio["series"]["values"][32] == 0.8
    ball = hit["modalities"]["ball_inflections"]
    assert ball["available"] is True
    assert [event["frame_offset"] for event in ball["events"]] == [33]
    assert ball["artifact"]["clip_time_mapping"]["verified"] is True
    wrist = hit["modalities"]["wrist_velocity_peaks"]
    assert wrist["available"] is False and wrist["reason"] == "no_artifact" and wrist["series"] is None

    negative = by_id["owner:lab_002"]
    assert negative["label"]["class"] == "negative"
    assert negative["label"]["event_frame"] is None and negative["label"]["dt_s"] is None
    assert negative["modalities"]["audio_onsets_v2"]["available"] is False
    assert negative["modalities"]["audio_onsets_v2"]["reason"] == "no_signal_in_window"

    val = by_id["owner:lab_003"]
    assert val["split"] == "val" and val["provenance"]["ground_truth"] is True

    teacher_record = teacher[0]
    assert teacher_record["split"] == "train"
    assert teacher_record["provenance"]["ground_truth"] is False
    assert teacher_record["provenance"]["label_provenance"] == "teacher_derived"
    assert teacher_record["modalities"]["ball_inflections"]["reason"] == "no_artifact"

    coverage = json.loads((out_dir / "coverage.json").read_text())
    assert coverage["assertions"]["audio_only_teacher_events"] == 0
    unbuildable = coverage["unbuildable_rows"]
    assert len(unbuildable) == 1
    assert unbuildable[0]["family"] == fixture_repo["teacher_missing_fam"]
    assert unbuildable[0]["reason"] == "missing_media"
    assert coverage["split_table"]["owner"] == {"train": 2, "val": 1}
    assert (out_dir / "MANIFEST.sha256.json").exists()
    assert (out_dir / "COVERAGE.md").exists()


def test_build_is_byte_deterministic(fixture_repo: dict, tmp_path: Path) -> None:
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    assert cli.main(_build_args(fixture_repo, out_a)) == 0
    assert cli.main(_build_args(fixture_repo, out_b)) == 0
    for rel in ("records/owner_records.jsonl", "records/teacher_records.jsonl",
                "coverage.json", "COVERAGE.md", "MANIFEST.sha256.json"):
        assert (out_a / rel).read_bytes() == (out_b / rel).read_bytes(), rel


def test_audio_only_teacher_events_refused(fixture_repo: dict, tmp_path: Path) -> None:
    root = fixture_repo["root"]
    manifest = json.loads((root / "teacher_manifest.json").read_text())
    manifest["rows"][0]["events"][0]["independent_agreements"] = [{"family": "audio_onset"}]
    _write_json(root / "teacher_manifest.json", manifest)
    with pytest.raises(SystemExit, match=cli.AUDIO_ONLY_ERROR):
        cli.main(_build_args(fixture_repo, tmp_path / "out"))


def test_owner_split_drift_refused(fixture_repo: dict, tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match=cli.OWNER_SPLIT_ERROR):
        cli.main(_build_args(fixture_repo, tmp_path / "out", expect_split="61:41"))


def test_teacher_family_overlapping_owner_val_is_quarantined(fixture_repo: dict, tmp_path: Path) -> None:
    root = fixture_repo["root"]
    manifest = json.loads((root / "teacher_manifest.json").read_text())
    # Rebind the buildable teacher family onto the owner VAL family id.
    manifest["rows"][0]["source_video"] = fixture_repo["val_family"]
    manifest["rows"][0]["video"] = f"{fixture_repo['val_family']}:evt1"
    _write_json(root / "teacher_manifest.json", manifest)
    out_dir = tmp_path / "out"
    assert cli.main(_build_args(fixture_repo, out_dir)) == 0
    teacher = load_records_jsonl(out_dir / "records/teacher_records.jsonl")
    assert teacher == []
    coverage = json.loads((out_dir / "coverage.json").read_text())
    assert coverage["split_table"]["teacher"]["quarantined"] == 1
    assert coverage["quarantined_rows"][0]["reason"] == "family_overlaps_owner_val"


def test_teacher_row_in_val_split_refused(fixture_repo: dict, tmp_path: Path) -> None:
    root = fixture_repo["root"]
    manifest = json.loads((root / "teacher_manifest.json").read_text())
    manifest["rows"][0]["split"] = "val"
    _write_json(root / "teacher_manifest.json", manifest)
    with pytest.raises(SystemExit, match=cli.TEACHER_VAL_ERROR):
        cli.main(_build_args(fixture_repo, tmp_path / "out"))


def test_protected_overlap_in_train_window_refused(fixture_repo: dict, tmp_path: Path) -> None:
    root = fixture_repo["root"]
    # Move the protected window onto lab_001's clip-axis window
    # (source [60.0, 62.1333) minus clip start 50.0 => clip [10.0, 12.1333)).
    selector = json.loads((root / "selector.json").read_text())
    selector["labels"][0]["window"] = {"end_pts_s": 11.1, "half_width_s": 0.1, "start_pts_s": 10.9}
    selector["labels"][0]["anchor"] = {"frame": 330, "pts_s": 11.0}
    _write_json(root / "selector.json", selector)
    with pytest.raises(SystemExit, match=cli.PROTECTED_TRAIN_OVERLAP_ERROR):
        cli.main(_build_args(fixture_repo, tmp_path / "out"))


def test_protected_overlap_in_val_window_is_measured_not_refused(fixture_repo: dict, tmp_path: Path) -> None:
    root = fixture_repo["root"]
    val_clip = f"{fixture_repo['val_family']}_rally_0001"
    selector = json.loads((root / "selector.json").read_text())
    # Val row window: source [30.0, 32.1333), clip start 0.0.
    selector["labels"].append({
        "anchor": {"frame": 930, "pts_s": 31.0},
        "label_id": f"{val_clip}__a__000002",
        "window": {"end_pts_s": 31.1, "half_width_s": 0.1, "start_pts_s": 30.9},
    })
    _write_json(root / "selector.json", selector)
    out_dir = tmp_path / "out"
    assert cli.main(_build_args(fixture_repo, out_dir)) == 0
    coverage = json.loads((out_dir / "coverage.json").read_text())
    measured = coverage["assertions"]["protected_val_window_overlaps_measured"]
    assert [item["row_key"] for item in measured] == ["lab_003"]


def test_protected_seed_row_key_identity_refused(fixture_repo: dict, tmp_path: Path) -> None:
    root = fixture_repo["root"]
    selector = json.loads((root / "selector.json").read_text())
    selector["labels"][0]["label_id"] = "lab_001"
    _write_json(root / "selector.json", selector)
    with pytest.raises(SystemExit, match=cli.PROTECTED_IDENTITY_ERROR):
        cli.main(_build_args(fixture_repo, tmp_path / "out"))


def test_teacher_row_on_protected_video_sha_refused(fixture_repo: dict, tmp_path: Path) -> None:
    root = fixture_repo["root"]
    manifest = json.loads((root / "teacher_manifest.json").read_text())
    manifest["rows"][0]["source_video_sha256"] = "1" * 64  # protected clip sha
    _write_json(root / "teacher_manifest.json", manifest)
    with pytest.raises(SystemExit, match=cli.PROTECTED_IDENTITY_ERROR):
        cli.main(_build_args(fixture_repo, tmp_path / "out"))


def test_unverified_clip_mapping_masks_ball_modality(fixture_repo: dict, tmp_path: Path) -> None:
    root = fixture_repo["root"]
    clip_id = fixture_repo["clip_id"]
    # Break the clip onsets so fewer than the minimum onset matches remain.
    _write_json(root / "cues" / f"{clip_id}.clip_onsets_v0.json", {"onsets": [_onset(2.22)]})
    out_dir = tmp_path / "out"
    assert cli.main(_build_args(fixture_repo, out_dir)) == 0
    owner = load_records_jsonl(out_dir / "records/owner_records.jsonl")
    hit = next(record for record in owner if record["record_id"] == "owner:lab_001")
    ball = hit["modalities"]["ball_inflections"]
    assert ball["available"] is False
    assert ball["reason"] == "clip_time_mapping_unverified"
    assert ball["series"] is None


# ---------------------------------------------------------------------------
# Checks against the real produced lane artifacts (skip when absent).
# ---------------------------------------------------------------------------

LANE_DIR = REPO / "runs/ball_lane_20260723/mm_dataset"
SELECTOR = REPO / "runs/lanes/event_bootstrap_20260713/spot_check_tier_a_50.json"
INVENTORY = REPO / "data/event_bootstrap_20260713/inventory_v0.json"

requires_lane = pytest.mark.skipif(
    not (LANE_DIR / "records/owner_records.jsonl").exists() or not SELECTOR.exists() or not INVENTORY.exists(),
    reason="lane records or protected selector not present in this checkout",
)


@requires_lane
def test_real_records_preserve_frozen_owner_split() -> None:
    owner = load_records_jsonl(LANE_DIR / "records/owner_records.jsonl")
    assert sum(1 for record in owner if record["split"] == "train") == 61
    assert sum(1 for record in owner if record["split"] == "val") == 41


@requires_lane
def test_real_records_never_contain_protected_seed_identities() -> None:
    selector = json.loads(SELECTOR.read_text())
    inventory = json.loads(INVENTORY.read_text())
    protected = cli.load_protected_identities(selector, inventory)
    assert len(protected["label_ids"]) == 50
    for name in ("owner_records.jsonl", "teacher_records.jsonl"):
        for record in load_records_jsonl(LANE_DIR / "records" / name):
            assert record["row_key"] not in protected["label_ids"]
            assert record["record_id"] not in protected["label_ids"]
            sha = record["provenance"].get("source_video_sha256")
            if record["label_set"] == "teacher":
                assert sha not in protected["video_sha256s"]


@requires_lane
def test_real_teacher_records_are_train_only_and_not_ground_truth() -> None:
    teacher = load_records_jsonl(LANE_DIR / "records/teacher_records.jsonl")
    assert teacher, "teacher records file is empty"
    for record in teacher:
        assert record["split"] == "train"
        assert record["provenance"]["ground_truth"] is False
        assert record["provenance"]["label_provenance"] == "teacher_derived"
