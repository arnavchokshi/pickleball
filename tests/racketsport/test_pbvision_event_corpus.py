from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

import cv2
import pytest

from scripts.racketsport.build_abc_arm_manifests import (
    COMPARE_ONLY_HOLDOUTS as ABC_ARM_COMPARE_ONLY_HOLDOUTS,
)
from scripts.racketsport.build_abc_arm_manifests import (
    _validate_teacher_manifest,
)
from scripts.racketsport.build_pbvision_ball_sst import (
    COMPARE_ONLY_IDS as SST_COMPARE_ONLY_IDS,
)
from scripts.racketsport.build_pbvision_ball_sst import (
    TEACHER_TEST_ONLY_IDS,
    TEACHER_VAL_ONLY_IDS,
    TRAIN_IDS,
)
from scripts.racketsport.build_pbvision_event_corpus import (
    ABC_WEIGHTING_POLICY,
    BUILDER_PATH,
    COMPARE_ONLY_HOLDOUTS,
    CorpusBuildError,
    FILTERING_POLICY,
    ROW_SCHEMA_KEYS,
    TEACHER_CONFIDENCE_MIN,
    assign_source_splits,
    build_corpus,
    parse_pbvision_video,
    write_corpus_artifacts,
)
from threed.racketsport.event_head.datasets import decode_video_frames


ROOT = Path(__file__).resolve().parents[2]
GALLERY_ROOT = ROOT / "data/pbvision_gallery_20260719"
LOCAL_MEDIA = ROOT / "data/pbv_replay_20260720/xkadsq9bli3h/max.mp4"
FRAME_TIMES = (
    ROOT
    / "runs/lanes/pbv_replay_20260720/vm_pull/"
    "process_video_pbv_replay_xkadsq9bli3h_20260720/"
    "xkadsq9bli3h/frame_times.json"
)
CLI = "scripts/racketsport/build_pbvision_event_corpus.py"
# This is the real, already-committed frozen split manifest that
# build_pbvision_ball_sst.py itself pins as FROZEN_SPLIT_RELATIVE_PATH /
# FROZEN_SPLIT_SHA256. Reusing it here (rather than a synthetic fixture)
# proves the override works against the actual authoritative artifact, not
# just a look-alike.
FROZEN_SPLIT_MANIFEST = ROOT / "runs/lanes/pbv_pickleball_corpus_20260720/manifest.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _source_sha(video_id: str) -> str:
    return hashlib.sha256(f"fixture-media:{video_id}".encode()).hexdigest()


def _frame(selected: str | None = None, *, confidence: float = 0.9) -> dict[str, object]:
    actions = {
        name: {"u": 0.25, "v": 0.75, "confidence": confidence}
        for name in ("ball", "shot", "bounce", "net")
    }
    balls: dict[str, object] = {}
    if selected is not None:
        balls = {
            "selected": selected,
            selected: {
                "court_position": {"x": 1.0, "y": 2.0, "z": 0.1},
                "interpolated": False,
            },
        }
    return {
        "actions": actions,
        "balls": balls,
        "court": {"player_points": [{"u": 0.1, "v": 0.2, "confidence": 0.8}]},
    }


def _write_video(
    root: Path,
    video_id: str,
    *,
    source_fps: float = 60.0,
    teacher_fps: float = 30.0,
    selected: tuple[tuple[int, str, float], ...] = (
        (1, "shot", 0.91),
        (5, "bounce", 0.49),
        (8, "net", 0.73),
    ),
) -> Path:
    video_dir = root / video_id
    video_dir.mkdir(parents=True)
    frames = [_frame() for _ in range(10)]
    for local_frame, event_type, confidence in selected:
        frames[local_frame] = _frame(event_type, confidence=confidence)
    (video_dir / "cv_export.json").write_text(json.dumps({
        "version": "fixture-v1",
        "camera": {"fps": teacher_fps, "cameraSegments": []},
        "sessions": [{
            "session_type": "game",
            "rallies": [{"frame_index": 30, "frames": frames}],
        }],
    }), encoding="utf-8")
    (video_dir / "api_get_metadata.json").write_text(json.dumps({
        "metadata": {"fps": source_fps, "secs": 4.0, "width": 1280, "height": 720},
    }), encoding="utf-8")
    (video_dir / "api_get_cv_version.json").write_text(json.dumps({
        "cvGitHash": "fixture-cv-hash",
    }), encoding="utf-8")
    (video_dir / "video_provenance.json").write_text(json.dumps({
        "video_id": video_id,
        "gallery_card": {"title": f"Fixture {video_id}"},
        "fps_reported": source_fps,
        "duration_sec_reported": 4.0,
    }), encoding="utf-8")
    return video_dir


def _write_source_manifest(root: Path, video_ids: Iterable[str]) -> None:
    (root / "MANIFEST.json").write_text(json.dumps({
        "lane": "fixture",
        "videos": [
            {
                "video_id": video_id,
                "video_sha256": _source_sha(video_id),
                "duration_s": 4.0,
                "resolution": "1280x720@60fps",
            }
            for video_id in sorted(video_ids)
        ],
    }), encoding="utf-8")


def _write_frame_times(video_dir: Path, *, fps: float, offset_s: float) -> None:
    frame_count = 240
    (video_dir / "frame_times.json").write_text(json.dumps({
        "schema_version": 1,
        "fps": fps,
        "frame_count": frame_count,
        "duration_s": 4.0,
        "source_video_sha256": _source_sha(video_dir.name),
        "frames": [
            {"frame": frame, "pts_s": frame / fps + offset_s}
            for frame in range(frame_count)
        ],
    }), encoding="utf-8")


def _fixture_gallery(root: Path) -> Path:
    ids = [f"eligible_{index:02d}" for index in range(10)]
    ids.extend(sorted(COMPARE_ONLY_HOLDOUTS))
    for video_id in ids:
        _write_video(root, video_id)
    _write_source_manifest(root, ids)
    return root


def _walk_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def test_provenance_only_timebase_converts_teacher_timestamp_not_frame_number(
    tmp_path: Path,
) -> None:
    video_dir = _write_video(tmp_path, "sample", source_fps=60.0, teacher_fps=30.0)
    _write_source_manifest(tmp_path, ["sample"])
    parsed = parse_pbvision_video(video_dir, media_root=None, frame_times_root=None)

    assert parsed.teacher_fps == 30.0
    assert parsed.fps == 60.0
    assert parsed.num_frames == 240
    assert [event["teacher_frame"] for event in parsed.events] == [31, 35, 38]
    assert [event["frame"] for event in parsed.events] == [62, 70, 76]
    assert parsed.events[0]["teacher_timestamp_s"] == 31 / 30.0
    assert parsed.events[0]["source_pts_s"] == 62 / 60.0
    assert parsed.timebase_conversion["needs_pts_verify"] is True
    assert parsed.timebase_conversion["source_timebase"].startswith(
        "provenance_declared_nominal_cfr"
    )


def test_encoded_pts_mapping_uses_nearest_pts_not_nominal_frame_math(
    tmp_path: Path,
) -> None:
    video_dir = _write_video(tmp_path, "sample", source_fps=60.0, teacher_fps=30.0)
    _write_frame_times(video_dir, fps=60.0, offset_s=0.01)
    _write_source_manifest(tmp_path, ["sample"])

    parsed = parse_pbvision_video(video_dir, media_root=None, frame_times_root=None)

    # 31/30s would round to nominal frame 62. The nearest encoded PTS is frame
    # 61 because this fixture's encoded clock is offset by 10ms.
    assert parsed.events[0]["teacher_timestamp_s"] == 31 / 30.0
    assert parsed.events[0]["frame"] == 61
    assert parsed.events[0]["frame"] != round((31 / 30.0) * 60.0)
    assert parsed.events[0]["source_pts_s"] == pytest.approx(61 / 60.0 + 0.01)
    assert parsed.timebase_conversion["source_timebase"] == "encoded_pts_frame_times"
    assert parsed.timebase_conversion["needs_pts_verify"] is False


def test_frame_times_declared_media_sha_must_match_source_identity(
    tmp_path: Path,
) -> None:
    video_dir = _write_video(tmp_path, "sample", source_fps=60.0, teacher_fps=30.0)
    _write_frame_times(video_dir, fps=60.0, offset_s=0.0)
    payload = json.loads((video_dir / "frame_times.json").read_text())
    payload["source_video_sha256"] = "0" * 64
    (video_dir / "frame_times.json").write_text(json.dumps(payload))
    _write_source_manifest(tmp_path, ["sample"])

    with pytest.raises(CorpusBuildError, match="frame-times media SHA-256 mismatch"):
        parse_pbvision_video(video_dir, media_root=None, frame_times_root=None)


def test_worked_example_maps_2_233_seconds_to_source_frame_134_not_67() -> None:
    manifest, _, contexts, _ = build_corpus(GALLERY_ROOT)
    row = next(row for row in manifest["rows"] if row["source_video"] == "xkadsq9bli3h")
    first = row["events"][0]
    full = next(event for event in contexts if event["event_id"] == first["event_id"])

    assert first["teacher_frame"] == 67
    assert first["teacher_fps"] == 30.0
    assert first["teacher_timestamp_s"] == pytest.approx(2.2333333333333334)
    assert first["frame"] == 134
    assert first["frame"] != 67
    assert first["source_pts_s"] == pytest.approx(2.233333, abs=1e-6)
    assert full["mapping_abs_error_s"] < 1e-6
    assert row["fps"] == 60.0
    assert row["num_frames"] == 11_168
    assert row["timebase_conversion"] == {
        "teacher_timebase": "cv_export.camera.fps",
        "teacher_fps": 30.0,
        "timestamp_formula": "teacher_frame / teacher_fps",
        "source_timebase": "encoded_pts_frame_times",
        "source_fps": 60.0,
        "mapping": "argmin(abs(encoded_source_pts - teacher_timestamp_s))",
        "needs_pts_verify": False,
        "frame_times_sha256": _sha256(FRAME_TIMES),
        "pts_media_binding": {
            "binding_schema_version": 1,
            "status": "sha256_bound",
            "source_video_sha256": _sha256(LOCAL_MEDIA),
            "media_path": "data/pbv_replay_20260720/xkadsq9bli3h/max.mp4",
            "media_sha256_verified_from_file": True,
            "frame_times_path": (
                "runs/lanes/pbv_replay_20260720/vm_pull/"
                "process_video_pbv_replay_xkadsq9bli3h_20260720/"
                "xkadsq9bli3h/frame_times.json"
            ),
            "frame_times_sha256": _sha256(FRAME_TIMES),
            "frame_times_declares_media_sha256": False,
            "binding_sha256": _canonical_sha({
                "source_video_sha256": _sha256(LOCAL_MEDIA),
                "frame_times_sha256": _sha256(FRAME_TIMES),
            }),
        },
    }


def test_real_xkadsq9bli3h_corrected_frame_decodes_at_correct_pts() -> None:
    assert LOCAL_MEDIA.is_file()
    assert _sha256(LOCAL_MEDIA) == (
        "5085ae6ed0813b2b05ce1d6fe752423506cdc3fb78ca751d185403889b47b181"
    )
    decoded = decode_video_frames(LOCAL_MEDIA, [134], image_size=32)
    assert tuple(decoded.shape) == (1, 3, 32, 32)

    capture = cv2.VideoCapture(str(LOCAL_MEDIA))
    assert capture.isOpened()
    try:
        assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(60.0)
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 11_168
        assert capture.set(cv2.CAP_PROP_POS_FRAMES, 134)
        ok, frame = capture.read()
        assert ok and frame.shape == (1080, 1920, 3)
        decoded_pts_s = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    finally:
        capture.release()
    assert decoded_pts_s == pytest.approx(2.2333333333333334, abs=1e-6)


def test_builder_totals_and_real_gallery_contract() -> None:
    manifest, report, contexts, compare_contexts = build_corpus(GALLERY_ROOT)
    assert manifest["verified"] is False
    assert manifest["training_ready"] is False
    assert manifest["teacher_derived"] is True
    assert manifest["ground_truth"] is False
    assert manifest["totals"] == {
        "source_videos": 10,
        "media_present_videos": 1,
        "pts_verified_videos": 1,
        "needs_pts_verify_videos": 9,
        "HIT": 2839,
        "BOUNCE": 1798,
        "manifest_events": 4637,
        "low_confidence_rejected_events": 294,
        "needs_agreement_pass_events": 4343,
        "training_eligible_events": 0,
    }
    assert report["input_gallery_videos"] == 12
    assert report["included_staging_videos"] == 10
    assert report["builder_totals"] == manifest["totals"]
    assert len(contexts) == 4801
    assert len(compare_contexts) == 918
    assert {row["split"] for row in manifest["rows"]} == {"train", "val", "test"}

    parsed = parse_pbvision_video(GALLERY_ROOT / "0tmdeghtfvjx")
    mapped = [event for event in parsed.events if event["manifest_class"] is not None]
    assert sum(event["manifest_class"] == "HIT" for event in mapped) == 397
    assert sum(event["manifest_class"] == "BOUNCE" for event in mapped) == 223
    first_hit = next(event for event in parsed.events if event["manifest_class"] == "HIT")
    assert first_hit["teacher_frame"] == 114
    assert first_hit["frame"] == 114
    assert first_hit["teacher_confidence"] == 0.9052530527114868


def test_lineage_splits_and_complete_provenance_are_content_addressed() -> None:
    manifest, _, _, _ = build_corpus(GALLERY_ROOT)
    expected_assignment = assign_source_splits(
        {row["source_video"]: row["source_lineage_key"] for row in manifest["rows"]},
        seed=20260720,
    )
    provenance = manifest["provenance"]
    assert provenance["builder"] == {
        "path": "scripts/racketsport/build_pbvision_event_corpus.py",
        "sha256": _sha256(BUILDER_PATH),
    }
    assert provenance["source_manifest_sha256"] == _sha256(GALLERY_ROOT / "MANIFEST.json")
    assert provenance["filtering_policy"] == FILTERING_POLICY
    assert len(provenance["filtering_policy_sha256"]) == 64
    source_records = {item["video_id"]: item for item in provenance["sources"]}
    assert len(source_records) == 12
    for video_id in COMPARE_ONLY_HOLDOUTS:
        if video_id in source_records:
            assert source_records[video_id]["compare_only"] is True
            assert source_records[video_id]["training_eligible"] is False

    for row in manifest["rows"]:
        assert set(row) == ROW_SCHEMA_KEYS
        assert row["split"] == expected_assignment[row["source_video"]]
        assert row["source_lineage_key"] == hashlib.sha256(
            row["parent_identity"].encode()
        ).hexdigest()
        assert row["source_video_sha256"] in row["parent_identity"]
        source = source_records[row["source_video"]]
        assert source["compare_only"] is False
        assert source["training_eligible"] is False
        assert source["source_video_sha256"] == row["source_video_sha256"]
        assert source["cv_export_version"] == "2.1.0"
        assert source["cv_version"]["cvGitHash"]
        assert len(source["cv_export_sha256"]) == 64
        assert len(source["get_cv_version_sha256"]) == 64
        binding = source["pts_media_binding"]
        if row["timebase_conversion"]["needs_pts_verify"]:
            assert binding["status"] == "missing_pts_and_media_binding"
            assert binding["binding_sha256"] is None
        else:
            assert binding["source_video_sha256"] == row["source_video_sha256"]
            assert len(binding["binding_sha256"]) == 64


def test_confidence_agreement_placeholders_and_unknown_masks_fail_closed() -> None:
    manifest, _, contexts, _ = build_corpus(GALLERY_ROOT)
    assert manifest["abc_weighting_policy"] == ABC_WEIGHTING_POLICY
    assert manifest["config"]["unknown_frame_mask_semantics"].startswith(
        "true means ignore frame for loss"
    )
    low = min(
        (
            event for row in manifest["rows"] for event in row["events"]
            if event["teacher_confidence"] < TEACHER_CONFIDENCE_MIN
        ),
        key=lambda event: event["teacher_confidence"],
    )
    row = next(
        row for row in manifest["rows"]
        if any(event["event_id"] == low["event_id"] for event in row["events"])
    )
    assert low["teacher_confidence"] == pytest.approx(0.0001211768903885968)
    assert low["agreement_count"] == 0
    assert low["pseudo_weight"] == 0.0
    assert low["needs_agreement_pass"] is False
    assert low["training_eligible"] is False
    assert low["unknown_for_loss"] is True
    assert low["filter_decision"] == "rejected_low_teacher_confidence"
    assert row["unknown_frame_mask"][low["frame"]] is True
    assert len(row["unknown_frame_mask"]) == row["num_frames"]
    assert row["loss_validity_mask"] == [True, True, True]
    assert row["sample_weight"] == 0.0
    assert all(event["training_eligible"] is False for event in contexts)


def test_holdouts_are_denylisted_in_every_derivative(tmp_path: Path) -> None:
    built = build_corpus(_fixture_gallery(tmp_path / "gallery"))
    output = tmp_path / "out"
    write_corpus_artifacts(output, *built)
    manifest, report, contexts, compare_contexts = built

    assert {row["source_video"] for row in manifest["rows"]}.isdisjoint(
        COMPARE_ONLY_HOLDOUTS
    )
    assert all(event["video_id"] not in COMPARE_ONLY_HOLDOUTS for event in contexts)
    assert {event["video_id"] for event in compare_contexts} == {
        "83gyqyc10y8f", "iottnc0h3ekn", "o4dee9dn0ccr"
    }
    assert all(event["compare_only"] is True for event in compare_contexts)
    assert all(event["training_eligible"] is False for event in compare_contexts)
    documented = {item["video_id"]: item for item in report["compare_only_holdouts"]}
    assert set(documented) == set(COMPARE_ONLY_HOLDOUTS)
    assert all(item["excluded_from_training_rows"] for item in documented.values())
    assert all(item["training_eligible"] is False for item in documented.values())

    for path in output.iterdir():
        if path.suffix == ".jsonl":
            payloads = [json.loads(line) for line in path.read_text().splitlines() if line]
        elif path.suffix == ".json":
            payloads = [json.loads(path.read_text())]
        else:
            continue
        for payload in payloads:
            for item in _walk_dicts(payload):
                identity = item.get("video_id", item.get("source_video"))
                if identity in COMPARE_ONLY_HOLDOUTS:
                    assert item.get("training_eligible", False) is False, (path, item)


def test_empty_filtered_manifest_and_manager_hunk_block_training(tmp_path: Path) -> None:
    built = build_corpus(_fixture_gallery(tmp_path / "gallery"))
    output = tmp_path / "out"
    write_corpus_artifacts(output, *built)
    filtered = json.loads((output / "pbvision_filtered_teacher_manifest.json").read_text())
    assert filtered["training_ready"] is False
    assert filtered["needs_agreement_pass"] is True
    assert filtered["rows"] == []
    assert filtered["pending_event_count"] > 0
    assert filtered["permanent_compare_only_denylist"] == sorted(
        COMPARE_ONLY_HOLDOUTS
    )
    assert filtered["provenance"] == built[0]["provenance"]
    assert filtered["config"]["unknown_frame_mask_semantics"].startswith(
        "true means ignore frame for loss"
    )

    abc_stage = (output / "ABC_STAGE.md").read_text()
    assert "count 0 -> `pseudo_weight=0`" in abc_stage
    assert "count 1 -> `0.25`" in abc_stage
    assert "count >=2 -> `0.5`" in abc_stage
    assert "ignored, never converted to background" in abc_stage
    assert "normalized aggregate pseudo loss capped at human loss" in abc_stage
    assert "fixed owner validation set" in abc_stage
    assert "same immutable source rally" in abc_stage

    vm_run = (output / "VM_ABC_RUN.md").read_text()
    assert "staged media SHA" in vm_run
    assert "build_audio_onsets_v2.py" in vm_run
    assert "build_ball_inflections.py" in vm_run
    assert "build_abc_arm_manifests.py" in vm_run
    assert "must not\nrun a GT scorer" in vm_run

    hunk = (output / "LOADER_CHANGE_REQUIRED.diff").read_text()
    assert "unknown_frame_mask" in hunk
    assert "frame_loss_mask" in hunk
    assert 'manifest.get("schema_version") not in {1, 2}' in hunk
    assert "current early continue must move below this" in hunk
    assert "valid_target &= frame_loss_mask.bool()" in hunk
    assert '"frame_loss_mask", "sample_weight"' in hunk
    assert "NOT APPLIED" in hunk


def test_build_and_artifacts_are_byte_deterministic(tmp_path: Path) -> None:
    gallery = _fixture_gallery(tmp_path / "gallery")
    first = build_corpus(gallery, media_root=None, frame_times_root=None, seed=20260720)
    second = build_corpus(gallery, media_root=None, frame_times_root=None, seed=20260720)
    assert first == second

    output_a = tmp_path / "out_a"
    output_b = tmp_path / "out_b"
    hashes_a = write_corpus_artifacts(output_a, *first)
    hashes_b = write_corpus_artifacts(output_b, *second)
    assert hashes_a == hashes_b
    expected_names = {
        "manifest.json",
        "corpus_report.json",
        "teacher_event_context.jsonl",
        "compare_only_teacher_event_context.jsonl",
        "pbvision_filtered_teacher_manifest.json",
        "ABC_STAGE.md",
        "VM_ABC_RUN.md",
        "CORPUS_NOTES.md",
        "LOADER_CHANGE_REQUIRED.diff",
    }
    assert {path.name for path in output_a.iterdir()} == expected_names
    for name in expected_names:
        assert (output_a / name).read_bytes() == (output_b / name).read_bytes()

    cli_output = tmp_path / "cli_output"
    completed = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            CLI,
            "--input-root", str(gallery),
            "--media-root", str(tmp_path / "absent_media"),
            "--frame-times-root", str(tmp_path / "absent_frame_times"),
            "--output-dir", str(cli_output),
            "--seed", "20260720",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["training_ready"] is False
    for name in expected_names:
        assert (cli_output / name).read_bytes() == (output_a / name).read_bytes()


# ---------------------------------------------------------------------------
# --split-manifest frozen-ledger override
#
# assign_source_splits() derives its own seed-hashed train/val/test split
# from whichever sources happen to be eligible. Against the real gallery that
# self-computed split disagrees with the authoritative ledger partition: it
# puts ledger-VAL ids pldtjpw3h0jw/utasf5hnozwz into TRAIN and demotes
# ledger-TRAIN ids bewqc0glhgpq/tqjlrcntpjvt out of TRAIN. These tests pin
# down the override that replaces it, and prove the refusal that originally
# caught the mismatch downstream in build_abc_arm_manifests.py still fires.
# ---------------------------------------------------------------------------


def _frozen_role_assignment() -> dict[str, str]:
    assignment = {video_id: "train" for video_id in TRAIN_IDS}
    assignment.update({video_id: "val" for video_id in TEACHER_VAL_ONLY_IDS})
    assignment.update({video_id: "test" for video_id in TEACHER_TEST_ONLY_IDS})
    return assignment


def _write_split_manifest(path: Path, assignment: dict[str, str]) -> Path:
    path.write_text(json.dumps({
        "rows": [
            {"video": video_id, "source_video": video_id, "split": split}
            for video_id, split in assignment.items()
        ],
    }))
    return path


def test_split_manifest_override_replaces_self_computed_split_with_ledger_roles() -> None:
    manifest, _, _, _ = build_corpus(GALLERY_ROOT, split_manifest=FROZEN_SPLIT_MANIFEST)
    by_id = {row["source_video"]: row["split"] for row in manifest["rows"]}

    for video_id in TRAIN_IDS:
        assert by_id[video_id] == "train"
    for video_id in TEACHER_VAL_ONLY_IDS:
        assert by_id[video_id] == "val"
    for video_id in TEACHER_TEST_ONLY_IDS:
        assert by_id[video_id] == "test"

    # Prove the override actually overrode something: the self-computed split
    # for this real gallery disagrees with the ledger for exactly the ids the
    # original defect flipped.
    self_computed = assign_source_splits(
        {row["source_video"]: row["source_lineage_key"] for row in manifest["rows"]},
        seed=20260720,
    )
    assert self_computed["pldtjpw3h0jw"] == "train"
    assert self_computed["bewqc0glhgpq"] == "test"
    assert self_computed != by_id

    assert manifest["config"]["split_manifest_override"] == {
        "path": "runs/lanes/pbv_pickleball_corpus_20260720/manifest.json",
        "sha256": _sha256(FROZEN_SPLIT_MANIFEST),
    }


def test_frozen_split_manifest_fixture_matches_sst_builders_pinned_identity() -> None:
    # Sanity check on the fixture itself: build_pbvision_ball_sst.py pins this
    # exact file's SHA-256 as FROZEN_SPLIT_SHA256. If the committed file ever
    # changes without updating that constant, both builders' tests should
    # fail loudly rather than silently validating against a stale artifact.
    from scripts.racketsport.build_pbvision_ball_sst import FROZEN_SPLIT_SHA256

    assert _sha256(FROZEN_SPLIT_MANIFEST) == FROZEN_SPLIT_SHA256


def test_cli_accepts_split_manifest_and_applies_ledger_split(tmp_path: Path) -> None:
    cli_output = tmp_path / "cli_split_output"
    completed = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            CLI,
            "--input-root", str(GALLERY_ROOT),
            "--media-root", str(tmp_path / "absent_media"),
            "--frame-times-root", str(tmp_path / "absent_frame_times"),
            "--output-dir", str(cli_output),
            "--seed", "20260720",
            "--split-manifest", str(FROZEN_SPLIT_MANIFEST),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    manifest = json.loads((cli_output / "manifest.json").read_text())
    by_id = {row["source_video"]: row["split"] for row in manifest["rows"]}
    for video_id in TRAIN_IDS:
        assert by_id[video_id] == "train"
    for video_id in TEACHER_VAL_ONLY_IDS:
        assert by_id[video_id] == "val"
    for video_id in TEACHER_TEST_ONLY_IDS:
        assert by_id[video_id] == "test"


def test_split_manifest_absent_is_byte_identical_to_pre_override_behavior(
    tmp_path: Path,
) -> None:
    gallery = _fixture_gallery(tmp_path / "gallery")
    without_kwarg = build_corpus(gallery, media_root=None, frame_times_root=None, seed=20260720)
    with_explicit_none = build_corpus(
        gallery, media_root=None, frame_times_root=None, seed=20260720, split_manifest=None
    )
    assert without_kwarg == with_explicit_none

    manifest, _, _, _ = without_kwarg
    assert "split_manifest_override" not in manifest["config"]
    expected_assignment = assign_source_splits(
        {row["source_video"]: row["source_lineage_key"] for row in manifest["rows"]},
        seed=20260720,
    )
    assert {row["source_video"]: row["split"] for row in manifest["rows"]} == expected_assignment


def test_split_manifest_override_refuses_compare_only_alias_before_any_source_read(
    tmp_path: Path,
) -> None:
    compare_only_id = sorted(SST_COMPARE_ONLY_IDS)[0]
    split_manifest = _write_split_manifest(
        tmp_path / "split.json",
        {**_frozen_role_assignment(), compare_only_id: "train"},
    )
    # Also swap a legitimate id's row out for the compare-only id under an
    # alias key, matching _validate_frozen_split's alias-scan shape.
    payload = json.loads(split_manifest.read_text())
    payload["rows"] = [
        row for row in payload["rows"] if row["video"] != TRAIN_IDS[0]
    ]
    payload["rows"].append(
        {"video": compare_only_id, "source_video": compare_only_id, "clip_id": compare_only_id, "split": "train"}
    )
    split_manifest.write_text(json.dumps(payload))

    bogus_input_root = tmp_path / "this_input_root_does_not_exist"
    with pytest.raises(CorpusBuildError, match="compare-only alias"):
        build_corpus(bogus_input_root, split_manifest=split_manifest)


def test_split_manifest_override_refuses_wrong_frozen_role_assignment(tmp_path: Path) -> None:
    # Reproduce the exact real defect this override exists to prevent: a
    # ledger-VAL id relabeled TRAIN and a ledger-TRAIN id relabeled VAL. The
    # refusal that originally caught this mismatch downstream must still
    # fire -- now directly in the corpus builder, before the buggy split
    # can ever reach build_abc_arm_manifests.py.
    assignment = _frozen_role_assignment()
    assignment[TEACHER_VAL_ONLY_IDS[0]] = "train"
    assignment[TRAIN_IDS[0]] = "val"
    split_manifest = _write_split_manifest(tmp_path / "split.json", assignment)

    with pytest.raises(CorpusBuildError, match="violates frozen source roles"):
        build_corpus(GALLERY_ROOT, split_manifest=split_manifest)


def test_split_manifest_override_refuses_incomplete_frozen_id_set(tmp_path: Path) -> None:
    assignment = _frozen_role_assignment()
    del assignment[TRAIN_IDS[-1]]
    split_manifest = _write_split_manifest(tmp_path / "split.json", assignment)

    with pytest.raises(CorpusBuildError, match="frozen ten"):
        build_corpus(GALLERY_ROOT, split_manifest=split_manifest)


def test_split_manifest_override_refuses_id_set_disagreeing_with_local_gallery(
    tmp_path: Path,
) -> None:
    # The frozen ten is internally consistent, but if the local gallery's
    # eligible videos are not exactly that set, the override must not be
    # silently applied to a mismatched population.
    gallery = _fixture_gallery(tmp_path / "gallery")
    split_manifest = _write_split_manifest(tmp_path / "split.json", _frozen_role_assignment())

    with pytest.raises(CorpusBuildError, match="must exactly cover the eligible source"):
        build_corpus(gallery, media_root=None, frame_times_root=None, split_manifest=split_manifest)


def test_compare_only_holdouts_agree_with_sst_builder_frozen_ids() -> None:
    # build_pbvision_ball_sst.py's COMPARE_ONLY_IDS is the frozen source of
    # truth the override validator (_validate_frozen_split) checks against.
    # If this builder's own denylist ever drifts from it, a compare-only id
    # could be silently accepted or rejected inconsistently between the two
    # builders. This is a standing regression guard, independent of whether
    # --split-manifest is ever used.
    assert frozenset(COMPARE_ONLY_HOLDOUTS) == frozenset(SST_COMPARE_ONLY_IDS)


def test_abc_arm_compare_only_holdouts_agree_with_sst_builder_frozen_ids() -> None:
    assert frozenset(ABC_ARM_COMPARE_ONLY_HOLDOUTS) == frozenset(SST_COMPARE_ONLY_IDS)


def test_split_manifest_override_makes_abc_arm_consume_ledger_train_rows() -> None:
    # build_abc_arm_manifests.py never recomputes or validates splits itself
    # -- _validate_teacher_manifest simply trusts row["split"] as written by
    # the corpus builder. That makes it correct by construction once the
    # corpus builder's split is correct, and it means there is no second
    # split opinion anywhere downstream to keep in sync.
    overridden_manifest, _, _, _ = build_corpus(
        GALLERY_ROOT, split_manifest=FROZEN_SPLIT_MANIFEST
    )
    rows = _validate_teacher_manifest(overridden_manifest)
    assert {row["source_video"] for row in rows} == set(TRAIN_IDS)

    # Without the override, abc-arm would have silently ingested the wrong
    # train population -- the original defect this override closes.
    buggy_manifest, _, _, _ = build_corpus(GALLERY_ROOT)
    buggy_train_ids = {
        row["source_video"] for row in buggy_manifest["rows"] if row["split"] == "train"
    }
    assert buggy_train_ids != set(TRAIN_IDS)
