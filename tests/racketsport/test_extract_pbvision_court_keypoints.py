from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

import scripts.racketsport.extract_pbvision_court_keypoints as pbv_extractor
from scripts.racketsport.extract_pbvision_court_keypoints import (
    _candidate_score_for_segments,
    _camera_segments,
    _mapping_from_cover,
    CANONICAL_WORLD_XY,
    COMPARE_ONLY_IDS,
    DROPPED_CORPUS_IDS,
    GROUND_NAMES,
    PENDING_STATUS,
    PROVENANCE,
    build_timebase_audit,
    choose_target_frames,
    discover_mapping,
    load_records,
    validate_per_video_assignments,
)
from scripts.racketsport.train_court_keypoint_heatmap import (
    OWNER_APPROVED_STATUS,
    OWNER_ELIGIBILITY_ACT_RELATIVE_PATH,
    OWNER_ELIGIBILITY_ACT_SHA256,
    _frame_dir,
    court_keypoint_label_rows,
    load_label_image,
    load_real_court_keypoint_labels,
)
from scripts.racketsport.train_court_model_v2 import load_real_training_rows


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/racketsport/extract_pbvision_court_keypoints.py"
LOCAL_ID = "83gyqyc10y8f"
ELIGIBLE_ID = "fixtureeligible"
REAL_CORPUS_ROOT = ROOT / "data/court_real_pbvision_20260722"
REAL_GALLERY_ROOT = ROOT / "data/pbvision_gallery_20260719"
OWNER_ELIGIBILITY_ACT = ROOT / OWNER_ELIGIBILITY_ACT_RELATIVE_PATH
OWNER_ACT_PROMOTION_IDS = frozenset(
    {
        "0tmdeghtfvjx",
        "143sf3gdwxsa",
        "98z43hspqz13",
        "pldtjpw3h0jw",
        "td2szayjwtrj",
        "tqjlrcntpjvt",
    }
)
EXPECTED_REAL_MAPPING = (
    "far_left_corner",
    "far_baseline_center",
    "far_right_corner",
    "far_nvz_left",
    "far_nvz_center",
    "far_nvz_right",
    "near_nvz_left",
    "near_nvz_center",
    "near_nvz_right",
    "near_left_corner",
    "near_baseline_center",
    "near_right_corner",
)
REAL_0TM_NORMALIZED_POINTS = (
    (0.68583, 0.458567),
    (0.827151, 0.472184),
    None,
    (0.569603, 0.487207),
    (0.729281, 0.507246),
    (0.954384, 0.534124),
    (0.3922, 0.525699),
    (0.563257, 0.562665),
    (0.844306, 0.620391),
    (0.025241, 0.609233),
    (0.20002, 0.697017),
    (0.538026, 0.860998),
)
REAL_BEWQC_NORMALIZED_POINTS = (
    (0.0345, 0.59516),
    (0.139048, 0.472978),
    (0.218023, 0.3984),
    (0.264074, 0.648427),
    (0.346165, 0.46666),
    (0.389392, 0.385966),
    (0.714128, 0.648447),
    (0.628466, 0.464286),
    (0.588736, 0.383151),
    (0.944741, 0.5927),
    (0.838918, 0.468052),
    (0.762699, 0.397792),
)
OLD_XKADS_NATIVE_BUG_INDICES = [
    120,
    352,
    585,
    817,
    1050,
    1282,
    1514,
    1747,
    1979,
    2212,
    2444,
    2676,
    2909,
    3141,
    3373,
    3606,
    3838,
    4071,
    4303,
    4535,
    4768,
    5000,
    5233,
    5465,
]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _real_payload(video_id: str = "0tmdeghtfvjx") -> dict:
    return json.loads(
        (REAL_CORPUS_ROOT / video_id / "labels/court_keypoints.json").read_text(encoding="utf-8")
    )


def _bind_to_owner_eligibility_act(payload: dict) -> dict:
    payload = json.loads(json.dumps(payload))
    video_id = str(payload["clip"])
    assert _sha256(OWNER_ELIGIBILITY_ACT) == OWNER_ELIGIBILITY_ACT_SHA256
    payload["status"] = OWNER_APPROVED_STATUS
    payload["training_eligibility"] = {
        "queued": True,
        "owner_adjudication": {
            "path": OWNER_ELIGIBILITY_ACT_RELATIVE_PATH,
            "sha256": _sha256(OWNER_ELIGIBILITY_ACT),
            "video_id": video_id,
            "decision": "APPROVE",
        },
    }
    for item in payload["annotation"]["items"]:
        item["pseudo_label_status"] = OWNER_APPROVED_STATUS
    return payload


def _write_single_payload_corpus(tmp_path: Path, payload: dict) -> Path:
    corpus_root = tmp_path / "court_corpus"
    _write_json(corpus_root / str(payload["clip"]) / "labels/court_keypoints.json", payload)
    return corpus_root


def _homography(variant: int, *, width: int, height: int) -> np.ndarray:
    far_left = CANONICAL_WORLD_XY["far_left_corner"]
    far_right = CANONICAL_WORLD_XY["far_right_corner"]
    near_right = CANONICAL_WORLD_XY["near_right_corner"]
    near_left = CANONICAL_WORLD_XY["near_left_corner"]
    world = np.asarray([far_left, far_right, near_right, near_left], dtype=np.float32)
    shift = float(variant * 2)
    image = np.asarray(
        [
            [100 + shift, 38 + variant],
            [220 + shift, 42 - variant],
            [285 - shift, 155 - variant],
            [35 + shift, 150 + variant],
        ],
        dtype=np.float32,
    )
    assert image[:, 0].max() < width and image[:, 1].max() < height
    return cv2.getPerspectiveTransform(world, image)


def _court_points(variant: int, *, width: int, height: int) -> list[dict[str, float]]:
    homography = _homography(variant, width=width, height=height)
    world = np.asarray([CANONICAL_WORLD_XY[name] for name in GROUND_NAMES], dtype=np.float64)
    pixels = cv2.perspectiveTransform(world.reshape(-1, 1, 2), homography).reshape(-1, 2)
    rng = np.random.default_rng(20260722 + variant)
    pixels += rng.normal(0.0, 0.08, pixels.shape)
    return [
        {
            "u": float(x / width),
            "v": float(y / height),
            "confidence": float(0.94 + index * 0.002),
            "spread": float(0.001 + index * 0.00005),
        }
        for index, (x, y) in enumerate(pixels)
    ]


def _write_export(path: Path, variant: int, *, width: int, height: int, frame_count: int) -> None:
    _write_json(
        path,
        {
            "camera": {
                "fps": 30.0,
                "cameraSegments": [
                    {
                        "s": 0,
                        "e": frame_count - 1,
                        "fov": 1.2,
                        "position": {"x": 2.0, "y": 52.0, "z": 5.0},
                        "orientation": {"pitch": -0.2, "roll": 0.0, "yaw": -2.4},
                        "court_points": _court_points(variant, width=width, height=height),
                    }
                ],
            }
        },
    )


def _build_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    width, height, frame_count = 320, 180, 150
    gallery_root = tmp_path / "gallery"
    gallery_ids = [ELIGIBLE_ID, "iottnc0h3ekn", "o4dee9dn0ccr"]
    rows = []
    for variant, video_id in enumerate(gallery_ids):
        video_root = gallery_root / video_id
        if video_id == ELIGIBLE_ID:
            _write_export(
                video_root / "cv_export.json",
                variant,
                width=width,
                height=height,
                frame_count=frame_count,
            )
            _write_json(
                video_root / "api_get_metadata.json",
                {
                    "metadata": {
                        "fps": 30.0,
                        "height": height,
                        "width": width,
                        "secs": frame_count / 30.0,
                    }
                },
            )
        else:
            # These deliberately invalid source payloads prove identity rejection happens
            # before parse. The unit-level access trap below separately covers stat/open/probe.
            video_root.mkdir(parents=True, exist_ok=True)
            (video_root / "cv_export.json").write_text("COMPARE_ONLY_MUST_NOT_PARSE\n", encoding="utf-8")
            (video_root / "api_get_metadata.json").write_text(
                "COMPARE_ONLY_MUST_NOT_PARSE\n", encoding="utf-8"
            )
        rows.append(
            {
                "video_id": video_id,
                "title": f"Fixture {video_id}",
                "duration_s": frame_count / 30.0,
                "resolution": f"{width}x{height}@30fps",
                "video_sha256": f"fixture-{video_id}",
            }
        )

    local_root = tmp_path / "local"
    local_root.mkdir(parents=True)
    rows.append(
        {
            "video_id": LOCAL_ID,
            "title": "Fixture Demo Vid",
            "duration_s": frame_count / 30.0,
            "resolution": f"{width}x{height}@30/1",
            "video_sha256": "COMPARE_ONLY_MUST_NOT_HASH",
        }
    )
    manifest = gallery_root / "MANIFEST.json"
    _write_json(manifest, {"videos": rows})

    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    return gallery_root, local_root, manifest, runs_root


def test_compare_only_identities_are_rejected_before_source_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gallery_root, local_root, manifest, _ = _build_fixture(tmp_path)
    protected_roots = {
        gallery_root / "iottnc0h3ekn",
        gallery_root / "o4dee9dn0ccr",
        local_root,
    }
    contacts: list[str] = []

    def protected(path: Path) -> bool:
        return any(path == root or root in path.parents for root in protected_roots)

    original_open = Path.open
    original_stat = Path.stat

    def guarded_open(path: Path, *args, **kwargs):
        if protected(path):
            contacts.append(f"open:{path}")
            raise AssertionError(f"compare-only source opened: {path}")
        return original_open(path, *args, **kwargs)

    def guarded_stat(path: Path, *args, **kwargs):
        if protected(path):
            contacts.append(f"stat:{path}")
            raise AssertionError(f"compare-only source stated: {path}")
        return original_stat(path, *args, **kwargs)

    original_run = pbv_extractor.subprocess.run

    def guarded_run(command, *args, **kwargs):
        command_text = " ".join(str(part) for part in command)
        if any(str(root) in command_text for root in protected_roots):
            contacts.append(f"probe:{command_text}")
            raise AssertionError(f"compare-only source probed: {command_text}")
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "stat", guarded_stat)
    monkeypatch.setattr(pbv_extractor.subprocess, "run", guarded_run)

    records = load_records(
        gallery_root=gallery_root,
        local_root=local_root,
        manifest_path=manifest,
        local_video_id=LOCAL_ID,
        ffprobe_bin="ffprobe",
    )

    assert [record.video_id for record in records] == [ELIGIBLE_ID]
    assert contacts == []


def test_compare_only_policy_is_not_caller_overridable() -> None:
    parser = pbv_extractor.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--compare-only-id", ELIGIBLE_ID])


@lru_cache(maxsize=1)
def _permitted_real_records():
    records = tuple(
        load_records(
            gallery_root=REAL_GALLERY_ROOT,
            local_root=ROOT / "data/pbvision_11min_20260713",
            manifest_path=REAL_GALLERY_ROOT / "MANIFEST.json",
            local_video_id=LOCAL_ID,
            ffprobe_bin="ffprobe",
        )
    )
    assert len(records) == 10
    assert {record.video_id for record in records}.isdisjoint(COMPARE_ONLY_IDS)
    return records


def _normalized_points(record) -> tuple[tuple[float, float] | None, ...]:
    points = _camera_segments(record)[0]["court_points"]
    return tuple(None if point is None else (float(point["u"]), float(point["v"])) for point in points)


def _assert_xkads_native_span(frame_indices: list[int]) -> None:
    assert len(frame_indices) == 24
    assert frame_indices[0] / 60.0 == pytest.approx(2.0)
    assert frame_indices[-1] / 60.0 >= 184.0
    assert frame_indices[-1] <= 11160


def test_target_frames_are_uniform_inside_two_second_guard() -> None:
    result = choose_target_frames(0, 899, 30.0, 24)
    assert len(result) == 24
    assert result[0] == 60
    assert result[-1] == 839
    assert result == sorted(set(result))


def test_xkads_60fps_pbv_ticks_convert_to_full_native_source_span() -> None:
    converted = choose_target_frames(
        0,
        5585,
        30.0,
        24,
        source_fps=60.0,
        frame_count=11161,
    )
    _assert_xkads_native_span(converted)
    assert converted[0] == 120
    assert converted[-1] == 11050

    # The reviewed old manifest used PBV ticks as native indices. This guard is
    # deliberately red on its 120..5465 half-duration span.
    with pytest.raises(AssertionError):
        _assert_xkads_native_span(OLD_XKADS_NATIVE_BUG_INDICES)


def test_real_timebase_audit_closes_all_ten_original_corpus_candidates() -> None:
    audit = build_timebase_audit(
        list(_permitted_real_records()),
        ffprobe_bin="ffprobe",
    )
    rows = {row["video"]: row for row in audit["videos"]}
    assert len(rows) == 10
    assert {video_id for video_id, row in rows.items() if row["conversion_applied"]} == {
        "xkadsq9bli3h"
    }
    xkads = rows["xkadsq9bli3h"]
    assert xkads["pbv_tick_rate"] == 30.0
    assert xkads["source_fps"] == 60.0
    assert xkads["conversion_factor"] == 2.0
    assert xkads["local_ffprobe"]["fps"] == 60.0
    assert xkads["local_ffprobe"]["frame_count"] == 11168
    assert all(
        row["pbv_tick_rate"] == row["source_fps"]
        for video_id, row in rows.items()
        if video_id != "xkadsq9bli3h"
    )


def test_real_pending_payload_is_rejected_by_legacy_loader_path() -> None:
    with pytest.raises(ValueError, match="diagnostic-only.*PENDING_SPOTCHECK"):
        load_real_court_keypoint_labels(REAL_CORPUS_ROOT)


def test_real_pending_payload_is_rejected_by_v2_loader_path() -> None:
    with pytest.raises(ValueError, match="diagnostic-only.*PENDING_SPOTCHECK"):
        load_real_training_rows([REAL_CORPUS_ROOT])


def test_real_pending_payload_requires_explicit_diagnostic_opt_in() -> None:
    legacy_rows = load_real_court_keypoint_labels(
        REAL_CORPUS_ROOT,
        allow_pending_diagnostic_only=True,
    )
    v2_rows = load_real_training_rows(
        [REAL_CORPUS_ROOT],
        allow_pending_diagnostic_only=True,
    )
    assert len(legacy_rows) == len(v2_rows) == 9
    assert {row["clip"] for row in legacy_rows} == {row["clip"] for row in v2_rows}
    assert "bewqc0glhgpq" not in {row["clip"] for row in legacy_rows}


@pytest.mark.parametrize("trainer_path", ["legacy", "v2"])
def test_stripped_real_external_payload_is_rejected_by_both_trainer_paths(
    trainer_path: str,
    tmp_path: Path,
) -> None:
    payload = _bind_to_owner_eligibility_act(_real_payload())
    payload.pop("status")
    payload.pop("training_eligibility")
    for item in payload["annotation"]["items"]:
        item.pop("pseudo_label_status")
    corpus_root = _write_single_payload_corpus(tmp_path, payload)

    loader = (
        (lambda: load_real_court_keypoint_labels(corpus_root))
        if trainer_path == "legacy"
        else (lambda: load_real_training_rows([corpus_root]))
    )
    with pytest.raises(ValueError, match="missing positive training-eligibility act"):
        loader()


@pytest.mark.parametrize("trainer_path", ["legacy", "v2"])
def test_status_preserved_provenance_stripped_external_payload_is_denied_with_zero_rows(
    trainer_path: str,
    tmp_path: Path,
) -> None:
    """Regress the reviewer fixture that exposed the status-only eligibility bypass."""

    payload = _real_payload()
    payload.pop("status", None)
    payload.pop("training_eligibility", None)
    payload.pop("provenance", None)
    for item in payload["annotation"]["items"]:
        item.pop("pseudo_label_status", None)
        item.pop("provenance", None)
        assert item["status"] == "reviewed_external_dataset"
    corpus_root = _write_single_payload_corpus(tmp_path, payload)

    loader = (
        (lambda: load_real_court_keypoint_labels(corpus_root))
        if trainer_path == "legacy"
        else (lambda: load_real_training_rows([corpus_root]))
    )
    result = {"outcome": "ADMITTED", "row_count": 0}
    try:
        rows = loader()
    except ValueError as exc:
        result = {"outcome": "DENIED", "row_count": 0, "typed_reason": str(exc)}
    else:
        result["row_count"] = len(rows)

    assert result["outcome"] == "DENIED"
    assert result["row_count"] == 0
    assert "missing positive training-eligibility act" in result["typed_reason"]


@pytest.mark.parametrize("missing_field", ["top_status", "eligibility", "item_status"])
def test_each_positive_external_eligibility_marker_is_required(missing_field: str) -> None:
    payload = _bind_to_owner_eligibility_act(_real_payload())
    if missing_field == "top_status":
        payload.pop("status")
    elif missing_field == "eligibility":
        payload.pop("training_eligibility")
    else:
        for item in payload["annotation"]["items"]:
            item.pop("pseudo_label_status")

    with pytest.raises(ValueError, match="missing positive training-eligibility act"):
        court_keypoint_label_rows(payload)


def test_owner_adjudication_positive_act_admits_external_row_through_both_trainers(
    tmp_path: Path,
) -> None:
    payload = _bind_to_owner_eligibility_act(_real_payload())
    corpus_root = _write_single_payload_corpus(tmp_path, payload)

    legacy_rows = load_real_court_keypoint_labels(corpus_root)
    v2_rows = load_real_training_rows([corpus_root])

    assert len(legacy_rows) == len(v2_rows) == 1
    assert legacy_rows[0]["clip"] == v2_rows[0]["clip"] == "0tmdeghtfvjx"
    assert legacy_rows[0]["label_status"] == "reviewed_external_dataset"


@pytest.mark.parametrize("mutation", ["sha256", "video_id", "decision"])
def test_owner_adjudication_binding_rejects_mutated_metadata(mutation: str) -> None:
    payload = _bind_to_owner_eligibility_act(_real_payload())
    owner_adjudication = payload["training_eligibility"]["owner_adjudication"]
    if mutation == "sha256":
        owner_adjudication["sha256"] = "0" * 64
    elif mutation == "video_id":
        owner_adjudication["video_id"] = "143sf3gdwxsa"
    else:
        owner_adjudication["decision"] = "DROP"

    with pytest.raises(ValueError, match="owner adjudication"):
        court_keypoint_label_rows(payload)


def test_owner_adjudication_final_drop_cannot_be_presented_as_positive() -> None:
    payload = _bind_to_owner_eligibility_act(_real_payload("utasf5hnozwz"))
    with pytest.raises(ValueError, match="final decision.*DROP"):
        court_keypoint_label_rows(payload)


def test_owner_adjudication_clip_binding_matches_loader_directory(tmp_path: Path) -> None:
    payload = _bind_to_owner_eligibility_act(_real_payload())
    corpus_root = tmp_path / "court_corpus"
    _write_json(
        corpus_root / "substituted_clip" / "labels/court_keypoints.json",
        payload,
    )

    with pytest.raises(ValueError, match="payload clip.*loader directory"):
        load_real_court_keypoint_labels(corpus_root)


def test_legacy_independently_reviewed_human_row_needs_no_new_act(tmp_path: Path) -> None:
    payload = _real_payload()
    payload.pop("clip", None)
    payload.pop("status", None)
    payload.pop("training_eligibility", None)
    payload.pop("provenance", None)
    for item in payload["annotation"]["items"]:
        item["status"] = "reviewed"
        item.pop("pseudo_label_status", None)
        item.pop("provenance", None)
    payload["clip"] = "legacy_human_clip"
    payload["review"] = {
        "status": "reviewed",
        "reviewer": "legacy-independent-human-review",
    }
    corpus_root = _write_single_payload_corpus(tmp_path, payload)

    legacy_rows = load_real_court_keypoint_labels(corpus_root)
    v2_rows = load_real_training_rows([corpus_root])

    assert len(legacy_rows) == len(v2_rows) == 1
    assert legacy_rows[0]["label_status"] == v2_rows[0]["label_status"] == "reviewed"


@pytest.mark.parametrize("only_marker", ["top_status", "queued_false", "item_pending"])
def test_each_real_pending_marker_independently_defaults_to_deny(only_marker: str) -> None:
    payload = json.loads(
        (REAL_CORPUS_ROOT / "0tmdeghtfvjx/labels/court_keypoints.json").read_text(encoding="utf-8")
    )
    payload.pop("status", None)
    payload.pop("training_eligibility", None)
    for item in payload["annotation"]["items"]:
        item.pop("pseudo_label_status", None)
    if only_marker == "top_status":
        payload["status"] = PENDING_STATUS
    elif only_marker == "queued_false":
        payload["training_eligibility"] = {"queued": False}
    else:
        payload["annotation"]["items"][0]["pseudo_label_status"] = PENDING_STATUS

    with pytest.raises(ValueError, match="diagnostic-only"):
        court_keypoint_label_rows(payload)
    assert len(court_keypoint_label_rows(payload, allow_pending_diagnostic_only=True)) == 1


def test_real_mapping_null_slot_and_side_view_constrained_caveat() -> None:
    records = list(_permitted_real_records())
    by_id = {record.video_id: record for record in records}
    assert _normalized_points(by_id["0tmdeghtfvjx"]) == REAL_0TM_NORMALIZED_POINTS
    assert _normalized_points(by_id["bewqc0glhgpq"]) == REAL_BEWQC_NORMALIZED_POINTS

    batch_mapping, _ = discover_mapping(records)
    asserted_mapping = list(batch_mapping.index_to_name)
    if os.environ.get("PBV_MAPPING_MUTATION") == "swap_0_1":
        asserted_mapping[0], asserted_mapping[1] = asserted_mapping[1], asserted_mapping[0]
    assert batch_mapping.groups == ((0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11))
    assert tuple(asserted_mapping) == EXPECTED_REAL_MAPPING

    validation = {
        row["video_id"]: row
        for row in validate_per_video_assignments(records, batch_mapping)["videos"]
    }
    assert validation["0tmdeghtfvjx"]["missing_indices"] == [2]
    assert validation["0tmdeghtfvjx"]["status"] == "supports_batch_mapping_with_missing_export_slot"
    assert validation["bewqc0glhgpq"]["consistent_with_batch_mapping"] is True
    assert validation["bewqc0glhgpq"]["structured_candidates_scored"] == 48

    with pytest.raises(ValueError, match="index 2 is null"):
        discover_mapping([by_id["0tmdeghtfvjx"]])
    # Pin the real unconstrained single-view result from the review without
    # repeating its unbounded 220-triple exact-cover search in every wide run.
    bewqc_points = [_camera_segments(by_id["bewqc0glhgpq"])[0]["court_points"]]
    unconstrained_cover = (
        (5, 8, 11),
        (4, 7, 10),
        (1, 2, 9),
        (0, 3, 6),
    )
    unconstrained_groups, unconstrained_mapping = _mapping_from_cover(
        unconstrained_cover,
        bewqc_points,
    )
    assert unconstrained_groups == unconstrained_cover
    assert unconstrained_mapping == (
        "near_left_corner",
        "near_nvz_left",
        "near_nvz_center",
        "near_baseline_center",
        "far_nvz_left",
        "far_left_corner",
        "near_right_corner",
        "far_nvz_center",
        "far_baseline_center",
        "near_nvz_right",
        "far_nvz_right",
        "far_right_corner",
    )
    assert unconstrained_mapping != EXPECTED_REAL_MAPPING
    assert _candidate_score_for_segments(unconstrained_mapping, bewqc_points) == pytest.approx(
        0.05191300609957279
    )
    assert _candidate_score_for_segments(EXPECTED_REAL_MAPPING, bewqc_points) == pytest.approx(
        0.020480063999121372
    )


def test_real_corpus_has_zero_compare_only_or_dropped_loader_rows() -> None:
    loader_video_ids = {
        path.parent.parent.name for path in REAL_CORPUS_ROOT.glob("*/labels/court_keypoints.json")
    }
    assert len(loader_video_ids) == 9
    assert loader_video_ids.isdisjoint(COMPARE_ONLY_IDS)
    assert loader_video_ids.isdisjoint(DROPPED_CORPUS_IDS)
    for video_id in COMPARE_ONLY_IDS | DROPPED_CORPUS_IDS:
        assert not (REAL_CORPUS_ROOT / video_id).exists()

    families = json.loads((REAL_CORPUS_ROOT / "families.json").read_text(encoding="utf-8"))
    assert families["bewqc0glhgpq"]["corpus_eligible"] is False
    assert "bad multi-point planar solve" in families["bewqc0glhgpq"]["reason"]
    assert all(families[video_id]["corpus_eligible"] is False for video_id in COMPARE_ONLY_IDS)


def test_real_corpus_binds_exact_six_video_promotion_set_to_owner_act() -> None:
    payloads = {
        path.parent.parent.name: json.loads(path.read_text(encoding="utf-8"))
        for path in REAL_CORPUS_ROOT.glob("*/labels/court_keypoints.json")
    }
    promoted_ids = {
        video_id
        for video_id, payload in payloads.items()
        if payload.get("status") == OWNER_APPROVED_STATUS
    }

    assert promoted_ids == OWNER_ACT_PROMOTION_IDS
    for video_id in sorted(promoted_ids):
        payload = payloads[video_id]
        assert payload["training_eligibility"] == {
            "owner_adjudication": {
                "decision": "APPROVE",
                "path": OWNER_ELIGIBILITY_ACT_RELATIVE_PATH,
                "sha256": OWNER_ELIGIBILITY_ACT_SHA256,
                "video_id": video_id,
            },
            "queued": True,
            "reason": (
                "final owner adjudication approves this PBV pseudo-label for the court "
                "training pool"
            ),
        }
        assert {
            item["pseudo_label_status"] for item in payload["annotation"]["items"]
        } == {OWNER_APPROVED_STATUS}
        assert payload["review"]["owner_approved_external_count"] == len(
            payload["annotation"]["items"]
        )
        assert payload["review"]["pending_spotcheck_count"] == 0

    assert payloads["st0epgnab7dr"]["status"] == PENDING_STATUS
    assert payloads["utasf5hnozwz"]["status"] == PENDING_STATUS
    assert payloads["xkadsq9bli3h"]["status"] == PENDING_STATUS
    assert sum(
        len(
            json.loads(
                (REAL_CORPUS_ROOT / video_id / "frames_needed.json").read_text(
                    encoding="utf-8"
                )
            )["frame_indices"]
        )
        for video_id in promoted_ids
    ) == 144


def test_extractor_direct_cli_uses_only_permitted_sources_and_emits_pending_corpus(
    tmp_path: Path,
) -> None:
    """Direct-CLI reference for scripts/racketsport/extract_pbvision_court_keypoints.py."""
    gallery_root, local_root, manifest, runs_root = _build_fixture(tmp_path)
    output_root = tmp_path / "court_real_pbvision"
    lane_dir = tmp_path / "lane"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--gallery-root",
            str(gallery_root),
            "--local-root",
            str(local_root),
            "--manifest",
            str(manifest),
            "--output-root",
            str(output_root),
            "--lane-dir",
            str(lane_dir),
            "--runs-root",
            str(runs_root),
            "--expected-video-count",
            "1",
            "--expected-corpus-videos",
            "1",
            "--frames-per-segment",
            "24",
            "--json",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    stdout = json.loads(completed.stdout)
    assert stdout["mapping_consistent"] == "1/1"
    assert stdout["corpus_videos"] == 1
    assert stdout["corpus_rows"] == 1
    assert stdout["target_frames"] == 24
    assert stdout["spotcheck_overlays"] == 0

    summary = json.loads((lane_dir / "extraction_summary.json").read_text(encoding="utf-8"))
    assert [row["canonical_name"] for row in summary["assignment"]["mapping"]] == list(GROUND_NAMES)
    assert summary["existing_solver_cross_reference"] == {
        "status": "not_run_no_permitted_local_source",
    }
    assert summary["spotcheck"] == {
        "overlay_count": 0,
        "status": "not_run_no_permitted_local_source",
    }
    assert not (lane_dir / "spotcheck_11min").exists()

    # The two quarantined gallery videos and the local compare-only video have
    # zero rows and no frames/labels directory under the training corpus.
    assert COMPARE_ONLY_IDS == {LOCAL_ID, "iottnc0h3ekn", "o4dee9dn0ccr"}
    for video_id in COMPARE_ONLY_IDS:
        assert not (output_root / video_id).exists()
    assert sorted(path.parent.parent.name for path in output_root.glob("*/labels/court_keypoints.json")) == [
        ELIGIBLE_ID
    ]

    labels_path = output_root / ELIGIBLE_ID / "labels" / "court_keypoints.json"
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    assert payload["status"] == PENDING_STATUS
    assert payload["provenance"] == PROVENANCE
    assert payload["training_eligibility"]["queued"] is False
    assert payload["frames"]["path_base"] == "corpus_root"
    assert payload["frames"]["frame_dir"] == f"{ELIGIBLE_ID}/frames"
    assert _frame_dir(payload, corpus_root=output_root) == output_root / ELIGIBLE_ID / "frames"
    item = payload["annotation"]["items"][0]
    assert item["pseudo_label_status"] == PENDING_STATUS
    assert set(item["keypoints"]) == set(CANONICAL_WORLD_XY)
    assert sum(value is not None for value in item["keypoints"].values()) == 12
    assert sum(value is not None for value in item["keypoint_confidence"].values()) == 12

    frames_needed = json.loads(
        (output_root / ELIGIBLE_ID / "frames_needed.json").read_text(encoding="utf-8")
    )
    assert len(frames_needed["frame_indices"]) == 24
    assert frames_needed["frame_dir"] == f"{ELIGIBLE_ID}/frames"
    assert frames_needed["gcs_url"].endswith(f"/{ELIGIBLE_ID}/max.mp4")

    # The default shared loader rejects the exact pending payload before and
    # after materialization. Explicit diagnostic inspection can still prove
    # corpus-root rebasing and the real image-load path without training it.
    with pytest.raises(ValueError, match="diagnostic-only"):
        load_real_court_keypoint_labels(output_root)
    before = load_real_court_keypoint_labels(output_root, allow_pending_diagnostic_only=True)
    assert len(before) == 1
    assert before[0]["image_path"] is None
    fake_frame = output_root / payload["frames"]["frame_dir"] / item["frame"]
    fake_frame.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (320, 180), (31, 101, 72)).save(fake_frame)
    with pytest.raises(ValueError, match="diagnostic-only"):
        load_real_court_keypoint_labels(output_root)
    after = load_real_court_keypoint_labels(output_root, allow_pending_diagnostic_only=True)
    assert after[0]["image_path"] == str(fake_frame)
    loaded = load_label_image(after[0], cv2=cv2, image_module=Image)
    assert loaded.size == (320, 180)
    assert loaded.mode == "RGB"

    assert summary["videos"][0]["segments"][0]["all_point_residual_estimator"] == {
        "accuracy_claim": "geometric self-consistency only; not independent ground truth",
        "fit": "all usable points with OpenCV method-0 refinement",
        "implementation": "cv2.findHomography",
        "method": 0,
    }
    assert "not an independent partition win" in summary["assignment"]["side_view_note"]

    families = json.loads((output_root / "families.json").read_text(encoding="utf-8"))
    assert len(families) == 1
    assert families[ELIGIBLE_ID] == {
        "corpus_eligible": True,
        "family": f"pbv_{ELIGIBLE_ID}",
        "title": f"Fixture {ELIGIBLE_ID}",
    }
    assert set(families).isdisjoint(COMPARE_ONLY_IDS)
