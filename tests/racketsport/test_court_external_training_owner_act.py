from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.racketsport.train_court_keypoint_heatmap import (
    FLOOR_ONLY_OWNER_ELIGIBILITY_ACT_RELATIVE_PATH,
    FLOOR_ONLY_OWNER_ELIGIBILITY_ACT_SHA256,
    FLOOR_ONLY_TRAINING_CONDITION,
    PBVISION_PSEUDO_PROVENANCE,
    court_keypoint_label_rows,
)
from scripts.racketsport.train_court_model_v2 import load_real_training_rows
from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINTS


ROOT = Path(__file__).resolve().parents[2]


def _payload(*, source: str = "roboflow", scope_id: str = "approved_dataset") -> dict:
    keypoints = {point.name: None for point in PICKLEBALL_KEYPOINTS}
    keypoints["near_left_corner"] = [12.0, 34.0]
    if source == "roboflow":
        clip = f"{scope_id}__train"
        scope_type = "roboflow_dataset"
        provenance: object = {"dataset": scope_id, "sha256": "fixture"}
        top_provenance = None
    else:
        clip = scope_id
        scope_type = "pbvision_video"
        provenance = PBVISION_PSEUDO_PROVENANCE
        top_provenance = PBVISION_PSEUDO_PROVENANCE
    return {
        "annotation": {
            "items": [
                {
                    "frame": "frame_000001.jpg",
                    "keypoints": keypoints,
                    "provenance": provenance,
                    "pseudo_label_status": "OWNER_APPROVED",
                    "status": "reviewed_external_dataset",
                }
            ]
        },
        "clip": clip,
        "frames": {
            "frame_dir": f"{clip}/frames",
            "label_coordinate_space": [640, 360],
            "path_base": "corpus_root",
            "source_resolution": [640, 360],
        },
        "provenance": top_provenance,
        "review": {"status": "reviewed"},
        "status": "OWNER_APPROVED",
        "training_eligibility": {
            "owner_adjudication": {
                "decision": "APPROVE",
                "path": FLOOR_ONLY_OWNER_ELIGIBILITY_ACT_RELATIVE_PATH,
                "scope_id": scope_id,
                "scope_type": scope_type,
                "sha256": FLOOR_ONLY_OWNER_ELIGIBILITY_ACT_SHA256,
                "training_condition": FLOOR_ONLY_TRAINING_CONDITION,
            },
            "queued": True,
        },
    }


def _load(payload: dict, loader: str, tmp_path: Path) -> list[dict]:
    if loader == "legacy":
        return court_keypoint_label_rows(payload, clip_root=tmp_path / payload["clip"], corpus_root=tmp_path)
    label_path = tmp_path / payload["clip"] / "labels" / "court_keypoints.json"
    label_path.parent.mkdir(parents=True)
    label_path.write_text(json.dumps(payload), encoding="utf-8")
    return load_real_training_rows([tmp_path])


def test_floor_only_owner_act_bytes_match_pinned_sha256() -> None:
    act = ROOT / FLOOR_ONLY_OWNER_ELIGIBILITY_ACT_RELATIVE_PATH
    assert hashlib.sha256(act.read_bytes()).hexdigest() == FLOOR_ONLY_OWNER_ELIGIBILITY_ACT_SHA256


@pytest.mark.parametrize("loader", ["legacy", "v2"])
@pytest.mark.parametrize(
    ("source", "scope_id"),
    [
        ("roboflow", "chetan-rajagiri-9abfm__pickleball-court-v2__v1"),
        ("pbvision", "xkadsq9bli3h"),
    ],
)
def test_floor_only_owner_act_admits_bound_null_net_row(
    loader: str, source: str, scope_id: str, tmp_path: Path
) -> None:
    rows = _load(_payload(source=source, scope_id=scope_id), loader, tmp_path)
    assert len(rows) == 1
    assert not any(name.startswith("net_") for name in rows[0]["keypoints"])


@pytest.mark.parametrize("loader", ["legacy", "v2"])
def test_floor_only_owner_act_rejects_nonnull_net_supervision(loader: str, tmp_path: Path) -> None:
    payload = _payload()
    payload["annotation"]["items"][0]["keypoints"]["net_left_sideline"] = [20.0, 30.0]
    with pytest.raises(ValueError, match="requires every external net channel to be null"):
        _load(payload, loader, tmp_path)


@pytest.mark.parametrize("loader", ["legacy", "v2"])
def test_floor_only_owner_act_rejects_wrong_scope_binding(loader: str, tmp_path: Path) -> None:
    payload = _payload()
    payload["training_eligibility"]["owner_adjudication"]["scope_id"] = "different_dataset"
    with pytest.raises(ValueError, match="scope_id must match item provenance dataset"):
        _load(payload, loader, tmp_path)


def test_floor_only_owner_act_rejects_wrong_sha(tmp_path: Path) -> None:
    payload = copy.deepcopy(_payload())
    payload["training_eligibility"]["owner_adjudication"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="does not match the pinned floor-only act"):
        _load(payload, "legacy", tmp_path)


def test_floor_only_owner_act_rejects_item_without_bound_dataset(tmp_path: Path) -> None:
    payload = _payload()
    payload["annotation"]["items"][0]["provenance"] = {"sha256": "fixture"}
    with pytest.raises(ValueError, match="every item to name exactly one provenance dataset"):
        _load(payload, "legacy", tmp_path)
