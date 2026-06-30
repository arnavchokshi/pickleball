from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_ROOT = REPO_ROOT / "eval_clips" / "ball"
MANIFEST_PATH = BUNDLE_ROOT / "manifest.json"
REQUIRED_LABELS = ("ball_points", "events", "foot_contact", "court_corners")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_dicts(child)


def _frame_index_from_frame_name(frame: str) -> int | None:
    match = re.search(r"frame_(\d+)", frame)
    if match is None:
        return None
    return int(match.group(1))


def _referenced_frame_indexes(payload: dict[str, Any]) -> list[int]:
    indexes: list[int] = []
    for item in _iter_dicts(payload):
        frame_index = item.get("frame_index")
        if isinstance(frame_index, int):
            indexes.append(frame_index)
        frame = item.get("frame")
        if isinstance(frame, str):
            parsed = _frame_index_from_frame_name(frame)
            if parsed is not None:
                indexes.append(parsed)
    return indexes


def test_committed_ball_eval_clip_bundle_has_decodable_labeled_clips():
    cv2 = pytest.importorskip("cv2")
    manifest = _load_json(MANIFEST_PATH)

    assert manifest["schema_version"] == 1
    assert manifest["artifact_type"] == "pickleball_local_eval_clip_bundle"
    assert manifest["pillar"] == "ball"
    assert 4 <= len(manifest["clips"]) <= 8
    assert len({clip["clip"] for clip in manifest["clips"]}) == len(manifest["clips"])

    for clip in manifest["clips"]:
        video_path = BUNDLE_ROOT / clip["source_video"]
        assert video_path.is_file(), clip["clip"]
        assert _sha256(video_path) == clip["source_sha256"]

        capture = cv2.VideoCapture(str(video_path))
        try:
            assert capture.isOpened(), clip["clip"]
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
        finally:
            capture.release()

        assert frame_count == clip["frame_count"]
        assert width == clip["width"]
        assert height == clip["height"]
        assert fps > 0
        assert clip["duration_s"] == pytest.approx(frame_count / fps, rel=1e-3)

        label_paths = clip["labels"]
        assert tuple(label_paths) == REQUIRED_LABELS
        for label_name in REQUIRED_LABELS:
            label_path = BUNDLE_ROOT / label_paths[label_name]
            assert label_path.is_file(), f"{clip['clip']}:{label_name}"
            payload = _load_json(label_path)
            frames = _referenced_frame_indexes(payload)
            assert frames, f"{clip['clip']}:{label_name}"
            assert min(frames) >= 0
            assert max(frames) < frame_count

        ball_payload = _load_json(BUNDLE_ROOT / label_paths["ball_points"])
        ball_items = ball_payload["items"]
        assert len(ball_items) >= 10
        for item in ball_items:
            if not item.get("visible", False):
                continue
            x, y = item["xy_px"]
            assert 0 <= x < width
            assert 0 <= y < height
