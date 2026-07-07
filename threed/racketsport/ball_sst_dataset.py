"""SST pseudo-label manifest and disagreement helpers for BALL stage-2/3 work."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .online_harvest_ingest import perceptual_hash_video
from .roboflow_corpus import (
    DEFAULT_DEDUP_THRESHOLD,
    DEFAULT_EVAL_SAMPLE_EVERY_S,
    DEFAULT_PROTECTED_EVAL_HASH_COUNT,
    assert_no_protected_eval_hash_collisions,
    load_protected_eval_hashes,
)


SST_MANIFEST_ARTIFACT_TYPE = "racketsport_ball_sst_manifest"
SST_DISAGREEMENT_ARTIFACT_TYPE = "racketsport_ball_sst_disagreement_queue"


def build_sst_manifest(
    *,
    prelabel_root: str | Path,
    rally_root: str | Path,
    out_path: str | Path | None = None,
    clips: Sequence[str] | None = None,
    max_samples_per_clip: int | None = None,
    protected_eval_hashes: Mapping[str, Sequence[int | str]] | None = None,
    expected_protected_eval_hash_count: int | None = DEFAULT_PROTECTED_EVAL_HASH_COUNT,
    eval_root: str | Path = "eval_clips/ball",
    eval_sample_every_s: float = DEFAULT_EVAL_SAMPLE_EVERY_S,
    collision_hamming_threshold: int = DEFAULT_DEDUP_THRESHOLD,
) -> dict[str, Any]:
    """Build a fail-closed student-training manifest from WASB teacher sidecars.

    Per-sample pseudo-label weight is intentionally boring:
    ``weight = clamp(score, 0.0, 1.0)``. The score is the teacher detection
    confidence from the ball track frame.
    """

    prelabels = Path(prelabel_root)
    rallies = Path(rally_root)
    selected = _selected_clip_dirs(prelabels, clips)
    if max_samples_per_clip is not None and max_samples_per_clip <= 0:
        raise ValueError("max_samples_per_clip must be positive when provided")

    clip_entries: list[dict[str, Any]] = []
    guard_samples: list[dict[str, Any]] = []
    for clip_dir in selected:
        clip_id = clip_dir.name
        ball_track_path = clip_dir / "ball_track.json"
        metadata_path = clip_dir / "ball_track_metadata.json"
        if not ball_track_path.is_file():
            raise FileNotFoundError(f"missing teacher ball_track.json: {ball_track_path}")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"missing teacher ball_track_metadata.json: {metadata_path}")
        video_path = _rally_video_path(rallies, clip_id)
        provenance_path = video_path.with_suffix(".provenance.json")
        if not provenance_path.is_file():
            raise FileNotFoundError(f"missing rally provenance: {provenance_path}")

        track = _read_json(ball_track_path)
        fps = _positive_float(track.get("fps"), f"{ball_track_path} fps")
        samples: list[dict[str, Any]] = []
        frames = _frame_list(track, ball_track_path)
        for ordinal, frame in enumerate(frames):
            if max_samples_per_clip is not None and len(samples) >= max_samples_per_clip:
                break
            if not bool(frame.get("visible")):
                continue
            xy = _xy(frame.get("xy"), f"{ball_track_path} frame {ordinal} xy")
            score = _clamp01(_finite_float(frame.get("conf"), f"{ball_track_path} frame {ordinal} conf"))
            t = _finite_float(frame.get("t"), f"{ball_track_path} frame {ordinal} t")
            frame_index = _frame_index(frame, fps=fps, ordinal=ordinal)
            samples.append(
                {
                    "sample_id": f"{clip_id}:{frame_index}",
                    "clip_id": clip_id,
                    "frame_index": frame_index,
                    "t": t,
                    "frame_ref": {
                        "video": str(video_path),
                        "frame_index": frame_index,
                        "t": t,
                    },
                    "teacher_xy": [xy[0], xy[1]],
                    "score": score,
                    "weight": score,
                    "weight_policy": "clamp(score, 0.0, 1.0)",
                    "teacher_source": "wasb_ball_track",
                    "teacher_ball_track": str(ball_track_path),
                }
            )

        for hash_index, dhash in enumerate(perceptual_hash_video(video_path, sample_every_s=eval_sample_every_s)):
            guard_samples.append(
                {
                    "sample_id": f"{clip_id}:video_dhash:{hash_index}",
                    "source_slug": clip_id,
                    "hashes": {"dhash": f"{int(dhash):016x}"},
                }
            )

        clip_entries.append(
            {
                "clip_id": clip_id,
                "teacher_ball_track": str(ball_track_path),
                "teacher_metadata": str(metadata_path),
                "rally_video": str(video_path),
                "rally_provenance": str(provenance_path),
                "fps": fps,
                "sample_count": len(samples),
                "samples": samples,
            }
        )

    guard = assert_no_sst_protected_eval_hash_collisions(
        guard_samples,
        protected_eval_hashes=protected_eval_hashes,
        expected_protected_eval_hash_count=expected_protected_eval_hash_count,
        eval_root=eval_root,
        eval_sample_every_s=eval_sample_every_s,
        threshold=collision_hamming_threshold,
    )
    manifest = {
        "schema_version": 1,
        "artifact_type": SST_MANIFEST_ARTIFACT_TYPE,
        "ball_verified": False,
        "promotion_claimed": False,
        "prelabel_root": str(prelabels),
        "rally_root": str(rallies),
        "teacher": "wasb_ball_track_sidecars",
        "weight_policy": "weight = clamp(score, 0.0, 1.0)",
        "protected_eval_hash_check": guard,
        "summary": {
            "clip_count": len(clip_entries),
            "sample_count": sum(int(entry["sample_count"]) for entry in clip_entries),
        },
        "clips": clip_entries,
    }
    if out_path is not None:
        _write_json(Path(out_path), manifest)
    return manifest


def load_sst_manifest(path: str | Path) -> dict[str, Any]:
    manifest = _read_json(path)
    if manifest.get("artifact_type") != SST_MANIFEST_ARTIFACT_TYPE:
        raise ValueError(f"unexpected SST manifest artifact_type: {manifest.get('artifact_type')}")
    clips = manifest.get("clips")
    if not isinstance(clips, list):
        raise ValueError("SST manifest requires clips list")
    return manifest


def iter_sst_manifest_samples(path: str | Path) -> list[dict[str, Any]]:
    manifest = load_sst_manifest(path)
    rows: list[dict[str, Any]] = []
    for clip_index, clip in enumerate(manifest["clips"]):
        if not isinstance(clip, Mapping):
            raise ValueError(f"SST manifest clip {clip_index} must be an object")
        samples = clip.get("samples")
        if not isinstance(samples, list):
            raise ValueError(f"SST manifest clip {clip_index} requires samples list")
        for sample_index, sample in enumerate(samples):
            if not isinstance(sample, Mapping):
                raise ValueError(f"SST manifest sample {clip_index}:{sample_index} must be an object")
            rows.append(dict(sample))
    return rows


def assert_no_sst_protected_eval_hash_collisions(
    samples: Sequence[Mapping[str, Any]],
    *,
    protected_eval_hashes: Mapping[str, Sequence[int | str]] | None = None,
    expected_protected_eval_hash_count: int | None = DEFAULT_PROTECTED_EVAL_HASH_COUNT,
    eval_root: str | Path = "eval_clips/ball",
    eval_sample_every_s: float = DEFAULT_EVAL_SAMPLE_EVERY_S,
    threshold: int = DEFAULT_DEDUP_THRESHOLD,
) -> dict[str, Any]:
    if protected_eval_hashes is None:
        eval_hashes, source = load_protected_eval_hashes(
            eval_root=eval_root,
            eval_sample_every_s=eval_sample_every_s,
        )
    else:
        eval_hashes = _normalize_hashes(protected_eval_hashes)
        source = "constructor_provided_protected_eval_hashes"
    eval_hash_count = sum(len(values) for values in eval_hashes.values())
    if expected_protected_eval_hash_count is not None and eval_hash_count != expected_protected_eval_hash_count:
        raise ValueError(
            "protected eval hash count mismatch: "
            f"expected {expected_protected_eval_hash_count}, got {eval_hash_count} from {source}"
        )
    result = assert_no_protected_eval_hash_collisions(
        samples,
        eval_hashes=eval_hashes,
        threshold=threshold,
    )
    result["hash_source"] = source
    return result


def build_sst_disagreement_queue(
    *,
    teacher_predictions: str | Path,
    student_predictions: str | Path,
    out_path: str | Path | None = None,
    large_offset_px: float = 25.0,
) -> dict[str, Any]:
    if large_offset_px <= 0.0:
        raise ValueError("large_offset_px must be positive")
    teacher = _load_prediction_set(Path(teacher_predictions))
    student = _load_prediction_set(Path(student_predictions))
    queue: list[dict[str, Any]] = []
    for clip_id in sorted(set(teacher) | set(student)):
        teacher_frames = teacher.get(clip_id, {})
        student_frames = student.get(clip_id, {})
        for frame_index in sorted(set(teacher_frames) | set(student_frames)):
            t = teacher_frames.get(frame_index)
            s = student_frames.get(frame_index)
            t_visible = bool(t and t.get("visible"))
            s_visible = bool(s and s.get("visible"))
            if t_visible and not s_visible:
                queue.append(_disagreement_row("teacher-only", clip_id, frame_index, t, s, rank=float(t.get("score", 0.0))))
            elif s_visible and not t_visible:
                queue.append(_disagreement_row("student-only", clip_id, frame_index, t, s, rank=float(s.get("score", 0.0))))
            elif t_visible and s_visible:
                assert t is not None and s is not None
                offset = _distance(t["xy"], s["xy"])
                if offset > large_offset_px:
                    queue.append(_disagreement_row("large-offset", clip_id, frame_index, t, s, rank=offset, offset=offset))
    queue = sorted(queue, key=lambda item: (-float(item["rank"]), item["clip_id"], int(item["frame_index"]), item["disagreement_type"]))
    summary = {
        "clip_count": len(set(item["clip_id"] for item in queue)),
        "disagreement_count": len(queue),
        "teacher_only_count": sum(1 for item in queue if item["disagreement_type"] == "teacher-only"),
        "student_only_count": sum(1 for item in queue if item["disagreement_type"] == "student-only"),
        "large_offset_count": sum(1 for item in queue if item["disagreement_type"] == "large-offset"),
    }
    payload = {
        "schema_version": 1,
        "artifact_type": SST_DISAGREEMENT_ARTIFACT_TYPE,
        "ball_verified": False,
        "promotion_claimed": False,
        "teacher_predictions": str(teacher_predictions),
        "student_predictions": str(student_predictions),
        "large_offset_px": float(large_offset_px),
        "summary": summary,
        "queue": queue,
    }
    if out_path is not None:
        _write_json(Path(out_path), payload)
    return payload


def _selected_clip_dirs(prelabels: Path, clips: Sequence[str] | None) -> list[Path]:
    if not prelabels.is_dir():
        raise FileNotFoundError(f"missing prelabel root: {prelabels}")
    if clips:
        dirs = [prelabels / clip for clip in clips]
    else:
        dirs = sorted(path for path in prelabels.iterdir() if path.is_dir())
    missing = [str(path) for path in dirs if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"missing prelabel clip dir(s): {', '.join(missing)}")
    return sorted(dirs, key=lambda path: path.name)


def _rally_video_path(rally_root: Path, clip_id: str) -> Path:
    source_id = _source_id_from_clip(clip_id)
    path = rally_root / source_id / f"{clip_id}.mp4"
    if not path.is_file():
        raise FileNotFoundError(f"missing rally video for {clip_id}: {path}")
    return path


def _source_id_from_clip(clip_id: str) -> str:
    marker = "_rally_"
    if marker not in clip_id:
        raise ValueError(f"clip id does not contain {marker!r}: {clip_id}")
    return clip_id.split(marker, 1)[0]


def _load_prediction_set(path: Path) -> dict[str, dict[int, dict[str, Any]]]:
    if path.is_dir():
        if (path / "ball_track.json").is_file():
            return {path.name: _prediction_frames_from_ball_track(path / "ball_track.json")}
        result: dict[str, dict[int, dict[str, Any]]] = {}
        for track_path in sorted(path.glob("*/ball_track.json")):
            result[track_path.parent.name] = _prediction_frames_from_ball_track(track_path)
        if not result:
            raise FileNotFoundError(f"no */ball_track.json files under {path}")
        return result
    payload = _read_json(path)
    if payload.get("artifact_type") == SST_MANIFEST_ARTIFACT_TYPE:
        result: dict[str, dict[int, dict[str, Any]]] = {}
        for sample in iter_sst_manifest_samples(path):
            clip_id = str(sample["clip_id"])
            frame_index = int(sample["frame_index"])
            result.setdefault(clip_id, {})[frame_index] = {
                "visible": True,
                "xy": [float(sample["teacher_xy"][0]), float(sample["teacher_xy"][1])],
                "score": float(sample["score"]),
                "t": float(sample["t"]),
            }
        return result
    return {path.stem: _prediction_frames_from_payload(payload, path)}


def _prediction_frames_from_ball_track(path: Path) -> dict[int, dict[str, Any]]:
    return _prediction_frames_from_payload(_read_json(path), path)


def _prediction_frames_from_payload(payload: Mapping[str, Any], path: Path) -> dict[int, dict[str, Any]]:
    fps = _positive_float(payload.get("fps"), f"{path} fps")
    frames = _frame_list(payload, path)
    result: dict[int, dict[str, Any]] = {}
    for ordinal, frame in enumerate(frames):
        frame_index = _frame_index(frame, fps=fps, ordinal=ordinal)
        visible = bool(frame.get("visible"))
        xy = _xy(frame.get("xy"), f"{path} frame {ordinal} xy") if visible else [0.0, 0.0]
        score = _clamp01(_finite_float(frame.get("conf", 0.0), f"{path} frame {ordinal} conf"))
        result[frame_index] = {
            "visible": visible,
            "xy": xy,
            "score": score,
            "t": _finite_float(frame.get("t", frame_index / fps), f"{path} frame {ordinal} t"),
        }
    return result


def _disagreement_row(
    kind: str,
    clip_id: str,
    frame_index: int,
    teacher: Mapping[str, Any] | None,
    student: Mapping[str, Any] | None,
    *,
    rank: float,
    offset: float | None = None,
) -> dict[str, Any]:
    return {
        "clip_id": clip_id,
        "frame_index": int(frame_index),
        "frame_ref": {"clip_id": clip_id, "frame_index": int(frame_index)},
        "disagreement_type": kind,
        "teacher": _prediction_snapshot(teacher),
        "student": _prediction_snapshot(student),
        "offset_px": float(offset) if offset is not None else None,
        "rank": float(rank),
    }


def _prediction_snapshot(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "visible": bool(value.get("visible")),
        "xy": [float(value["xy"][0]), float(value["xy"][1])] if value.get("visible") else [0.0, 0.0],
        "score": float(value.get("score", 0.0)),
        "t": float(value.get("t", 0.0)),
    }


def _frame_list(payload: Mapping[str, Any], path: str | Path) -> list[Mapping[str, Any]]:
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"ball track frames must be a list: {path}")
    if any(not isinstance(frame, Mapping) for frame in frames):
        raise ValueError(f"ball track frames must be objects: {path}")
    return frames


def _frame_index(frame: Mapping[str, Any], *, fps: float, ordinal: int) -> int:
    raw = frame.get("frame_index", frame.get("frame"))
    if raw is not None:
        frame_index = int(raw)
    else:
        frame_index = int(round(_finite_float(frame.get("t"), f"frame {ordinal} t") * fps))
    if frame_index < 0:
        raise ValueError("frame_index must be nonnegative")
    return frame_index


def _xy(value: Any, field: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{field} must be [x, y]")
    x = _finite_float(value[0], f"{field}[0]")
    y = _finite_float(value[1], f"{field}[1]")
    return [x, y]


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2)


def _positive_float(value: Any, field: str) -> float:
    number = _finite_float(value, field)
    if number <= 0.0:
        raise ValueError(f"{field} must be positive")
    return number


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_hashes(values: Mapping[str, Sequence[int | str]]) -> dict[str, list[int]]:
    normalized: dict[str, list[int]] = {}
    for clip_id, hashes in values.items():
        normalized[str(clip_id)] = [_parse_hash(value) for value in hashes]
    return normalized


def _parse_hash(value: int | str) -> int:
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    if text.startswith("0x"):
        text = text[2:]
    return int(text, 16)


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
