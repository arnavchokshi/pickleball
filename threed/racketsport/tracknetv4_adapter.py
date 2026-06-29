"""TrackNetV4 prediction adapters for schema-valid ball tracks."""

from __future__ import annotations

import csv
import hashlib
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

from .ball_tracknet import ball_frame
from .schemas import BallTrack

CONFIDENCE_SEMANTICS = "official/simple visibility mapped to conf 1.0/0.0"
TRACKNETV4_SOURCE_URL = "https://github.com/TrackNetV4/TrackNetV4"

_FRAME_COLUMNS = ("frame", "frame_id", "frame_index")
_VISIBILITY_COLUMNS = ("visibility", "visible", "is_visible")
_X_COLUMNS = ("x", "x_px", "ball_x")
_Y_COLUMNS = ("y", "y_px", "ball_y")


def tracknetv4_csv_to_ball_track(csv_path: str | Path, *, fps: float) -> dict[str, Any]:
    """Convert TrackNetV4 official/simple CSV predictions into BallTrack JSON."""

    fps = _require_positive_float(fps, "fps")
    rows = _read_tracknetv4_rows(Path(csv_path))
    frames = [
        ball_frame(
            t=float(row["frame"]) / fps,
            xy=[row["x"], row["y"]],
            conf=1.0 if row["visible"] else 0.0,
            visible=row["visible"],
            approx=False,
        )
        for row in sorted(rows, key=lambda item: item["frame"])
    ]
    payload = {"schema_version": 1, "fps": fps, "source": "tracknet", "frames": frames, "bounces": []}
    BallTrack.model_validate(payload)
    return payload


def write_ball_track_from_csv(
    *,
    predictions_csv: str | Path,
    fps: float,
    out: str | Path,
    metadata_out: str | Path | None = None,
    source_mode: str = "tracknetv4_csv",
    runtime: dict[str, Any] | None = None,
    verified: bool = False,
) -> dict[str, Any]:
    """Write ``ball_track.json`` and a sidecar run summary from TrackNetV4 predictions."""

    out_path = Path(out)
    payload = tracknetv4_csv_to_ball_track(predictions_csv, fps=fps)
    _write_json(out_path, payload)
    visible = sum(1 for frame in payload["frames"] if frame["visible"])
    metadata = {
        "schema_version": 1,
        "artifact_type": "racketsport_tracknetv4_ball_run",
        "source_mode": source_mode,
        "predictions_csv": str(predictions_csv),
        "out": str(out_path),
        "fps": float(fps),
        "frame_count": len(payload["frames"]),
        "visible_frame_count": visible,
        "confidence_semantics": CONFIDENCE_SEMANTICS,
        "runtime": runtime or {},
        "not_ground_truth": True,
        "verified": bool(verified),
        "verification_note": _verification_note(verified),
    }
    if metadata_out is not None:
        _write_json(Path(metadata_out), metadata)
    return metadata


def run_external_tracknetv4_predict(
    *,
    tracknetv4_repo: str | Path,
    video: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    command: str | Sequence[str] | None = None,
    queue_length: int = 5,
    expected_csv: str | Path | None = None,
) -> Path:
    """Run an external TrackNetV4 predictor and return its predictions CSV path."""

    repo = _validate_tracknetv4_repo(tracknetv4_repo, require_predict_py=command is None)
    predict_py = repo / "src" / "predict.py"
    video_path = _require_file(video, "video")
    checkpoint_path = _require_file(checkpoint, "TrackNetV4 checkpoint")
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    normalized_queue_length = _require_positive_int(queue_length, "queue_length")
    cmd = _build_predict_command(
        command,
        repo=repo,
        predict_py=predict_py,
        video=video_path,
        checkpoint=checkpoint_path,
        output_dir=output_path,
        queue_length=normalized_queue_length,
    )

    subprocess.run(cmd, cwd=repo, check=True)
    return _resolve_predictions_csv(output_path, expected_csv=expected_csv, video=video_path)


def checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
    return {"path": str(checkpoint), "sha256": _sha256(checkpoint)}


def run_tracknetv4_or_convert(
    *,
    out: str | Path,
    fps: float,
    metadata_out: str | Path | None = None,
    predictions_csv: str | Path | None = None,
    video: str | Path | None = None,
    checkpoint: str | Path | None = None,
    tracknetv4_repo: str | Path | None = None,
    prediction_dir: str | Path | None = None,
    command: str | Sequence[str] | None = None,
    queue_length: int = 5,
    expected_csv: str | Path | None = None,
    mark_real_run_succeeded: bool = False,
) -> dict[str, Any]:
    """CLI-oriented entrypoint that converts CSV or runs external TrackNetV4."""

    if predictions_csv is not None:
        return write_ball_track_from_csv(
            predictions_csv=predictions_csv,
            fps=fps,
            out=out,
            metadata_out=metadata_out,
            source_mode="tracknetv4_csv",
            verified=False,
        )

    if video is None or checkpoint is None or tracknetv4_repo is None:
        raise ValueError("either --predictions-csv or --video/--checkpoint/--tracknetv4-repo is required")

    repo = _validate_tracknetv4_repo(tracknetv4_repo, require_predict_py=command is None)
    runtime = {
        "tracknetv4_repo": str(repo),
        "tracknetv4_source_url": TRACKNETV4_SOURCE_URL,
        "tracknetv4_checkpoint": checkpoint_metadata(checkpoint),
        "video": str(video),
        "command_configured": command is not None,
        "queue_length": _require_positive_int(queue_length, "queue_length"),
        "run_succeeded": False,
    }

    if prediction_dir is None:
        with tempfile.TemporaryDirectory(prefix="tracknetv4_") as tmp_dir:
            csv_path = run_external_tracknetv4_predict(
                tracknetv4_repo=repo,
                video=video,
                checkpoint=checkpoint,
                output_dir=tmp_dir,
                command=command,
                queue_length=queue_length,
                expected_csv=expected_csv,
            )
            runtime["run_succeeded"] = True
            return write_ball_track_from_csv(
                predictions_csv=csv_path,
                fps=fps,
                out=out,
                metadata_out=metadata_out,
                source_mode="tracknetv4_predict",
                runtime=runtime,
                verified=mark_real_run_succeeded,
            )

    csv_path = run_external_tracknetv4_predict(
        tracknetv4_repo=repo,
        video=video,
        checkpoint=checkpoint,
        output_dir=prediction_dir,
        command=command,
        queue_length=queue_length,
        expected_csv=expected_csv,
    )
    runtime["run_succeeded"] = True
    return write_ball_track_from_csv(
        predictions_csv=csv_path,
        fps=fps,
        out=out,
        metadata_out=metadata_out,
        source_mode="tracknetv4_predict",
        runtime=runtime,
        verified=mark_real_run_succeeded,
    )


def _read_tracknetv4_rows(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing TrackNetV4 predictions CSV: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        column_map = _column_map(fieldnames)
        _require_column(column_map, "frame", csv_path)
        _require_column(column_map, "x", csv_path)
        _require_column(column_map, "y", csv_path)
        rows = []
        for index, row in enumerate(reader):
            rows.append(_normalize_prediction_row(row, column_map=column_map, row_index=index))
    if not rows:
        raise ValueError(f"TrackNetV4 predictions CSV is empty: {csv_path}")
    return rows


def _normalize_prediction_row(
    row: dict[str, str | None],
    *,
    column_map: dict[str, str],
    row_index: int,
) -> dict[str, Any]:
    frame = _parse_nonnegative_int(row.get(column_map["frame"]), f"frame/{row_index}")
    x = _parse_optional_coordinate(row.get(column_map["x"]), f"x/{row_index}")
    y = _parse_optional_coordinate(row.get(column_map["y"]), f"y/{row_index}")

    visibility_column = column_map.get("visibility")
    if visibility_column is None:
        visible = _infer_visibility(x, y)
    else:
        visible = _parse_visibility(row.get(visibility_column), f"visibility/{row_index}")

    if visible and (x is None or y is None):
        raise ValueError(f"visible row {row_index} requires finite x and y")
    return {
        "frame": frame,
        "visible": visible,
        "x": float(x) if x is not None else 0.0,
        "y": float(y) if y is not None else 0.0,
    }


def _column_map(fieldnames: Sequence[str]) -> dict[str, str]:
    normalized = {_normalize_column_name(name): name for name in fieldnames}
    mapping: dict[str, str] = {}
    for canonical, candidates in {
        "frame": _FRAME_COLUMNS,
        "visibility": _VISIBILITY_COLUMNS,
        "x": _X_COLUMNS,
        "y": _Y_COLUMNS,
    }.items():
        for candidate in candidates:
            if candidate in normalized:
                mapping[canonical] = normalized[candidate]
                break
    return mapping


def _normalize_column_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _require_column(column_map: dict[str, str], name: str, csv_path: Path) -> None:
    if name not in column_map:
        accepted = {
            "frame": ", ".join(_FRAME_COLUMNS),
            "x": ", ".join(_X_COLUMNS),
            "y": ", ".join(_Y_COLUMNS),
        }[name]
        raise ValueError(f"missing TrackNetV4 {name} column in {csv_path}; accepted names: {accepted}")


def _parse_visibility(value: object, name: str) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "visible"}:
            return True
        if normalized in {"0", "false", "no", "n", "hidden", "invisible", "not_visible", "none", ""}:
            return False
    number = _parse_float(value, name)
    if number not in {0.0, 1.0}:
        raise ValueError(f"{name} must be 0/1 or true/false")
    return number == 1.0


def _parse_nonnegative_int(value: object, name: str) -> int:
    number = _parse_float(value, name)
    if int(number) != number or number < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(number)


def _parse_optional_coordinate(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "none", "null"}:
        return None
    return _parse_float(value, name)


def _infer_visibility(x: float | None, y: float | None) -> bool:
    if x is None or y is None:
        return False
    return not (x == -1.0 and y == -1.0)


def _require_positive_float(value: object, name: str) -> float:
    number = _parse_float(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _require_positive_int(value: object, name: str) -> int:
    number = _parse_nonnegative_int(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _parse_float(value: object, name: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if number != number or number in {float("inf"), float("-inf")}:
        raise ValueError(f"{name} must be finite")
    return number


def _validate_tracknetv4_repo(path: str | Path, *, require_predict_py: bool) -> Path:
    repo = Path(path).resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"missing TrackNetV4 repo: {repo}")
    predict_py = repo / "src" / "predict.py"
    if require_predict_py and not predict_py.is_file():
        raise FileNotFoundError(f"missing TrackNetV4 predict.py: {predict_py}")
    return repo


def _require_file(path: str | Path, label: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    return resolved


def _build_predict_command(
    command: str | Sequence[str] | None,
    *,
    repo: Path,
    predict_py: Path,
    video: Path,
    checkpoint: Path,
    output_dir: Path,
    queue_length: int,
) -> list[str]:
    if command is None:
        return [
            sys.executable,
            str(predict_py),
            "--video_path",
            str(video),
            "--model_weights",
            str(checkpoint),
            "--output_dir",
            str(output_dir),
            "--queue_length",
            str(queue_length),
        ]
    parts = shlex.split(command) if isinstance(command, str) else list(command)
    replacements = {
        "python": sys.executable,
        "repo": str(repo),
        "predict_py": str(predict_py),
        "video": str(video),
        "checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "queue_length": str(queue_length),
    }
    return [part.format(**replacements) for part in parts]


def _resolve_predictions_csv(output_dir: Path, *, expected_csv: str | Path | None, video: Path) -> Path:
    if expected_csv is not None:
        expected_path = Path(expected_csv)
        csv_path = expected_path if expected_path.is_absolute() else output_dir / expected_path
        if not csv_path.is_file():
            raise FileNotFoundError(f"TrackNetV4 completed without expected predictions CSV: {csv_path}")
        return csv_path

    candidates = [
        output_dir / f"{video.stem}.csv",
        output_dir / f"{video.stem}_predictions.csv",
        output_dir / f"{video.stem}_ball.csv",
        output_dir / "predictions.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    csv_files = sorted(output_dir.glob("*.csv"))
    if len(csv_files) == 1:
        return csv_files[0]
    if not csv_files:
        raise FileNotFoundError(f"TrackNetV4 completed without writing a predictions CSV in {output_dir}")
    raise ValueError(f"TrackNetV4 wrote multiple CSV files in {output_dir}; pass expected_csv")


def _verification_note(verified: bool) -> str:
    if verified:
        return "external TrackNetV4 command reported success; still not ground truth or accuracy-gate verified"
    return "not verified; CSV conversion or unmarked external run only"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "CONFIDENCE_SEMANTICS",
    "checkpoint_metadata",
    "run_external_tracknetv4_predict",
    "run_tracknetv4_or_convert",
    "tracknetv4_csv_to_ball_track",
    "write_ball_track_from_csv",
]
