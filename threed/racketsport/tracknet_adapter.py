"""TrackNetV3 prediction adapters for schema-valid ball tracks."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .ball_tracknet import ball_frame
from .schemas import BallTrack


TRACKNET_COLUMNS = ("Frame", "Visibility", "X", "Y")
CONFIDENCE_SEMANTICS = "official visibility mapped to conf 1.0/0.0"


def tracknet_csv_to_ball_track(csv_path: str | Path, *, fps: float) -> dict[str, Any]:
    """Convert official TrackNetV3 ``Frame,Visibility,X,Y`` CSV into BallTrack JSON."""

    fps = _require_positive_float(fps, "fps")
    rows = _read_tracknet_rows(Path(csv_path))
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
    source_mode: str = "tracknet_csv",
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``ball_track.json`` and a sidecar run summary from official predictions."""

    out_path = Path(out)
    payload = tracknet_csv_to_ball_track(predictions_csv, fps=fps)
    _write_json(out_path, payload)
    visible = sum(1 for frame in payload["frames"] if frame["visible"])
    metadata = {
        "schema_version": 1,
        "artifact_type": "racketsport_tracknet_ball_run",
        "source_mode": source_mode,
        "predictions_csv": str(predictions_csv),
        "out": str(out_path),
        "fps": float(fps),
        "frame_count": len(payload["frames"]),
        "visible_frame_count": visible,
        "confidence_semantics": CONFIDENCE_SEMANTICS,
        "runtime": runtime or {},
        "not_ground_truth": True,
    }
    if metadata_out is not None:
        _write_json(Path(metadata_out), metadata)
    return metadata


def run_official_tracknet_predict(
    *,
    tracknet_repo: str | Path,
    video: str | Path,
    tracknet_file: str | Path,
    inpaintnet_file: str | Path,
    save_dir: str | Path,
    batch_size: int = 16,
    video_range: tuple[int, int] | list[int] | None = None,
    large_video: bool = True,
) -> Path:
    """Run official TrackNetV3 ``predict.py`` and return its CSV path."""

    repo = Path(tracknet_repo).resolve()
    predict_py = repo / "predict.py"
    if not predict_py.is_file():
        raise FileNotFoundError(f"missing TrackNetV3 predict.py: {predict_py}")
    video_path = Path(video).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"missing video: {video_path}")
    tracknet_path = Path(tracknet_file).resolve()
    if not tracknet_path.is_file():
        raise FileNotFoundError(f"missing TrackNet checkpoint: {tracknet_path}")
    inpaintnet_path = Path(inpaintnet_file).resolve()
    if not inpaintnet_path.is_file():
        raise FileNotFoundError(f"missing InpaintNet checkpoint: {inpaintnet_path}")

    save_path = Path(save_dir).resolve()
    save_path.mkdir(parents=True, exist_ok=True)
    normalized_video_range = _normalize_video_range(video_range)
    cmd = [
        sys.executable,
        str(predict_py),
        "--video_file",
        str(video_path),
        "--tracknet_file",
        str(tracknet_path),
        "--inpaintnet_file",
        str(inpaintnet_path),
        "--save_dir",
        str(save_path),
        "--batch_size",
        str(batch_size),
    ]
    if normalized_video_range is not None:
        cmd.extend(["--video_range", f"{normalized_video_range[0]},{normalized_video_range[1]}"])
    if large_video:
        cmd.append("--large_video")
    subprocess.run(cmd, cwd=repo, check=True)
    csv_path = save_path / f"{video_path.stem}_ball.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"TrackNetV3 completed without writing predictions CSV: {csv_path}")
    return csv_path


def checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
    return {"path": str(checkpoint), "sha256": _sha256(checkpoint)}


def _read_tracknet_rows(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing TrackNet predictions CSV: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in TRACKNET_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"missing TrackNet column(s): {', '.join(missing)}")
        rows = []
        for index, row in enumerate(reader):
            rows.append(
                {
                    "frame": _parse_nonnegative_int(row["Frame"], f"Frame/{index}"),
                    "visible": _parse_visibility(row["Visibility"], f"Visibility/{index}"),
                    "x": _parse_float(row["X"], f"X/{index}"),
                    "y": _parse_float(row["Y"], f"Y/{index}"),
                }
            )
    if not rows:
        raise ValueError(f"TrackNet predictions CSV is empty: {csv_path}")
    return rows


def _parse_visibility(value: object, name: str) -> bool:
    number = _parse_float(value, name)
    if number not in {0.0, 1.0}:
        raise ValueError(f"{name} must be 0 or 1")
    return number == 1.0


def _parse_nonnegative_int(value: object, name: str) -> int:
    number = _parse_float(value, name)
    if int(number) != number or number < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(number)


def _require_positive_float(value: object, name: str) -> float:
    number = _parse_float(value, name)
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _normalize_video_range(video_range: tuple[int, int] | list[int] | None) -> tuple[int, int] | None:
    if video_range is None:
        return None
    if len(video_range) != 2:
        raise ValueError("video_range must contain START_S and END_S")
    start, end = int(video_range[0]), int(video_range[1])
    if start < 0 or end <= start:
        raise ValueError("video_range must satisfy 0 <= START_S < END_S")
    return start, end


def run_tracknet_or_convert(
    *,
    out: str | Path,
    fps: float,
    metadata_out: str | Path | None = None,
    predictions_csv: str | Path | None = None,
    video: str | Path | None = None,
    tracknet_file: str | Path | None = None,
    inpaintnet_file: str | Path | None = None,
    tracknet_repo: str | Path | None = None,
    prediction_dir: str | Path | None = None,
    batch_size: int = 16,
    video_range: tuple[int, int] | list[int] | None = None,
    large_video: bool = False,
) -> dict[str, Any]:
    """CLI-oriented entrypoint that converts CSV or runs official TrackNetV3."""

    if predictions_csv is not None:
        return write_ball_track_from_csv(
            predictions_csv=predictions_csv,
            fps=fps,
            out=out,
            metadata_out=metadata_out,
            source_mode="tracknet_csv",
        )

    if video is None or tracknet_file is None or inpaintnet_file is None or tracknet_repo is None:
        raise ValueError("either --predictions-csv or --video/--tracknet-file/--inpaintnet-file/--tracknet-repo is required")

    predict_py = Path(tracknet_repo) / "predict.py"
    if not predict_py.is_file():
        raise FileNotFoundError(f"missing TrackNetV3 predict.py: {predict_py}")

    normalized_video_range = _normalize_video_range(video_range)
    runtime = {
        "tracknet_repo": str(tracknet_repo),
        "tracknet_checkpoint": checkpoint_metadata(tracknet_file),
        "inpaintnet_checkpoint": checkpoint_metadata(inpaintnet_file),
        "video": str(video),
    }
    if normalized_video_range is not None:
        runtime["video_range_seconds"] = list(normalized_video_range)
        runtime["video_range_semantics"] = (
            "official TrackNetV3 background median sampling range; does not trim prediction frames"
        )
    if prediction_dir is None:
        with tempfile.TemporaryDirectory(prefix="tracknetv3_") as tmp_dir:
            csv_path = run_official_tracknet_predict(
                tracknet_repo=tracknet_repo,
                video=video,
                tracknet_file=tracknet_file,
                inpaintnet_file=inpaintnet_file,
                save_dir=tmp_dir,
                batch_size=batch_size,
                video_range=normalized_video_range,
                large_video=large_video,
            )
            return write_ball_track_from_csv(
                predictions_csv=csv_path,
                fps=fps,
                out=out,
                metadata_out=metadata_out,
                source_mode="tracknet_predict",
                runtime=runtime,
            )

    csv_path = run_official_tracknet_predict(
        tracknet_repo=tracknet_repo,
        video=video,
        tracknet_file=tracknet_file,
        inpaintnet_file=inpaintnet_file,
        save_dir=prediction_dir,
        batch_size=batch_size,
        video_range=normalized_video_range,
        large_video=large_video,
    )
    return write_ball_track_from_csv(
        predictions_csv=csv_path,
        fps=fps,
        out=out,
        metadata_out=metadata_out,
        source_mode="tracknet_predict",
        runtime=runtime,
    )


__all__ = [
    "CONFIDENCE_SEMANTICS",
    "checkpoint_metadata",
    "run_official_tracknet_predict",
    "run_tracknet_or_convert",
    "tracknet_csv_to_ball_track",
    "write_ball_track_from_csv",
]
