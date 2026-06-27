#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

BACKEND_CHOICE_NOTE = (
    "Backend choice is empirical per real clip set; this report does not define a global default backend."
)
NUMERIC_FIELDS = ("elapsed_s", "duration_s", "frame_count", "decode_fps", "realtime_factor")
CLIP_STRING_FIELDS = ("clip_path", "source_relpath", "source_path", "path")


@dataclass(frozen=True)
class DecodeBenchmarkRecord:
    clip: str
    backend: str
    elapsed_s: float
    duration_s: float
    frame_count: int
    decode_fps: float
    realtime_factor: float
    source: str


def summarize_benchmarks(paths: list[Path]) -> dict[str, Any]:
    records: list[DecodeBenchmarkRecord] = []
    for path in paths:
        records.extend(_load_records(path))

    if not records:
        raise ValueError("at least one benchmark record is required")

    fastest_by_clip: list[dict[str, Any]] = []
    for clip in sorted({record.clip for record in records}):
        clip_records = [record for record in records if record.clip == clip]
        fastest = max(
            clip_records,
            key=lambda record: (record.decode_fps, record.realtime_factor, record.backend),
        )
        fastest_by_clip.append(
            {
                "clip": fastest.clip,
                "backend": fastest.backend,
                "decode_fps": fastest.decode_fps,
                "realtime_factor": fastest.realtime_factor,
                "source": fastest.source,
            }
        )

    return {
        "schema_version": 1,
        "backend_choice_note": BACKEND_CHOICE_NOTE,
        "benchmark_count": len(records),
        "clip_count": len({record.clip for record in records}),
        "fastest_backend_by_clip": fastest_by_clip,
        "aggregate_by_backend": _aggregate_by_backend(records),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    rows = [
        "# Decode Benchmark Summary",
        "",
        f"- Benchmarks: {summary['benchmark_count']}",
        f"- Clips: {summary['clip_count']}",
        f"- Backend choice note: {summary['backend_choice_note']}",
        "",
        "## Fastest Backend By Clip",
        "",
        "| Clip | Backend | Decode FPS | Realtime Factor |",
        "|---|---|---:|---:|",
    ]
    for entry in summary["fastest_backend_by_clip"]:
        rows.append(
            "| `{clip}` | `{backend}` | {decode_fps:.3f} | {realtime_factor:.3f} |".format(
                clip=entry["clip"],
                backend=entry["backend"],
                decode_fps=entry["decode_fps"],
                realtime_factor=entry["realtime_factor"],
            )
        )

    rows.extend(
        [
            "",
            "## Aggregate By Backend",
            "",
            "| Backend | Runs | Clips | Mean Decode FPS | Mean Realtime Factor | Total Frames |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for backend, aggregate in summary["aggregate_by_backend"].items():
        rows.append(
            "| `{backend}` | {runs} | {clips} | {mean_decode_fps:.3f} | "
            "{mean_realtime_factor:.3f} | {total_frames} |".format(
                backend=backend,
                runs=aggregate["runs"],
                clips=aggregate["clips"],
                mean_decode_fps=aggregate["mean_decode_fps"],
                mean_realtime_factor=aggregate["mean_realtime_factor"],
                total_frames=aggregate["total_frames"],
            )
        )
    rows.append("")
    return "\n".join(rows)


def _aggregate_by_backend(records: list[DecodeBenchmarkRecord]) -> dict[str, Any]:
    aggregates: dict[str, Any] = {}
    for backend in sorted({record.backend for record in records}):
        backend_records = [record for record in records if record.backend == backend]
        aggregates[backend] = {
            "runs": len(backend_records),
            "clips": len({record.clip for record in backend_records}),
            "mean_decode_fps": _mean(record.decode_fps for record in backend_records),
            "mean_realtime_factor": _mean(record.realtime_factor for record in backend_records),
            "total_duration_s": sum(record.duration_s for record in backend_records),
            "total_elapsed_s": sum(record.elapsed_s for record in backend_records),
            "total_frames": sum(record.frame_count for record in backend_records),
        }
    return aggregates


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items)


def _load_records(path: Path) -> list[DecodeBenchmarkRecord]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name}: invalid JSON: {exc.msg}") from exc

    raw_records = _extract_payload_records(payload, path)
    return [_validate_record(raw, path, index=index) for index, raw in enumerate(raw_records)]


def _extract_payload_records(payload: Any, path: Path) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        raw_records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("benchmarks"), list):
        raw_records = payload["benchmarks"]
    elif isinstance(payload, dict):
        raw_records = [payload]
    else:
        raise ValueError(f"{path.name}: benchmark payload must be a JSON object or list")

    if not raw_records:
        raise ValueError(f"{path.name}: benchmark payload must contain at least one record")
    if not all(isinstance(record, dict) for record in raw_records):
        raise ValueError(f"{path.name}: every benchmark record must be a JSON object")
    return raw_records


def _validate_record(raw: dict[str, Any], path: Path, *, index: int) -> DecodeBenchmarkRecord:
    label = path.name if index == 0 else f"{path.name}[{index}]"
    backend = _required_string(raw, "backend", label)
    values = {field: _required_number(raw, field, label) for field in NUMERIC_FIELDS}
    _validate_positive(values, label)
    _validate_clip_metadata(raw, label)

    return DecodeBenchmarkRecord(
        clip=_clip_name(raw, fallback=path.stem, label=label),
        backend=backend,
        elapsed_s=float(values["elapsed_s"]),
        duration_s=float(values["duration_s"]),
        frame_count=int(values["frame_count"]),
        decode_fps=float(values["decode_fps"]),
        realtime_factor=float(values["realtime_factor"]),
        source=str(path),
    )


def _required_string(raw: dict[str, Any], field: str, label: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: {field} must be a non-empty string")
    return value


def _required_number(raw: dict[str, Any], field: str, label: str) -> float:
    if field not in raw:
        raise ValueError(f"{label}: missing required field {field}")
    value = raw[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label}: {field} must be numeric")
    return float(value)


def _validate_positive(values: dict[str, float], label: str) -> None:
    for field, value in values.items():
        if value <= 0:
            raise ValueError(f"{label}: {field} must be positive")
    if not values["frame_count"].is_integer():
        raise ValueError(f"{label}: frame_count must be an integer")


def _validate_clip_metadata(raw: dict[str, Any], label: str) -> None:
    for field in CLIP_STRING_FIELDS:
        if field in raw and (not isinstance(raw[field], str) or not raw[field].strip()):
            raise ValueError(f"{label}: {field} must be a non-empty string when present")

    if "fps" in raw:
        value = raw["fps"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{label}: fps must be positive numeric when present")

    if "resolution" in raw:
        resolution = raw["resolution"]
        if (
            not isinstance(resolution, list)
            or len(resolution) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in resolution)
        ):
            raise ValueError(f"{label}: resolution must be [width, height] positive integers when present")


def _clip_name(raw: dict[str, Any], *, fallback: str, label: str) -> str:
    clip = raw.get("clip")
    if isinstance(clip, str):
        if not clip.strip():
            raise ValueError(f"{label}: clip must be a non-empty string when present")
        return clip
    if clip is not None:
        raise ValueError(f"{label}: clip must be a string when present")

    for field in CLIP_STRING_FIELDS:
        value = raw.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return fallback


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize decode benchmark JSON artifacts.")
    parser.add_argument("benchmarks", type=Path, nargs="+", help="Benchmark JSON artifact paths.")
    parser.add_argument("--markdown-out", type=Path, help="Optional Markdown report path.")
    args = parser.parse_args()

    try:
        summary = summarize_benchmarks(args.benchmarks)
        if args.markdown_out:
            args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
            args.markdown_out.write_text(render_markdown(summary), encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
