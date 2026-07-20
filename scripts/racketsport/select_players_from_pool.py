#!/usr/bin/env python3
"""Apply the preview player-selection layer to association output."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unicodedata
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.player_selection import (
    PlayerSelectionConfig,
    select_players_payload,
)  # noqa: E402
from threed.racketsport.schemas import CourtCalibration, Tracks, validate_artifact_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Source-only four-slot player selection with soft court presence, "
            "open-set identity decisions, stitch vetoes, and honest micro-fill provenance."
        )
    )
    parser.add_argument(
        "--tracks", type=Path, required=True, help="Associated tracks.json input."
    )
    parser.add_argument(
        "--raw-pool",
        type=Path,
        help="Raw tracked_detections.json; required when enabled.",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        help="Source-only ReID embedding export; required when enabled.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        help="court_calibration.json; required when enabled.",
    )
    parser.add_argument("--out-tracks", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--enable-selection",
        action="store_true",
        help="Enable preview selection. Omit for a byte-identical tracks.json no-op.",
    )
    args = parser.parse_args()

    try:
        _require_pairwise_distinct_paths(
            {
                "--tracks": args.tracks,
                "--raw-pool": args.raw_pool,
                "--embeddings": args.embeddings,
                "--calibration": args.calibration,
                "--out-tracks": args.out_tracks,
                "--report": args.report,
            }
        )
        tracks_payload = _read_object(args.tracks)
        config = PlayerSelectionConfig()
        if not args.enable_selection:
            selected, report = select_players_payload(
                tracks_payload, enabled=False, config=config
            )
            del selected
            report_text = (
                json.dumps(
                    report,
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            _stage_and_publish_report(
                staged_tracks=_stage_copy(args.tracks, args.out_tracks),
                tracks_destination=args.out_tracks,
                report_text=report_text,
                report_destination=args.report,
            )
        else:
            missing = [
                flag
                for flag, value in (
                    ("--raw-pool", args.raw_pool),
                    ("--embeddings", args.embeddings),
                    ("--calibration", args.calibration),
                )
                if value is None
            ]
            if missing:
                raise ValueError("enabled selection requires " + ", ".join(missing))
            calibration = validate_artifact_file("court_calibration", args.calibration)
            if not isinstance(calibration, CourtCalibration):
                raise ValueError(
                    f"{args.calibration} did not parse as CourtCalibration"
                )
            selected, report = select_players_payload(
                tracks_payload,
                raw_pool_payload=_read_object(args.raw_pool),
                embedding_payload=_read_object(args.embeddings),
                calibration=calibration,
                enabled=True,
                config=config,
            )
            validated = Tracks.model_validate(selected)
            selected_text = (
                json.dumps(
                    validated.model_dump(mode="json"),
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            report_text = (
                json.dumps(
                    report,
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            _stage_and_publish_report(
                staged_tracks=_stage_text(args.out_tracks, selected_text),
                tracks_destination=args.out_tracks,
                report_text=report_text,
                report_destination=args.report,
            )
    except Exception as exc:
        print(f"player selection failed: {exc}", file=sys.stderr)
        return 1

    print(args.out_tracks)
    print(args.report)
    return 0


def _read_object(path: Path | None) -> dict[str, Any]:
    if path is None:
        raise ValueError("required JSON path is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _require_pairwise_distinct_paths(paths: dict[str, Path | None]) -> None:
    present = [(flag, path) for flag, path in paths.items() if path is not None]
    for index, (left_flag, left_path) in enumerate(present):
        assert left_path is not None
        for right_flag, right_path in present[index + 1 :]:
            assert right_path is not None
            left_resolved = left_path.resolve()
            right_resolved = right_path.resolve()
            same_resolved_path = left_resolved == right_resolved
            same_casefolded_path = (
                unicodedata.normalize("NFC", str(left_resolved)).casefold()
                == unicodedata.normalize("NFC", str(right_resolved)).casefold()
            )
            same_existing_file = (
                left_path.exists()
                and right_path.exists()
                and os.path.samefile(left_path, right_path)
            )
            if same_resolved_path or same_casefolded_path or same_existing_file:
                raise ValueError(
                    f"{left_flag} and {right_flag} must be pairwise-distinct paths"
                )


def _stage_copy(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with (
            source.open("rb") as input_file,
            os.fdopen(file_descriptor, "wb") as output_file,
        ):
            shutil.copyfileobj(input_file, output_file)
            output_file.flush()
            os.fsync(output_file.fileno())
        return temporary_path
    except BaseException:
        try:
            os.close(file_descriptor)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)
        raise


def _stage_text(destination: Path, content: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(
            file_descriptor, "w", encoding="utf-8", newline=""
        ) as output_file:
            output_file.write(content)
            output_file.flush()
            os.fsync(output_file.fileno())
        return temporary_path
    except BaseException:
        try:
            os.close(file_descriptor)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)
        raise


def _publish_output_pair(
    *,
    staged_tracks: Path,
    tracks_destination: Path,
    staged_report: Path,
    report_destination: Path,
) -> None:
    """Publish the report first, tracks last, and roll back report on failure."""

    report_existed = report_destination.exists()
    report_backup: Path | None = None
    report_published = False
    try:
        if report_existed:
            report_backup = _stage_copy(report_destination, report_destination)
        os.replace(staged_report, report_destination)
        report_published = True
        os.replace(staged_tracks, tracks_destination)
    except BaseException as publish_error:
        rollback_error: BaseException | None = None
        if report_published:
            try:
                if report_existed:
                    assert report_backup is not None
                    os.replace(report_backup, report_destination)
                    report_backup = None
                else:
                    report_destination.unlink(missing_ok=True)
            except BaseException as exc:
                rollback_error = exc
        staged_report.unlink(missing_ok=True)
        staged_tracks.unlink(missing_ok=True)
        if report_backup is not None:
            report_backup.unlink(missing_ok=True)
        if rollback_error is not None:
            raise RuntimeError(
                "tracks publication failed and report rollback also failed"
            ) from publish_error
        raise
    if report_backup is not None:
        report_backup.unlink(missing_ok=True)


def _stage_and_publish_report(
    *,
    staged_tracks: Path,
    tracks_destination: Path,
    report_text: str,
    report_destination: Path,
) -> None:
    try:
        staged_report = _stage_text(report_destination, report_text)
    except BaseException:
        staged_tracks.unlink(missing_ok=True)
        raise
    _publish_output_pair(
        staged_tracks=staged_tracks,
        tracks_destination=tracks_destination,
        staged_report=staged_report,
        report_destination=report_destination,
    )


if __name__ == "__main__":
    raise SystemExit(main())
