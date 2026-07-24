#!/usr/bin/env python3
"""Apply the preview player-selection layer to association output."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unicodedata
from typing import Any, Sequence

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
        help=(
            "Optional source-only ReID embedding export. Missing provider evidence "
            "is reported loudly as reid_unavailable."
        ),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        help="court_calibration.json; required when enabled.",
    )
    parser.add_argument("--out-tracks", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--scoring-projection",
        type=Path,
        help=(
            "Field-stripped measured-only Tracks input for later scoring. "
            "Enabled selection defaults to <out-tracks stem>.scoring_projection.json; "
            "a sibling .sha256 file hashes the exact emitted bytes."
        ),
    )
    parser.add_argument(
        "--enable-selection",
        action="store_true",
        help="Enable preview selection. Omit for a byte-identical tracks.json no-op.",
    )
    selection_flags = parser.add_mutually_exclusive_group()
    selection_flags.add_argument(
        "--player-selection",
        dest="player_selection",
        action="store_true",
        default=None,
        help="Explicit alias for --enable-selection.",
    )
    selection_flags.add_argument(
        "--no-player-selection",
        dest="player_selection",
        action="store_false",
        help="Explicitly keep selection disabled.",
    )
    parser.add_argument(
        "--embedding-bbox-scale",
        type=float,
        default=1.0,
        help=(
            "Runtime raw-pool coordinate scale joining geometry boxes to embedding boxes. "
            "The production raw-pool authority default is identity when no scale metadata exists."
        ),
    )
    args = parser.parse_args()

    try:
        if args.enable_selection and args.player_selection is False:
            raise ValueError(
                "--enable-selection and --no-player-selection cannot be combined"
            )
        selection_enabled = bool(
            args.enable_selection or args.player_selection is True
        )
        if args.scoring_projection is not None and not selection_enabled:
            raise ValueError("--scoring-projection requires --enable-selection")
        scoring_projection = (
            args.scoring_projection
            if args.scoring_projection is not None
            else _default_scoring_projection_path(args.out_tracks)
            if selection_enabled
            else None
        )
        scoring_projection_sha256 = (
            Path(f"{scoring_projection}.sha256")
            if scoring_projection is not None
            else None
        )
        _require_pairwise_distinct_paths(
            {
                "--tracks": args.tracks,
                "--raw-pool": args.raw_pool,
                "--embeddings": args.embeddings,
                "--calibration": args.calibration,
                "--out-tracks": args.out_tracks,
                "--report": args.report,
                "--scoring-projection": scoring_projection,
                "--scoring-projection-sha256": scoring_projection_sha256,
            }
        )
        tracks_payload = _read_object(args.tracks)
        config = PlayerSelectionConfig()
        if not selection_enabled:
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
                embedding_payload=(
                    _read_object(args.embeddings)
                    if args.embeddings is not None
                    else None
                ),
                calibration=calibration,
                enabled=True,
                config=config,
                embedding_bbox_scale=args.embedding_bbox_scale,
                reid_provider_available=args.embeddings is not None,
                reid_provider_reason=(
                    None
                    if args.embeddings is not None
                    else "embedding_provider_artifact_absent"
                ),
            )
            unbound_observations = selected.get("unbound_observations")
            if not isinstance(unbound_observations, list):
                raise ValueError(
                    "enabled selection must emit an unbound_observations list"
                )
            canonical_tracks_input = dict(selected)
            del canonical_tracks_input["unbound_observations"]
            validated = Tracks.model_validate(canonical_tracks_input)
            canonical_output = validated.model_dump(mode="json")
            canonical_output["unbound_observations"] = unbound_observations
            scoring_projection_payload = _field_stripped_scoring_projection(
                canonical_output
            )
            selected_text = (
                json.dumps(
                    canonical_output,
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            scoring_projection_text = (
                json.dumps(
                    scoring_projection_payload,
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            scoring_projection_digest = hashlib.sha256(
                scoring_projection_text.encode("utf-8")
            ).hexdigest()
            report_text = (
                json.dumps(
                    report,
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            assert scoring_projection is not None
            assert scoring_projection_sha256 is not None
            _stage_and_publish_enabled_outputs(
                tracks_text=selected_text,
                tracks_destination=args.out_tracks,
                report_text=report_text,
                report_destination=args.report,
                projection_text=scoring_projection_text,
                projection_destination=scoring_projection,
                projection_sha256_text=f"{scoring_projection_digest}\n",
                projection_sha256_destination=scoring_projection_sha256,
            )
    except Exception as exc:
        print(f"player selection failed: {exc}", file=sys.stderr)
        return 1

    print(args.out_tracks)
    print(args.report)
    if selection_enabled:
        assert scoring_projection is not None
        assert scoring_projection_sha256 is not None
        print(scoring_projection)
        print(scoring_projection_sha256)
        print(f"scoring_projection_sha256={scoring_projection_digest}")
        reid_summary = _selective_reid_summary(report)
        if int(reid_summary["reid_unavailable"]) > 0:
            print(
                "player selection warning: "
                f"reid_unavailable={reid_summary['reid_unavailable']}",
                file=sys.stderr,
            )
    return 0


def _read_object(path: Path | None) -> dict[str, Any]:
    if path is None:
        raise ValueError("required JSON path is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _default_scoring_projection_path(output: Path) -> Path:
    suffix = output.suffix or ".json"
    return output.with_name(f"{output.stem}.scoring_projection{suffix}")


def _selective_reid_summary(report: dict[str, Any]) -> dict[str, Any]:
    summaries = [
        row
        for row in report.get("decisions", [])
        if isinstance(row, dict)
        and row.get("action") == "selective_reid_policy_summary"
    ]
    if len(summaries) != 1:
        raise ValueError(
            "enabled selection must emit exactly one selective_reid_policy_summary"
        )
    return summaries[0]


def _field_stripped_scoring_projection(
    authoritative_output: dict[str, Any],
) -> dict[str, Any]:
    """Build the exact measured-only legacy Tracks input; never retain synthesis."""

    projection = copy.deepcopy(authoritative_output)
    projection.pop("unbound_observations", None)
    players = projection.get("players")
    if not isinstance(players, list):
        raise ValueError("authoritative selection output must contain a players list")
    for player in players:
        if not isinstance(player, dict) or not isinstance(player.get("frames"), list):
            raise ValueError("authoritative selection players must contain frame lists")
        measured_frames: list[dict[str, Any]] = []
        for frame in player["frames"]:
            if not isinstance(frame, dict):
                raise ValueError("authoritative selection frames must be objects")
            if frame.get("interpolated") is True:
                continue
            measured = dict(frame)
            measured.pop("interpolated", None)
            measured_frames.append(measured)
        player["frames"] = measured_frames
    # Validate the field-stripped payload without reserializing it through the
    # model, which would reintroduce optional default fields.
    Tracks.model_validate(projection)
    return projection


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

    _publish_output_bundle(
        (
            (staged_report, report_destination),
            (staged_tracks, tracks_destination),
        )
    )


def _publish_output_bundle(
    outputs: Sequence[tuple[Path, Path]],
) -> None:
    """Publish a staged bundle in order and restore every destination on failure."""

    if not outputs:
        raise ValueError("output bundle cannot be empty")
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for _staged, destination in outputs:
            if destination.exists():
                backups[destination] = _stage_copy(destination, destination)
        for staged, destination in outputs:
            os.replace(staged, destination)
            published.append(destination)
    except BaseException as publish_error:
        rollback_errors: list[BaseException] = []
        for destination in reversed(published):
            backup = backups.pop(destination, None)
            try:
                if backup is not None:
                    os.replace(backup, destination)
                else:
                    destination.unlink(missing_ok=True)
            except BaseException as exc:
                rollback_errors.append(exc)
        for staged, _destination in outputs:
            staged.unlink(missing_ok=True)
        for backup in backups.values():
            backup.unlink(missing_ok=True)
        if rollback_errors:
            raise RuntimeError(
                "output publication failed and one or more destinations could not be restored"
            ) from publish_error
        raise
    for backup in backups.values():
        backup.unlink(missing_ok=True)


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


def _stage_and_publish_enabled_outputs(
    *,
    tracks_text: str,
    tracks_destination: Path,
    report_text: str,
    report_destination: Path,
    projection_text: str,
    projection_destination: Path,
    projection_sha256_text: str,
    projection_sha256_destination: Path,
) -> None:
    output_texts = (
        (projection_destination, projection_text),
        (projection_sha256_destination, projection_sha256_text),
        (report_destination, report_text),
        # The authoritative full output is the final publication/commit point.
        (tracks_destination, tracks_text),
    )
    staged_outputs: list[tuple[Path, Path]] = []
    try:
        for destination, content in output_texts:
            staged_outputs.append((_stage_text(destination, content), destination))
    except BaseException:
        for staged, _destination in staged_outputs:
            staged.unlink(missing_ok=True)
        raise
    _publish_output_bundle(staged_outputs)


# Atomic publication helpers shared by the process_video selection seam.
stage_copy = _stage_copy
stage_text = _stage_text
publish_output_bundle = _publish_output_bundle


if __name__ == "__main__":
    raise SystemExit(main())
