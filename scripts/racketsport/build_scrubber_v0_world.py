#!/usr/bin/env python3
"""Build a W3-SCRUBBER-V0 `virtual_world.json` with trust-band provenance.

This assembles a scrubber-ready virtual world from already-produced stage
artifacts (never re-running any gate) and wires a per-entity trust band onto
players/court/ball/paddles from the real gate-report state of those
artifacts, per NORTH_STAR_ROADMAP.md's "3D world / scrubber surfaces" section and
the W3-SCRUBBER-V0 milestone row.

BODY world joints can come from either a full `skeleton3d.json`/
`smpl_motion.json` artifact, or -- when only the compact review packet is
available locally (mesh vertices may be VM-only) -- from
`body_world_label_packet.json`, which this script converts into a
`Skeleton3D`-shaped preview via
`threed.racketsport.body_world_label_packet_skeleton`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.body_world_label_packet_skeleton import (  # noqa: E402
    skeleton3d_from_body_world_label_packet,
)
from threed.racketsport.trust_band import (  # noqa: E402
    derive_ball_trust_band,
    derive_body_trust_band,
    derive_court_trust_band,
    derive_paddle_trust_band,
    derive_track_trust_band,
)
from threed.racketsport.virtual_world import build_virtual_world_state, write_virtual_world  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", required=True, help="Clip id (used only for the printed summary).")
    parser.add_argument("--court-calibration", type=Path, required=True, help="court_calibration.json artifact.")
    parser.add_argument("--tracks", type=Path, help="tracks.json artifact (full-clip TRK coverage).")
    parser.add_argument("--smpl-motion", type=Path, help="Optional smpl_motion.json artifact.")
    parser.add_argument("--skeleton3d", type=Path, help="Optional skeleton3d.json artifact.")
    parser.add_argument(
        "--body-world-label-packet",
        type=Path,
        help=(
            "Optional compact body_world_label_packet.json to convert into a preview "
            "skeleton3d when no full skeleton3d.json/smpl_motion.json is available locally."
        ),
    )
    parser.add_argument("--ball-track", type=Path, help="Optional ball_track.json artifact.")
    parser.add_argument("--racket-pose", type=Path, help="Optional racket_pose.json artifact.")
    parser.add_argument("--physics-footlock", type=Path, help="Optional physics_footlock.json artifact.")
    parser.add_argument("--ball-track-physics-filled", type=Path, help="Optional ball_track_physics_filled.json artifact.")
    parser.add_argument("--racket-pose-estimate", type=Path, help="Optional racket_pose_estimate.json artifact.")
    parser.add_argument(
        "--body-gate-report",
        type=Path,
        help="Optional body_gate_report.json used to derive the BODY trust band from real gate state.",
    )
    parser.add_argument(
        "--body-gate-report-clip",
        default=None,
        help="Substring to select a clip entry from body_gate_report.json when it has more than one.",
    )
    parser.add_argument("--track-idf1", type=float, default=None, help="Measured TRK IDF1 for the track-trust-band reason text.")
    parser.add_argument("--track-evidence", default=None, help="Evidence path/run dir backing --track-idf1.")
    parser.add_argument("--out", type=Path, required=True, help="Output virtual_world.json path.")
    parser.add_argument(
        "--trust-band-report-out",
        type=Path,
        default=None,
        help="Optional path to also dump the derived trust bands as their own JSON for audit.",
    )
    args = parser.parse_args(argv)

    try:
        court_calibration = _read_json(args.court_calibration)
        tracks = _read_optional_json(args.tracks)
        smpl_motion = _read_optional_json(args.smpl_motion)
        skeleton3d = _read_optional_json(args.skeleton3d)
        ball_track = _read_optional_json(args.ball_track)
        racket_pose = _read_optional_json(args.racket_pose)
        physics_footlock = _read_optional_json(args.physics_footlock)
        ball_track_physics_filled = _read_optional_json(args.ball_track_physics_filled)
        racket_pose_estimate = _read_optional_json(args.racket_pose_estimate)

        if skeleton3d is None and smpl_motion is None and args.body_world_label_packet is not None:
            packet = _read_json(args.body_world_label_packet)
            fps = float((tracks or {}).get("fps") or 30.0)
            skeleton3d = skeleton3d_from_body_world_label_packet(packet, fps=fps)

        trust_bands, trust_band_sources = _derive_trust_bands(
            args=args,
            court_calibration=court_calibration,
            ball_track=ball_track,
            racket_pose=racket_pose,
        )

        payload = build_virtual_world_state(
            court_calibration=court_calibration,
            tracks=tracks,
            smpl_motion=smpl_motion,
            skeleton3d=skeleton3d,
            ball_track=ball_track,
            racket_pose=racket_pose,
            trust_bands=trust_bands,
            physics_footlock=physics_footlock,
            ball_track_physics_filled=ball_track_physics_filled,
            racket_pose_estimate=racket_pose_estimate,
            placement_calibration_path=args.court_calibration,
            artifact_paths={
                "physics_footlock": args.physics_footlock,
                "ball_track_physics_filled": args.ball_track_physics_filled,
                "racket_pose_estimate": args.racket_pose_estimate,
            },
        )
        write_virtual_world(args.out, payload)
        if args.trust_band_report_out is not None:
            args.trust_band_report_out.parent.mkdir(parents=True, exist_ok=True)
            args.trust_band_report_out.write_text(
                json.dumps({"clip": args.clip, "trust_bands": trust_band_sources}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"ERROR: scrubber world build failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "schema_version": 1,
                "clip": args.clip,
                "out": str(args.out),
                "summary": payload["summary"],
                "trust_bands": trust_band_sources,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _derive_trust_bands(
    *,
    args: argparse.Namespace,
    court_calibration: Mapping[str, Any],
    ball_track: Mapping[str, Any] | None,
    racket_pose: Mapping[str, Any] | None,
) -> tuple[dict[str, dict[str, Any] | None], dict[str, dict[str, Any] | None]]:
    trust_bands: dict[str, dict[str, Any] | None] = {
        "court": derive_court_trust_band(court_calibration, evidence_path=str(args.court_calibration)),
    }

    if args.body_gate_report is not None:
        gate_report = _read_json(args.body_gate_report)
        clip_entry = _select_gate_report_clip(gate_report, args.body_gate_report_clip)
        trust_bands["body"] = derive_body_trust_band(clip_entry, evidence_path=str(args.body_gate_report))

    trust_bands["track"] = derive_track_trust_band(
        idf1=args.track_idf1,
        evidence_path=args.track_evidence or "no --track-evidence supplied",
    )

    if ball_track is not None:
        trust_bands["ball"] = derive_ball_trust_band(
            source=ball_track.get("source"),
            evidence_path=str(args.ball_track),
        )

    if racket_pose is not None:
        trust_bands["paddle"] = derive_paddle_trust_band(evidence_path=str(args.racket_pose))

    return trust_bands, trust_bands


def _select_gate_report_clip(gate_report: Mapping[str, Any], clip_substring: str | None) -> dict[str, Any]:
    clips = gate_report.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError("body_gate_report.json has no clips entries")
    if clip_substring is None:
        if len(clips) == 1:
            return clips[0]
        raise ValueError(
            "body_gate_report.json has multiple clips; pass --body-gate-report-clip to select one"
        )
    matches = [clip for clip in clips if clip_substring in str(clip.get("clip", ""))]
    if len(matches) != 1:
        raise ValueError(
            f"--body-gate-report-clip {clip_substring!r} matched {len(matches)} clip entries, expected exactly 1"
        )
    return matches[0]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return _read_json(path)


if __name__ == "__main__":
    raise SystemExit(main())
