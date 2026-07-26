#!/usr/bin/env python3
"""Launch the local ball-labelling studio for one pipeline run directory.

Click the ball in the video to fix the camera ray (2 of 3 degrees of freedom),
then set the remaining degree of freedom -- depth -- in a live 3D court view
with the tracked player skeletons drawn at their known 3D positions.

    .venv/bin/python scripts/racketsport/ball_label_studio.py \
        --run-dir runs/.../indoor_doubles_20s_fullmesh_final \
        --out runs/lanes/ball_label_tool_20260726/labels/indoor_doubles

CPU-only, no network, no GPU. Reads the run directory read-only and refuses to
write anything inside it: labels land in ``--out`` as ``ball_human_labels.json``
and are autosaved after every single label.

VERIFIED=0. This produces human review-only labels. Only ``bounce`` labels have
a geometrically solved depth; ``near_player`` and ``free_flight`` depths are
human estimates and are stored and reported as such.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_label_geometry import (  # noqa: E402
    DEFAULT_CLICK_SIGMA_PX,
    FREE_FLIGHT_DEPTH_SIGMA_M,
    NEAR_PLAYER_DEPTH_SIGMA_M,
)
from threed.racketsport.ball_label_studio import (  # noqa: E402
    StudioError,
    extract_frames,
    load_clip_bundle,
    open_session,
    run_studio_server,
    summarize_session,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Human ball-labelling studio: click the ray in 2D, set depth in 3D.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Pipeline run directory holding court_calibration.json, skeleton3d.json, "
        "ball_track.json and the source video. Read-only; never written to.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Where labels, the frame cache and the session file are written. "
        "Must be outside --run-dir.",
    )
    parser.add_argument("--clip-id", default=None, help="Override the inferred clip id.")
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Source video override. Needed for archived runs whose source.mp4 symlinks "
        "to the GPU box that produced them.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--no-open", action="store_true", help="Do not open a browser.")
    parser.add_argument(
        "--frame-quality",
        type=int,
        default=3,
        help="ffmpeg -q:v for the extracted frame cache (2 best .. 8 smallest).",
    )
    parser.add_argument(
        "--reextract-frames", action="store_true", help="Rebuild the frame cache."
    )
    parser.add_argument(
        "--click-sigma-px",
        type=float,
        default=DEFAULT_CLICK_SIGMA_PX,
        help="Assumed click precision in pixels; feeds every label's uncertainty.",
    )
    parser.add_argument(
        "--near-player-sigma-m",
        type=float,
        default=NEAR_PLAYER_DEPTH_SIGMA_M,
        help="Prior on how well a human resolves depth against a tracked skeleton.",
    )
    parser.add_argument(
        "--free-flight-sigma-m",
        type=float,
        default=FREE_FLIGHT_DEPTH_SIGMA_M,
        help="Prior on an unreferenced human depth guess. Keep this honest and large.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Load the run, extract frames, print a JSON readiness summary and exit "
        "without serving.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Also write the readiness summary to this path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        bundle = load_clip_bundle(args.run_dir, clip_id=args.clip_id, video_path=args.video)
        session = open_session(
            bundle,
            args.out,
            click_sigma_px=args.click_sigma_px,
            near_player_sigma_m=args.near_player_sigma_m,
            free_flight_sigma_m=args.free_flight_sigma_m,
        )
    except StudioError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"clip           {bundle.clip_id}")
    print(f"run dir        {bundle.run_dir}  (read-only)")
    print(f"video          {bundle.video_path}")
    print(f"frames         {bundle.frame_count} @ {bundle.fps} fps")
    print(f"players        {len(bundle.players)} skeleton tracks")
    print(f"detections     {len(bundle.detections)} frames with a ball guess")
    print(f"prefill        {len(bundle.prefill)} frames with a solver 3D position")
    print(f"bounce cands   {len(bundle.bounce_candidate_frames)}")
    if bundle.missing_artifacts:
        print(f"missing        {', '.join(bundle.missing_artifacts)}")
    residuals = bundle.plane_residuals
    if residuals.get("available"):
        print(
            "accuracy floor bounce labels inherit "
            f"{residuals['median_m']:.3f} m median / {residuals['max_m']:.3f} m worst "
            f"court-plane error from this clip's calibration"
        )

    try:
        extraction = extract_frames(
            bundle.video_path,
            session.frames_dir,
            expected_count=bundle.frame_count,
            quality=args.frame_quality,
            force=args.reextract_frames,
        )
    except StudioError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"frame cache    {extraction['frame_count']} jpgs in {extraction['dir']}"
        + (f" ({extraction['elapsed_s']}s)" if extraction.get("extracted") else " (cached)")
    )

    summary = summarize_session(session)
    summary["frame_cache"] = extraction
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"summary        {args.summary_json}")

    if args.check:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    server = run_studio_server(session, host=args.host, port=args.port)
    url = f"http://{args.host}:{server.server_address[1]}/"
    print(f"labels         {session.label_path}  (autosaved after every label)")
    print(f"studio         {url}")
    print("VERIFIED=0 - review-only human labels. Only bounce labels have a solved depth.")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\nstopping. {len(session.label_set.labels)} labels in {session.label_path}")
    finally:
        threading.Thread(target=server.server_close).start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
