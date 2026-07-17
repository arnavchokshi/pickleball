#!/usr/bin/env python3
"""Create the immutable first-70s demo slice used for no-soft byte parity."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = (
    ROOT.parent
    / "pbv11_headtohead_20260713/rerun_20260715/vm_pull_partial/"
    "pbvision_11min_20260713"
)
OUTPUT = ROOT / "demo_slice_input"
FRAME_END_EXCLUSIVE = 2100
SHORT_OUTPUT = ROOT / "demo_short_slice_input"
SHORT_FRAME_START = 4750
SHORT_FRAME_END_EXCLUSIVE = 4900


def read(name: str) -> dict:
    return json.loads((SOURCE / name).read_text())


def write(name: str, value: dict) -> None:
    (OUTPUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    track = read("ball_track.json")
    candidates = read("ball_candidates.json")
    frame_times = read("frame_times.json")
    track["frames"] = track["frames"][:FRAME_END_EXCLUSIVE]
    candidates["frames"] = candidates["frames"][:FRAME_END_EXCLUSIVE]
    frame_times["frames"] = frame_times["frames"][:FRAME_END_EXCLUSIVE]
    frame_times["frame_count"] = FRAME_END_EXCLUSIVE
    frame_times["duration_s"] = frame_times["frames"][-1]["pts_s"]
    frame_times["clip_path"] = "pbvision_11min_20260713:R&D-reference:first-2100-frames"
    write("ball_track.json", track)
    write("ball_candidates.json", candidates)
    write("frame_times.json", frame_times)
    shutil.copyfile(SOURCE / "court_calibration.json", OUTPUT / "court_calibration.json")
    shutil.copyfile(SOURCE / "net_plane.json", OUTPUT / "net_plane.json")
    write(
        "slice_provenance.json",
        {
            "source": str(SOURCE),
            "frame_start": 0,
            "frame_end_exclusive": FRAME_END_EXCLUSIVE,
            "pbvision_policy": "R&D reference only; never GT, training, or redistribution",
            "purpose": "byte-identity regression only",
        },
    )
    short_track = read("ball_track.json")
    short_candidates = read("ball_candidates.json")
    short_times = read("frame_times.json")
    t0 = float(short_times["frames"][SHORT_FRAME_START]["pts_s"])
    short_track["frames"] = short_track["frames"][SHORT_FRAME_START:SHORT_FRAME_END_EXCLUSIVE]
    for frame in short_track["frames"]:
        frame["t"] = round(float(frame["t"]) - t0, 6)
    short_candidates["frames"] = short_candidates["frames"][
        SHORT_FRAME_START:SHORT_FRAME_END_EXCLUSIVE
    ]
    for frame in short_candidates["frames"]:
        frame["frame"] = int(frame["frame"]) - SHORT_FRAME_START
    short_times["frames"] = short_times["frames"][SHORT_FRAME_START:SHORT_FRAME_END_EXCLUSIVE]
    for frame in short_times["frames"]:
        frame["frame"] = int(frame["frame"]) - SHORT_FRAME_START
        frame["pts_s"] = round(float(frame["pts_s"]) - t0, 6)
    short_times["frame_count"] = len(short_times["frames"])
    short_times["duration_s"] = short_times["frames"][-1]["pts_s"]
    short_times["source_start_pts_s"] = t0
    short_times["clip_path"] = "pbvision_11min_20260713:R&D-reference:frames-4750-4899"
    SHORT_OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("ball_track.json", short_track),
        ("ball_candidates.json", short_candidates),
        ("frame_times.json", short_times),
    ):
        (SHORT_OUTPUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    shutil.copyfile(SOURCE / "court_calibration.json", SHORT_OUTPUT / "court_calibration.json")
    shutil.copyfile(SOURCE / "net_plane.json", SHORT_OUTPUT / "net_plane.json")
    (SHORT_OUTPUT / "slice_provenance.json").write_text(
        json.dumps(
            {
                "source": str(SOURCE),
                "source_frame_start": SHORT_FRAME_START,
                "source_frame_end_exclusive": SHORT_FRAME_END_EXCLUSIVE,
                "rebase_time_origin_s": t0,
                "pbvision_policy": "R&D reference only; never GT, training, or redistribution",
                "purpose": "byte-identity regression only",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
