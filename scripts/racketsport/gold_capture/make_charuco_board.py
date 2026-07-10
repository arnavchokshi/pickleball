#!/usr/bin/env python3
"""Generate the repo-compatible A3 ChArUco board and its physical-size spec."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

SQUARES_X = 5
SQUARES_Y = 7
SQUARE_LENGTH_MM = 40.0
MARKER_LENGTH_MM = 30.0
PAGE_WIDTH_MM = 297.0
PAGE_HEIGHT_MM = 420.0
BOARD_WIDTH_MM = SQUARES_X * SQUARE_LENGTH_MM
BOARD_HEIGHT_MM = SQUARES_Y * SQUARE_LENGTH_MM
DICTIONARY_NAME = "DICT_4X4_50"
REPO_CALIBRATION_TOOL = "scripts/racketsport/calibrate_charuco_device.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def projected_marker_pixels(*, distance_m: float, frame_width_px: int, horizontal_fov_deg: float) -> float:
    """Pinhole planning estimate; the printed spec labels this as non-acceptance math."""
    span_m = 2.0 * distance_m * math.tan(math.radians(horizontal_fov_deg) / 2.0)
    return frame_width_px * (MARKER_LENGTH_MM / 1000.0) / span_m


def build_board(*, output: Path, spec_output: Path, dpi: int) -> dict[str, object]:
    if dpi < 72:
        raise ValueError("dpi must be at least 72")

    import cv2  # type: ignore[import-not-found]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV aruco support is required")

    dictionary_id = getattr(cv2.aruco, DICTIONARY_NAME)
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_LENGTH_MM / 1000.0,
        MARKER_LENGTH_MM / 1000.0,
        dictionary,
    )
    board_width_px = round(BOARD_WIDTH_MM / 25.4 * dpi)
    board_height_px = round(BOARD_HEIGHT_MM / 25.4 * dpi)
    board_image = board.generateImage((board_width_px, board_height_px), marginSize=0, borderBits=1)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(PAGE_WIDTH_MM / 25.4, PAGE_HEIGHT_MM / 25.4), dpi=dpi, facecolor="white")
    left_mm = (PAGE_WIDTH_MM - BOARD_WIDTH_MM) / 2.0
    bottom_mm = 70.0
    board_ax = fig.add_axes(
        [
            left_mm / PAGE_WIDTH_MM,
            bottom_mm / PAGE_HEIGHT_MM,
            BOARD_WIDTH_MM / PAGE_WIDTH_MM,
            BOARD_HEIGHT_MM / PAGE_HEIGHT_MM,
        ]
    )
    board_ax.imshow(board_image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    board_ax.set_axis_off()

    fig.text(0.5, 0.968, "DinkVision Gold Capture - ChArUco A3", ha="center", va="top", fontsize=14, weight="bold")
    fig.text(
        0.5,
        0.944,
        "5 x 7 squares | 40.0 mm square | 30.0 mm marker | DICT_4X4_50 | print at 100%",
        ha="center",
        va="top",
        fontsize=8.5,
    )
    fig.text(
        0.5,
        0.060,
        "Mount flat on matte rigid board. Do not crop. Measure the 200 mm bar and one 40 mm square before use.",
        ha="center",
        va="bottom",
        fontsize=7.5,
    )
    fig.text(
        0.5,
        0.049,
        "The product remains monocular; extra cameras, markers, and surveys are GT-only.",
        ha="center",
        va="bottom",
        fontsize=6.5,
    )

    ruler_ax = fig.add_axes([0.5 - 100.0 / PAGE_WIDTH_MM, 0.024, 200.0 / PAGE_WIDTH_MM, 0.018])
    for index in range(10):
        ruler_ax.add_patch(
            plt.Rectangle((index * 20.0, 0.0), 20.0, 10.0, color="black" if index % 2 == 0 else "white", ec="black")
        )
    ruler_ax.set_xlim(0.0, 200.0)
    ruler_ax.set_ylim(0.0, 10.0)
    ruler_ax.set_xticks([0.0, 100.0, 200.0], labels=["0", "100 mm", "200 mm"])
    ruler_ax.set_yticks([])
    ruler_ax.tick_params(axis="x", labelsize=7, pad=1)
    for spine in ruler_ax.spines.values():
        spine.set_visible(False)

    fig.savefig(output, format="pdf", dpi=dpi)
    plt.close(fig)

    planning = [
        {
            "distance_m": distance,
            "projected_marker_width_px": round(
                projected_marker_pixels(distance_m=distance, frame_width_px=1920, horizontal_fov_deg=70.0), 2
            ),
        }
        for distance in (0.75, 1.5, 2.0)
    ]
    spec: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "gold_capture_charuco_board_spec",
        "output_pdf": output.as_posix(),
        "output_pdf_sha256": _sha256(output),
        "page": {"standard": "ISO_A3_portrait", "width_mm": PAGE_WIDTH_MM, "height_mm": PAGE_HEIGHT_MM},
        "board": {
            "squares_x": SQUARES_X,
            "squares_y": SQUARES_Y,
            "square_length_mm": SQUARE_LENGTH_MM,
            "marker_length_mm": MARKER_LENGTH_MM,
            "marker_to_square_ratio": MARKER_LENGTH_MM / SQUARE_LENGTH_MM,
            "printed_width_mm": BOARD_WIDTH_MM,
            "printed_height_mm": BOARD_HEIGHT_MM,
            "dictionary": DICTIONARY_NAME,
            "border_bits": 1,
        },
        "repo_compatibility": {
            "calibration_tool": REPO_CALIBRATION_TOOL,
            "contract": "Matches the read-only tool's 5x7, 0.04 m square, 0.03 m marker, DICT_4X4_50 board.",
        },
        "print_acceptance": {
            "scale_percent": 100,
            "measure_square_mm": SQUARE_LENGTH_MM,
            "measure_five_square_span_mm": BOARD_WIDTH_MM,
            "absolute_tolerance_mm": 0.5,
            "surface": "matte_flat_rigid",
        },
        "court_distance_planning_math": {
            "formula": "marker_px = frame_width_px * marker_width_m / (2 * distance_m * tan(horizontal_fov_deg / 2))",
            "assumptions": {"frame_width_px": 1920, "horizontal_fov_deg": 70.0},
            "estimates": planning,
            "use_limit_m": 2.0,
            "honesty_note": "Planning estimate only; actual on-device ChArUco detections are the acceptance check.",
        },
        "product_boundary": "The product remains monocular; extra cameras, markers, and surveys are GT-only.",
    }
    spec_output.parent.mkdir(parents=True, exist_ok=True)
    spec_output.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return spec


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Output printable PDF path.")
    parser.add_argument("--spec-output", type=Path, required=True, help="Output JSON physical-size spec path.")
    parser.add_argument("--dpi", type=int, default=300, help="Embedded raster resolution; physical PDF size is fixed.")
    args = parser.parse_args(argv)
    spec = build_board(output=args.output, spec_output=args.spec_output, dpi=args.dpi)
    print(json.dumps({"status": "pass", "pdf": str(args.output), "spec": str(args.spec_output), "sha256": spec["output_pdf_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
