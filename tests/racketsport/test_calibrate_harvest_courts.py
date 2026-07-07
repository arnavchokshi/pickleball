from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.racketsport.calibrate_harvest_courts import (
    DECLARED_SKIP_SOURCE_IDS,
    GRADE_AUTO_BAR,
    GRADE_FAILED,
    GRADE_MANUAL_BAR,
    build_coverage_report,
    gather_source_frames,
    grade_calibration,
    parse_cvat_image_points,
)


def _write_annotations(path: Path) -> None:
    labels = ["near_baseline_center", "far_left_corner", "far_baseline_center", "far_right_corner"]
    extra = ["net_left_sideline", "net_center", "net_right_sideline"]
    full_labels = labels + extra + [
        "near_left_corner",
        "near_right_corner",
        "near_nvz_left",
        "near_nvz_center",
        "near_nvz_right",
        "far_nvz_left",
        "far_nvz_center",
        "far_nvz_right",
    ]

    def points_xml(names: list[str]) -> str:
        return "\n".join(
            f'    <points label="{name}" source="manual" occluded="0" points="{100 + idx:.2f},{200 + idx:.2f}" z_order="0" />'
            for idx, name in enumerate(names)
        )

    path.write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <image id="0" name="fullSrc__fullSrc_rally_0001__abs_000010.png" width="1920" height="1080">
{points_xml(full_labels)}
  </image>
  <image id="1" name="partialSrc__partialSrc_rally_0001__abs_000020.png" width="1920" height="1080">
{points_xml(labels + extra)}
  </image>
  <image id="2" name="Ezz6HDNHlnk__Ezz6HDNHlnk_rally_0004__abs_010677.png" width="1920" height="1080">
{points_xml(["near_baseline_center"])}
  </image>
</annotations>
""",
        encoding="utf-8",
    )


def test_gather_source_frames_applies_owner_stray_drop_and_keeps_partial_subsets(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.xml"
    _write_annotations(annotations)

    parsed = parse_cvat_image_points(annotations)
    grouped, dropped = gather_source_frames(
        parsed,
        declared_skip_source_ids=DECLARED_SKIP_SOURCE_IDS,
        tennis_overlay_partial_source_ids={"partialSrc"},
    )

    assert sorted(grouped) == ["fullSrc", "partialSrc"]
    assert grouped["fullSrc"][0].point_count == 15
    assert grouped["partialSrc"][0].point_count == 7
    assert grouped["partialSrc"][0].quality_flags == ["tennis_overlay_partial"]
    assert dropped == [
        {
            "source_id": "Ezz6HDNHlnk",
            "frame_name": "Ezz6HDNHlnk__Ezz6HDNHlnk_rally_0004__abs_010677.png",
            "reason": "owner_declared_skip_stray_drop",
            "dropped_point_count": 1,
        }
    ]
    assert "near_left_corner" not in grouped["partialSrc"][0].points


def test_gather_source_frames_refuses_heldout_source_labels(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.xml"
    annotations.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <image id="0" name="pwxNwFfYQlQ__pwxNwFfYQlQ_rally_0001__abs_000010.png" width="1280" height="720">
    <points label="near_baseline_center" points="10.0,20.0" />
  </image>
</annotations>
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="held-out source"):
        gather_source_frames(parse_cvat_image_points(annotations))


def test_grade_calibration_thresholds_and_failures() -> None:
    assert grade_calibration(median_px=4.8, p95_px=12.3) == (GRADE_MANUAL_BAR, None)
    assert grade_calibration(median_px=4.81, p95_px=12.3) == (GRADE_AUTO_BAR, None)
    assert grade_calibration(median_px=9.0, p95_px=20.0) == (GRADE_AUTO_BAR, None)
    assert grade_calibration(median_px=4.0, p95_px=20.01) == (GRADE_FAILED, "reprojection_above_auto_bar")
    assert grade_calibration(median_px=None, p95_px=None, failure_reason="insufficient_points: 4 < 6") == (
        GRADE_FAILED,
        "insufficient_points: 4 < 6",
    )


def test_build_coverage_report_maps_prelabels_to_source_grades(tmp_path: Path) -> None:
    prelabels = tmp_path / "prelabels"
    for clip_id in ["fullSrc_rally_0001", "partialSrc_rally_0002", "_L0HVmAlCQI_rally_0019"]:
        clip_dir = prelabels / clip_id
        clip_dir.mkdir(parents=True)
        (clip_dir / "ball_track.json").write_text("{}\n", encoding="utf-8")

    report = build_coverage_report(
        prelabels_dir=prelabels,
        source_grades={
            "fullSrc": {"calibration_grade": GRADE_MANUAL_BAR, "artifact_path": "cal/fullSrc.json"},
            "partialSrc": {"calibration_grade": GRADE_FAILED, "failure_reason": "insufficient_points: 4 < 6"},
            "_L0HVmAlCQI": {"calibration_grade": GRADE_AUTO_BAR, "artifact_path": "cal/_L0HVmAlCQI.json"},
        },
    )

    assert report["summary"]["clip_count"] == 3
    assert report["summary"]["covered_by_grade"] == {GRADE_AUTO_BAR: 1, GRADE_MANUAL_BAR: 1}
    assert report["summary"]["failed_clip_count"] == 1
    rows = {row["clip_id"]: row for row in report["clips"]}
    assert rows["fullSrc_rally_0001"]["calibrated"] is True
    assert rows["partialSrc_rally_0002"]["calibrated"] is False
    assert rows["_L0HVmAlCQI_rally_0019"]["source_id"] == "_L0HVmAlCQI"


def test_calibrate_harvest_courts_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/racketsport/calibrate_harvest_courts.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "scripts/racketsport/calibrate_harvest_courts.py"
    assert "--cvat-export-xml" in result.stdout
