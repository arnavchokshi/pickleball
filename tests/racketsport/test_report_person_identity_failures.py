from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


CLI_PATH = "scripts/racketsport/report_person_identity_failures.py"


def _tracks() -> dict:
    fps = 60.0
    return {
        "schema_version": 1,
        "fps": fps,
        "players": [
            {
                "id": 2,
                "side": "near",
                "role": "left",
                "frames": [
                    {"t": 812 / fps, "bbox": [0, 0, 20, 80], "world_xy": [0.0, 0.0], "conf": 0.9},
                    {"t": 813 / fps, "bbox": [0, 0, 20, 80], "world_xy": [2.0, 0.0], "conf": 0.9},
                ],
            },
            {
                "id": 3,
                "side": "far",
                "role": "right",
                "frames": [
                    {"t": frame / fps, "bbox": [100, 0, 130, 80], "world_xy": [1.0, 4.0], "conf": 0.8}
                    for frame in range(120, 126)
                ],
            },
            {
                "id": 4,
                "side": "far",
                "role": "left",
                "frames": [
                    {
                        "t": frame / fps,
                        "bbox": [140, 0, 170, 80],
                        "world_xy": [0.0, 0.02495 * (frame - 120)],
                        "conf": 0.8,
                    }
                    for frame in range(120, 126)
                ],
            },
        ],
    }


def test_identity_failure_reporter_surfaces_required_trk_symptoms(tmp_path: Path) -> None:
    tracks_path = tmp_path / "tracks.json"
    sidecar_path = tmp_path / "native2d_sidecar.json"
    membership_path = tmp_path / "court_membership.json"
    sam3d_evidence_path = tmp_path / "sam3d_identity_evidence.json"
    out_json = tmp_path / "identity_failures.json"
    out_md = tmp_path / "IDENTITY_FAILURES.md"

    tracks_path.write_text(json.dumps(_tracks(), indent=2) + "\n", encoding="utf-8")
    sidecar_path.write_text(
        json.dumps(
            {
                "players": [
                    {
                        "id": 1,
                        "frames": [
                            {"frame_idx": 120, "bbox": [100, 0, 130, 80]},
                            {"frame_idx": 121, "bbox": [100, 0, 130, 80]},
                        ],
                    }
                ]
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    membership_path.write_text(
        json.dumps(
            {
                "artifact_type": "racketsport_person_court_membership",
                "fragments": [
                    {"fragment_id": "track_3", "source_tracker_id": 3, "membership_class": "adjacent_court"},
                    {"fragment_id": "track_4", "source_tracker_id": 4, "membership_class": "adjacent_court"},
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    sam3d_evidence_path.write_text(
        json.dumps(
            {
                "artifact_type": "racketsport_sam3d_identity_evidence",
                "body_observations": [
                    {
                        "body_observation_id": "813:2",
                        "frame_idx": 813,
                        "player_id": 2,
                        "risk_flags": ["placement_track_world_xy_anchor"],
                        "root_track_residual_m": 1.25,
                        "transl_world_independent": False,
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            CLI_PATH,
            "--tracks",
            str(tracks_path),
            "--sidecar",
            str(sidecar_path),
            "--court-membership",
            str(membership_path),
            "--sam3d-identity-evidence",
            str(sam3d_evidence_path),
            "--clip-id",
            "owner_IMG_1605_8a193402780b",
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["artifact_type"] == "racketsport_person_identity_failure_report"
    assert report["track_summary"]["track_count"] == 3
    assert report["speed_teleport_summary"]["teleport_count"] == 1
    assert report["speed_teleport_summary"]["teleports"][0]["to_frame"] == 813
    assert report["watched_failures"]["outdoor_p2_frame_813"]["status"] == "flagged"
    assert report["constant_speed_summary"]["constant_speed_span_count"] == 1
    assert report["constant_speed_summary"]["spans"][0]["player_id"] == 4
    assert report["id_sidecar_mismatch_summary"]["mismatches"][0]["sidecar_id"] == 1
    assert report["id_sidecar_mismatch_summary"]["mismatches"][0]["majority_track_id"] == 3
    assert report["court_membership_violation_summary"]["violating_track_ids"] == [3, 4]
    assert report["watched_failures"]["img1605_p3_p4_adjacent_constant_speed"]["status"] == "flagged"
    assert report["body_sam3d_risk_summary"]["inherited_anchor_risk_count"] == 1
    assert report["body_sam3d_risk_summary"]["root_track_residual_conflict_count"] == 1

    markdown = out_md.read_text(encoding="utf-8")
    assert "Outdoor p2 frame 813" in markdown
    assert "IMG_1605 p3/p4" in markdown
    assert "SAM3D root/track residual conflicts" in markdown
    assert CLI_PATH in Path("tests/racketsport/test_report_person_identity_failures.py").read_text(encoding="utf-8")
