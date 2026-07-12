from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

from threed.racketsport.rally_metrics import build_rally_metrics


def test_audited_facts_are_deterministic_complete_and_fail_closed(tmp_path: Path) -> None:
    run_dir = _write_fact_run(tmp_path / "run")

    first = build_rally_metrics(run_dir)["coaching_card_facts"]
    second = build_rally_metrics(run_dir)["coaching_card_facts"]

    first_bytes = (json.dumps(first, indent=2, sort_keys=True) + "\n").encode("utf-8")
    second_bytes = (json.dumps(second, indent=2, sort_keys=True) + "\n").encode("utf-8")
    assert first_bytes == second_bytes
    assert first["build_order"] == "coaching_facts_before_manifest"
    assert first["compatibility"]["user_facing"] is False
    assert {fact["fact_type"] for fact in first["audited_facts"]} == {
        "rally",
        "movement",
        "positioning",
        "recovery",
    }
    assert {omission["fact_type"] for omission in first["omissions"]} == {"shot", "landing", "contact"}
    assert all(omission["status"] == "absent" for omission in first["omissions"])
    assert all(omission["reason_code"] == "required_artifact_missing" for omission in first["omissions"])

    for fact in first["audited_facts"]:
        assert fact["fact_id"].startswith("ns051.")
        assert fact["interval"]["frame_end_exclusive"] > fact["interval"]["frame_start"]
        assert fact["interval"]["pts_end_s"] >= fact["interval"]["pts_start_s"]
        assert fact["entity"]["id"]
        assert fact["coordinate_space"] in {"court_Z0_xy_m", "not_applicable"}
        assert fact["time_space"] == "source_video_pts_s"
        assert fact["trust"]["authority_band"] == "preview"
        assert fact["trust"]["gate_status"] == "unpassed"
        assert fact["coverage"]["frames_total"] >= fact["coverage"]["frames_used"]
        assert fact["rule"]["version"] == "ns051.1"
        assert fact["numeric_lineage"]
        for source in fact["source_artifacts"]:
            source_path = Path(source["path"])
            assert source_path.is_file()
            assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source["sha256"]
        parsed = urlsplit(fact["evidence_locator"]["uri"])
        assert parsed.scheme == "file"
        assert Path(unquote(parsed.path)).is_file()
        assert parsed.fragment == fact["evidence_locator"]["json_pointer"]


def test_legacy_projection_remains_additive_for_existing_builder_consumers(tmp_path: Path) -> None:
    facts = build_rally_metrics(_write_fact_run(tmp_path / "run"))["coaching_card_facts"]

    assert facts["facts"][0]["metric"] == "contact_count"
    assert facts["facts"][0]["trust"] == "unverified_cue"
    assert facts["compatibility"] == {
        "facts_field": "legacy_v1_projection",
        "authoritative_field": "audited_facts",
        "user_facing": False,
    }
    assert all(fact["fact_type"] != "contact" for fact in facts["audited_facts"])


def _write_fact_run(run_dir: Path) -> Path:
    run_dir.mkdir(parents=True)
    world = {
        "schema_version": 1,
        "artifact_type": "racketsport_virtual_world",
        "fps": 10.0,
        "summary": {"ball_frame_count": 4},
        "players": [
            {
                "id": "p1",
                "frames": [
                    {"t": 0.0, "track_world_xy": [0.0, -5.0], "trust_band": {"gate_status": "preview"}},
                    {"t": 0.1, "track_world_xy": [0.0, -4.0], "trust_band": {"gate_status": "preview"}},
                    {"t": 0.2, "track_world_xy": [0.0, -2.3], "trust_band": {"gate_status": "preview"}},
                    {"t": 0.3, "track_world_xy": [0.0, -2.0], "trust_band": {"gate_status": "preview"}},
                ],
            }
        ],
        "ball": {"frames": []},
    }
    zones = {
        "court": [[-3.0, -6.0], [3.0, -6.0], [3.0, 6.0], [-3.0, 6.0]],
        "near_nvz": [[-3.0, -2.0], [3.0, -2.0], [3.0, 0.0], [-3.0, 0.0]],
        "far_nvz": [[-3.0, 0.0], [3.0, 0.0], [3.0, 2.0], [-3.0, 2.0]],
    }
    spans = {"schema_version": 1, "artifact_type": "racketsport_rally_spans", "spans": [{"id": "r0", "t0": 0.0, "t1": 0.4}]}
    contacts = {
        "schema_version": 1,
        "artifact_type": "racketsport_contact_windows",
        "events": [
            {
                "type": "contact",
                "player_id": "p1",
                "frame": 1,
                "t": 0.1,
                "trust_band_note": "wrist-cue-only, unverified",
            }
        ],
    }
    _write_json(run_dir / "virtual_world.json", world)
    _write_json(run_dir / "court_zones.json", {"schema_version": 1, "artifact_type": "racketsport_court_zones", "zones": zones})
    _write_json(run_dir / "rally_spans.json", spans)
    _write_json(run_dir / "contact_windows.json", contacts)
    return run_dir


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
