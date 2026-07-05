from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from threed.racketsport import orchestrator


def test_compact_json_writer_round_trips_nested_payload_and_terminates_with_newline(tmp_path: Path) -> None:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "representative_nested_payload",
        "players": [
            {
                "id": 7,
                "frames": [
                    {
                        "frame_idx": 3,
                        "joints_world": [[0.1, 0.2, 1.0], [0.4, 0.5, 1.3]],
                        "confidence": {"band": "preview", "reasons": ["unit_test"]},
                    }
                ],
            }
        ],
        "summary": {"mesh_frame_count": 1, "player_count": 1},
    }
    out = tmp_path / "smpl_motion.json"

    timing = orchestrator._write_compact_json(out, payload)

    raw = out.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    assert "\n " not in raw
    assert ": " not in raw
    assert ", " not in raw
    assert json.loads(raw) == payload
    assert timing["bytes"] == out.stat().st_size
    assert timing["serialization_seconds"] >= 0.0
