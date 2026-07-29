"""CPU-local full-scale repro for the player_selection disjointness bug.

Reproduces the exact `ValueError: bound, unbound, and dropped raw UID sets
must be disjoint` observed at `threed/racketsport/player_selection.py:3605`
on the real 20,922-frame pb.vision full-game run (fullgame_demo_20260728,
FAILED at player_selection after 448.7s). Confirmed to raise on pre-fix
code and succeed on post-fix code, both against this real data.

Requires the large recovered artifacts pulled from court23 into
`recovered/` (tracks.json, tracked_detections.json, reid_embeddings.json,
court_calibration.json) -- NOT committed to git (reid_embeddings.json alone
is ~1GB); pull them again from the preserved VM disk
(`/home/arnavchokshi/coldstart_20260706/fullgame_demo_20260728/out/pbvision_11min_20260713_demo_seed/`)
to rerun this script. Only the small evidence files
(PIPELINE_SUMMARY.json, player_selection.json, raw_pool_authority_summary.json,
run_stdout.log, run_stderr.log) are committed alongside this script.

Run from the repo root: `.venv/bin/python runs/lanes/fullgame_selection_fix_20260728/repro_full_scale_selection.py`
"""

import json
import sys
import time

sys.path.insert(0, ".")
from threed.racketsport.player_selection import select_players_payload
from threed.racketsport.schemas import validate_artifact_file

BASE = "runs/lanes/fullgame_selection_fix_20260728/recovered"

tracks_payload = json.load(open(f"{BASE}/tracks.json"))
raw_pool_payload = json.load(open(f"{BASE}/tracked_detections.json"))
calibration = validate_artifact_file("court_calibration", f"{BASE}/court_calibration.json")
embedding_payload = json.load(open(f"{BASE}/reid_embeddings.json"))

print("tracks players:", len(tracks_payload.get("players", [])))
print("raw pool frames:", len(raw_pool_payload.get("frames", [])))

t0 = time.time()
try:
    selected, report = select_players_payload(
        tracks_payload,
        raw_pool_payload=raw_pool_payload,
        embedding_payload=embedding_payload,
        calibration=calibration,
        enabled=True,
        embedding_bbox_scale=1.0,
        reid_provider_available=True,
        reid_provider_reason=None,
        auto_player_count=True,
    )
    print("SUCCEEDED in", time.time() - t0, "s")
except Exception as e:
    print("FAILED after", time.time() - t0, "s:", type(e).__name__, e)
    raise
