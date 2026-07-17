#!/usr/bin/env python3
"""Track K manager probe: is the contact co-location anchor class blocked by GEOMETRY or by EVENT TIMING?

For each declared contact, search a +/-15 frame window for the closest approach of the 3D ball
to ANY player's wrist (BODY_17 idx 9/10). If a near-approach exists at dt != 0, the anchor class
is real and the events are mistimed. Read-only; no artifacts mutated. VERIFIED=0, diagnostic only.
"""
import json, sys
from pathlib import Path

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else
           "runs/lanes/oneworld_impl_20260716/wolverine")
FPS = 30.0
WRIST = (9, 10)
WIN = 15

arc = json.load(open(RUN / "ball_track_arc_solved.json"))
body = json.load(open(RUN / "smpl_motion.json"))
events = json.load(open(RUN / "contact_windows.json"))

ball = {}
for f in arc.get("frames", []):
    if f.get("world_xyz"):
        ball[round(f["t"] * FPS)] = (f["world_xyz"], f.get("conf", 0.0))

wrists = {}  # frame -> list[(pid, widx, xyz, conf)]
for p in body.get("players", []):
    pid = p["id"]
    for fr in p.get("frames", []):
        k = fr.get("frame_idx")
        if k is None:
            k = round(fr["t"] * FPS)
        jw = fr.get("joints_world") or []
        jc = fr.get("joint_conf") or []
        for wi in WRIST:
            if wi < len(jw) and jw[wi]:
                wrists.setdefault(k, []).append(
                    (pid, wi, jw[wi], jc[wi] if wi < len(jc) else 0.0))


def d3(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


rows = []
for i, e in enumerate(events.get("events", [])):
    if e.get("type") != "contact":
        continue
    k0 = e.get("frame")
    if k0 is None:
        k0 = round(e["t"] * FPS)
    best = None          # (dist, dt, pid, widx)
    best_at_declared = None
    for dt in range(-WIN, WIN + 1):
        k = k0 + dt
        if k not in ball or k not in wrists:
            continue
        bxyz = ball[k][0]
        for (pid, wi, wxyz, wconf) in wrists[k]:
            dist = d3(bxyz, wxyz)
            if best is None or dist < best[0]:
                best = (dist, dt, pid, wi, wconf)
            if dt == 0 and (best_at_declared is None or dist < best_at_declared[0]):
                best_at_declared = (dist, pid, wi)
    if best is None:
        continue
    rows.append({
        "event_index": i,
        "declared_frame": k0,
        "declared_player_id": e.get("player_id"),
        "min_dist_m": round(best[0], 3),
        "at_dt_frames": best[1],
        "at_dt_seconds": round(best[1] / FPS, 3),
        "nearest_player_id": best[2],
        "nearest_wrist_idx": best[3],
        "dist_at_declared_frame_m": round(best_at_declared[0], 3) if best_at_declared else None,
        "declared_player_matches_nearest": (e.get("player_id") == best[2]),
    })

if not rows:
    print(json.dumps({"error": "no rows"}))
    sys.exit(2)

dists = sorted(r["min_dist_m"] for r in rows)
n = len(dists)
med = dists[n // 2]
p90 = dists[min(n - 1, int(round(0.9 * n)) - 1 if n > 1 else 0)]
declared = sorted(r["dist_at_declared_frame_m"] for r in rows
                  if r["dist_at_declared_frame_m"] is not None)
dmed = declared[len(declared) // 2] if declared else None

out = {
    "artifact_type": "trackK_anchor_window_probe",
    "purpose": "geometry-vs-timing diagnosis for the contact co-location anchor class",
    "verified": 0,
    "preview_band": True,
    "raw_inputs_mutated": False,
    "run_dir": str(RUN),
    "window_frames": WIN,
    "contact_count": n,
    "windowed_min_distance_m": {"median": med, "p90": p90,
                                "min": dists[0], "max": dists[-1]},
    "declared_frame_distance_m": {"median": dmed},
    "anchor_yield_at_gate": {
        "within_0.30m": sum(1 for d in dists if d <= 0.30),
        "within_0.50m": sum(1 for d in dists if d <= 0.50),
        "within_1.20m": sum(1 for d in dists if d <= 1.20),
    },
    "timing_offsets_frames": sorted(r["at_dt_frames"] for r in rows),
    "declared_player_matches_nearest_count":
        sum(1 for r in rows if r["declared_player_matches_nearest"]),
    "rows": rows,
}
print(json.dumps(out, indent=2, sort_keys=True))
