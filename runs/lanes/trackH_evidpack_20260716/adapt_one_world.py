#!/usr/bin/env python3
"""Track H: adapt Track K's one_world_v1 (wolverine) into the viewer's virtual_world schema
so the owner can click through the FUSED world in the real replay viewer.

HONESTY RULES (enforced, not decorative):
 - Player positions/joints come ONLY from one_world_v1 fused output (placement_tier=placement_fused).
 - Ball comes ONLY from one_world_v1 (estimate_tier arc_measured|physics_predicted); tier is carried
   into the viewer's provenance so banding is truthful.
 - Frames the fusion did NOT produce stay MISSING (no carry-forward, no interpolation).
 - Paddles are emitted EMPTY: one_world reports resolved_fraction=0.0 (1102/1102 legacy wrist proxy,
   never resolved). Rendering a paddle as solved would be a lie, so the viewer shows it absent.
 - Court geometry + joint_names are copied from the container world (geometry, not estimates).
"""
import json

LANE = "/Users/arnavchokshi/Desktop/pickleball/runs/lanes/oneworld_impl_20260716/wolverine"
OUT = "/Users/arnavchokshi/Desktop/pickleball/runs/lanes/trackH_evidpack_20260716"
CONTAINER = f"{LANE}/confidence_gated_world.json"

ow = json.load(open(f"{LANE}/one_world_v1.json"))
cw = json.load(open(CONTAINER))
assert ow["artifact_type"] == "racketsport_one_world_v1" and ow["world_frame"] == "court_Z0"
assert ow["preview_only"] and ow["render_only"] and ow["VERIFIED"] == 0

FUSED_BAND = {
    "stage": ow["trust_band"].get("stage", "CAL"),
    "badge": "preview",
    "reason": ("FUSED one_world_v1 (Track K): confidence-weighted joint fusion, render-only preview. "
               "Not a promoted result; VERIFIED=0."),
    "gate_id": ow["trust_band"].get("gate_id"),
    "gate_status": ow["trust_band"].get("gate_status"),
    "evidence_path": ow["trust_band"].get("evidence_path"),
}

def band_for(tier, extra):
    return {"stage": FUSED_BAND["stage"], "badge": "preview",
            "reason": f"{extra} (one_world_v1 tier: {tier})",
            "gate_id": FUSED_BAND["gate_id"], "gate_status": FUSED_BAND["gate_status"],
            "evidence_path": FUSED_BAND["evidence_path"]}

# ---------- players: fused only ----------
by_pid = {}
for fr in ow["frames"]:
    for p in fr["players"]:
        by_pid.setdefault(p["player_id"], []).append((fr["t"], p))

players = []
for pid, rows in sorted(by_pid.items()):
    frames = []
    for t, p in rows:
        joints = p.get("joints_world") or []
        root = p.get("root_world")
        frames.append({
            "t": t,
            "mesh_ref": None,
            "track_world_xy": [root[0], root[1]] if root else None,
            "track_conf": p.get("confidence"),
            "bbox": None,
            "transl_world": root,
            "joints_world": joints,
            "joint_conf": p.get("joint_conf") or [],
            "mesh_vertices_world": [],
            "joint_count": len(joints),
            "mesh_vertex_count": 0,
            "floor_world_xyz": [root[0], root[1], 0.0] if root else None,
            "floor_source": "one_world_v1_fused_placement",
            "confidence_provenance": p.get("confidence_provenance"),
            "trust_band": band_for(p.get("placement_tier"), "Player placement fused from tracker+body+court evidence"),
        })
    frames.sort(key=lambda f: f["t"])
    players.append({
        "id": int(pid), "side": None, "role": None, "representation": "joints",
        "frames": frames, "trust_band": FUSED_BAND,
        "coverage_fraction": len(frames) / len(ow["frames"]),
    })

# ---------- ball: fused only, tier carried ----------
ball_frames = []
for fr in ow["frames"]:
    b = fr.get("ball") or {}
    xyz = b.get("world_xyz")
    tier = b.get("estimate_tier")
    xy = b.get("xy_observed_px")
    ball_frames.append({
        "t": fr["t"],
        "xy": [float(xy[0]), float(xy[1])] if xy else [0.0, 0.0],
        "conf": float(b.get("confidence") or 0.0),
        "visible": bool(xyz is not None),
        "world_xyz": xyz,
        "approx": bool(b.get("approx") or tier != "arc_measured"),
        "confidence_provenance": b.get("confidence_provenance"),
        "trust_band": band_for(tier, "Ball position from fused arc/physics chain"),
    })
tiers = {}
for fr in ow["frames"]:
    t = (fr.get("ball") or {}).get("estimate_tier")
    tiers[t] = tiers.get(t, 0) + 1

world = {
    "schema_version": 1,
    "artifact_type": "racketsport_virtual_world",
    "world_frame": "court_Z0",
    "fps": ow["fps"],
    "court": cw["court"],
    "joint_names": cw.get("joint_names"),
    "players": players,
    "ball": {
        "source": "one_world_v1 fused (arc_measured/physics_predicted tiers)",
        "frames": ball_frames,
        "trust_band": band_for("/".join(f"{k}:{v}" for k, v in tiers.items()), "Fused ball continuity chain"),
    },
    # HONEST ABSENCE: fusion never resolved a paddle pose (resolved_fraction 0.0)
    "paddles": [],
    "summary": {
        **cw["summary"],
        "player_count": len(players),
        "mesh_player_count": 0,
        "mesh_player_frame_count": 0,
        "joint_player_frame_count": sum(len(p["frames"]) for p in players),
        "track_only_player_frame_count": 0,
        "notes": [
            "SOURCE: one_world_v1 (Track K fused artifact), adapted by Track H for viewer rendering.",
            "Players + ball are fused estimates. Paddles intentionally absent: one_world reports "
            "resolved_fraction=0.0 (every paddle pose is an unresolved legacy wrist proxy).",
            "Preview band, render-only, VERIFIED=0. Not for detection metrics, not for training.",
        ],
    },
}
if world["joint_names"] is None:
    del world["joint_names"]

out_world = f"{OUT}/one_world_v1_viewer.json"
with open(out_world, "w") as f:
    json.dump(world, f)

src = json.load(open(f"{LANE}/replay_viewer_manifest.json"))
manifest = {
    "schema_version": 1,
    "artifact_type": "racketsport_replay_viewer_manifest",
    "clip": src.get("clip", "wolverine_mixed_0200_mid_steep_corner"),
    "video_url": "/@fs//Users/arnavchokshi/Desktop/pickleball/eval_clips/ball/wolverine_mixed_0200_mid_steep_corner/source.mp4",
    "virtual_world_url": f"/@fs/{out_world}",
    "body_mesh_url": None, "body_mesh_index_url": None,
    "ball_arc_render_url": None, "ball_arc_solved_url": None,
    "ball_bounce_candidates_url": None, "auto_bounce_candidates_url": None,
    "ball_flight_sanity_url": None, "ball_inflections_url": None,
    "contact_windows_url": None, "physics_refinement_url": None,
    "rally_spans_url": None, "replay_scene_url": None, "reviewed_bounces_url": None,
    "coaching_card_facts_url": None,
    "mesh_status": "absent",
    "label_overlays": [], "annotation_sources": [],
    "notes": [
        "FUSED WORLD (Track K one_world_v1) rendered by Track H. Preview band, render-only, VERIFIED=0.",
        "Marker/arc sources deliberately NULL: one_world emits its own typed events; baseline arc/contact "
        "artifacts are NOT mixed in, so nothing shown here is from a different pipeline.",
        "Paddles absent by design: fusion resolved 0 of 1102 paddle poses (legacy wrist proxy).",
    ],
}
out_manifest = f"{OUT}/one_world_viewer_manifest.json"
with open(out_manifest, "w") as f:
    json.dump(manifest, f, indent=1)

print(f"players: {[(p['id'], len(p['frames'])) for p in players]}")
print(f"ball frames: {len(ball_frames)} | visible(world_xyz): {sum(1 for b in ball_frames if b['visible'])} | tiers: {tiers}")
print(f"paddles: [] (honest absence; resolved_fraction={ow['summary']['paddle_resolution']['resolved_fraction']})")
print(f"events in source: {len(ow['events'])} (contacts {len(ow['contacts'])}, bounces {len(ow['bounces'])})")
print("world  ->", out_world)
print("manifest ->", out_manifest)
