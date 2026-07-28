#!/usr/bin/env python3
"""Scratch evidence-extraction helper for the alwayson_fresh_wave_20260728 lane.
Not part of the pipeline; reads PIPELINE_SUMMARY.json + sidecars for a clip dir
and prints a compact JSON of the fields needed for REPORT.md. VERIFIED=0 throughout;
this script performs no computation of its own beyond field extraction.
"""
import json
import sys
from pathlib import Path


def find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            yield obj[key]
        for v in obj.values():
            yield from find_key(v, key)
    elif isinstance(obj, list):
        for v in obj:
            yield from find_key(v, key)


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def main(clip_dir):
    clip_dir = Path(clip_dir)
    out = {"clip_dir": str(clip_dir)}
    summ = load(clip_dir / "PIPELINE_SUMMARY.json")
    if summ is None:
        out["error"] = "no PIPELINE_SUMMARY.json"
        print(json.dumps(out, indent=1))
        return
    out["status"] = summ.get("status")
    out["wall_seconds"] = summ.get("wall_seconds")
    out["stages"] = {s["stage"]: {"status": s["status"], "wall_seconds": s.get("wall_seconds")} for s in summ.get("stages", [])}
    out["degraded_reasons"] = summ.get("degraded_reasons")
    out["missing_capabilities"] = summ.get("missing_capabilities")

    bspt = load(clip_dir / "body_stage_phase_timing.json")
    if bspt:
        out["body_phase_timing"] = {
            k: bspt.get(k)
            for k in (
                "model_load_s",
                "compile_warmup_s",
                "inference_s",
                "attributed_s",
                "ms_per_person_steady",
                "person_frame_count",
            )
        }

    bgq = load(clip_dir / "body_grounding_quality.json")
    if bgq:
        slides = list(find_key(bgq, "max_foot_lock_slide_m"))
        out["max_foot_lock_slide_m"] = slides[0] if slides else None
        samples = list(find_key(bgq, "body_samples"))
        out["body_samples"] = samples[0] if samples else None

    rbdt = load(clip_dir / "remote_body_dispatch_timing.json")
    if rbdt:
        out["remote_body_dispatch_timing"] = {
            "remote_host": rbdt.get("remote_host"),
            "phases": rbdt.get("phases"),
            "status": rbdt.get("status"),
        }

    bfcg = load(clip_dir / "body_full_clip_gate.json")
    if bfcg:
        out["body_full_clip_gate_status"] = bfcg if isinstance(bfcg, str) else bfcg.get("status", bfcg)

    placement_refine = (clip_dir / "placement_refined.json").is_file()
    placement_traj = (clip_dir / "placement_trajectory_refined.json").is_file()
    out["placement_refined_artifact_present"] = placement_refine
    out["placement_trajectory_refined_artifact_present"] = placement_traj

    sel = load(clip_dir / "selection_report.json")
    if sel:
        out["players_retained"] = (sel.get("output_counts") or {}).get("players")
        out["selection_status"] = sel.get("status")

    czd = load(clip_dir / "court_zones.json")
    # NVZ/kitchen decision counts: search placement_refined.json / placement.json for court_contact_state histogram
    for pfile in ("placement_refined.json", "placement.json"):
        pd = load(clip_dir / pfile)
        if pd is None:
            continue
        states = list(find_key(pd, "court_contact_state"))
        if states:
            hist = {}
            for s in states:
                hist[s] = hist.get(s, 0) + 1
            out[f"nvz_kitchen_decision_hist_{pfile}"] = hist
            break

    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1])
