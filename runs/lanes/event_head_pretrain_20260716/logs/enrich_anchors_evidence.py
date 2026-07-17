#!/usr/bin/env python3
"""Attach a per-candidate multi-signal EVIDENCE VECTOR to an event-head anchor file.

Owner directive 2026-07-16 (relayed via coordinator): contact anchors delivered to
Track A must carry a mix of evidence (audio, visual head, wrist/pose, ball-track),
not one scalar. This script ENRICHES an existing anchor JSON produced by
scripts/racketsport/build_event_head_anchor_candidates.py. It never creates or
removes candidates: the visual temporal head is the only emitter; audio and
ball-track signals ANNOTATE existing candidates (owner rule: no audio-only typed
contacts from this artifact).

Lane-local tool (runs/lanes/event_head_pretrain_20260716/) — not a registered
repo CLI; produces lane evidence only. VERIFIED=0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

AUDIO_WINDOW_S = 0.15  # co-location window vs pts_s (same convention both sides)
KINK_HALF_SPAN = 4     # frames each side for velocity estimation
KINK_MIN_CONF = 0.05


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _audio_evidence(onsets: list[dict], pts_s: float) -> dict:
    if not onsets:
        return {"available": False, "reason": "no audio onset artifact supplied"}
    nearest = min(onsets, key=lambda o: abs(float(o["corrected_time_s"]) - pts_s))
    dt = float(nearest["corrected_time_s"]) - pts_s
    within = abs(dt) <= AUDIO_WINDOW_S
    out = {
        "available": True,
        "nearest_onset_dt_s": round(dt, 6),
        "within_window": within,
        "window_s": AUDIO_WINDOW_S,
    }
    if within:
        out["onset_strength"] = nearest.get("onset_strength")
        feats = nearest.get("features") or {}
        out["pop_band_ratio"] = feats.get("pop_band_ratio")
        out["spectral_flux"] = feats.get("spectral_flux")
        out["onset_order"] = nearest.get("onset_order")
    return out


def _kink_evidence(frames: list[dict], frame_idx: int) -> dict:
    lo, hi = frame_idx - KINK_HALF_SPAN, frame_idx + KINK_HALF_SPAN
    if lo < 0 or hi >= len(frames):
        return {"available": False, "reason": "frame span outside track"}
    span = frames[lo : hi + 1]
    usable = [f for f in span if f.get("visible") and float(f.get("conf", 0.0)) >= KINK_MIN_CONF]
    if len(usable) < 5:
        return {
            "available": False,
            "reason": f"insufficient visible track points ({len(usable)}/9 usable)",
        }
    mid = frame_idx - lo
    before = [f for f in span[: mid + 1] if f.get("visible") and float(f.get("conf", 0.0)) >= KINK_MIN_CONF]
    after = [f for f in span[mid:] if f.get("visible") and float(f.get("conf", 0.0)) >= KINK_MIN_CONF]
    if len(before) < 2 or len(after) < 2:
        return {"available": False, "reason": "one-sided visibility around candidate"}
    vin = (
        before[-1]["xy"][0] - before[0]["xy"][0],
        before[-1]["xy"][1] - before[0]["xy"][1],
    )
    vout = (
        after[-1]["xy"][0] - after[0]["xy"][0],
        after[-1]["xy"][1] - after[0]["xy"][1],
    )
    speed_in = math.hypot(*vin)
    speed_out = math.hypot(*vout)
    if speed_in < 1e-9 or speed_out < 1e-9:
        return {"available": False, "reason": "degenerate (near-zero) velocity segment"}
    cosang = max(-1.0, min(1.0, (vin[0] * vout[0] + vin[1] * vout[1]) / (speed_in * speed_out)))
    return {
        "available": True,
        "direction_change_deg": round(math.degrees(math.acos(cosang)), 3),
        "speed_in_px_per_span": round(speed_in, 6),
        "speed_out_px_per_span": round(speed_out, 6),
        "usable_points": len(usable),
        "half_span_frames": KINK_HALF_SPAN,
        "min_conf": KINK_MIN_CONF,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--audio-onsets", default=None)
    ap.add_argument("--ball-track", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    anchors_path = Path(args.anchors)
    doc = json.loads(anchors_path.read_text())
    if doc.get("artifact_type") != "event_head_contact_anchor_candidates":
        raise SystemExit("input is not an event_head anchor artifact")

    onsets: list[dict] = []
    sources: dict = {}
    if args.audio_onsets:
        ap_path = Path(args.audio_onsets)
        audio_doc = json.loads(ap_path.read_text())
        onsets = audio_doc.get("onsets") or []
        sources["audio_onsets"] = {
            "path": str(ap_path),
            "md5": _md5(ap_path),
            "count": len(onsets),
            "review_only": True,
            "never_training": True,
            "not_gate_verified": bool(audio_doc.get("not_gate_verified", True)),
        }
    frames: list[dict] = []
    if args.ball_track:
        bt_path = Path(args.ball_track)
        track_doc = json.loads(bt_path.read_text())
        frames = track_doc.get("frames") or []
        sources["ball_track_2d"] = {
            "path": str(bt_path),
            "md5": _md5(bt_path),
            "frames": len(frames),
            "fps": track_doc.get("fps"),
            "review_only": True,
        }
    sources["wrist_pose"] = {
        "available": False,
        "reason": "no BODY/pose artifacts exist for this video (full-scale BODY pass never completed; Track I ledger: demo video out of scope for BODY)",
    }

    for ev in doc.get("events", []):
        ev["evidence"] = {
            "event_head_score": ev.get("score"),
            "audio": _audio_evidence(onsets, float(ev["pts_s"])),
            "ball_track_kink": _kink_evidence(frames, int(ev["frame_idx"])) if frames else {"available": False, "reason": "no ball track artifact supplied"},
            "wrist_swing_proximity": {"available": False, "reason": sources["wrist_pose"]["reason"]},
        }

    doc["evidence_enrichment"] = {
        "owner_directive": "2026-07-16 multi-signal contact evidence (neighboring-court audio bleed): anchors carry an evidence vector; the visual temporal head is the sole emitter; audio/ball-track annotate only",
        "audio_window_s": AUDIO_WINDOW_S,
        "sources": sources,
        "no_audio_only_candidates": True,
    }
    hl = doc.setdefault("honest_limits", [])
    hl.append("Evidence vector is annotation-only: candidates were emitted solely by the visual temporal head; audio-only events are intentionally NOT emitted (neighboring-court audio bleed).")
    hl.append("wrist_swing_proximity unavailable: no pose/BODY artifacts exist for this video; conditioning channels (track/wrist) are also NOT part of the trained model as committed (RGB-only per E2E-Spot reference recipe).")

    out_path = Path(args.out)
    out_path.write_text(json.dumps(doc, indent=2) + "\n")
    n = len(doc.get("events", []))
    aud = sum(1 for e in doc.get("events", []) if e["evidence"]["audio"].get("within_window"))
    kink = sum(1 for e in doc.get("events", []) if e["evidence"]["ball_track_kink"].get("available"))
    print(json.dumps({"events": n, "audio_within_window": aud, "kink_available": kink, "out": str(out_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
