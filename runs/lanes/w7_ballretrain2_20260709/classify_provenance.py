#!/usr/bin/env python3
"""Provenance-split classifier for a ball_loso_validation.py report.

Per manager ruling 2026-07-09 (w7_ballretrain2_20260709): clips whose per-clip
median_error_px < 1.0 for a WASB-lineage candidate are "confirmed-prelabel" rows
(owner's documented prelabel-confirm workflow: model prelabel confirmed when
right, corrected when wrong -- SESSION_NOTES_20260709.md "461 prelabels
confirmed <=2px; 108 corrected, median move 985px"). This script reproduces the
classifier used to find that pattern (raw byte-level coordinate match, ~0.003-
0.006px, not achievable by independent inference) and reports pooled metrics
for: full block / clean subset / confirmed-prelabel subset, labeled
"mixed-provenance, non-comparable absolutes, ordering-only" per the ruling.

Usage: python3 classify_provenance.py <loso_report.json> <candidate_name>
"""
import json
import sys


def pooled(per_source, clip_list):
    tp = fp = fn = 0
    rows = 0
    for clip in clip_list:
        m = per_source[clip]
        tp += m["f1_true_positive_count"]
        fp += m["f1_false_positive_count"]
        fn += m["f1_false_negative_count"]
        rows += m.get("visible_label_count", 0) + m.get("hidden_label_count", 0)
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else None
    return {"tp": tp, "fp": fp, "fn": fn, "rows": rows, "pooled_f1": f1}


def main() -> int:
    report_path, candidate = sys.argv[1], sys.argv[2]
    d = json.load(open(report_path))
    c = d["candidates"][candidate]
    per = c["per_source_metrics"]
    contaminated = []
    clean = []
    for clip, m in per.items():
        me = m.get("median_error_px")
        if me is not None and me < 1.0:
            contaminated.append(clip)
        else:
            clean.append(clip)
    out = {
        "artifact_type": "racketsport_ball_provenance_split",
        "classifier": "median_error_px < 1.0px on a fresh independent-inference run",
        "provenance_label": "mixed-provenance, non-comparable absolutes, ordering-only",
        "candidate": candidate,
        "report_path": report_path,
        "confirmed_prelabel_clips": sorted(contaminated),
        "confirmed_prelabel_clip_count": len(contaminated),
        "clean_clips": sorted(clean),
        "clean_clip_count": len(clean),
        "pooled_full_block": pooled(per, list(per.keys())),
        "pooled_clean_subset": pooled(per, clean),
        "pooled_confirmed_prelabel_subset": pooled(per, contaminated),
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
