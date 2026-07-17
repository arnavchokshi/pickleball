#!/usr/bin/env python3
"""Section 3: ball 2D + arc recovery — baseline (1 fitted of 188) vs anchor-fused balanced
preset (85 fitted of 361, KILLED: 18 flight-sanity violations). Mechanism preview, not promoted.
CPU-only; reads locked artifacts only."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE_ART = "/Users/arnavchokshi/Desktop/pickleball/runs/lanes/ballarc_scale_guard_20260715/full_guard5_r4/ball_track_arc_solved.json"
FUSE_DIR = "/Users/arnavchokshi/Desktop/pickleball/runs/lanes/ballarc_anchorfusion_20260716"
TRACK = "/Users/arnavchokshi/Desktop/pickleball/runs/lanes/pbv11_headtohead_20260713/rerun_20260715/vm_pull_partial/pbvision_11min_20260713/ball_track.json"
OUT = "/Users/arnavchokshi/Desktop/visual_evidence_20260716/assets"
BLUE, GREEN, RED, GRAY = "#2a78d6", "#008300", "#e34948", "#d9dce1"

base = json.load(open(BASE_ART))["segments"]
fuse = json.load(open(f"{FUSE_DIR}/preset_balanced/ball_track_arc_solved.json"))["segments"]
sanity = json.load(open(f"{FUSE_DIR}/preset_balanced/ball_flight_sanity.json"))
metrics = json.load(open(f"{FUSE_DIR}/preset_balanced_metrics.json"))
track = json.load(open(TRACK))

bf = [s for s in base if s["status"].startswith("fit")]
ff = [s for s in fuse if s["status"].startswith("fit")]
fails = [s for s in sanity["segments"] if s["verdict"] == "fail"]
print(f"baseline fitted {len(bf)}/{len(base)}; fused fitted {len(ff)}/{len(fuse)}; sanity fails {len(fails)}/{len(sanity['segments'])}")
assert len(bf) == 1 and len(ff) == 85 and len(fails) == 18, "numbers drifted from scored artifacts"

fig, axes = plt.subplots(3, 1, figsize=(11.5, 6.4), sharex=True,
                         gridspec_kw={"height_ratios": [1.15, 0.7, 1.0], "hspace": 0.42})
axA, axB, axC = axes

# Row A: salvaged 2D ball track (visible frames) + rally shading
ts = [f["t"] for f in track["frames"] if f["visible"]]
ys = [f["xy"][1] for f in track["frames"] if f["visible"]]
for s0, s1 in metrics["rally_active_spans"]["spans"]:
    axA.axvspan(s0, s1, color="#f1f3f6", zorder=0)
axA.scatter(ts, ys, s=0.6, color="#6b7280", zorder=2)
axA.invert_yaxis()
axA.set_ylabel("ball y (px)", fontsize=8.5)
axA.set_title(f"salvaged 2D ball track (WASB): {sum(1 for f in track['frames'] if f['visible'])} visible frames of "
              f"{len(track['frames'])} — shaded = 52 rally-active spans", fontsize=9.5, loc="left")

def spans(ax, segs, color, y=0.5, h=0.68):
    for s in segs:
        ax.add_patch(mpatches.Rectangle((s["t0"], y - h/2), max(s["t1"] - s["t0"], 0.4), h,
                                        color=color, lw=0, zorder=3))

# Row B: baseline
spans(axB, [s for s in base if not s["status"].startswith("fit")], GRAY)
spans(axB, bf, BLUE)
axB.set_ylim(0, 1); axB.set_yticks([])
axB.set_title(f"BASELINE solver (no audio anchors): {len(bf)} fitted segment of {len(base)} — "
              f"gray = unfittable (budget exceeded)", fontsize=9.5, loc="left", color="#1f2937")

# Row C: anchor-fused balanced (killed)
spans(axC, [s for s in fuse if not s["status"].startswith("fit")], GRAY)
spans(axC, ff, GREEN)
for s in fails:
    axC.add_patch(mpatches.Rectangle((s["t_start"], 0.05), max(s["t_end"] - s["t_start"], 0.4), 0.9,
                                     fill=False, ec=RED, lw=1.3, hatch="///", zorder=4))
axC.set_ylim(0, 1); axC.set_yticks([])
axC.set_xlabel("clip time (s) — pbvision 11-min demo, 30 fps, 697 s", fontsize=9)
axC.set_title(f"ANCHOR-FUSED balanced preset: {len(ff)} fitted of {len(fuse)} — KILLED by flight-sanity rule "
              f"(red hatched = {len(fails)} physics-violating windows)", fontsize=9.5, loc="left", color="#b91c1c")

axC.legend(handles=[
    mpatches.Patch(color=BLUE, label="baseline fitted arc"),
    mpatches.Patch(color=GREEN, label="anchor-fused fitted arc"),
    mpatches.Patch(facecolor="white", edgecolor=RED, hatch="///", label="flight-sanity violation"),
    mpatches.Patch(color=GRAY, label="unfitted"),
], frameon=False, fontsize=8.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.42))

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, 700)

fig.suptitle("Ball arc recovery — segmentation coverage mechanism (PREVIEW EVIDENCE, preset killed, nothing promoted)",
             fontsize=11, y=0.99)
fig.savefig(f"{OUT}/ballarc_timeline.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# violation numbers panel (text figure)
summ = sanity["summary"]
viol_lines = [
    "Why the kill rule fired (balanced preset, 19 arcs evaluated, 0 passed, 18 failed):",
    "  - outside_court_volume: 18 arcs; worst leaves the court volume by 78.5 m (bound +4 m margin)",
    "  - speed_jump: 4 arcs; worst 30.8 m/s frame-to-frame vs 8.75 m/s limit",
    "  - horizontal_direction_reversal: 2 arcs; worst 164.6 deg heading change vs 120 deg limit",
    "  - vertical_multi_apex: 2 arcs; 3 apex sign changes vs 1 allowed",
    "",
    "What the mechanism DID show: audio-onset soft splits raised fitted arcs 1 -> 85 and",
    "in-rally fitted coverage 0.3% -> 29.7% on a 697 s clip - but the fits are not yet",
    "physically sane, so ALL presets were killed. Conservative: 53 fitted, 16 violations,",
    "also killed. Broad: crashed before scoring. VERIFIED=0. Nothing was promoted.",
]
fig2, ax2 = plt.subplots(figsize=(11.5, 2.6))
ax2.axis("off")
ax2.text(0.01, 0.95, "\n".join(viol_lines), va="top", ha="left", fontsize=10.5,
         family="monospace", color="#1f2937",
         bbox=dict(boxstyle="round,pad=0.6", fc="#fef2f2", ec=RED, lw=1.2))
fig2.savefig(f"{OUT}/ballarc_kill_reasons.png", dpi=150, bbox_inches="tight"); plt.close(fig2)
print("wrote ballarc_timeline.png + ballarc_kill_reasons.png")
print("summary block:", {k: summ[k] for k in sorted(summ)[:6]})
