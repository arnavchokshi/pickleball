#!/usr/bin/env python3
"""Section 2: foot-slide/placement — baseline vs fused, from trackI_placefuse_20260716.
CPU-only; reads existing JSON artifacts. Baseline foot per frame = fused - rigid_correction."""
import json, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

LANE = "/Users/arnavchokshi/Desktop/pickleball/runs/lanes/trackI_placefuse_20260716"
OUT = "/Users/arnavchokshi/Desktop/visual_evidence_20260716/assets"
BLUE, GREEN, INKMUT = "#2a78d6", "#008300", "#6b7280"
CLIPS = ["burlington", "outdoor", "wolverine", "img1605"]

base = json.load(open(f"{LANE}/baseline_metrics.json"))
fused = json.load(open(f"{LANE}/fused_metrics.json"))

# ---------- summary grouped bars ----------
bmm = [base["clips"][c]["accepted_phase"]["max_slide_m"] * 1000 for c in CLIPS]
fmm = [fused["clips"][c]["accepted_phase"]["max_slide_m"] * 1000 for c in CLIPS]
fig, ax = plt.subplots(figsize=(8.6, 4.4))
x = range(len(CLIPS)); w = 0.34
for i, (b, f) in enumerate(zip(bmm, fmm)):
    ax.bar(i - w/2, b, w * 0.94, color=BLUE, zorder=3)
    ax.bar(i + w/2, f, w * 0.94, color=GREEN, zorder=3)
    ax.text(i - w/2, b + 0.8, f"{b:.1f}", ha="center", fontsize=10, color="#111")
    ax.text(i + w/2, f + 0.8, f"{f:.1f}", ha="center", fontsize=10, color="#111")
ax.set_xticks(list(x)); ax.set_xticklabels(CLIPS, fontsize=11)
ax.set_ylabel("max foot-slide in accepted plant phases (mm)", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e5e7eb", lw=0.7, zorder=0)
ax.legend(handles=[mpatches.Patch(color=BLUE, label="baseline skeleton"),
                   mpatches.Patch(color=GREEN, label="fused (placement-refined)")],
          frameon=False, fontsize=10, loc="upper left")
ax.set_title("Planted-foot slide, baseline vs fused — 4 real clips (lower is better)", fontsize=12)
fig.tight_layout(); fig.savefig(f"{OUT}/placement_slide_summary.png", dpi=150); plt.close(fig)
print("summary bars written", [round(v,1) for v in bmm], "->", [round(v,1) for v in fmm])

# ---------- per-clip court + worst-phase zoom ----------
def court(ax):
    ax.add_patch(mpatches.Rectangle((-3.048, -6.705), 6.096, 13.41, fill=False, ec="#9ca3af", lw=1.2))
    ax.axhline(0, color="#4b5563", lw=1.6)                      # net
    for y in (-2.134, 2.134):                                   # kitchen lines
        ax.plot([-3.048, 3.048], [y, y], color="#c4c9d0", lw=0.9)
    for y0, y1 in ((-6.705, -2.134), (2.134, 6.705)):           # centerlines
        ax.plot([0, 0], [y0, y1], color="#c4c9d0", lw=0.9)
    ax.set_aspect("equal")

sanity = []
for clip in CLIPS:
    traj = json.load(open(f"{LANE}/{clip}/placement_trajectory_refined.json"))
    phases = base["clips"][clip]["accepted_phase"]["per_phase"]
    worst = max(phases, key=lambda p: p["slide_m"])
    pid, foot, s, e = worst["player_id"], worst["foot"], worst["start_frame_index"], worst["end_frame_index"]
    # match fused phase for the same player+foot with max overlap
    fphases = [p for p in fused["clips"][clip]["accepted_phase"]["per_phase"]
               if int(p["player_id"]) == int(pid) and p["foot"] == foot and p["start_frame_index"] <= e and p["end_frame_index"] >= s]
    fslide = max((p["slide_m"] for p in fphases), default=None)

    player = next(p for p in traj["players"] if int(p["id"]) == int(pid))
    frames = [fr for fr in player["frames"] if s <= fr["frame_idx"] <= e]
    fus, basel = [], []
    for fr in frames:
        ref = fr["placement_trajectory_refinement"]
        fp = ref["refined_foot_positions"][foot]
        corr = ref["rigid_correction_xyz_m"]
        fus.append((fp[0], fp[1]))
        basel.append((fp[0] - corr[0], fp[1] - corr[1]))
    # sanity: recomputed baseline max pairwise-from-first displacement vs scored slide_m
    def spread(pts):
        return max(math.dist(a, b) for a in pts for b in pts) if len(pts) > 1 else 0.0
    sanity.append((clip, pid, foot, worst["slide_m"], spread(basel), fslide, spread(fus)))

    fig = plt.figure(figsize=(9.4, 6.2))
    axc = fig.add_axes((0.04, 0.07, 0.36, 0.86)); court(axc)
    for pl in traj["players"]:
        xs = [fr["transl_world"][0] for fr in pl["frames"]]
        ys = [fr["transl_world"][1] for fr in pl["frames"]]
        axc.plot(xs, ys, color="#d1d5db", lw=1.0, zorder=2)
    fx = [fr["placement_trajectory_refinement"]["refined_foot_positions"][foot][0] for fr in frames]
    fy = [fr["placement_trajectory_refinement"]["refined_foot_positions"][foot][1] for fr in frames]
    axc.plot(fx, fy, color=GREEN, lw=2.0, zorder=4)
    axc.scatter([fx[0]], [fy[0]], s=44, color=GREEN, zorder=5)
    axc.set_xlim(-4.0, 4.0); axc.set_ylim(-7.6, 7.6)
    axc.set_xticks([]); axc.set_yticks([])
    for sp in axc.spines.values(): sp.set_visible(False)
    axc.set_title(f"{clip}: top-down court\ngray = player roots, green dot = plant under zoom", fontsize=9.5)

    axz = fig.add_axes((0.50, 0.13, 0.46, 0.74))
    bx, by = [p[0]*1000 for p in basel], [p[1]*1000 for p in basel]
    gx, gy = [p[0]*1000 for p in fus], [p[1]*1000 for p in fus]
    cx, cy = sum(gx)/len(gx), sum(gy)/len(gy)
    bspread, gspread = spread(basel) * 1000, spread(fus) * 1000
    axz.plot([v-cx for v in bx], [v-cy for v in by], color=BLUE, lw=2.0, marker="o", ms=3.5,
             mec="white", mew=0.5, label=f"baseline foot: wanders {bspread:.1f} mm")
    axz.plot([v-cx for v in gx], [v-cy for v in gy], color=GREEN, lw=2.0, marker="o", ms=3.5,
             mec="white", mew=0.5, label=f"fused foot: wanders {gspread:.1f} mm")
    axz.set_aspect("equal")
    axz.set_xlabel("mm (court X, centered on plant)", fontsize=9)
    axz.set_ylabel("mm (court Y)", fontsize=9)
    axz.grid(color="#eceef1", lw=0.7)
    axz.spines[["top", "right"]].set_visible(False)
    axz.legend(frameon=False, fontsize=9.5, loc="upper right")
    axz.set_title(f"worst accepted plant phase (player {pid}, {foot} foot, frames {s}-{e}):\n"
                  f"the planted foot should NOT move", fontsize=10)
    fig.savefig(f"{OUT}/placement_{clip}_plant_zoom.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", clip)

print("\nSANITY (clip, player, foot, scored_baseline_m, recomputed_m, scored_fused_m, recomputed_fused_m):")
for row in sanity:
    print(" ", row)
