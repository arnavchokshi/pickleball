#!/usr/bin/env python3
"""Section 1: people detection — YOLO26m baseline (arm0a) vs RF-DETR-L (arm1) overlays.
CPU-only. Uses scored final tracks for the coverage story (honest: the burlington gap is a
tracking/pooling effect downstream of the detector) and raw RF-DETR dets for the FP frame."""
import json, subprocess
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VP = "/Users/arnavchokshi/Desktop/pickleball/runs/lanes/trk_detbench_20260716/vm_pull/detbench_out"
EC = "/Users/arnavchokshi/Desktop/pickleball/eval_clips/ball"
OUT = "/Users/arnavchokshi/Desktop/visual_evidence_20260716/assets"
BURL = "burlington_gold_0300_low_steep_corner"
WOLV = "wolverine_mixed_0200_mid_steep_corner"
BLUE, GREEN, RED = (214, 120, 42), (0, 131, 0), (72, 73, 227)  # BGR for cv2
BLUE_H, GREEN_H, RED_H = "#2a78d6", "#008300", "#e34948"

def load_tracks(clip, arm):
    d = json.load(open(f"{VP}/scored/{clip}/{arm}/tracks.json"))
    return d["players"]

def boxes_at(players, k):
    out = []
    for p in players:
        fr = p["frames"][k] if k < len(p["frames"]) else None
        if fr and fr.get("bbox"):
            out.append(fr["bbox"])
    return out

def grab(clip, k):
    cap = cv2.VideoCapture(f"{EC}/{clip}/source.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, k)
    ok, img = cap.read(); cap.release()
    assert ok, f"frame {k} decode failed"
    return img

def draw(img, boxes, color, label):
    for b in boxes:
        x0, y0, x1, y1 = [int(round(v)) for v in b]
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 5)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 3)
    cv2.rectangle(img, (0, 0), (760, 54), (24, 24, 24), -1)
    cv2.putText(img, label, (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return img

# ---------- burlington side-by-side frame in the baseline gap ----------
b0 = load_tracks(BURL, "arm0a_repro")
b1 = load_tracks(BURL, "arm1_rfdetr_l")
K = 510
n0, n1 = boxes_at(b0, K), boxes_at(b1, K)
left = draw(grab(BURL, K), n0, BLUE, f"YOLO26m baseline track: {len(n0)}/4 players  (frame {K})")
right = draw(grab(BURL, K), n1, GREEN, f"RF-DETR-L track: {len(n1)}/4 players  (frame {K})")
side = cv2.hconcat([left, right])
cv2.imwrite(f"{OUT}/detect_burlington_side_by_side.png", side, [cv2.IMWRITE_PNG_COMPRESSION, 6])
print(f"burlington frame {K}: baseline {len(n0)}/4 vs rfdetr {len(n1)}/4")

# ---------- burlington coverage timeline ----------
cov0 = [len(boxes_at(b0, k)) for k in range(600)]
cov1 = [len(boxes_at(b1, k)) for k in range(600)]
fig, ax = plt.subplots(figsize=(9.6, 2.9))
ax.step(range(600), cov0, where="post", color=BLUE_H, lw=1.8, label="YOLO26m baseline (cov4 0.712)")
ax.step(range(600), cov1, where="post", color=GREEN_H, lw=1.8, label="RF-DETR-L (cov4 0.997)")
ax.axvspan(427, 597, color="#e34948", alpha=0.08, zorder=0)
ax.text(512, 1.45, "baseline loses a player\nframes 427-597", ha="center", fontsize=8.5, color="#b91c1c")
ax.set_ylim(0.5, 4.5); ax.set_yticks([1, 2, 3, 4])
ax.set_xlabel("frame (60 fps, 10 s clip)", fontsize=9)
ax.set_ylabel("players tracked", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#eceef1", lw=0.7)
ax.legend(frameon=False, fontsize=9, loc="lower left")
ax.set_title("burlington: players held per frame — final scored tracks", fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/detect_burlington_coverage_timeline.png", dpi=150); plt.close(fig)
lt4_0 = sum(1 for v in cov0 if v == 4); lt4_1 = sum(1 for v in cov1 if v == 4)
print(f"timeline: baseline 4/4 on {lt4_0}/600 frames; rfdetr on {lt4_1}/600 (scorer: 427 vs 598)")

# ---------- wolverine honest FP frame (raw RF-DETR dets) ----------
raw = json.load(open(f"{VP}/raw/arm1_rfdetr_l/{WOLV}.json"))
KW = 24
dets = [d for d in raw["frames"][KW]["detections"] if d["conf"] >= 0.7]
# player identity comes from the scored final tracks (the artifact's own 4-player set),
# NOT a size heuristic: raw dets that IoU-match a track box are players, rest are FPs.
wtracks = load_tracks(WOLV, "arm1_rfdetr_l")
tboxes = boxes_at(wtracks, KW)
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0
matched = [t for t in tboxes if any(iou(d["bbox"], t) > 0.5 for d in dets)]
drifted = [t for t in tboxes if t not in matched]
extras = [d for d in dets if not any(iou(d["bbox"], t) > 0.5 for t in tboxes)]
assert len(tboxes) == 4, f"expected 4 track boxes, got {len(tboxes)}"
img = grab(WOLV, KW)
img = draw(img, matched, GREEN,
           f"RF-DETR-L weaknesses, frame {KW}: {len(extras)} non-player FPs (red) + drifted track (amber)")
AMBER = (0, 160, 237)
for t in drifted:
    x0, y0, x1, y1 = [int(round(v)) for v in t]
    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 5)
    cv2.rectangle(img, (x0, y0), (x1, y1), AMBER, 3)
    cv2.putText(img, "track drifted - no player here", (x0, min(1060, y1 + 30)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, AMBER, 2, cv2.LINE_AA)
for d in extras:
    x0, y0, x1, y1 = [int(round(v)) for v in d["bbox"]]
    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 5)
    cv2.rectangle(img, (x0, y0), (x1, y1), RED, 3)
    cv2.putText(img, "non-player FP", (x0, max(24, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2, cv2.LINE_AA)
cv2.imwrite(f"{OUT}/detect_wolverine_fp_frame.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
print(f"wolverine frame {KW}: {len(matched)} matched players + {len(drifted)} drifted + {len(extras)} non-player FPs")

# ---------- short side-by-side clip (burlington 380-600, 2x downsample) ----------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
tmp = f"{OUT}/_tmp_detect_clip.mp4"
vw = cv2.VideoWriter(tmp, fourcc, 30.0, (1920, 540))
cap = cv2.VideoCapture(f"{EC}/{BURL}/source.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 380)
for k in range(380, 600):
    ok, frame = cap.read()
    if not ok: break
    L = draw(frame.copy(), boxes_at(b0, k), BLUE, f"YOLO26m baseline: {len(boxes_at(b0,k))}/4")
    R = draw(frame.copy(), boxes_at(b1, k), GREEN, f"RF-DETR-L: {len(boxes_at(b1,k))}/4")
    sm = cv2.resize(cv2.hconcat([L, R]), (1920, 540))
    vw.write(sm)
cap.release(); vw.release()
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-crf", "23", f"{OUT}/detect_burlington_clip.mp4"], check=True)
import os; os.remove(tmp)
print("clip written: detect_burlington_clip.mp4 (frames 380-599 side by side)")
