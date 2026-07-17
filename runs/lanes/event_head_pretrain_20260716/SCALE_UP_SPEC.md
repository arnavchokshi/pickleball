# Lane spec (READY TO DISPATCH, next session) — event_head_corpus_20260717: full-corpus event-head pretrain

**Status:** SPEC ONLY — not dispatched. Authored by the Track G2 manager 2026-07-17 from the
measured facts of `event_head_pretrain_20260716` (T4 run, best_val_f1 0.3631 @±2f). VERIFIED=0
binding; this lane produces a BASELINE checkpoint, never a promotion.

**Why this lane exists:** the 2026-07-16 pretrain was *data-starved, not architecture-limited*.
The measurements below locate the starvation precisely and the levers are ordered by yield/effort.
The prior run's late collapse (val F1 0.363 @step1976 → 0.009 @step3956) is the classic signature
of overfit-then-diverge on a starved set — a DATA problem with a known fix.

---

## 1. Measured baseline (all numbers from the 07-16 run; reproduce before trusting)

| fact | value | source |
|---|---|---|
| label reach | **2.4% — 1,793 of 74,546 events** actually reachable by training windows | ops-lane diagnostic; denominator = 33,791 jhong93 + 36,484 ShuttleSet + 4,271 OpenTT |
| media coverage | **18.1% — 6/28 jhong93 videos + 2/12 OpenTT games staged** | `jhong93_spot/manifest.json` (6 shortest LIVE videos, ≤360p, disk-controlled); manager-verified: 643 of 3,561 manifest rows have `media_present` |
| window yield | **1 window per row** — `manifest_windows()` centres ONE window on `row["events"][0]` | `threed/racketsport/event_head/datasets.py:328-341`; manager-verified: train_windows 226 == media-present train rows 226 exactly |
| split imbalance | **226 train / 282 val** (val > train) | train_manifest.json |
| throughput | **0.619 steps/s realized** (batch 4 × window 64 @224), **~20% GPU util — decode-bound** | ops-lane VM measurement |
| events stranded | 2,010 labeled events sit inside the 226 media-present train rows; only ~1,793 land inside their 64-frame windows | manager manifest analysis (mean 8.9 events/row, median 5.5, max 404) |

**Read:** the head sees a rounding error of the corpus. Three independent multipliers are being
left on the table, and they compose.

---

## 2. The three levers, in dispatch order (do them in this order — each unblocks the next)

### Lever 1 — stage the remaining videos (×~5 rows) — BIGGEST, do first
- 22 of 28 jhong93 videos unstaged; 10 of 12 OpenTT games unstaged.
- Effect: media-present train rows **226 → ~1,301** (manifest knows clip lengths for 1,301 train
  rows; only 226 have the file).
- **Acquisition:** all 28 jhong93 sources probed LIVE 2026-07-13 via yt-dlp (`runs/lanes/
  eventdata_acquire_20260713/jhong93_probe.tsv`) — **re-probe first, 4+ days stale.**
- **Do it VM-side.** Mac disk was ~99% full at scaffold time; ~3-4GB of new video at ≤360p does not
  belong there.
- **Fetch h264 directly** (`yt-dlp -S vcodec:h264` or equivalent): the 07-16 run lost ~40 min to an
  AV1 wall (VM `cv2`/bundled ffmpeg cannot decode AV1; 5 of 6 pilots were AV1 → Mac-side transcode →
  re-ship). Fetching h264 at the source removes that entire class of failure. **Verify decodability
  on the VM immediately after fetch** (`cv2` open + decode 10 frames) before training — that check
  cost 30 seconds and would have saved 40 minutes.
- License posture unchanged: broadcast pixels = RD_ONLY, never redistributed, NS-07.3 review before
  any commercial use.

### Lever 2 — multi-window-per-row (×~3-12 windows/row)
- `manifest_windows()` takes only `row["events"][0]`. A sliding extractor (stride ≈ window/2 = 32)
  over each row's full `num_frames` reaches the other ~8 events/row.
- Manager projection @stride 32 over full clips: **226 → ~15,317 train windows (~68×)** once Lever 1
  also lands (mean clip 403 frames → ~11.6 windows/clip; conservative event-reachability read gives
  ~×3 — either way it is multiples, not percent).
- Keep source-disjoint splits and the loss-masked union exactly as-is. **Fix the split imbalance
  while here** (226 train / 282 val is backwards; target ≈ 70/15/15 by parent video).
- This is a `datasets.py` change with a determinism test — the scaffold's byte-identical-manifest
  gate must stay green.

### Lever 3 — dataloader workers BEFORE any bigger batch
- 20% GPU util means the T4 was starved by on-the-fly decode, not compute. **Do not raise batch size
  first** — it will not help a decode-bound loop.
- Add `DataLoader(num_workers=N, prefetch_factor=…)` (or a decode-ahead ring buffer) over the
  existing on-the-fly decode contract (no frame cache — the disk rule stands). Expect util 20% →
  60-80%, i.e. **~3-4× throughput at identical cost**.
- Only after util is high does a batch-size sweep mean anything.

---

## 3. Cost / GPU-hour estimate (full corpus, post-Levers 1+2, with Lever 3 applied)

Basis: ~15,317 train windows; batch 4 → 3,829 steps/epoch.

| scenario | steps/s | h/epoch | 15 epochs | spot $/hr | est. cost |
|---|---|---|---|---|---|
| T4, decode-bound (as-is) | 0.619 | 1.72 | ~26 h | $0.2-0.4 | $5-10 (over target wall) |
| T4 + workers (Lever 3) | ~2.0-2.5 | ~0.45 | **~7-9 h** | $0.2-0.4 | **$1.4-3.6** |
| L4 + workers | ~3-4 | ~0.3 | ~4-5 h | $0.3-0.7 | $1.2-3.5 |
| A100-40 + workers | ~6-8 | ~0.15 | **~2-3 h** | $1.1-1.5 | **$2.2-4.5** |

Plus one-time: video acquisition + staging ~1-1.5 h VM wall (22+10 videos, ≤360p h264, VM-side
fetch). **Recommended: A100-40 or L4 with workers, wall cap 5h, $10 cap → comfortable.**
Note the 07-16 stockout reality: L4 was exhausted across 7 zones; T4 succeeded. Ladder accordingly
(T4 is a legitimate first rung, not a fallback).

---

## 4. Mandatory fixes carried from the 07-16 run (cheap, high-value)

1. **`eval_event_head.py` window mismatch (REAL DEFECT, diagnosed 2026-07-17).** The CLI hardcodes
   `window_frames=15` (lines 68-69) while training used 64. Manager's controlled re-eval on the SAME
   checkpoint and SAME 16 clips:
   | window | preds @thr 0.5 | TPs | max class prob |
   |---|---|---|---|
   | 15f (committed CLI) | 2 | **0** | 0.556 |
   | 64f (matched) | 9 | **9 (0 FP)** | 0.937 |
   The committed CLI's "0 TP" was **a measurement artifact, not a model verdict**. FIX: make the eval
   window a parameter defaulting to the checkpoint's `train_manifest.config.window_frames`, and
   assert it matches at load. Evidence: `eval/matched_window64_eval.json` +
   `eval/control_window15_eval.json`; harness `logs/matched_window_eval.py`.
2. **Public eval breadth:** 3 hardcoded clips is not an eval set. Score ≥50 clips across both
   sources with a threshold sweep (the 07-16 harness already sweeps 0.5→0.05 × tol 1/2/5).
3. **Boot-armed rails** (booked ops lesson): arm `shutdown -P +N` from the VM's own startup script,
   never via post-RUNNING ssh — on fresh DLVM images the first-boot driver install races ssh and the
   arm fails closed (cost the 07-16 lane one fail-closed DELETE cycle).
4. **AppleDouble hygiene:** tar from macOS injects `._*` files that broke the ShuttleSet CSV glob
   (UnicodeDecodeError). Use `COPYFILE_DISABLE=1 tar …` or strip on arrival.
5. **`git rev-parse HEAD` is unguarded** in `train_event_head.py` provenance capture — it crashes on
   a VM mirror shipped without `.git`. Either guard it or `git init` the mirror (the 07-16 lane did
   the latter and recorded the true Mac HEAD in the report).

---

## 5. Acceptance for the next lane

- Builder totals reconcile to the Mac manifest exactly (the 07-16 gate held: 33,791/4,271/36,484).
- train_windows ≥ 10,000 with a documented split rebalance; val/test source-disjoint.
- GPU util ≥60% sustained (report it — Lever 3's whole point).
- Matched-window public eval over ≥50 clips with a threshold sweep + honest precision/recall.
- Protected 50-row owner seed: EVAL ONLY, never training (unchanged, hard-failed in code).
- Two-sided md5 on every pulled artifact; DELETE + list+disks confirm; cost vs cap.
- VERIFIED=0; no promotion claims; `models/MANIFEST.json` PENDING entry if the checkpoint is to be
  used as lineage.
