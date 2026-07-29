# Ball bounce labelling — round 2 (4 clips live)

**Lane:** `ball_labels_round2_20260728` · **Date:** 2026-07-28 · **VERIFIED=0** — every server below
produces human review-only labels, not verified ground truth.

All four servers are CPU-only, localhost-only, no GPU, no cloud, $0. Each opens its run
directory **read-only** and writes only into its own `--out` directory below.

---

## The 4 servers

| # | Clip | URL | Calibration floor (median court-plane error a *perfect* bounce click still inherits) | Frames | Notes |
|---|---|---|---|---|---|
| 1 | `wolverine_mixed_0200_mid_steep_corner` | **http://127.0.0.1:8801** | **0.127 m** median / 0.612 m worst | 300 @ 30 fps (10 s) | 215/300 frames have a solver 3D prefill; 7 bounce candidates; no tracked skeletons, so `near_player` is unusable here — bounce + free-flight only. |
| 2 | `burlington_gold_0300_low_steep_corner` | **http://127.0.0.1:8802** | **0.191 m** median / 0.535 m worst | 600 @ 30 fps (20 s) | 402/600 frames have a solver 3D prefill; 8 bounce candidates; no tracked skeletons — bounce + free-flight only. This clip was not part of the original 3-clip pilot; its floor was measured fresh for this round. |
| 3 | `outdoor_webcam_20s_fullmesh_final` | **http://127.0.0.1:8803** | **0.101 m** median / 0.447 m worst | 600 @ 30 fps (20 s) | 186/600 frames have a solver 3D prefill; 9 bounce candidates; 4 tracked skeletons — the only clip where `near_player` is meaningfully usable. Best-calibrated clip; start here if you want the cleanest bounce truth. |
| 4 | `pbvision_11min_20260713` (the pb.vision 11-minute demo) | **http://127.0.0.1:8804** | **0.144 m** median (in-sample) / **0.177 m** (held-out) / 0.386 m worst | **20,922 @ 30 fps (697.4 s ≈ 11.6 min)** | Newly assembled this session — see "About clip 4" below. Sits between wolverine and outdoor on calibration accuracy, so it's a real, usable source, but it is missing several conveniences the other three have (also below). |

Every number above was read directly off each clip's own `--check` run against the exact directory
each server is serving (or, for clip 4, its freshly-assembled run directory), using the tool's own
`pixel_ray_world -> intersect_ray_z(z=0)` method on that calibration's reviewed correspondences —
the same method the 2026-07-26 pilot used for its outdoor/wolverine/indoor table. It is **not**
copied from any report.

---

## About clip 4 (pb.vision demo) — what's different from the other three

This run directory did not exist before this session; it was assembled at
`runs/lanes/ball_labels_round2_20260728/pbv11_runassembly/` from three pieces that already existed
elsewhere in the repo (full chain of custody in that directory's `provenance.json`):

- **Calibration** — your reviewed 15-point taps (`1075cee57`), refit by the 2026-07-26 calibration
  lane to fix two bugs (net keypoints declared at the wrong height; the distortion fit scored on
  the wrong residual). Converted into `court_calibration.json` by validating it through the repo's
  own `CourtCalibration` pydantic schema (the exact class the pipeline uses to write that file) —
  nothing was hand-written. The in-sample floor this produces, 0.144 m, matches the refit lane's
  own report line-for-line; the held-out (more honest, slightly worse) estimate from that lane was
  0.177 m, and both numbers are shown above rather than only the friendlier one.
- **Ball detections** — an existing WASB run (`ball_hitdetect_20260713/pbv11_wasb`), CPU, tennis
  checkpoint. **Important limit: it only processed the first 5,400 of 20,922 frames — roughly the
  first 180 seconds (26%) of the 11-minute video.** Frames 5,400–20,921 (the remaining ~74%) have
  zero detector guidance: no yellow detection dot, no candidate rings, no `P`-key pixel guess.
  You can still label anywhere in the clip, but past the 180 s mark you're finding the ball
  yourself with no assist. If you want the easiest labelling, work the first 180 s of this clip
  first.
- **Source video** — `data/pbvision_11min_20260713/source_video.mp4` (not under `eval_clips/`,
  despite what you might expect from the folder naming; that `eval_clips` directory only holds your
  15-point review and court-keypoint frame crops, no video). Frame count/fps/resolution were
  cross-checked against the WASB run's own recorded provenance with `ffprobe` and match exactly, so
  this is confirmed to be the same video the ball track was computed from.

What clip 4 does **not** have, and why that's an honest omission rather than an oversight:

- **No `skeleton3d.json`** — no BODY-stage run has ever completed against the full 11-minute clip
  (a 302-frame/~10 s BODY slice exists from an unrelated demo, `runs/demo_court_people_20260726/
  outdoor_pbvision/`, but its frame numbering doesn't line up with this clip's frame numbering, so
  splicing it in would silently put player skeletons on the wrong frames — worse than not having
  them, so it was left out). Effect: `near_player` labelling is unusable on this clip (the tool
  refuses a near-player judgement with no tracked joint within 2.5 m of the ray) — bounce and
  free-flight only, same as clips 1 and 2.
- **No 3D solver prefill at all** (`ball_track_arc_solved.json` doesn't exist for this clip) — the
  `P` key won't offer a solved depth to correct on any frame of this clip. This does **not** change
  bounce accuracy: a bounce's depth is always solved live by ray-plane intersection the moment you
  click it, prefill or not — it only means you don't get a pre-filled starting point to correct,
  you place the click yourself every time.
- **No bounce-candidate file** — `B` / `Shift+B` (jump to next bounce candidate) has nothing to jump
  to on this clip. You'll need to scrub or use `N`/`Shift+N` (next frame with a detection, only
  useful in the first 180 s) to find bounces here.

None of this was fabricated to make the clip look more complete than it is — the tool's own
`--check` readiness summary (saved in `provenance.json`) reports all of it, and the server itself
reports the same `missing_artifacts` list live at `http://127.0.0.1:8804/api/state`.

---

## The round target

Per `NORTH_STAR_ROADMAP.md` §5 (queue row 5, "Owner bounce labelling at scale"):

> Bounce is the only kind with a solved depth, so it is the only kind that can become truth;
> free-flight and near-player stay review-only estimates.
>
> **Target: ≥150 bounce labels across ≥4 source-disjoint clips**, each carrying its calibration
> floor, click sensitivity and realised sigma; a source-disjoint held-out split declared before any
> scoring.

The four clips above are four different source videos/venues (wolverine, burlington, outdoor
webcam, pb.vision), which is what makes this the 4th-clip milestone for that target. `near_player`
and `free_flight` labels are still worth recording as you go — they cost almost nothing once you're
already looking at a frame — but only `bounce` labels count toward the ≥150.

Practical order, from the 2026-07-26 pilot: press `B` to jump to a bounce candidate (clips 1–3
only), press `P` to load the pipeline's own pixel guess where one exists, nudge the click if it's
off, press `Enter`. Those are the highest-value two-second labels. On clip 4 past the 180 s mark,
skip straight to clicking the ball yourself since there's no guess to load.

---

## Where labels land

Each server autosaves every label immediately to its own `--out` directory as
`ball_human_labels.json`, plus a session file that resumes you at your last frame if you quit and
relaunch:

- `runs/lanes/ball_labels_round2_20260728/wolverine/ball_human_labels.json`
- `runs/lanes/ball_labels_round2_20260728/burlington/ball_human_labels.json`
- `runs/lanes/ball_labels_round2_20260728/outdoor_webcam/ball_human_labels.json`
- `runs/lanes/ball_labels_round2_20260728/pbv11/ball_human_labels.json`

All four are writable (checked directly, not just assumed) and each write is atomic (temp file +
rename), so a crash mid-save can't truncate your work.

---

## Server logs / process notes

`server_wolverine.log`, `server_burlington.log`, `server_outdoor.log` were started in an earlier
part of this session and are still running untouched (PIDs 19761/19763/19765, launched with
`--no-open`, all HTTP 200 as of this write-up). `server_pbv11.log` is this session's 4th server
(background `nohup`, `--no-open`); all four server log files are normally near-empty in `--no-open`
mode — that's expected, not a symptom of anything wrong — confirm liveness with the URLs above or
`curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:<port>/`.

If you restart any of these, relaunch with the same `--run-dir`/`--out`/`--port` so labels resume
in place. Command used for clip 4:

```bash
.venv/bin/python scripts/racketsport/ball_label_studio.py \
  --run-dir runs/lanes/ball_labels_round2_20260728/pbv11_runassembly \
  --clip-id pbvision_11min_20260713 \
  --out runs/lanes/ball_labels_round2_20260728/pbv11 \
  --host 127.0.0.1 --port 8804 --no-open
```
