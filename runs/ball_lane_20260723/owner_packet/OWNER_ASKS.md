# Owner asks — 2026-07-23 (ball lane, WS4 packet)

`VERIFIED=0` binding everywhere; nothing below claims completion. Exactly 4 asks, in priority order.

## 1. Record a real game (standing ask #1)
**What**: 1–2 full games at your usual court, per `runs/manager/capture_recording_guide_20260709.md`
§B-2 (tripod, landscape, full court + both baselines, 1080p60, audio on).
**Why**: it is the top line of `OWNER_CHECKIN.md` (2026-07-23): nothing passes `VERIFIED=0` without
owner-shot ground truth. The data ledger (`runs/manager/data_ledger.json`, 34 assets) registers
**zero owner-recorded game footage** — owner-origin assets are labels/stills only (102 event labels,
IMG_1605 review, protected seed 50). A 2026-07-16 probe cited `runs/owner_footage_intake_20260702/raw/`
(39 clips), but that directory **no longer exists on disk** — so the effective owned-game corpus
today is zero.
**Urgency**: highest; unblocks fresh holdout + labeling for every lane (court, ball, person, events).

## 2. T1 gold multi-view capture session
**What**: execute `T1_SHOT_LIST.md` in this folder — one 60–90 min visit, 100–300 controlled flights,
baseline iPhone + 2 temporary synced cameras, marked bounce points.
**Why**: Phase A of `runs/ball3d_lifting_plan_20260723/PLAN.md` gates everything downstream in 3D
("nothing downstream is trustworthy without this"), and only you can produce it (PLAN.md: "I have no
cameras/court"). Until T1 exists, the current solver cannot even be scored in metric 3D.
**Urgency**: high; the whole ball-3D critical path runs through this physical capture.

## 3. Contamination ruling on the 3,026 reviewed ball centers
**What**: read `CONTAMINATION_MEMO_3026.md` (this folder) and pick Option 1 / 2 / 3. Recommendation
there: **Option 2** (clear with judge-parent families excluded — the already-built B0 split).
**Why**: `ball_reviewed_corpus_chain_1121_3026` is QUARANTINED in the ledger with the 74.8% finding
UNRULED; Track B may not reuse the corpus until "an explicit contamination ruling" lands (ledger
`next_check`). The B0 split lane already built and byte-bound the clean construction; it only lacks
your ruling.
**Urgency**: medium-high; it is the difference between 2,249 usable in-domain training rows and zero.

## 4. Confirm B1 GPU-resume authorization still stands
**What**: a yes/no that the 2026-07-22 grant (A100-40 SPOT slot, budget $3.20–5.90, rail
`shutdown -P +300` — `runs/tracks/trackB_ball_20260722/STATUS.md`) survives this week's changes:
your check-in (2026-07-23) records ball **VMs+disks all torn down** (~$1.30 spent) and re-scopes the
work to "profile WASB inference → batch it → rebuild SST cheap → short GPU run for B2".
**Why**: the grant was tied to resuming on the kept `pickleball-gpu-ball-disk-f` disk, which no
longer exists; dispatching on a stale authorization would violate the money discipline rules
(`runs/handoff_20260722/STATE.md` §GPU money discipline).
**Urgency**: medium; blocks the first row of the cost ledger below.

## Budget note — pre-approved ~$65 GPU envelope (2026-07-23)
Controller decision of 2026-07-23 (relayed in this packet's tasking; **not yet recorded in any
committed repo file this lane could read** — `GPU_COST_LEDGER.md` in this folder is the seed record):
per-run estimates **B1 finish $6–10 · B2 pair $8–18 · E-v2 $3–6 · multimodal v3 $5–9 ·
outdoor-night ensemble $11–21** (sum $33–64, inside ~$65). Every dispatch appends a ledger row
before launch; ceiling stays $65 until you raise it.
