# Lane spec — trk_detbench_20260716 (Track F, owner-directed GPU execution of benchmark_spec_trk.md)

Owner directive 2026-07-16: real numbers on real video. Execute the FINAL frozen protocol in
`runs/research_trk_rkt_20260716/benchmark_spec_trk.md` (authoritative for anything unstated here),
zero-shot arms + baseline reproduction. Diagnostic only; VERIFIED=0; no best_stack change; no
promotion language. This card is load-bearing because NO official crowded-person numbers exist for
these detectors.

## Budget + rails (HARD)

**AMENDMENT 1 (Track F manager, 2026-07-16 ~09:3x, after attempt-1 NO-ATTEMPT on fleet-wide H100
SPOT stockout — 6/6 zones exhausted, $0, no orphans):** SKU fallback authorized per the fleet
ledger's standing SKU ladder (H100 default → A100-80GB middle tier → A100-40GB proven fallback;
A100 quota filed in ase1/usc1/use4/euw4). Rationale: this workload is inference-only on 900
frames with small detectors — the H100 pin was the default heavy-worker SKU, not a requirement;
the boot snapshot is A100-native (source disk pickleball-a100-fleet1); A100 spot is CHEAPER, so
every cost rail tightens. Manager assumption noted: the owner directive's material terms are the
caps + discipline, not the SKU name. Attempt order this dispatch: H100 a3-highgpu-1g SPOT in
ase1-b and usc1-a ONLY (2 quick attempts), then A100-80GB (a2-ultragpu-1g) SPOT ladder
ase1 → usc1 → use4 → euw4, then A100-40GB (a2-highgpu-1g) same ladder. Same 120s backoff,
same 30-min total no-attempt cap. All other rails unchanged.

- One SPOT GPU VM per the amendment ladder above. ≤$5/hr. **$15 total cap → wall cap 3.5h from
  RUNNING** (A100 is ~2.4x slower than H100 on BODY-class work, but this lane is light inference —
  3.5h stands).
- Provision gate: fresh `gcloud compute instances list --filter=labels.fable-fleet=pickleball`
  FIRST (Track G may hold a second VM — that is fine, ≤5 concurrent; a pre-existing
  trk_detbench VM = reconcile, don't duplicate). Dead auth = typed STOP.
- Create per fleet precedent (runs/lanes/trk_reid_apron_20260712/spec.md + pbv11 lane): boot disk
  pd-balanced 200GB FROM snapshot `pickleball-fleet-snap-20260709-w7close`, zone ladder
  ase1-b → ase1-c → us-central1-a → us-central1-b → europe-west4-b, 120s inter-attempt backoff,
  6-attempt/30-min no-attempt cap, `--provisioning-model=SPOT --instance-termination-action=STOP`,
  labels `fable-lane=trk_detbench_20260716,fable-fleet=pickleball,owner=arnavchokshi`.
- **On-VM teardown rail at t0, VERIFIED ARMED before any work** (ops lesson 2026-07-15: Mac-side
  watchers die on laptop sleep): `sudo shutdown -P +210` scheduled immediately after first ssh;
  print the shutdown schedule confirmation into the lane log as proof. Also arm the 60-min
  no-heartbeat self-stop per boot ritual. Manager will check the log for the armed proof.
- END NO MATTER WHAT: `gcloud compute instances delete` + instances list-confirm + disks
  list-confirm 0 lane disks + wall-hours and $ estimate → fleet ledger.
- Write the VM row into `runs/manager/gpu_fleet.md` BEFORE dispatching work on it; update at
  teardown.

## Boot ritual (fleet ledger standing policy)

reset --hard if dirty beyond the 2 vendor-submodule lines; code identity via git bundle from Mac
HEAD (record SHA + bundle sha256, two-sided); fresh ssh-keyscan SELF entry into
configs/ssh/a100_known_hosts after checkout; python3 explicitly; compute-mode DEFAULT (single
lane). Known snapshot gaps: scp `models/checkpoints/osnet_x1_0_market1501.pt` (sha256 must match
2809d322…9154) and `pip install torchreid` in the pipeline venv.

## Transfer manifest (Mac → VM, tarball, two-sided md5)

- eval videos: `eval_clips/ball/{wolverine_mixed_0200_mid_steep_corner,burlington_gold_0300_low_steep_corner}/source.mp4`
  (1920x1080; 300f@30 / 600f@59.94)
- frozen GT: `runs/lanes/trk_flip_20260713/frozen_gt/*/person_ground_truth.json`
- frozen baseline pools + calibrations + metrics: `runs/lanes/trk_flip_20260713/{default,preflip}_production/<clip>/`
  (tracked_detections.json, raw_tracked_detections.json, metrics.json, court_calibration.json if
  present; else calibrations from `runs/trk_live_rescore_20260702T2200Z/score_inputs/<clip>/court_calibration.json`)
- reproduction target: `runs/lanes/trk_flip_20260713/preflip_score/person_track_gt_scoring_report.json`
- OSNet ckpt (above)

## PREFLIGHT (STOP on any mismatch)

Byte-verify on VM vs these Mac md5s (re-confirmed 2026-07-16):
- scripts/racketsport/benchmark_person_trackers.py = 07deba04bc00f9eaff9670676ac3ec45
- scripts/racketsport/score_person_track_sources.py = cd7ae4891c482a257807761f3b934a90
- threed/racketsport/person_track_gt_scoring.py = be38f76547d05d8ac7b12274de5b659d
Association identity pins (Mac md5s, computed 2026-07-16 at spec time; VM copies must match):
- threed/racketsport/raw_pool_person_authority.py = ea30bfdf3a57bf7e2fff06476ec6295c
- threed/racketsport/player_global_association.py = 5e761c5db3327a1841fc0e54281bb9d7 yolo26m.pt sha256 401cea9a…5d0b7 (snapshot copy must match).

## Arms (in this order; score incrementally so early arms survive any later failure)

Association step for EVERY arm = the frozen champion:
`python3 scripts/racketsport/run_raw_pool_person_authority.py --clip-id <clip> --candidate <arm>
 --video <source.mp4> --raw-pool-dir <arm pool dir> --calibration <court_calibration.json>
 --out-dir <scored/<clip>/<arm>> --reid-model models/checkpoints/osnet_x1_0_market1501.pt
 --reid-backend osnet --court-margin-m 1.0 --expected-players 4` — all other knobs DEFAULT
(that IS owner_directed_margin1p0_osnet). Compute OSNet embeddings once per pool; reuse via
`--embedding-export` where applicable.

- **Arm 0a — association+scorer reproduction from FROZEN pools.** Run the association on the
  transferred trk_flip production pools → scorer. MUST match preflip_score within 0.0001:
  burlington IDF1 0.8830775881 / cov4 0.7116666667; wolverine IDF1 0.8515962036 / cov4 0.76;
  0 switches. Mismatch = STOP + diagnose (do not proceed to candidates on a broken harness).
- **Arm 0b — feeder validation (the confound check).** Fresh YOLO26m detections at the PRODUCTION
  operating point (conf=0.05, imgsz=1536, classes=[0], per orchestrator.py:617-629) fed through
  `ultralytics` BOTSORT constructed from `configs/racketsport/botsort_no_reid_loose.yaml` via its
  per-frame `update()` API (same pinned ultralytics as the snapshot venv) → pool JSON in the exact
  schema ({"fps":…, "frames":[{"frame":i,"detections":[{"bbox":[x1,y1,x2,y2],"class":"person",
  "conf":c,"track_id":t}]}]} + metrics.json counts block w/ source/calibration dims + bbox_scale)
  → association → scorer. Compare to Arm 0a: |ΔIDF1| and |Δcov4| ≤0.005 per clip → feeder CLEAN;
  larger → flag FEEDER_DRIFT and interpret all candidate arms as paired-protocol vs 0b (report
  both deltas). The feeder script lives in the LANE DIR (scripts stay out of pipeline dirs).
- **Arm 1 — RF-DETR-L zero-shot.** `pip install rfdetr` (pin the installed version in report);
  weights `rf-detr-large-2026.pth` (auto-download URL storage.googleapis.com/rfdetr/…; record
  sha256). Native 704, keep ALL person detections conf ≥0.05 (match the production pool floor;
  VERIFY the person class id from the model's class map and record it). → feeder → association →
  scorer. Record batch-1 ms/frame.
- **Arm 2 — RF-DETR-Seg-L zero-shot.** `rf-detr-seg-l-ft.pth`, 504 native. Score its BOXES
  identically; ARCHIVE masks (per-frame RLE or polygons, conf, class) to the lane dir for the
  future mask-cue lane. Record ms/frame.
- **Arm 3 — D-FINE-L control.** Repo github.com/Peterande/D-FINE pinned to current HEAD (record
  SHA), ckpt `dfine_l_obj2coco_e25.pth` (exact URL in benchmark_spec_trk.md; record sha256).
  Integration budget 45 min; if exceeded → `no-attempt` with the exact blocker, move on.
- **Arm 4 — DEIMv2-L control.** Repo github.com/Intellindust-AI-Lab/DEIMv2 pinned (record SHA),
  HF `Intellindust/DEIMv2_DINOv3_L_COCO` (NOT `_S_COCO`). Same 45-min budget / no-attempt rule.

Scoring (once per new arm, cumulative): `python3 scripts/racketsport/score_person_track_sources.py
--cvat-root runs/lanes/trk_flip_20260713/frozen_gt --runs-root <scored root> --out-dir <scores
dir> --iou-threshold 0.5 --expected-players 4` with layout `<scored>/<clip>/<arm>/tracks.json`
(+ sibling metrics.json + court_calibration.json so bbox-scale and calibration resolve).

## Metrics + verdicts (per arm per clip)

IDF1, cov4, id_switches, true_spectator_or_background FP, far_off_court FP frames, near-miss/
no-gt-frame FP (diagnostic), HOTA/DetA/AssA (diagnostic), detector batch-1 ms/frame, arm wall.
Verdict per benchmark_spec_trk.md stop rules: reject on any new switch / spectator FP /
far-off-court FP or >20% detector wall increase without gate-relevant gain; coverage (cov4) is
the decision metric for the fine-tune go/no-go. Verdict vocabulary: `adopt-next-step` /
`reject` / `no-attempt`. NO promotion; this is a historical-internal card.

## Deliverables

- `runs/lanes/trk_detbench_20260716/report.json`: provision/teardown proof (list outputs), rail
  armed proof, preflight md5 table, arm 0a reproduction table, feeder delta, per-arm decision
  table, runtimes, checkpoint sha256s + versions, cost estimate.
- `DECISION_TABLE.md`: the human table (arms x {burl,wolv} x {IDF1,cov4,switches,spectFP,farFP},
  baseline row first) + go/no-go recommendation w/ evidence for the fine-tune decision arm.
- Pulled artifacts two-sided md5: per-arm pools, tracks, person_track_gt_scoring_report.json,
  seg mask archive, logs.
- Fleet ledger rows updated (provision + teardown); heartbeat file while running.

## Fence

Write ONLY: runs/lanes/trk_detbench_20260716/**, runs/manager/gpu_fleet.md (rows). All adapter/
feeder scripts live in the lane dir. NO pipeline-code, config, or best_stack edits. NO git
commits. Read-only everything else. Never wait passively (poll w/ timeouts; if the VM ladder
no-attempts out, STOP with the evidence).
