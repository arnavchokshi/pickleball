# RE-RUN ADDENDUM 2026-07-15 — pbv11_headtohead MOVE-1 (Sonnet GPU ops lane)

The original `spec.md` in this directory remains the mission definition (READ IT FIRST, plus its
"Ground truth READ FIRST" list). This addendum supplies the re-run deltas after the 2026-07-13
attempt died mid-run (Fable spend limit; VM deleted 2026-07-14; NO scorecard was ever produced).

## HARD RULES (supplement to spec.md's)
- Ops lane: Bash writes only. NO repo source edits, NO commits, NO branches. Do NOT touch
  `scripts/racketsport/process_video.py` or any pipeline source (a parallel track owns NS-01
  surfaces). The VM runs PINNED code via git bundle only.
- `runs/lanes/pbv11_headtohead_20260713/report.json` is a STALE STOP report from the dead-auth
  attempt. NEVER read it as a result. Your fresh report goes to
  `runs/lanes/pbv11_headtohead_20260713/rerun_20260715/report.json`.
- All new artifacts under `runs/lanes/pbv11_headtohead_20260713/rerun_20260715/` (plus the owner
  pack at `runs/lanes/pbv11_headtohead_20260713/owner_event_review/` per original spec step 5).
- pb.vision demo video + exports = R&D reference ONLY: never ground truth, never training data,
  never redistributed. Protected denylist scan (Outdoor/Indoor) over all inputs/commands before
  copy and again before scoring; save scan outputs in the rerun dir.
- VERIFIED=0 is binding. The scorecard is DIAGNOSTIC, not a promotion. Honest status vocabulary
  only (adopt/reject/partial/no-attempt for the lane objective; scoped language elsewhere).
- Spend guard: if projected total lane spend risks exceeding ~$20 (e.g., preemption churn),
  STOP, tear everything down, record cost, and report rather than continuing.

## FILE OWNERSHIP
This lane owns ONLY: `runs/lanes/pbv11_headtohead_20260713/rerun_20260715/**` and
`runs/lanes/pbv11_headtohead_20260713/owner_event_review/**`. Other lanes' run dirs are READ-ONLY
evidence. `runs/manager/*.md` is the manager's — do not edit; report facts instead.

## PIN + IDENTITY (deltas from the 07-13 attempt)
- Code pin: `ac0b14ab0d3a5c00418671f84a725affc54a8213` (current main tip; ancestor pin 541f89d9a
  superseded — ac0b14ab0 includes the 03a0085ab audio normal-path chain and the rev-12 TRK flip
  9739bd8fc). Build a fresh git bundle at this exact SHA on the Mac, transfer, verify bundle sha,
  checkout the PIN on the VM (never tip). Record every hash in the report.
- best_stack revision consumed: **rev 12** (margin-1.0m + OSNet WIRED_DEFAULT, preview band).
- Source video: `data/pbvision_11min_20260713/source_video.mp4`, sha256
  `272a2132ce7c72ea31fe6351c9ea05ac3016bbbfed0a5801d9c3a973ec628383`, 697.4s 1280x720@30 + AAC.
  Verify sha BOTH sides after transfer.
- KNOWN SNAPSHOT GAP (will otherwise fail loud as `best_stack_asset_missing` in tracking):
  transfer `models/checkpoints/osnet_x1_0_market1501.pt` (10,399,605 bytes, sha256
  `2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154`) to the same relative path in
  the VM checkout; verify sha on the VM BEFORE starting the run.

## PROVISION (manager already reconciled ledger vs live 2026-07-15: fleet 0/5 RUNNING; gate PASS)
- Re-check cheaply at create time: `gcloud compute instances list --filter=labels.fable-fleet=pickleball`
  — if ≥5 fleet VMs RUNNING, typed needs-capacity STOP.
- VM `pickleball-h100-pbv11r`: a3-highgpu-1g SPOT (one H100-80GB), boot disk pd-balanced 200GB
  FROM snapshot `pickleball-fleet-snap-20260709-w7close`.
- `--provisioning-model=SPOT --instance-termination-action=STOP` (never DELETE on preempt);
  labels `fable-lane=pbv11-rerun,fable-fleet=pickleball,owner=arnavchokshi`.
- Zone ladder: asia-southeast1-b → asia-southeast1-c → us-central1-a → us-central1-b →
  europe-west4-b; 120s inter-attempt backoff; max 6 attempts; 30 min with no successful attempt =
  typed `no-attempt` STOP (record + exit; that is NOT negative model evidence). Prior attempt's
  `create_loop.sh` in this lane dir is reusable scaffolding — adapt, don't trust blindly.
- Wall cap 5h from RUNNING (write `rerun_20260715/wall_cap_start.txt`); arm the in-VM 60-min
  no-heartbeat idle self-stop; compute-mode DEFAULT (self-dispatch lane, per fleet boot ritual).
- Preemption mid-run: restart the STOPped VM once and resume (disk persists); if repeated churn
  threatens the ~$20 guard or the 5h wall cap, abandon, teardown, report `partial`.
- Boot ritual (gpu_fleet.md): reset --hard if dirty beyond the 2 vendor-submodule lines; fresh
  `ssh-keyscan` SELF-entry into `configs/ssh/a100_known_hosts` AFTER every checkout/reset; use
  `python3` (bare `python` absent); fleet IPs RECYCLE — always fresh keyscan + explicit host.

## RUN (per original spec.md steps 1-5, with these clarifications)
1. Transfer video (chunked/bwlimit fallback if uplink flaky) + sha verify both sides. FRESH
   content-addressed generation — use explicit `--clip pbvision_11min_20260713` and a fresh
   `--out` under the VM-side lane work dir. No reuse of any prior run dir.
2. ONE full promoted-stack run: best_stack rev-12 defaults, `--body-local`, audio ON (do NOT pass
   `--skip-audio`), `--sport pickleball --max-players 4`. Expect the 1200-frame BODY cap to bind
   on ~20,922 frames — fine; record exclusion provenance. Record per-stage wall times (this is
   full-game-scale speed evidence too). `nohup` the pipeline; poll bounded (foreground until-loop,
   120-300s intervals, each poll a cheap ssh tail of the stage log). Ending your turn to wait =
   lane death; you will NOT be re-woken. End only with the final report or a hard blocker.
3. Pull ALL ball/world/event artifacts + per-stage summary; md5 BOTH sides for every pulled file;
   save the md5 manifests in the rerun dir.
4. TEARDOWN NO MATTER WHAT (success, failure, preemption-abandon, spend-guard STOP): DELETE the
   VM, `gcloud compute instances list` confirm gone, `gcloud compute disks list` confirm 0
   lane-created disks remain, record wall hours × spot rate as the cost estimate. Save all three
   command outputs in the rerun dir as teardown proof.
5. MAC-SIDE scoring (after teardown): first run the harness self-test
   `MPLBACKEND=Agg .venv/bin/python -m pytest runs/research_pbv_reveng_20260712/test_compare_vs_pbvision.py -q`;
   then `runs/research_pbv_reveng_20260712/compare_vs_pbvision.py` with
   `--pb-export data/pbvision_11min_20260713/cv_export.json --ours <pulled run dir> --frame-offset auto`
   → ONE immutable per-rally scorecard JSON + a human md summary in the rerun dir. PB per-rally 3D
   baseline for the headline comparison: 183/252 (from the frozen 07-13 forensics).
6. OWNER EVENT-REVIEW PACK (quality gates the video event head — treat as a first-class
   deliverable, not an afterthought): merge OUR event/contact/bounce proposals + PB typed events
   (each row flagged `ours` / `pb_reference` / `both`) into a compact reviewable list with
   video timestamp (mm:ss.ff), type, source, confidence, thumbnail frame idx. Extract a small
   JPEG thumbnail per row from the LOCAL source video (ffmpeg) so the owner pass needs no
   scrubbing. If the union exceeds ~150 rows, stratify: ALL disagreement rows + a documented
   uniform sample of agreement rows; record the sampling rule in the pack. One-screen owner
   instructions + an explicit estimated review time (target ≤15 min). Stage under
   `runs/lanes/pbv11_headtohead_20260713/owner_event_review/`. PB rows remain reference-only and
   must be visually marked as such.

## MANDATORY STRUCTURED REPORT — `rerun_20260715/report.json`
{objective_result: adopt|reject|partial|no-attempt vs the original spec's deliverables,
run_identity: {pin_sha, bundle_sha, video_sha, osnet_sha, best_stack_rev, clip_id, out_dir},
provision: {vm, zone, attempts, running_at, deleted_at},
per_stage_timing: [...], body_cap_provenance,
scorecard_summary: {our per-rally 3D coverage vs PB 183/252, physics/bounce/reproj headline rows,
scorecard_json_path, scorecard_md_path},
owner_pack: {path, row_count, disagreement_count, sampling_rule, estimated_review_minutes},
teardown_proof: {delete_output, list_confirm, disks_confirm},
cost: {wall_hours, rate_range, estimate_usd},
full_suite: "not-required (ops lane, zero source edits) — harness self-test result instead",
honest_issues: [...]}
BEST-STACK DELTA (mandatory): (c) NONE — diagnostic eval lane consuming best_stack rev 12 on the
promoted upstream stack; no manifest entry changes; not a promotion; VERIFIED=0 stands.
