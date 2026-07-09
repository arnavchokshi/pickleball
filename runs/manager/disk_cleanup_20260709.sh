#!/bin/bash
# disk_cleanup_20260709.sh — OWNER-REVIEW deletion script (wave-7 boot; Mac at 99%, 5.2GB free).
# Written by the manager from the read-only disk-triage scout (facts) + manager rulings (decisions).
# EVERY line states size + why it is safe + how to regenerate. Comment out anything you want to keep,
# then run:  bash runs/manager/disk_cleanup_20260709.sh
#
# ============================ NEVER DELETE (for the record) ============================
#   runs/cvat_imports/                      — contains a STRICT HELD-OUT eval clip's labels (protected data)
#   runs/lanes/w6_labelingest_20260708/     — the 1,121-row reviewed corpus + LoSO folds (critical path)
#   runs/lanes/w5_closeproof_20260708/      — phase_b_predictions/ is the REGENERATION INPUT for all labelpacks
#   data/roboflow_universe_20260706/        — P1-0 training corpus source of truth (also baked in fleet snapshot)
#   data/online_harvest_20260706/           — P0-1b harvest fuel (active train/test source)
#   runs/lanes/wave2_freshworlds_20260707/  — referenced by tests/racketsport/test_frame_rating.py
#   runs/owner_footage_intake_20260702/     — YOUR original camera footage (5.4GB); delete yourself ONLY if
#                                             these 4 .movs are in Photos/iCloud (IMG_7768/5014/4983/1014)
# =======================================================================================
set -euo pipefail
cd /Users/arnavchokshi/Desktop/pickleball
df -h /System/Volumes/Data | tail -1

# ---------- SECTION A — consumed intermediates (recommended; ~19GB) ----------
# A1 (13GB) w6 labelpack FRAME STAGING — intermediate frames used to build the CVAT zips; packages built
#           AND 68/68 sessions already imported+verified into your CVAT (wave-6). Regen: the lane's own
#           build_labelpack.py from w5_closeproof phase_b_predictions (kept above).
rm -rf runs/lanes/w6_labelpack_20260708/frame_staging
# A2 (~1.9GB) w5 labelpack staging + task export — superseded by w6 lineage; session-01 already labeled+ingested.
rm -rf runs/lanes/w5_labelpack_20260708/frame_staging runs/lanes/w5_labelpack_20260708/cvat_task_export
# A3 (1.5GB) i1 grounding-unification wolverine run (Jul 4) — full-pipeline output; ruling booked; regen = process_video.py.
rm -rf runs/i1_grounding_unification_a100_wolverine_20260704T0718Z
# A4 (1.1GB) process_video glue wolverine2 (Jul 2) — its fixture excerpts are already committed under ios/ + tests/.
rm -rf runs/process_video_glue_20260702T_live_wolverine2
# A5 (954MB) fix3_stance_slide offline replay artifacts — proof superseded by the wave-3 slide-gate close; keep report.md.
rm -rf runs/lanes/fix3_stance_slide/offline_replay_attempt4
# A6 (932MB) payload-collapse debug repro (Jul 5) — bug fixed and booked; ad hoc dir, no consumers.
rm -rf runs/payload_collapse_isolation_20260705T223802Z
# A7 (~490MB) usdz-compress INTERMEDIATE bakes — keeps the final 11MB usdz + summaries the iOS manifests reference.
find runs/usdz-compress_20260702T113045Z -name "*.usdz" ! -name "body_mesh_animated_budget53.usdz" -delete
# A8 (~450MB) w3 teachertune evidence/sweep bulk — teacher approach killed by your 480-frame measurement; REPORT.md stays.
rm -rf runs/lanes/w3_teachertune_20260707/evidence runs/lanes/w3_teachertune_20260707/sweep
# A9 (621MB) p06 freshworlds attempt-1 (PARTIAL; superseded by later fresh-worlds proofs; report.md stays).
rm -rf runs/lanes/p06_freshworlds_20260706/outputs runs/lanes/p06_freshworlds_20260706/worlds 2>/dev/null || true

# ---------- SECTION B — regenerable corpora/packages (bigger; ~17GB) ----------
# B1 (13GB) w6 CVAT upload PACKAGES — the zips are already inside your CVAT (68/68 import verified).
#           Regen if ever needed: runs/lanes/w6_labelpack_20260708/build_labelpack.py. Comment out if you
#           want the zips as an extra CVAT backup.
rm -rf cvat_upload/w6_labelpack_20260708/packages
# B2 (982MB) w5 CVAT upload packages — same reasoning, older lineage.
rm -rf cvat_upload/w5_labelpack_20260708/packages
# B3 (4.1GB) flight corpus JSONLs (P0-7 simulator output) — deterministic regen:
#           scripts/racketsport/generate_flight_corpus.py (corpus_report.json stays as the recipe record).
rm -f runs/flight_corpus_20260707/flight_corpus_train_50000.jsonl runs/flight_corpus_20260707/flight_corpus_val_5000.jsonl

# ---------- SECTION C — OPTIONAL, outside the working tree (commented out; your call) ----------
# C1 (249MB) stray git worktree from the closed product-infra session:
# git worktree remove --force .claude/worktrees/product-infra-spec
# C2 (~900MB) old Claude session data dirs (breaks transcript-resume for THOSE old sessions only):
# rm -rf /Users/arnavchokshi/.claude/projects/-Users-arnavchokshi-Desktop-pickleball/35f43026-eaf1-4a04-87e1-066f679f3a70
# rm -rf /Users/arnavchokshi/.claude/projects/-Users-arnavchokshi-Desktop-pickleball/084d26f0-72ef-4b7b-b235-8a23d2c621a8
# rm -rf /Users/arnavchokshi/.claude/projects/-Users-arnavchokshi-Desktop-pickleball/70940ed3-b51c-4e20-ad03-dba2fb5d203a
# C3 (22GB!) ~/Desktop/CV_pipeline — a DIFFERENT project; only you know if it's live:
# (no command staged — handle manually if you want it gone)

df -h /System/Volumes/Data | tail -1
echo "CLEANUP DONE — sections A+B target ~35GB reclaimed"
