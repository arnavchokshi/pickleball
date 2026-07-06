# VM Archive Manifest — 2026-07-05

Source VM: `arnavchokshi@34.126.67.233` (about to be terminated). Archive is
read-only w.r.t. the VM — nothing was deleted or modified there.

Local free space at start: 11 GiB avail / 460 GiB total (98% used on
`/dev/disk3s5`) — above the 5GB cap-trigger, so no 1GB cap was applied.
Total transferred this session: ~279 MB checkpoints + 69.3 MB git bundle +
~23 KB text captures ≈ 348 MB.

## Step 1 — Checkpoints already present locally (SKIPPED, byte-identical mirror path + size match, no transfer needed)

| VM path | Local path | Size (bytes) |
|---|---|---|
| `~/pickleball_train_main/models/checkpoints/osnet_x1_0_market1501.pt` | `models/checkpoints/osnet_x1_0_market1501.pt` | 10399605 |
| `~/pickleball/runs/detect/runs/phase2/trk_people_id_goal_20260701T030347Z/yolo26m_detector_train/a100_player_yolo26m_e30/weights/best.pt` | `runs/phase2/trk_people_id_goal_20260701T030347Z/yolo26m_detector_train/a100_player_yolo26m_e30/weights/best.pt` | 44082073 |
| ...same dir/last.pt | ...same dir/last.pt | 44082073 |
| `~/pickleball/runs/detect/runs/gpu/train_player_yolo26/a100_player_yolo26n_20260630_044525/weights/best.pt` | `runs/detect/runs/gpu/train_player_yolo26/a100_player_yolo26n_20260630_044525/weights/best.pt` | 5453893 |
| ...same dir/last.pt | ...same dir/last.pt | 5453893 |
| `~/pickleball/runs/a100_detector_eval_20260630/ultralytics/player_yolo26n_smoke_train/weights/best.pt` | `runs/a100_detector_eval_20260630/ultralytics/player_yolo26n_smoke_train/weights/best.pt` | 5356933 |
| ...same dir/last.pt | ...same dir/last.pt | 5356933 |

## Step 1 — Checkpoints archived (transferred to `models/checkpoints/vm_archive_20260705/`)

| VM path | Local path | Size | SHA256 local | SHA256 VM | Match |
|---|---|---|---|---|---|
| `~/pickleball_train_main/runs/detect/runs/trk_det_retrain_20260702T004615Z/training/configA_baseline_img1536/weights/best.pt` | `train_main__trk_det_retrain_20260702T004615Z__configA_baseline_img1536__best.pt` | 44266905 | bd3d592d...6bbb3 | bd3d592d...6bbb3 | Y |
| `.../configA_baseline_img1536/weights/last.pt` | `train_main__trk_det_retrain_20260702T004615Z__configA_baseline_img1536__last.pt` | 44266905 | 1f575400...47824 | 1f575400...47824 | Y |
| `.../configB_boxloss_upweighted/weights/best.pt` | `train_main__trk_det_retrain_20260702T004615Z__configB_boxloss_upweighted__best.pt` | 44267417 | e5b77733...22f81 | e5b77733...22f81 | Y |
| `.../configB_boxloss_upweighted/weights/last.pt` | `train_main__trk_det_retrain_20260702T004615Z__configB_boxloss_upweighted__last.pt` | 44267417 | 7705739a...6fb713 | 7705739a...6fb713 | Y |
| `.../configC_highres_copypaste/weights/best.pt` | `train_main__trk_det_retrain_20260702T004615Z__configC_highres_copypaste__best.pt` | 44432793 | 945d6b98...28f1e9 | 945d6b98...28f1e9 | Y |
| `.../configC_highres_copypaste/weights/last.pt` | `train_main__trk_det_retrain_20260702T004615Z__configC_highres_copypaste__last.pt` | 44432793 | 4ecdc055...9b4a65 | 4ecdc055...9b4a65 | Y |
| `~/pickleball_train_main/runs/court_keypoint_detector_20260701_a100_frameholdout_e400/court_keypoint_heatmap.pt` | `train_main__court_keypoint_detector_20260701_a100_frameholdout_e400__court_keypoint_heatmap.pt` | 2080426 | d9b1fef0...67d06 | d9b1fef0...67d06 | Y |
| `~/pickleball_git/runs/cvat_imports/2026_06_30/racket_yolo_train_gpu/yolo11n_paddle_img960_e50/weights/best.pt` | `git__racket_yolo_train_gpu_yolo11n_paddle_img960_e50__best.pt` | 5485203 | 7134203e...4bdbe | 7134203e...4bdbe | Y |
| `.../yolo11n_paddle_img960_e50/weights/last.pt` | `git__racket_yolo_train_gpu_yolo11n_paddle_img960_e50__last.pt` | 5485203 | fcb756fa...3bbf2 | fcb756fa...3bbf2 | Y |
| `~/pickleball_git/runs/pickleball_pretraining/court_keypoint_20260628/court_keypoint_heatmap.pt` | `git__pickleball_pretraining_court_keypoint_20260628__court_keypoint_heatmap.pt` | 134772 | f7505683...6740f8 | f7505683...6740f8 | Y |

All 10 archived checkpoint files: **SHA256 match Y (10/10)**. Total archived
checkpoint bytes: ~279 MB. Full-length SHA256 values were diffed byte-for-byte
via a paired local `shasum -a 256` / VM `sha256sum` run (see session log);
truncated above for table readability only.

## Step 2 — Git history insurance

### `~/pickleball_git` (VM)
- `git remote -v`: `origin https://github.com/arnavchokshi/pickleball.git` (fetch+push) — same remote as local repo.
- `git count-objects -vH`: size-pack **67.36 MiB** (count 141, in-pack 4272, packs 3) — under the 1.5GB bundle threshold.
- Working tree: **dirty** — `git status --short` = 320 lines (files), `git diff --stat HEAD` = 66 lines changed. Captured as text only (`pg_status.txt`, `pg_dirtystat.txt`) per task scope — the actual diff content/hunks were NOT captured (only file-list/stat), so uncommitted edits beyond what's already in the local working tree are not fully insured by this archive. Refs (branches/commits) ARE fully insured via the bundle.
- **Bundle created**: `git bundle create --all` → `/tmp/pickleball_git_all.bundle` on VM, 69,345,260 bytes. `git bundle verify` on VM: OK, "records a complete history." Contains 5 refs: `refs/heads/main`, `refs/remotes/origin/HEAD`, `refs/remotes/origin/main`, `HEAD`, and `worktrees/pickleball_ball_regen_20260704T0807Z/HEAD` (commits `fb2169d7...` and `bf25a98d...`).
- Transferred to `runs/lanes/vm_archive_20260705/pickleball_git_all.bundle`. SHA256 local = VM: `fb8d0b0d1a61df0116587acbcbd55e067d84abf656a64ce83ef35166be346598` — **Match Y**.
- Text captures transferred: `pg_log.txt` (30 commits), `pg_status.txt` (320 lines dirty status), `pg_dirtystat.txt` (66 lines diffstat).

### `~/pickleball_train_main` (VM)
- `git remote -v`: `origin https://github.com/arnavchokshi/pickleball.git` (fetch+push) — same remote.
- `git count-objects -vH`: size-pack 66.50 MiB (count 493, in-pack 4215, packs 2, prune-packable 365).
- Per task instructions, only the three text captures were requested for this repo (no bundle instructed) — captured and transferred: `pt_log.txt` (30 commits), `pt_status.txt` (small — repo mostly clean), `pt_dirtystat.txt` (small).

## Skipped-on-purpose (large data/venv dirs — not archived, rationale below)

| VM path | Size | Rationale |
|---|---|---|
| `~/ball_training_data` | 19G | Bulk training data, reproducible from source video + labeling pipeline; not unique model output. |
| `~/aspset_body_ext_20260702` | 9.5G | External ASPset body dataset extract — re-downloadable/re-buildable from public source, not VM-unique. |
| `~/body_runtime` | 8.4G | Runtime/inference working dir incl. its own venv (`fast_sam_venv`); regenerable from checkpoints + code, not itself a unique artifact. |
| `~/pickleball_train_main/runs/process_video_body_dispatch` | 7.1G | Dispatch/scratch working directory for batch video processing runs; intermediate outputs, regenerable by re-running the pipeline against source clips, not unique checkpoints. |
| venvs (`~/rf_venv_t1a2`, `~/pickleball_git/.venv`, `~/pickleball/.venv`, `~/pickleball_ball_regen_20260704T0807Z/.venv`, `~/body_runtime/fast_sam_venv`) | 490M+ combined | Reproducible from `requirements`/lockfiles; no unique state. |

## Failures
None. All 10 targeted checkpoint transfers, the git bundle transfer, and all 6 text-capture transfers completed with verified matches.
