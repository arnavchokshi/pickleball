# GPU fleet ledger (FABLE_OPERATING_MANUAL §12)

Live source of truth for every fleet VM. One row per VM; update on provision / dispatch / preempt /
teardown. A session MUST reconcile this against `gcloud compute instances list
--filter=labels.fable-fleet=pickleball` at start (orphaned VM = resume its lane or tear it down).

| vm_name | zone | gpu | model | status (provisioning/idle/busy/preempted/tearing-down) | lane | $/hr | created_at | notes |
|---|---|---|---|---|---|---|---|---|
| pickleball-a100-fleet1 | asia-southeast1-a | A100-SXM4-40GB (a2-highgpu-1g) | SPOT | STOPPED (wave-1 complete 2026-07-06; disk persists — restart: gcloud compute instances start) | p06-ground DONE -> idle (4/4 fresh BODY re-dispatches: stance_aware ACTIVE on all 4 (anchor_source=placement_track_world_xy); foot-slide PASS wolverine ~0m + img1605 25.6mm, FAIL burlington 46.9mm + outdoor 40.5mm w/ root_motion_temporal_jump blockers; body gate PASS 2/4. VM left RUNNING per manager — fleet decision at closeout. Details: runs/lanes/p06_freshworlds_20260706/) | ~$1.1-1.3/hr (est., matches prior same-zone/same-shape SPOT rate per RESET_HANDOFF §7; well under $5/hr cap) | 2026-07-06T18:55:16Z | P0-1 cold start DONE: nvidia-smi OK, EXCLUSIVE_PROCESS set, vendor pins restored, 27/27 BODY pytest GPU tests pass (0 skipped), inference smoke produced pred_keypoints_3d w/ GPU util confirmed. IP 34.143.175.207. Full detail: `runs/lanes/gpu_coldstart_20260706/report.md`. P0-6 lane (2026-07-06) added SSH host key to configs/ssh/a100_known_hosts + a VM-local symlink (~/body_runtime/Fast-SAM-3D-Body/checkpoints -> ~/coldstart_20260706/body_runtime/checkpoints) so the checked-in models/MANIFEST.json's hardcoded checkpoint local_path resolves; ran 4 fresh clips, all `partial` (BODY blocked by a local-machine rsync/openrsync transport bug, NOT a VM problem -- VM confirmed healthy post-run, nvidia-smi 0% util/0 MiB). Full detail: `runs/lanes/p06_freshworlds_20260706/report.md`. Left RUNNING + idle per mission. |

Fleet cost cap (owner 2026-07-06): ≤$5/GPU/hr, max 4 concurrent GPUs; teardown/DELETE the moment a
lane ends (idle spend never acceptable); 5th GPU or >$5/hr = `needs-purchase-approval` STOP.
Auth: owner gcloud refresh token (hello@; SA key creation org-blocked); dead auth = typed STOP for one owner login.
