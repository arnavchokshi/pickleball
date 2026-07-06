# GPU fleet ledger (FABLE_OPERATING_MANUAL §12)

Live source of truth for every fleet VM. One row per VM; update on provision / dispatch / preempt /
teardown. A session MUST reconcile this against `gcloud compute instances list
--filter=labels.fable-fleet=pickleball` at start (orphaned VM = resume its lane or tear it down).

| vm_name | zone | gpu | model | status (provisioning/idle/busy/preempted/tearing-down) | lane | $/hr | created_at | notes |
|---|---|---|---|---|---|---|---|---|
| pickleball-a100-fleet1 | asia-southeast1-a | A100-SXM4-40GB (a2-highgpu-1g) | SPOT | idle | coldstart→idle (fleet GPU #1, ready for next wave's dispatch) | ~$1.1-1.3/hr (est., matches prior same-zone/same-shape SPOT rate per RESET_HANDOFF §7; well under $5/hr cap) | 2026-07-06T18:55:16Z | P0-1 cold start DONE: nvidia-smi OK, EXCLUSIVE_PROCESS set, vendor pins restored, 27/27 BODY pytest GPU tests pass (0 skipped), inference smoke produced pred_keypoints_3d w/ GPU util confirmed. IP 34.143.175.207. Full detail: `runs/lanes/gpu_coldstart_20260706/report.md`. |

Fleet cost cap (owner 2026-07-06): ≤$5/GPU/hr, max 4 concurrent GPUs; teardown/DELETE the moment a
lane ends (idle spend never acceptable); 5th GPU or >$5/hr = `needs-purchase-approval` STOP.
Auth: owner gcloud refresh token (hello@; SA key creation org-blocked); dead auth = typed STOP for one owner login.
