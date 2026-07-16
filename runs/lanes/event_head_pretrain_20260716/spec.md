# Lane event_head_pretrain_20260716 — Track G GPU ops: public-data pretrain + pbvision anchor inference (Sonnet, network-capable)

You are a GPU-ops lane for the DinkVision pickleball repo at /Users/arnavchokshi/Desktop/pickleball.
You hold GPU morning slot 1-of-2 (owner directive 2026-07-16). VERIFIED=0 binding; this produces a
BASELINE pretrain checkpoint + anchor candidates, never a promotion.

## HARD RULES
- Budget: one spot GPU, ≤$5/hr, HARD CAP $10 total (the ORIGINAL user-authorized cap — a
  coordinator-relayed $15 raise was not honored: only the user/permission system authorizes spend).
  Target total VM wall ≤4h; the on-VM rail is 5.5h (worst case A100 ≈$1.5/hr × 5.5h = $8.25 < $10).
- ON-VM TEARDOWN RAIL IS MANDATORY AT PROVISION (yesterday a VM stall burned money; Mac-side
  watchers die on laptop sleep): within 2 minutes of first SSH, run `sudo shutdown -P +330` on the
  VM and VERIFY it is scheduled (`cat /run/systemd/shutdown/scheduled` or `sudo shutdown --show`
  equivalent); ALSO install an idle watchdog (background loop: if no python train/infer process
  for 25 consecutive minutes → `sudo shutdown -P now`). Show both as evidence BEFORE starting any
  data transfer or training. If you cannot arm the rail, DELETE the VM immediately and stop.
- Every gcloud create/start uses labels fable-lane=event_head_pretrain_20260716,
  fable-fleet=pickleball,owner=arnavchokshi; spot/preemptible;
  --instance-termination-action=STOP; metadata-from-file startup-script=scripts/fleet/lane_vm_startup.sh
  (for create).
- NO training on: Tier-A bootstrap labels (data/event_bootstrap_20260713 — dead label source), the
  50-row protected seed, owner harvest videos, or the pb.vision video (R&D reference INFERENCE
  ONLY — never training, never redistributed).
- No repo source edits. Your writes on the Mac side go ONLY under
  runs/lanes/event_head_pretrain_20260716/ (+ rsync temp). The manager owns ledger/commits.
- Honest reporting; real unpiped exit codes; two-sided md5 for every artifact pulled back.

## Provision (auth is LIVE, verified 09:3x; fleet EMPTY confirmed)
1. Preflight: `gcloud compute instances list --filter=labels.fable-fleet=pickleball` (reconcile).
2. Try REUSE first: `gcloud compute instances start pickleball-a100-fleet1 --zone=asia-southeast1-a`
   (TERMINATED, disk intact, baked fleet env). If it starts: update its fable-lane label to this
   lane, verify `nvidia-smi` + torch CUDA, arm the rail. Spot A100 ase1 ≈ $1.1-1.5/hr.
3. Fallback (stockout/quota): create fresh spot g2-standard-8 (1x NVIDIA L4, ≈$0.3-0.7/hr) in
   us-central1-a → -b → -c, image family common-cu124 project deeplearning-platform-release,
   boot disk 200GB pd-balanced. One attempt per zone, no retry storms.
4. Record for the manager (report lines): vm name/zone/type/$hr estimate/created_at + rail
   evidence. The manager writes the fleet ledger row.

## Data + code to the VM (uplink is assumed-unreliable — chunked, resumable)
- Code: tar (from the Mac working tree) threed/racketsport/event_head/,
  scripts/racketsport/{build_event_head_dataset,train_event_head,eval_event_head,build_event_head_anchor_candidates}.py,
  plus tests/racketsport/fixtures/event_head/ → scp to VM; VM: python3 -m venv or use the image's
  torch env; pip install opencv-python-headless numpy if missing. Recreate the repo-relative layout
  under ~/pickleball_g/ (package import path matters: threed/racketsport/event_head).
- Labels (small, tar+scp): data/event_public_20260713/jhong93_spot/data/tennis/,
  openttgames/markup/extracted/, extended_openttgames/data/, plus each dataset's manifest.json.
- jhong93 pilot videos (963MB): rsync -av --partial --append-verify --bwlimit=8000 the 6 files in
  data/event_public_20260713/jhong93_spot/videos_pilot/ (retry loop ≤5; 50MB-chunk split+cat
  fallback if append-verify stalls twice).
- OpenTT videos: VM-side fetch https://lab.osai.ai/datasets/openttgames/data/game_4.zip and
  test_2.zip, unzip into the mirrored layout (videos/game_4.mp4, videos/test_2.mp4). Fallback:
  rsync from Mac (they exist locally).
- pbvision demo video: VM-side fetch https://storage.googleapis.com/pbv-pro/83gyqyc10y8f/max.mp4,
  verify sha256 == 272a2132… (full value in data/pbvision_11min_20260713/video_provenance.json —
  read it before you go). Fallback: scp the local file (114MiB).

## Run (all VM-side steps nohup-detached; poll with bounded loops; never passive-wait)
1. Builder on VM: build_event_head_dataset.py --public-root <mirror> --out manifest.json — counts
   must reconcile with the Mac manifest (spot-check totals).
2. PROBE: train_event_head.py --full --device cuda --weights imagenet <pinned config: image-size
   224, window-frames 64, batch fit-to-memory> for 100 steps → record steps/s.
3. CAP FORMULA (pre-approved, no round-trip): train_steps = min(20000,
   floor(2.5h * 3600 * measured_steps_per_s)). Run --full with --max-wall-minutes 150 and
   --val-every so ≥4 val evals happen. Keep best + last checkpoints.
4. Public held-out eval on VM (eval_event_head.py --mode public w/ the best checkpoint if the
   mode runs there; if its public-data paths resist the mirror layout, pull checkpoints back and
   note it honestly — the Mac re-runs public eval locally).
5. Anchor inference: build_event_head_anchor_candidates.py --checkpoint <best> --video max.mp4
   --video-provenance pbvision_demo_rd_reference_only --device cuda --out
   anchors/pbvision_11min_event_head_anchors.json (typed schema; NEVER training).
6. Pull back (two-sided md5): best.pt, last.pt, train_manifest.json, probe + train logs, public
   eval JSON, the anchor JSON → runs/lanes/event_head_pretrain_20260716/{checkpoints,eval,anchors,logs}/.
7. TEARDOWN NO MATTER WHAT: `gcloud compute instances delete <vm> --zone=<zone> --quiet`, then
   `gcloud compute instances list --filter=labels.fable-fleet=pickleball` AND
   `gcloud compute disks list` — paste both outputs. If the reused a100-fleet1 was used, DELETE is
   replaced by STOP ONLY IF the manager pre-authorized preserving the historical disk — default for
   this lane: if reused, `gcloud compute instances stop` + report (its disk is a historical
   snapshot source; do not delete it); if fresh-created, DELETE + confirm 0 lane disks remain.
- Wall/cost accounting in the report: provision→teardown wall clock, $/hr band, estimated spend.

## Report (structured, honest)
Provision path taken + rail evidence; probe steps/s + chosen step cap; final train/val losses +
val F1@±2; artifact list w/ md5 pairs; anchor event counts (HIT/BOUNCE); teardown outputs; spend
estimate; honest issues (expect weak transfer — do not dress it up).
