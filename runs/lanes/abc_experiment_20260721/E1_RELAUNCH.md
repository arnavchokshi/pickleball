# E1 relaunch runbook — corrected B/C rerun (post abc_audiofix)

Preconditions: abc_audiofix_20260721 landed + ultra-reviewed + committed/pushed as <AUDIOFIX_COMMIT>.
Executor: Sonnet GPU lane. VM: pickleball-gpu-abc us-central1-a (STOPPED, disk kept). Budget: ~3.2h VM ≈ $3.5-4.

1. START + ACCESS
   gcloud compute instances start pickleball-gpu-abc --zone=us-central1-a
   IP=$(gcloud compute instances describe pickleball-gpu-abc --zone=us-central1-a --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
   ssh-keygen -R <old-ips>; ssh -o StrictHostKeyChecking=no -i ~/.ssh/google_compute_engine arnavchokshi@$IP
   Confirm boot rail armed (startup script shutdown -P). Confirm staged artifacts exist:
   ~/pbv_media_root/{143sf3gdwxsa,98z43hspqz13,st0epgnab7dr,td2szayjwtrj,utasf5hnozwz,xkadsq9bli3h}/max.mp4,
   ~/pbv_frame_times_root/*/frame_times.json, ~/audio_onsets/*_audio_onsets_v2.json, ~/ball_kinks/*_ball_inflections.json,
   ~/pickleball/runs/lanes/ball_event_abc_20260720/inputs/* and seed_20260720/A_owner_only/*.

2. CODE
   cd ~/pickleball && git fetch origin && git checkout <AUDIOFIX_COMMIT>
   (VM was at e3f47d651.)

3. REBUILD MANIFESTS (fresh dir, old abc_out untouched as evidence)
   Reconstruct the exact builder command from runs/lanes/abc_experiment_20260721/abc_out/VM_ABC_NEEDS.json
   (6 clips; one --media/--frame-times/--audio-onsets/--ball-velocity-kinks flag per clip),
   --teacher-manifest = the SAME corpus manifest as the seed-20260720 build (verify against
   abc_out provenance/SHA before building), --seed 20260720,
   --output-dir runs/lanes/abc_experiment_20260721/abc_out_v2
   HARD ASSERTS before training: audio-only accepted == 0; accepted rows == 1,189
   (773 @0.25-tier + 416 kink+audio rows tiered by per-video audio_time_shift_null);
   every input SHA recomputes; C mirrors corrected B row count.

4. TRAIN — SEQUENTIAL, never concurrent (concurrent B+C nearly breached the 90-min wall)
   B: frozen ABC_READY command, --pseudo-manifest runs/lanes/abc_experiment_20260721/abc_out_v2/arm_b_manifest.json,
      --out .../seed_20260720/B_pbvision_teacher (dir must be emptied first: rm -rf of the killed partials
      — they are preserved on the Mac in vm_pull/), nohup + bounded foreground poll.
      Require finetune_manifest.json completed_steps==target_steps==1000. Wall exit = failed arm, report honestly.
   C: same with arm_c_manifest.json -> C_placebo. Then sha256 all outputs.

5. SCORE (on VM, frozen threshold 0.5, checkpoint = each arm's best_event_head_finetuned.pt)
   eval_event_head.py --mode owner-val --device cuda --manifest runs/lanes/ball_event_abc_20260720/inputs/owner_102_manifest.json \
     --arm {A|B|C} --seed 20260720 --checkpoint <arm-best> --threshold 0.5 --out <arm>_owner41.json
   A scored too (checkpoint already on disk) so all three JSONs share one scorer run/env.

6. E1 SCREEN (computed from the three JSONs; the 3-seed abc_decision_gate.py stays untouched until E2)
   PASS requires ALL: (B-A) macro-F1@±2 >= +0.10; B > C; B negative FP <= 2/22 AND <= A+1;
   B timing p90 non-worse than A; B full-video event rate in 0.3-1.0/s.
   Verdict: DIRECTIONAL_PASS -> E2 (seeds 20260721/20260722). Otherwise EVENT_PBV_SEED1_NO_LIFT
   -> experiment closes honestly, no more seeds.

7. PULL + STOP
   sha256 manifest on VM -> pull abc_out_v2 + seed_20260720 B/C + eval JSONs + screen JSON to
   runs/lanes/abc_experiment_20260721/vm_pull_v2/ -> verify two-sided -> gcloud stop (disk kept
   until E2 concludes or experiment closes). Ledger row + spend estimate.
