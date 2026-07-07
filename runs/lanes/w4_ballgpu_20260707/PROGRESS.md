# w4_ballgpu PROGRESS — 2026-07-07T20:28Z (post-stall reconcile)
- STEP: data staging just completed; NO training launched yet — VM GPU was IDLE for the full 2h43m uptime so far (env setup ~20min + WASB sha-verified prestage + code-bundle sync w/ version_stamp.json + 1.1GB harvest transfer that fought a REPRODUCIBLE macOS ssh EMSGSIZE flake needing bwlimit+append-verify+50MB chunking, then the API-spend stall). Honest cost finding: ~2h of that uptime was transfer/stall, not compute.
- DONE: VM pickleball-h100-w4ballgpu up (H100, EXCLUSIVE_PROCESS, preemption watcher); venv torch 2.9.1+cu129 OK; WASB sha256 MATCH 9d391239… (also copied to Mac models/checkpoints/wasb/ — LOCAL BLOCKER closed); repo @ 5b268aa6d bundle-synced, 10-file md5 stamp all-match (version_stamp.json); owner CVAT export + stage-1 latest.pt + eval clips/labels landed; all 22 harvest chunks byte-complete on VM.
- NEXT (immediate): reassemble+md5-verify harvest tar → extract → build-sst-manifest (protected-hash 35/0 proof BEFORE any training) → seed fine-tune (nohup on VM so Mac-side outages can't kill it) → bridge-score stage-1+seed → SST ruling+train → bridge → threshold sweep → disagreements → pull+md5 → DELETE VM.
- ETA: seed fine-tune ~1830 steps (486 CVAT rows ×30ep÷8) ≈ fast; SST size unknown until manifest built (could be large — will report the real step count before launching); bridge runs ~min each. Optimistic lane completion in ~2-4h if SST corpus is modest; will update here after manifest.
- COST SO FAR: 2h43m × ~$2-4/hr ≈ $5.50-11 (incl. idle window; full-lifetime accounting will be in the final report).
- BLOCKERS: none — proceeding.

## UPDATE 2026-07-07T20:52Z
- SST manifest BUILT: 40 clips, 58,353 samples; protected-hash assert = **35 eval hashes / 0 collisions** (sample_count 1823 video dhashes) — banked before any training, per spec.
- Real SST step math: 30 epochs ≈ 220k steps → manager 12k-step cap BINDS; SST student will run `--steps 12000` (~26 min proven H100 budget).
- ffmpeg was missing on the fresh VM (manifest build failed fast once); apt-installed, relaunched, clean.
- NOW: seed fine-tune launching (486 owner rows → 1830 steps, nohup).

## UPDATE 2026-07-07T21:38Z — CRITICAL FINDING + SST launch
- SEED FINE-TUNE DONE (1830 steps, 34 min, init key-diff empty, ckpts banked; loss flat-noisy 0.00110→0.00117).
- CRITICAL CODE FINDING: harness-trained ckpts CANNOT be validly bridge-scored. Harness preprocessing = plain resize + /255 (roboflow_corpus.py:981); bridge official-inference = affine warp + ImageNet mean/std (wasb_adapter._preprocess_wasb_window). Lane ckpts through the bridge → degenerate output (stage-1: constant point, F1 0.0 both clips, hidden-FP 1.0). CONTROL PROVES bridge healthy: official tennis ckpt through the identical path scores F1@20 0.714/0.783 (burl/wolv). Also hit+worked-around a second bug: wasb_adapter torch.load lacks weights_only=False → torch 2.9 refuses lane ckpts (PosixPath in payload); sanitized tensor-only copies (bitwise-verified) used. BOTH need a manager-routed code fix; bridge scores for lane candidates are BLOCKED until then.
- SST-init ruling: IF (seed >= stage-1 on bridge) is UNMEASURABLE (above) → ELSE branch fires mechanically → SST student inits from STAGE-1 checkpoint.
- SST student LAUNCHING: --steps 12000 (manager cap; 30 epochs would be ~220k). HONEST ETA WARNING: video-seek loader measured ~0.9 steps/s on the seed run → 12k steps ≈ 3.5-4h (manager's ~26min estimate was for the corpus-image loader). checkpoint-every 500 = an early kill still banks progress. Flag me via SendMessage to shorten if 4h × ~$2-4/hr is not worth it given bridge-validation is blocked anyway.

## FINAL 2026-07-07T23:45Z — LANE COMPLETE
VM DELETED (list-confirmed). All artifacts pulled + md5-verified. See REPORT.md.
