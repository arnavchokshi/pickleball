# ULTRA ADVERSARIAL REVIEW of lane abc_audiofix_20260721 (read-only; do NOT modify any file)

HIGHEST-STAKES review of this wave: these two files gate the EVENT A/B/C causal experiment. A wrong
ACCEPT means the corrected B/C arms train on an invalid manifest and burn the last ~$3 of VM budget;
a missed loophole means the causal claim is contaminated.

Ground truth: runs/lanes/abc_audiofix_20260721/spec.md (contract), report.json (claim), the diff on
scripts/racketsport/build_abc_arm_manifests.py + tests/racketsport/test_abc_arm_manifests.py, log.txt.
Doctrine: runs/regroup_20260721/EXACT_PLAN.md §2.1 + §3.1 E0/E1; runs/lanes/abc_experiment_20260721/E0_VERDICT.md;
runs/lanes/w1b_abc_loader_20260721/VM_ABC_RUN.md §5-6.

Attack surfaces to probe hard:
1. Eligibility: can ANY code path accept a row without a ball_velocity_kink agreement? (audio-only,
   zero-agreement, exotic family values, duplicate families, case variants.)
2. Weight: can audio lift weight to 0.5 in a video whose null was NOT beaten? Is the null computation
   itself sound — deterministic offsets, |offset|>=1.0s enforced, circular wrap correct, observed vs
   max(null) comparison strict, per-VIDEO not global? Could the null be gamed by degenerate onset
   lists (empty, single onset, dense onsets ~29 Hz where null≈observed)?
3. C placebo integrity: does C still mirror corrected B exactly (rows/pixels/classes/weights/
   loss-valid frames) with only focal-time shuffling within source/rally? Did the fix change any C
   semantics?
4. Determinism + SHA chain: byte-identical rebuilds; input_bindings/dependency hashes still enforced;
   no relaxation of the materializer refusal checks in VM_ABC_RUN §5-6.
5. The 1,189 recount: is the offline recount in the tests actually applying the NEW rule to the real
   pulled decisions file (sha 3a3463565e57a5cd909eaad01f2ddf6fa66f23468396f7162a94c85f8b1bf4f1), or a
   circular restatement?
6. CLI/contract: flags unchanged; VM rebuild command from VM_ABC_NEEDS.json will work unmodified.
7. Wide-suite attribution: 31 failures, 30 reproduced without the diff, 1 transient timeout — challenge
   if anything could actually be caused by these two files.

VERDICT in final JSON: ACCEPT | ACCEPT_WITH_FIXES (exact fixes) | REJECT (why). Be adversarial.
