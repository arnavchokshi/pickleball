# E0 verdict — seed-20260720 A/B/C recovery + method audit

Date: 2026-07-21T20:05Z. Status: `VERIFIED=0`. Plan ref: `runs/regroup_20260721/EXACT_PLAN.md` §3.1 E0.

## VERDICT: `METHOD_INVALID_AUDIO_ONLY=292`

The B-arm manifest materialized on the VM (`abc_out/arm_b_manifest.json`, built at repo
commit `e3f47d651` by `scripts/racketsport/build_abc_arm_manifests.py`) accepted **292 rows
whose ONLY independent agreement family is `audio_onset`** (pseudo_weight 0.25). This
violates the binding ruling `PBV-FULL-USAGE-20260720` quality correction (EXACT_PLAN §2.1):
an audio-only match may not make a row eligible; audio alone stays weight zero.

Audit of `vm_pull/abc_out/agreement_decisions.jsonl` (2,192 decisions):

| bucket | count |
|---|---|
| accepted into arm B | 1,481 |
| — accepted with audio_onset ONLY (**invalid**) | **292** |
| — accepted with ball_velocity_kink only | 773 |
| — accepted with both families | 416 |
| rejected: zero_independent_agreements | 585 |
| rejected: rejected_low_teacher_confidence | 126 |
| accepted weights | 0.25 -> 1,065 rows; 0.5 -> 416 rows |

Defect location: `scripts/racketsport/build_abc_arm_manifests.py` — acceptance rule is
`accepted = needs_agreement_pass and count > 0`, so a single-family audio_onset match
qualifies. C mirrors B's rows, so both arms are method-invalid.

**Consequence:** the B/C training runs in flight at audit time (launched 18:34Z on
`pickleball-gpu-abc`, ~75 min elapsed of a 90-min wall) were killed at 19:52Z per E0
("Do not score a method-invalid B/C as causal evidence"). Their partial outputs
(`best_event_head_finetuned.pt` only, no `finetune_manifest.json`) are preserved in
`vm_pull/` as forensic evidence, never scoreable. A corrected B retains 1,189 eligible
rows (773 kink-only + 416 both) — the experiment remains viable.

## Also established (recovery facts)

- **Arm A recovered and hashed**: `completed_steps=1000 target_steps=1000 status=complete`,
  `best_val_macro_f1_at_2 = 0.0` — **0.0 at every one of 11 validations** (steps 0..1000),
  final train losses ~0.005, `best_val_max_positive_class_probability` 0.799, steps/s 0.279
  (elapsed 3,585s). The previously user-reported A=0.0 is now locally verified from durable
  artifacts. Owner rows 61 train / 41 val as frozen.
- **Two-sided integrity**: 31/31 files sha256-verified VM->local, 0 mismatches
  (`vm_pull/vm_pull_sha256.txt`; verification run 2026-07-21T20:00Z).
- Pulled: `seed_20260720/{A_owner_only,B_pbvision_teacher,C_placebo}`, frozen `inputs/`
  (T20 checkpoint, owner_102 manifest, original teacher/placebo manifests),
  `abc_out/` (arm manifests + agreement_decisions + input_bindings + VM_ABC_NEEDS),
  `corpus_out/` + `corpus_out_6clip/`, `train_A_20260720.log`.
- **VM**: `pickleball-gpu-abc` STOPPED (TERMINATED, disk kept) at 20:03Z after pull
  verification. Boot rail had been re-armed +120 min as a belt-and-braces bound before pull.

## Next (E0 repair path, per plan)

1. Codex lane fixes `build_abc_arm_manifests.py`: `accepted` requires a
   `ball_velocity_kink` agreement; audio_onset alone -> weight 0 / ineligible; audio may
   count toward the 2-family 0.5 weight tier only where its per-video match rate beats a
   preregistered time-shift null (new recorded field); tests added.
2. Restart VM, rebuild B/C manifests with the fixed builder (media/PTS/audio/kink
   artifacts persist on the VM disk), rerun B and C **sequentially** (concurrent B+C
   shared one A100 and was on pace to breach the 90-min wall; A solo = 60 min).
3. Score A/B/C owner-41 at frozen threshold 0.5; run `abc_decision_gate.py` E1 screen.

A is unaffected by the defect (no pseudo manifest) and is NOT rerun.
