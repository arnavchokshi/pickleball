# FIX ROUND for data_steward_ledger_20260721 — address ultra-review REJECT (runs/lanes/data_steward_ledger_20260721_review/review.json)

Same HARD RULES + FILE OWNERSHIP as spec.md. Read the full review JSON first. Core defect:
DISPATCH-GATE-FAIL-OPEN — the audit trusts caller-supplied declarations. Rebuild enforcement to be
LEDGER-DERIVED and fail-closed:

- Reachability: derive from the ACTUAL command argv — scan every argv token/path against ledger
  asset paths (and protected/compare registries); an argv referencing a protected/compare asset
  fails regardless of caller flags. trainer_reachable as caller input is abolished (or advisory-only).
- Identity/partition: train/holdout overlap computed from ledger lineage/family/partition data,
  never from caller strings. Mixed-identity assets (e.g., a gallery containing compare-only IDs)
  fail unless the command's asset selection provably excludes the quarantined identities.
- Authority: teacher-as-GT rejection derived from ledger label_authority vs the row's use, not the
  exact string 'human_gt'.
- State/rulings: enforce ledger state (REJECTED/QUARANTINED/BLOCKED refuse train use) and
  component_rulings (e.g., EVENT=FORBID) for the dispatching component.
- GPU/decoded: zero-decoded refusal must not depend on voluntary command.asset_ids — resolve
  consumed assets from argv; unknown/unresolvable data-looking argv paths = refusal, not a pass.
- Tests: replace self-certifying tests with the reviewer's eight independent probes as fixtures
  (protected_event_seed_50 GPU dispatch, eval_clips protected, NC-licensed, compare-only gallery,
  IYnbdRs1Jdk-derived court pack, state=REJECTED tt_sounds, zero-decoded with empty asset_ids,
  argv-pointing-at-protected-with-benign-contract) — ALL must refuse.

Keep: ledger schema/round-trip determinism, never-queued report, 25-asset snapshot (extend rows
with the fields the new enforcement needs). Report to report_fix1.json (schema-valid). Wide suite:
no NEW failures vs the known environmental set.
