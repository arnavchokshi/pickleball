# LANE: beststack_verify_20260709 — ADVERSARIAL VERIFY of the best-stack landing (commit range 9308c875d..a8082598e)

## ROLE + HARD RULES
You are an INDEPENDENT verifier with fresh eyes. Your job is to BREAK the landing, not to confirm it.
The implementing lane self-verified; you attack what self-verification structurally cannot see:
tests that don't test the right thing, vacuous assertions, replicas of production paths, silent
default drift. NO edits to repo source/tests/docs — you may write ONLY under
runs/lanes/beststack_verify_20260709/ (your own executable harness + evidence) and /tmp copies.
NO branches, NO commits. Protected clips untouched. If you need a repo change to prove a defect,
write a FAILING test file under YOUR lane dir + inline hunks in the report.

## WHAT LANDED (read first)
git show a8082598e --stat; runs/lanes/beststack_core_20260708/report.json + report_r2.json;
configs/racketsport/best_stack.json (rev 3); the doctrine docs (NORTH_STAR Part IV rule 15).
Claimed: single default-selection surface; sanctioned default changes = mesh byte-budget-300 +
events-before-frames ONLY; CLI-override-wins; server parity with one declared override; BODY
local==remote unification; no-orphan audit; fixture diffs fully attributable.

## ATTACKS (each = executable evidence, PASS/FAIL verdict; your harness stays UNMODIFIED once written)
A1 Fixture attribution: diff the deterministic no-flag fixture behavior at a8082598e vs 9308c875d
   (worktree-detached copies): EVERY changed field must trace to the two sanctioned changes. Any
   third behavioral change = FAIL with the field named.
A2 Resolution precedence: prove explicit CLI flags beat manifest on ≥3 surfaces (mesh budget, WASB
   ckpt, association profile); prove a deleted/missing manifest entry on a decision path = HARD
   error, not silent constant fallback (mutate a /tmp manifest copy via env/arg redirection if the
   loader supports it, else spawn python with a patched path — do not edit the repo manifest).
A3 Events-before-frames correctness: on the deterministic fixture, events/contact outputs at
   a8082598e must be CONTENT-IDENTICAL to 9308c875d (ordering changed, results must not); mesh plan
   must actually become contact-dense when contact windows exist. Either deviation = FAIL.
A4 Vacuousness/mutation testing: (a) flip mesh.byte_budget_mib to 250 in a scratch copy — does the
   equivalence/resolution test catch it? (b) add a fake DEFAULT_FAKE_MODEL constant to a scratch
   process_video copy — does the no-orphan audit fail? (c) break the declared server_override —
   does the parity test fail? Run tests against scratch copies (pytest from the copied tree).
A5 Parity realism: confirm tests/server/test_best_stack_parity.py exercises the REAL GpuRunRequest /
   daemon default path (import-level), not a re-declared dict replica.
A6 BODY unification: confirm remote fixture-copy fallback semantics preserved (the recon claim) by
   tracing remote_body_dispatch.py resolution with manifest values empty vs set.
A7 Manifest truth: rev-3 semantic spot-audit vs boards — ball default ckpt UNCHANGED (raw WASB
   zero-shot), P2-2 DORMANT citing GATE-1b FAIL 262mm, byte-budget provenance cites the outdoor
   4.1x proof, PENDING entries all carry non-null gates.
A8 Doctrine docs: the 4 sentinels exist at HEAD and rule-15 text does not contradict the manifest
   invariants (read both, quote any drift).

## REPORT (self-write to runs/lanes/beststack_verify_20260709/report.json, lane_report schema shape)
objective_result = PASS only if ALL attacks fail to break it; otherwise FAIL/PARTIAL with per-attack
verdict table + executable repro paths under your lane dir. HONEST ISSUES unsoftened. Full-suite line
not required (read-only lane) — but list any test you ran with its census.
