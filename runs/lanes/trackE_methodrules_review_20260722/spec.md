# trackE_methodrules_review_20260722 — XHIGH adversarial review of the method-rules skill update

Codex gpt-5.6-sol, effort xhigh (standard review tier per the 2026-07-22 economy ladder — this is
governance text, no production code, no judge arming). You are the DECIDING reviewer before Track
E commits. Adversarial: correctness, spec-match, hidden contradictions, honesty.

## Scope under review (exact)
1. `.claude/skills/run-lane/SKILL.md` working-tree diff vs HEAD (`git diff HEAD -- .claude/skills/run-lane/SKILL.md`).
   Authorship note: the 6 STANDING METHOD RULES + the LONG-RUNNING CODEX DISPATCH block came from
   the trackE_methodrules_20260722 lane; the speed/tier retier (NORMAL-only, xhigh-default,
   ultra=blast-radius-reviews-only), the >120s/SALVAGED_VERDICT additions, and the SUBAGENT OPS
   block are Track E coordination edits per owner policy 2026-07-22. Review the WHOLE final text
   as one artifact.
2. `runs/lanes/person_mixed_20260722/NAMED_NEGATIVE.md` (new file).
3. `runs/lanes/trackE_methodrules_20260722/{spec.md,RESUME_BRIEF.md,report.json}` — did the lane
   deliver its spec + all addenda; is its attribution table honest (43 failures, self-labeled
   iteration evidence under its own rule 6)?

## Obligations
- Verify every required content item is PRESENT and internally consistent: 6 standing rules
  (exposure-matched arms w/ PERSON precedent; per-family metrics; ensemble teachers; license=FYI
  w/ protocol-quarantine carve-out; attribution convention w/ the 3 sources + reject/never-blocks
  semantics; immutable-revision promotion runs w/ A2 incident) + dispatch gotchas (nohup-detached,
  ppid test, >120s wrappers, SALVAGED_VERDICT) + SUBAGENT OPS (isolation=worktree at spawn; commit-inside-worktree/copy-out-before-complete artifact preservation; unblocks-as-respawn-never-peer-chat) + NORMAL-speed/tier-economy text.
- Hunt CONTRADICTIONS with the rest of SKILL.md and with AGENTS.md/CLAUDE.md (e.g. stale 'ultra'
  references elsewhere in the file that now conflict with the economy ladder; the header line
  still says reviews happen — make sure no text still mandates ultra for ordinary lanes).
- NAMED_NEGATIVE.md fidelity vs runs/lanes/person_mixed_20260722/gpu_phase_report.json +
  runs/handoff_20260722/ORCHESTRATOR_STATE.md section 5: every number verbatim-checkable
  (13.5x, 6.75x, 1066/14400, 21320/288000, -0.1924, -0.0842, +0.046). Signed deltas MUST appear.
- Confirm zero out-of-scope file changes by the lane (its report claims none; verify via git
  status attribution against inflight_lanes.md).
- Wide-suite: do NOT rerun it. The lane's 4432P/43F iteration run + attribution table is the
  record; verify the attribution table's logic spot-wise (>=3 failures traced to their attributed
  causes), per the standing attribution convention.
## Output
runs/lanes/trackE_methodrules_review_20260722/{review.json,findings.md} + report.json per schema:
verdict ADOPT_COMMIT | ADOPT_WITH_FIXES(list) | REJECT(blockers) + exact commit file-list.
No fixes yourself; findings only. Long commands under nohup/timeout wrappers.

## FIX-VERIFICATION ROUND (Track E, post-ADOPT_WITH_FIXES)
Your F1-F6 have been applied by Track E, plus three coordinator additions (log-sweep review
obligation; contamination-adjacent lane standard; revert discipline) and the worktree
artifact-preservation phase-boundary rewrite (F4 shape). Re-review ONLY the current combined
SKILL.md working-tree text + runs/lanes/trackE_methodrules_20260722/SUPERSESSION_NOTE.md (F6):
verify each F1-F6 is genuinely resolved (not cosmetically), the new additions introduce NO fresh
contradictions, and issue the final verdict + exact commit file-list. Write review_fix.json +
report.json (overwrite).
ADDENDUM to fix-verification: one more Track E coordination addition is in the text since your
dispatch — the CONTENT-FILTER GOTCHA (neutral data-integrity phrasing for integrity-test review
specs + banked pre-verified context; Track B 157k-token loss). Include it in scope.

## ROUND 3 — BOUNDED FINAL RE-REVIEW (current-hash only)
Track E applied your review_fix.json residuals: F1 (both undefined section refs removed at what
were lines 8/13; hard-rules reading set now names NORTH_STAR_ROADMAP.md, AGENTS.md, the relevant
RUNBOOK.md section, and inflight_lanes.md), F2 (TIER variable — literal placeholder gone, block
now zsh-parses clean with only the pre-existing <short-name> placeholder substituted; ppid check
moved to an explicitly SEPARATE later shell with the reparent-after-launcher-exit rationale
stated), F6 (SUPERSESSION_NOTE.md records the final combined numstat + final SKILL.md sha256 and
binds to review.json + review_fix.json + this round's record), F7 (objective_result enum stated
as PASS | BLOCKED | PARTIAL; PASS/FAIL reserved for acceptance rows). Scope: verify these four on
the CURRENT hashes only — no wide rerun, no reopening PASSed items. Write review_fix2.json with
final verdict + confirm/amend the 13-file commit whitelist. Overwrite report.json (schema-valid).
