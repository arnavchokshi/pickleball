# Track E method-rules deciding review

Reviewed: 2026-07-22

Branch: `main`

HEAD: `7cfa02258fbfb69ff3f43e10a62afcb46f12fb3b`

SKILL.md snapshot SHA-256: `df576d7674aeb51169192473cd0b6209a7dbe859bef93b81171619bb04c57ddd`

Product status: `VERIFIED=0`

## Verdict

**ADOPT_WITH_FIXES. Do not commit the current snapshot.**

All requested new content is present. The named negative is numerically faithful, the lane delivered the original spec and all addenda, the report is honestly `PARTIAL`, and the recorded 4432P/43F/25S suite is correctly labeled iteration-only. Six governance contradictions still need textual correction before the combined skill is safe to adopt.

## Required fixes

| ID | Severity | Finding | Required correction |
|---|---|---|---|
| F1 | HIGH | Lines 11-17, 80, 92, 102, and 159 still invoke `FABLE_OPERATING_MANUAL`, `BUILD_CHECKLIST`, manual sections, and a task board. The root documents are absent; their only copies are archived. `AGENTS.md:6-10` and `CLAUDE.md:3-15` make the North Star the sole current authority and forbid reviving a build checklist or operating manual. | Replace them with `NORTH_STAR_ROADMAP.md`, `AGENTS.md`, the relevant `RUNBOOK.md` section, and manager inflight/fleet files only for volatile coordination. Remove the dated BUILD_CHECKLIST deliverable. |
| F2 | HIGH | The canonical command at line 108 hardcodes `high` although lines 138-147 make `xhigh` the default. The command is foreground and records no PID, while line 126 still says `run_in_background: true` directly after forbidding harness-background long lanes. | Use `xhigh` or an explicit chosen-tier placeholder. Add an executable nohup-detached/PID/PPID example for long lanes. Remove or narrowly qualify `run_in_background: true`. |
| F3 | HIGH | Rule 5 says attributed noise never blocks acceptance, but lines 90-92 and `lane_report.schema.json` forbid `objective_result=PASS` when failures remain and are not all proven pre-existing. Ownership and `KNOWN_ATTRIBUTED(...)` labels are leads, not causal proof; a lane can break an unowned test. | Require move-aside, clean-HEAD, or independently evidenced equivalent reproduction for nonblocking certification. Keep lane acceptance rows separate if useful, but require overall `PARTIAL/BLOCKED` whenever the schema condition is unmet. |
| F4 | HIGH | Line 80 makes no commits a standing lane rule, while lines 127-133 make an inside-worktree commit one preservation route. Rule 6 also requires a pinned commit that cannot contain an uncommitted pre-review candidate. | State the phase boundary: under default no-commit specs, parent copy-out before completion is mandatory; commits require explicit spec authority. Promotion-grade runs happen on the integration commit or an expressly authorized temporary immutable commit. |
| F5 | MEDIUM | Lines 90-92 say Fable rules only from `report.json`; lines 123-125 allow `SALVAGED_VERDICT.md` without requiring a replacement structured report. | Make the salvage document provenance input only and require the countersigning coordinator to emit a schema-valid `report.json` before ruling. |
| F6 | MEDIUM | The lane report accurately describes its own +40/0 slice, but the final combined skill is +72/-18 and includes later Track E coordination edits. Its `next` still mandates an ultra review even though this is a standard xhigh review under the new ladder. | Add a post-lane coordination/supersession note and the combined numstat to `report.json`; change the next-review tier to xhigh. Preserve the historical spec rather than silently rewriting it. |

## Required-content audit

| Obligation | Result | Evidence |
|---|---|---|
| Exposure-matched arms with PERSON precedent | PASS | `SKILL.md:46-52` |
| Per-family metrics | PASS | `SKILL.md:53-55` |
| Two-plus independent teachers with explicit agreement | PASS | `SKILL.md:56-58` |
| License FYI only with protocol quarantine intact | PASS | `SKILL.md:59-61` |
| Three attribution sources plus reject/nonblocking semantics | PASS as requested text; correctness fix F3 required | `SKILL.md:62-70` |
| Immutable-revision promotion suite with A2 incident | PASS as requested text; phase fix F4 required | `SKILL.md:71-77` |
| Nohup/PID and PPID exposure check | PASS as requested text; executable dispatch fix F2 required | `SKILL.md:114-119` |
| Commands over 120 seconds use wrapper/output file | PASS | `SKILL.md:120-123` |
| `SALVAGED_VERDICT.md` pattern | PASS as requested text; report-only fix F5 required | `SKILL.md:123-125` |
| Worktree isolation at spawn | PASS | `SKILL.md:127-129` |
| Commit/copy-out artifact preservation | PASS as requested text; commit-policy fix F4 required | `SKILL.md:129-133` |
| Unblock by respawn/coordinator, never peer chat | PASS | `SKILL.md:133-135` |
| NORMAL speed, xhigh default, ultra only for blast-radius review | PASS | `SKILL.md:138-157` |
| No stale ordinary-lane ultra mandate elsewhere in final SKILL.md | PASS | Remaining `ultra` references are the ladder and blast-radius exception. |

## Named-negative fidelity

`NAMED_NEGATIVE.md:7-19` is faithful to `gpu_phase_report.json` and `ORCHESTRATOR_STATE.md` section 5:

| Fact | Recorded | Check |
|---|---:|---|
| Control rows | 1,066 | GPU report says 1066. |
| Mixed exposures | 14,400 | GPU report says 14400. |
| Update ratio | about 13.5x | 14400 / 1066 = 13.5084. |
| Anchor ratio | 6.75x | 7200 / 1066 = 6.7542. |
| Control total exposures | 21,320 | 1066 x 20. |
| Mixed total exposures | 288,000 | 14400 x 20. |
| od8al precision delta | -0.1924 | Exact signed source value. |
| od8al F1 delta | -0.0842 | Exact signed source value. |
| hemel F1 delta | +0.0460 | Signed rendering of source 0.046 and handoff +0.0460. |

No sign was dropped. The “undercontrolled” interpretation is appropriately limited to unattributability among the three named hypotheses; it does not claim which hypothesis is true.

## Lane delivery and report honesty

- The original four rules and named-negative record are present.
- Resume addenda for PPID, rule 5, rule 6, commands over 120 seconds, and salvage are all present in the final skill.
- The later NORMAL/xhigh retier and SUBAGENT OPS block are owner-attributed coordination edits and were reviewed as part of the combined artifact.
- `report.json` is honest where it matters: `objective_result=PARTIAL`, 4432 passed, 43 failed, 25 skipped, `failures_all_preexisting=false`, and explicit `ITERATION EVIDENCE ONLY` wording.
- The +40/0 line describes the lane-authored slice, not the final combined diff. F6 requires that scope to be explicit before commit.

## Wide-suite attribution spot-check

The wide suite was not rerun. Direct parsing of `wide_postchange.junit.xml` confirmed 4,500 tests, 43 failure cases, and 25 skips. The set-diff arithmetic is coherent: 32 of 33 baseline failures persisted, 11 appeared, and one disappeared, giving 43; collection grew by 119 tests.

| Failure | Recorded failure | Attribution check | Result |
|---|---|---|---|
| `test_best_stack_manifest_integrity` | revision 15 versus expected 14 | Track C's spec owns `best_stack.json`; the live Track C diff changes revision 14 to 15. | Supported. |
| `test_direct_cli_refuses_compare_only_id_before_path_access` | stale expected refusal string versus registry refusal | `ball_audio_ttcal_20260722/spec.md:21-25` owns the CLI and its test. | Supported. |
| `test_train_court_model_v2_cli_cpu_smoke_beats_random_init` | subprocess timeout after 540 seconds | The independent data-debt wide run records the same runtime-sensitive node; method governance files cannot enter this path. | Supported as unrelated/runtime noise, not proven at clean HEAD. |
| `test_wolverine_seg6_fixture_falls_back_to_anchor_bvp_and_render_samples_stay_in_bounds` | `degraded` versus `ran` | The independent data-debt wide run records the same node; no method-rules file enters BALL execution. | Supported as unrelated/runtime noise, not proven at clean HEAD. |

The table is honest because it does not set `failures_all_preexisting=true` and does not promote the shared-tree run. F3 is still necessary: the standing rule must not turn future labels into self-certifying proof.

## Scope fence

The surviving worktree attribution supports zero lane writes outside its declared fence:

- the only tracked diff inside the method-rules fence is `.claude/skills/run-lane/SKILL.md`;
- `NAMED_NEGATIVE.md` and the method-rules lane directory exist under their declared ignored run paths;
- `inflight_lanes.md:559-570` assigns these paths solely to `trackE_methodrules_20260722` and separately assigns the ledger/auditor paths to `trackE_datadebt_20260722`;
- the user-provided authorship note accounts for the later Track E coordination hunks inside the same SKILL.md path.

Because this is a shared dirty checkout, status attribution is strong evidence about the surviving diff, not cryptographic proof of every historical process write.

## Exact commit file list after F1-F6 and bounded rereview

1. `.claude/skills/run-lane/SKILL.md`
2. `runs/lanes/person_mixed_20260722/NAMED_NEGATIVE.md`
3. `runs/lanes/trackE_methodrules_20260722/spec.md`
4. `runs/lanes/trackE_methodrules_20260722/RESUME_BRIEF.md`
5. `runs/lanes/trackE_methodrules_20260722/junit_failure_set_diff.json`
6. `runs/lanes/trackE_methodrules_20260722/wide_suite_attribution.json`
7. `runs/lanes/trackE_methodrules_20260722/report.json`
8. `runs/lanes/trackE_methodrules_review_20260722/spec.md`
9. `runs/lanes/trackE_methodrules_review_20260722/review.json`
10. `runs/lanes/trackE_methodrules_review_20260722/findings.md`
11. `runs/lanes/trackE_methodrules_review_20260722/report.json`

Do not include PIDs, logs, either raw wide JUnit XML, or `RESUME.md`. No commit is authorized until the six fixes are applied and the final hashes receive a bounded rereview; no new wide-suite run is requested for those governance-only repairs.
