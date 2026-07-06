# FINAL-PASS REVIEW — doc set readiness (2026-07-06)

5 lenses, 6 agents. Lens grades: coherence=B+ day1-simulation=C+ for "can a fresh Fable-5 session run day 1 purely from these docs." Below is the actual bootstrap I ran (read CLAUDE.md -> FABLE

**Readiness verdict:** not-ready — hold Day 1 fleet dispatch until two owner decisions land (docs-commit authorization and the biometric-consent-scope for P0-1b harvest) and a short list of apply-now doc/hook fixes are applied. The underlying roadmap content is unusually thorough and mostly executable (the day1-sim lens rates the actual lane specs as 5-6 fully or mostly spec-able lanes with clear gates), but every lens independently found the same class of problem: the doc set is being handed off in a self-contradictory state relative to its own stated rules. The critical blocker is structural, not editorial — PART 0 requires the doc-of-record set (CLAUDE.md, NORTH_STAR_ROADMAP.md, EDGE_PLAYBOOK.md, .claude/skills/**, the manual addendum) committed+pushed before a run can rely on fresh clones/VMs having them, and verified git status shows all of these are currently untracked or uncommitted, directly colliding with the hard "no commit without owner say-so" rule. A fresh Fable-5 session literally cannot resolve this on its own without either violating a rule or stalling. Layered on top: a proposed Notification hook for auto-resuming stalled subagents is technically infeasible as designed (would silently no-op rather than automate anything), the GPU-provisioning delegation in §12 sends network-requiring gcloud calls to the one execution context (Codex) that's documented to have no network, and §4 vs §14 give literally opposite verification instructions with no reconciliation. None of these are hard to fix — most are text edits or a hook redesign — but shipping the doc set to an autonomous manager before they're fixed means Day 1 opens on a coin-flip: either the manager stalls on the contradiction, or it "resolves" it by guessing, which is exactly the failure mode PART 0 was written to prevent. Once the two owner decisions are made and the ~20 apply-now fixes are applied (mostly 5-30 minutes of doc editing plus one hook redesign), this becomes ready-with-fixes turning into ready.

## Report

## Final-pass synthesis: pickleball roadmap doc set, pre-handoff to Fable-5

Six lenses reviewed the ~1200-line NORTH_STAR_ROADMAP.md plus its supporting docs (CLAUDE.md, FABLE_OPERATING_MANUAL.md, EDGE_PLAYBOOK.md, BUILD_CHECKLIST.md, three drafted skills, seven proposed hooks). All six converge on the same root problem from different angles: the content is strong and largely executable, but the doc set is being handed off in a state that contradicts its own governing rules, and a fresh autonomous session would either stall or silently paper over that contradiction.

**Coherence (B+).** The document is structurally sound at scale — task-ID sequences are complete and non-duplicated, the Map matches Part order, most cross-references resolve. The real defects are small but sharp: an unclosed parenthesis breaks a sentence in the GPU-fleet rules (PART IV rule 7); "P5-7" is cited twice as a real dependency-gating task but was never defined in the Phase 5 checklist; "P1-4b" is referenced as a dependency but P1-4's two tracks were never formally lettered; and I.6's quick-start step still describes P4-1's patch application as a trivial one-liner even though a later-added harsh-review note on the same task says it doesn't apply cleanly at all — an internal staleness the doc's own "later passage corrects an earlier one" pattern warns about elsewhere. Three more low-severity items (a stray horizontal rule splitting a numbered list, a missing I.1 in the Map, ambiguous "P4 ball lane" naming) round this out. None of this blocks Day 1, but a "clean final read" bar isn't met.

**Day-1 simulation (C+).** Running the actual mandated read-order and grounding it against real repo state surfaces the headline problem: PART 0's own gate ("commit the docs of record") is unsatisfied on this exact checkout, contradicting CLAUDE.md's no-commit rule — a genuine STOP, not a hypothetical. Compounding it: `runs/manager/gpu_fleet.md`, which four separate places in the read-order treat as existing, does not exist; BUILD_CHECKLIST's "last ~15 bullets" convention misses HEAD commit e5789028's own stated open action item ("manager browser verification next"); and PART 0's biometric-consent item contradicts its own harvest-approval item on whether third-party scraped video is in scope. Despite this, the lens found 5 lanes fully or mostly spec-able for immediate dispatch (GPU coldstart, tree hygiene, flight-sim, registry-schema, VFR-audit), with only Lane F (harvest) correctly queued pending the consent/ToS decisions.

**Manual audit (C+).** The operating manual's reasoning is sound but has drifted from both git reality and its own addendum. Three high-severity contradictions: PART 0's docs are uncommitted (same root issue as above); §12 delegates `gcloud compute instances create` to a Codex lane despite the manual's own rule that Codex has no network access; and §4's "never re-verify a lane's report" stance directly contradicts §14's "run the decisive check yourself." Secondary drift: §9 still points sessions at a demoted historical doc, §14 names tools ("ScheduleWakeup," "Workflow") that don't exist in this environment, referenced script paths (`scripts/fleet/`) don't exist, there's no Codex-quota-exhaustion fallback despite two real dated occurrences, and no SLA for a silently-stalled lane.

**Hooks/skills feasibility (C+, mixed).** The three drafted skills pass cleanly — correct frontmatter, correct discovery path, ready to ship as-is. The seven hooks are the weakest link: the Notification-hook auto-resume design is flatly infeasible (wrong event, no message-send capability); the Stop-hook design conflates Codex (invisible to hooks except via its wrapping Bash call) with Sonnet subagents (which need SubagentStop, not Stop); the model-pin PreToolUse hook targets an unconfirmed tool_input schema. Four other hooks (destructive-git/gcloud guard, PostToolUse report audit, new-root-.md guard, gcloud spot-create guard, SessionStart inject) are feasible as specified or with a minor transcript-parsing/scoping caveat.

**Cross-doc (B-).** No contradictions in governing rules across the four core docs. The gap is completeness: roughly a third of EDGE_PLAYBOOK §5's roadmap-delta items (P3-4b face-texture anchors, H17 cache/cascade, H18 factor-graph polish, and partial misses on H9/H10/H13/H11/H8/H4) were never actually threaded into NORTH_STAR_ROADMAP.md's task list, and BUILD_CHECKLIST's landing bullet overclaims that they were. An agent following NORTH_STAR alone would silently miss hacks the playbook says are "applied."

**Bottom line:** two items need an owner decision before dispatch (doc-commit authorization; biometric-consent scope for harvest), and roughly 20 apply-now text/hook fixes should land first — mostly cheap, none requiring new design work beyond the Notification/Stop hook redesigns already specified above.

## Fix list (30/32 applied by the manager 2026-07-06; #1 and #6/#18 = owner decisions)

### #1 [CRITICAL] apply_now=False — NORTH_STAR_ROADMAP.md PART 0 (commit-the-docs-of-record gate) vs CLAUDE.md hard no-commit rule; live git status shows CLAUDE.md, NORTH_STAR_ROADMAP.md, EDGE_PLAYBOOK.md, .claude/skills/** untracked and BUILD_CHECKLIST.md/FABLE_OPERATING_MANUAL.md/test_truthful_capabilities.py modified-uncommitted
- issue: PART 0 requires the doc-of-record set committed+pushed before a run starts ('the one thing that breaks finish-from-the-docs-alone'), but CLAUDE.md forbids any commit/push without explicit owner say-so. A fresh clone/VM tomorrow gets none of these docs.
- fix: Owner must explicitly authorize (or personally perform) committing CLAUDE.md, NORTH_STAR_ROADMAP.md, EDGE_PLAYBOOK.md, .claude/skills/**, FABLE_OPERATING_MANUAL.md addendum, and the test allowlist diff as one joint commit before Day 1 dispatch; alternatively bake a standing exception into the commit rule for exactly this doc set.

### #2 [CRITICAL] apply_now=True — runs/research_sota_20260705/fable5_manager_setup.md:258 (Notification hook, auto-resume design)
- issue: As specified, this hook cannot work: Notification hooks can't send messages/resume a conversation, exit code 2 is non-blocking for this event, and the matcher filters on notification_type not message-text regex. Enabling it as literal spec gives silent no-op automation the manager will believe is active.
- fix: Do not ship as designed. Redesign as: (a) CronCreate/`/loop` for periodic resume prompts in a kept-open session, or (b) rely on native background-task-completion re-invocation, or (c) if only a deterrent is wanted, a SubagentStop hook that greps transcript_path for passive-wait language and blocks (exit 2) rather than trying to actively re-wake an exited agent.

### #3 [HIGH] apply_now=True — PART III / PHASE 5 header (lines 1027, 1032) — cites 'P5-7' as real; checklist (lines 1034-1069) only defines P5-1..P5-6
- issue: P5-7 is referenced twice as a load-bearing planned task (the thing that unlocks the '≤1x stretch' gate) but was never written as a checklist item.
- fix: Add a P5-7 checklist bullet (BODY-runtime redesign) with its own gate/kill criteria, or replace both references with prose noting it's an unscoped future item.

### #4 [HIGH] apply_now=True — NORTH_STAR_ROADMAP.md PART IV rule 7 (GPU FLEET), lines 1194-1198, parenthetical starting '(Legacy single-GPU serialize via scripts/gpu-train-lock.sh...'
- issue: The outer parenthetical is never closed (verified: only paragraph in the file with nonzero paren balance) — a grammatically broken sentence in the rules section every agent reads.
- fix: Add the missing closing paren at the end of line 1198, or restructure so the outer aside closes before the two nested asides.

### #5 [HIGH] apply_now=True — I.6 'Direct next steps' item 3 (line 222) vs P4-1's harsh-review note (lines 923-934, added 2026-07-06)
- issue: I.6 tells the owner the P4-1 patch application is a trivial one-step action, but P4-1's own later-added note says the patch does NOT apply as-is (git apply --check fails on 4 hunks, BUILD_CHECKLIST/package.json conflicts, non-ancestor baseline).
- fix: Update I.6 item 3 to: 'Court Wave B kickoff: P4-1 regenerate+land the Wave A patch (does NOT apply as-is — see P4-1's harsh-review note), then P4-2 train court_unet_v2.'

### #6 [HIGH] apply_now=False — NORTH_STAR_ROADMAP.md PART 0 item 5 (biometric consent, non-owner footage) vs item 3 (P0-1b harvest listed as zero-owner-time)
- issue: Item 5's plain text ('ANY non-owner footage') would seem to cover P0-1b's third-party video harvest, while item 3 lists that same harvest as approvable with no owner time needed — the two items contradict on scope.
- fix: Owner must state whether item 5 covers only footage-of-identifiable-friends (ReID/body-shape persistence) or also third-party harvested video; until answered default to NOT dispatching the harvest lane.

### #7 [HIGH] apply_now=True — runs/manager/gpu_fleet.md (referenced in CLAUDE.md, FABLE_OPERATING_MANUAL.md §12/§14, NORTH_STAR Part IV rules 1/7) and scripts/fleet/{lane_vm_startup.sh,reconcile.sh}
- issue: Four places in the mandatory read-order assume these files exist; none do (verified with find/ls). A fresh session following the read-order literally stalls trying to read/use nonexistent files.
- fix: Create runs/manager/gpu_fleet.md as an empty/template ledger and scripts/fleet/{lane_vm_startup.sh,reconcile.sh} now, in the same lane that lands the doc commit; or change the read-order text to 'create if absent.'

### #8 [HIGH] apply_now=True — runs/lanes/ball_p4_render_fix_20260706/REPORT.md (HEAD commit e5789028) vs BUILD_CHECKLIST.md tail
- issue: HEAD's own report flags 'manager browser verification next' as outstanding, but BUILD_CHECKLIST has no bullet for this commit — the 'read last ~15 bullets' instruction misses the single most recent open action item.
- fix: Add the missing BUILD_CHECKLIST bullet for e5789028 now, and add a standing rule: every landing commit must add its own BUILD_CHECKLIST bullet in the same commit (enforce via pre-push hook or doc-consistency test).

### #9 [HIGH] apply_now=True — NORTH_STAR_ROADMAP.md PART 0 checkboxes (all unchecked) vs BUILD_CHECKLIST.md tail (GPU access, data flow, Roboflow key all evidently already working/resolved)
- issue: All 5 PART 0 items read as blank/blocking STOPs, but 3 of them are functionally already resolved per BUILD_CHECKLIST evidence — a literal fresh session either wrongly stops on settled facts or the STOP rule is silently being ignored, with no way to tell which.
- fix: Check off items already evidenced (GPU access, Roboflow key, first-data) with a one-line evidence pointer each; leave only genuinely open items (biometric consent scope, docs-commit gate) as live STOPs. Add a rule that PART 0 checkboxes are re-verified against BUILD_CHECKLIST each session, not one-time historical ticks.

### #10 [HIGH] apply_now=True — FABLE_OPERATING_MANUAL.md §12 (GPU provisioning delegated to 'a Codex/script lane') and .claude/skills/gpu-fleet-provision/SKILL.md vs §2/§8 role table and NORTH_STAR Part IV known-traps ('Codex sandbox has no network')
- issue: §12 tells Fable to delegate `gcloud compute instances create` (needs network) to a Codex-sandboxed lane, directly contradicting the manual's own rule that Codex has no network access.
- fix: Change §12 and the gpu-fleet-provision skill to delegate gcloud provisioning calls to a Sonnet subagent (or a Fable-run Bash step outside the Codex sandbox), keeping Codex only for network-free work.

### #11 [HIGH] apply_now=True — FABLE_OPERATING_MANUAL.md §4 ('never re-read artifacts to Verify') vs §14 step 6 ('run the decisive diff/browser check yourself') and .claude/skills/run-lane/SKILL.md's 'After' section
- issue: §4 and §14 give literally opposite instructions for verifying a completed lane's report, with no reconciliation — a session can't tell which rule governs.
- fix: Add an explicit line in both §4 and §14: '§14 supersedes §4's never-verify stance specifically for fleet/high-stakes decisions (spot-check the decisive number/screenshot once); §2's harder rule that Fable never re-runs the full suite still stands.'

### #12 [HIGH] apply_now=True — runs/research_sota_20260705/fable5_manager_setup.md:257 (Stop hook, 'end of a Codex/Sonnet lane turn') and §16 line 322
- issue: Conflates two incompatible targets: Claude Code hooks have zero visibility into Codex's internal turns (only the wrapping Bash call), and Stop hooks apply only to the top-level session, not named subagents (that's SubagentStop). As specified this hook cannot fire as intended for either case.
- fix: Split into two mechanisms: (a) SubagentStop matched on agent_type for genuine Sonnet subagent lanes, reading transcript_path and blocking (exit 2) if required report headings are missing, with a hand-rolled retry-cap counter file (no built-in cap exists); (b) for Codex lanes, enforce report format via the existing PostToolUse hook on the Bash call that invokes/reads `codex exec ... -o report.json`, not via Stop.

### #13 [HIGH] apply_now=True — runs/research_sota_20260705/fable5_manager_setup.md:251 (PreToolUse model-pin hook on the Agent tool) and §16 line 322-323
- issue: The exact tool_input schema for the Agent tool (renamed from Task in v2.1.63) isn't documented, and the 'model' field actually lives in the named subagent's own frontmatter (default inherit), not necessarily a flat tool_input key — a naive grep can false-negative on subagent_type-only dispatches.
- fix: Before enabling as a hard block: capture a real PreToolUse event via `claude --debug` first; resolve subagent_type -> agent definition file -> check frontmatter model is set and not 'inherit'; ship log-only for one session before flipping to a hard deny.

### #14 [HIGH] apply_now=True — EDGE_PLAYBOOK.md:457-459 (§5 items 8-10: H7/P3-4b, H17, H18) vs NORTH_STAR_ROADMAP.md (grep for P3-4b/H17/H18 = zero hits)
- issue: Three roadmap-delta task IDs the playbook promises (P3-4b face-texture anchors, a new P5 task for H17 cache/cascade, a new P1/P3 integration task for H18 factor-graph polish) were never created in NORTH_STAR, and BUILD_CHECKLIST's landing bullet implies they were applied.
- fix: For each: either create the promised checkbox task with its stated gate, or edit EDGE_PLAYBOOK §5 to say 'not yet applied to NORTH_STAR' / relabel the closest existing task (e.g. PF-3 for H18) with an explicit cross-reference — and correct the overclaiming BUILD_CHECKLIST bullet.

### #15 [MEDIUM] apply_now=True — Line 749 (P1-5 depends on 'P1-4b') vs P1-4's bullet (lines 737-748, tracks (a)/(b) never formally lettered P1-4a/P1-4b)
- issue: P1-5 cites a dependency task ID that was never formally defined, unlike other sub-lettered tasks (P4-6a/b/c, P0-10a-d) elsewhere in the doc.
- fix: Either label P1-4's two tracks explicitly as P1-4a/P1-4b, or change P1-5's reference to plain prose naming track (b), the learned-lift path.

### #16 [MEDIUM] apply_now=True — Line 638 ('P4 ball lane found wolverine contact anchors blocked on this') and line 1087 ('court-map view already landed in P4 ball lane')
- issue: 'P4' is reserved elsewhere in the doc for the Court/net phase; using 'P4 ball lane' to mean a legacy/unrelated ball-tracking lane name is undisambiguated and will confuse a reader.
- fix: Replace both instances with the actual lane/run path (e.g. runs/lanes/ball_tracking_long_run_.../), or add a one-time disambiguating parenthetical on first use.

### #17 [MEDIUM] apply_now=True — PART II-B (lines 498-499, SAT-HMR vs SAM-3D-Body verification pointer) vs P0-6's actual gate list (lines 636-645, no such item) — P0-2 already owns vendor reconciliation
- issue: A promised verification ('confirm which body backbone is live, don't assume — verify in P0-6') was never actually placed into P0-6's task/gate list.
- fix: Add the explicit backbone-verification sub-item/gate to P0-6, or retarget the II-B pointer to P0-2, which already reconciles SAT-HMR vendor content.

### #18 [MEDIUM] apply_now=False — NORTH_STAR_ROADMAP.md P0-1b task block (Part III / I.7 line 254-255)
- issue: Bulk yt-dlp scraping of third-party YouTube pickleball games for a training corpus carries real copyright/ToS exposure never named in the doc as a risk category, unlike other license considerations (Part IV rule 6 covers datasets/models we consume, not scraping video we don't own).
- fix: Owner decision needed: proceed broadly under a private-use-only stance, or restrict sourcing to Creative-Commons/explicitly-licensed channels by default (recommended default until answered).

### #19 [MEDIUM] apply_now=True — requirements-racketsport.txt:12 + scripts/racketsport/install_mujoco_mjx_env.sh vs NORTH_STAR_ROADMAP.md P0-7 (line 650, 'build on MuJoCo')
- issue: MuJoCo is referenced only in a requirements comment, is not installed in .venv (import fails), and Codex's sandbox has no network to install it — a Codex lane dispatched per the literal task text fails at step one with no self-recovery.
- fix: Note the install-script prerequisite directly in P0-7's task block; require a one-time network-capable prestage (Sonnet or Fable) before the Codex build lane starts, or scope P0-7's first cut to a pure-numpy physics model needing no new install.

### #20 [MEDIUM] apply_now=True — NORTH_STAR_ROADMAP.md I.7 line 256 ('P2-1 SMART hardening, eval clips suffice') vs P2-1 task block (lines 781-791, RAFT optical-flow scheduled via gpu-train-lock)
- issue: I.7's critical-path summary lists P2-1 among 'start today' items in a way that reads CPU-only/zero-resource, but the task itself needs a GPU slot and gpu-train-lock coordination.
- fix: Reword I.7 to 'GPU-light, schedulable without blocking P1 training' instead of implying no GPU dependency.

### #21 [MEDIUM] apply_now=True — RESET_HANDOFF_20260705.md §8 (label-file schema owner decision blocking P0-2's hygiene gate) — not surfaced in NORTH_STAR_ROADMAP.md, which the read-order calls the durable source of truth
- issue: A concrete blocker for one of Part III's first-wave tasks (P0-2) lives only in a doc explicitly marked 'historical context only,' not in the mandatory-read NORTH_STAR.
- fix: Surface this specific blocker as a typed item directly under P0-2 in NORTH_STAR_ROADMAP.md (loader compat shim vs label-file re-export), not only in RESET_HANDOFF.

### #22 [MEDIUM] apply_now=True — FABLE_OPERATING_MANUAL.md §9 (still points to JOINT_DETECTION_AND_PLACEMENT_HANDOFF.md §6/§9 as the session-start doc) vs line 7 and §14 (NORTH_STAR + BUILD_CHECKLIST now the durable state)
- issue: §9 was never updated when the §12-16 addendum landed and points a fresh session at the wrong doc for its first decision.
- fix: Rewrite §9 step 1/6 to reference NORTH_STAR_ROADMAP.md Part III + BUILD_CHECKLIST.md, matching §14, or delete §9 since §14 supersedes it.

### #23 [MEDIUM] apply_now=True — FABLE_OPERATING_MANUAL.md §14 step 4 ('Workflow tool') and step 5 ('ScheduleWakeup')
- issue: Neither tool name exists in this Claude Code environment's actual/deferred tool set (real names: CronCreate/CronList/CronDelete via /loop or /schedule for scheduling; Task/Agent for subagent fan-out). Cloud Routines also cannot reach local gcloud/SSH/fleet state without extra provisioning.
- fix: Replace 'ScheduleWakeup' with CronCreate/`/loop`, replace 'Workflow' with the actual Task/Agent tool name, and add a caveat that fleet-reconcile jobs needing live gcloud/SSH access should run as a /loop job in a kept-open session, not a Cloud Routine, unless GCP credentials/network are separately provisioned.

### #24 [MEDIUM] apply_now=True — FABLE_OPERATING_MANUAL.md — no section on Codex quota exhaustion, despite two dated real occurrences (RACKET_6DOF_GOAL.md 2026-07-05, RESET_HANDOFF_20260705.md reset Jul 9)
- issue: The manual claims Codex credits are 'abundant' with no detection/fallback procedure, even though quota exhaustion has already forced ad hoc Sonnet-implements-a-leg workarounds that otherwise conflict with §2's rule.
- fix: Add a 'Codex-quota fallback' subsection: how to detect exhaustion, default action (queue non-urgent lanes, log reset time), and the narrow labeled exception for Sonnet to implement a bounded leg under the same verification bar.

### #25 [MEDIUM] apply_now=True — .claude/skills/run-lane/SKILL.md 'Dispatch' section vs FABLE_OPERATING_MANUAL.md §10 (mandates --output-schema + -o, forbids Monitor watchers)
- issue: The skill's own dispatch template omits both required flags, pipes to a plain log file, and recommends a Monitor watcher the manual explicitly forbids — the skill has drifted from the manual it operationalizes.
- fix: Update the Dispatch block to match §10 verbatim (add --output-schema and -o report.json paths, drop the Monitor suggestion, rely on background-task notification).

### #26 [LOW] apply_now=True — NORTH_STAR_ROADMAP.md Line 1208, between PART IV rule 8 and rule 9
- issue: A bare '---' mid-list splits the 'STANDING RULES' numbered list, reading as a stray patch artifact since --- is used elsewhere only for major section boundaries.
- fix: Remove the '---' so rule 9 flows as item 9 of the same list.

### #27 [LOW] apply_now=True — 'Map of this document' note, lines 25-31
- issue: The Map skips I.1 ('the one-paragraph verdict', lines 113-133), a real substantive subsection between I.0 and I.2.
- fix: Insert 'I.1 one-paragraph verdict' into the Map's Part I list.

### #28 [LOW] apply_now=True — NORTH_STAR_ROADMAP.md PART 0 item 4 (Roboflow API key) vs runs/lanes/ball_tracking_long_run_STATUS.md:207 and live ~/.roboflow_key (dated Jul 4)
- issue: PART 0 phrases this as still-pending, but the key was already received/stored — risks a false STOP on something already satisfied.
- fix: Tick off satisfied PART 0 items with a dated evidence note, or point checkboxes at a single live-state file instead of static prose.

### #29 [LOW] apply_now=True — NORTH_STAR_ROADMAP.md P0-9 task block (lines 663-671)
- issue: Specifies profile types/behavior but no storage technology, forcing a Codex lane to invent architecture rather than execute a decision already made.
- fix: Pre-rule and record the storage choice in P0-9's own text (e.g. flat per-account JSON under runs/profiles/<account_id>/, matching the pipeline's existing artifact convention).

### #30 [LOW] apply_now=True — local disk (df -h: 95% used, 23GiB free) vs FABLE_OPERATING_MANUAL.md §12 (covers only VM-disk hygiene)
- issue: No documented local-disk budget/cleanup rule; a day-1 harvest or ingest lane could exhaust remaining headroom with no guardrail.
- fix: Add a local-disk preflight (df -h check + per-lane budget cap) mirroring the existing VM-side rule.

### #31 [LOW] apply_now=True — FABLE_OPERATING_MANUAL.md §15 item 12 and .claude/skills/run-lane/SKILL.md line 11, both cite '§12.1'
- issue: §12 has no numbered subsections; '§12.1' is a dead in-document cross-reference.
- fix: Number §12's bullets, or change both citations to plain '§12.'

### #32 [LOW] apply_now=True — FABLE_OPERATING_MANUAL.md §14 step 9 (session-end state) — no schema for lanes still RUNNING at session end
- issue: No mechanic records in-flight lane session_id/resume command/expected-done time, risking double-dispatch or lost resume state on the next session.
- fix: Add a runs/manager/inflight_lanes.md with one row per in-flight lane (session_id, resume command, owned files, expected-done time), written at session end, read at next session's step 1.

## Hooks feasibility verdicts (claude-code-guide)

- **Notification hook (auto-resume subagents that say 'waiting for'/'will check back')** → no
  - corrected design: Notification hooks can't send conversation messages or resume an agent, and exit 2 is non-blocking for this event; matcher filters on notification_type, not message-text regex. Replace with: (a) CronCreate/`/loop` periodic resume prompt in a kept-open session, or (b) rely on native background-task-completion re-invocation, or (c) as a deterrent only, a SubagentStop hook that greps transcript_path for passive-wait phrasing and blocks (exit 2) so the agent can't end on a passive wait — it cannot proactively re-wake something already exited.
- **Stop hook: enforce report format at 'end of a Codex/Sonnet lane turn'** → yes-with-changes
  - corrected design: Split by target: Codex lanes are invisible to Claude Code hooks except via the wrapping Bash call — enforce via the existing PostToolUse hook on that Bash call, not Stop. Genuine Sonnet subagents need SubagentStop (matched on agent_type), not Stop (Stop only covers the top-level session and can't target specific subagents); grep transcript_path for required report headings and block (exit 2/decision:block) if missing, with a hand-rolled retry-cap counter file since there's no built-in cap.
- **PreToolUse model-pin check on the Agent/Task tool** → yes-with-changes
  - corrected design: Agent tool_input schema and the exact location of the model field (subagent's own frontmatter, default inherit) aren't confirmed. Capture a real PreToolUse event via `claude --debug` first; resolve subagent_type -> agent definition file -> check frontmatter model is set and not 'inherit'; ship log-only for one session before flipping to a hard deny.
- **PreToolUse destructive git/gcloud guard (block unless a preceding git status this turn)** → yes-with-changes
  - corrected design: Matcher and deny mechanics are fine as specified. The 'preceding check in the same turn' clause isn't a first-class primitive — hand-parse transcript_path for a recent git-status tool call, and fail closed (deny) if the transcript can't be parsed.
- **PostToolUse lane-report audit (after the Bash call that runs codex exec / reads report.json)** → yes
  - corrected design: No change needed — correctly targets the Bash wrapper since that's all Claude Code can see of Codex. Just scope the matcher/if tightly to Bash calls referencing report.json or `codex exec` so it doesn't add overhead on unrelated Bash calls.
- **PreToolUse new-root-.md allowlist guard on Write** → yes-with-changes
  - corrected design: Matcher on Write is fine, but the matcher's own if-pattern syntax (Write(*.md)) can't express 'root-only, no nested .md' precisely. Match broadly on Write, then do the exact regex test (^[^/]+\.md$ at repo root) against tool_input.file_path inside the hook script itself, then check the allowlist condition.
- **PreToolUse gcloud spot-create guard (require SPOT/STOP-on-preempt/fable-lane label + a preceding safe-parallelism note)** → yes-with-changes
  - corrected design: Matcher and substring checks on tool_input.command work as specified. The 'preceding note this turn' clause has the same caveat as the git/gcloud guard: implement as a transcript_path scan for a required marker string, fail closed if the scan errors.
- **SessionStart context inject (refresh gpu_fleet.md / doc state on session start and --resume)** → yes
  - corrected design: Confirmed feasible exactly as specified via hookSpecificOutput.additionalContext, and SessionStart re-fires on --resume. No change needed; optionally set reloadSkills:true if new skill files are added, and consider watchPaths on gpu_fleet.md for FileChanged events.