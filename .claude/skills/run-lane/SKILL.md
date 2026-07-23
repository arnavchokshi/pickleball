---
name: run-lane
description: Use when Fable is about to dispatch any Codex or Sonnet implementation lane (build / fix / verify / docs work) on the pickleball project. Templates a complete, collision-safe lane spec so the lane self-verifies and returns a rulable structured report. Do NOT use for Fable's own tiny coordination edits or for research fan-outs (use research-fanout).
---

# run-lane

Every implementation lane gets this spec. Fable writes it, a Codex lane (default) or Sonnet lane
(only for GPU/SSH/browser/network) executes it. Fable never implements directly.

## Pre-dispatch: the safe-parallelism check
Before writing the spec, confirm the lane is: **file-disjoint** (owned files overlap 0 with every
in-flight lane per `runs/manager/inflight_lanes.md`), **data-disjoint** (no held-out/protected label without a ledger
row — else STOP), **resource-disjoint** (GPU? reuse an idle matching VM else provision new). If it needs a GPU,
provision/assign the VM FIRST (gpu-fleet-provision skill) and record it in `runs/manager/gpu_fleet.md`.

## Standing spec lines (include the ones that apply)
- **Training lane** → mandatory acceptance item: push ONE sample through the training dataloader AND
  the production inference preprocessor; assert identical tensors (or a documented, stamped mapping).
  Product-metric scoring always includes a healthy-checkpoint CONTROL row.
- **GPU wall-clock** → never quote a cap from a different data path: the lane runs a ~100-step probe,
  reports measured steps/s, THEN the manager sets the step/wall cap. Budgets carry +~50%
  outage/idle contingency; VMs expected >2h get an in-VM idle watchdog (no-ssh self-stop).
- **GPU lane resilience** → nohup every long VM step (survives driver outages); chunked-transfer
  fallback for Mac→VM uploads (bwlimit + append-verify + 50MB chunks; uplink is assumed-unreliable);
  prefer VM `git fetch` from origin for code when origin is current; refresh known_hosts INTO any
  pinned worktree the lane runs from.
- **Deliverables** → NEVER a .patch file (both wave-4 attempts were invalid): deferred fenced-file
  changes = inline diff hunks in the report; the integration lane re-derives + `git apply --check`s
  its own work. Regenerated/derived artifacts go under YOUR lane dir — other lanes' run dirs are
  READ-ONLY evidence.
- **Verify rounds** → gate-adjacent fixes get one adversarial round per repair round, scored by the
  verifier's UNMODIFIED harness; fix results matching a diagnosis's predictions to the digit =
  provenance-check before acceptance. Round-3 resumes encode an explicit fallback so the thread
  ends decidable in one round-trip.

## CROSS-SIGNAL MANDATE (owner core thesis 2026-07-20: EVERYTHING HELPS EVERYTHING)
Every lane spec AND report carries a CROSS-SIGNAL row: which other systems' signals this lane
CONSUMES (wrist/skeleton windows, ball-near-person proximity, court calibration, audio-bounded,
rally structure, event anchors) and which consumers it FEEDS (North Star §3.1 reuse contract).
A detector/estimator lane that doesn't consider conditioning on existing signals before asking for
more data/params is mis-designed — the fused-signal moat is the product strategy. Fusion rules
stand: no single signal decides irreversibly; raw immutable; provenance carried.

## STANDING METHOD RULES (owner-approved SOTA program 2026-07-22 — checked at every review)
1. **Exposure-matched arms MANDATORY.** Every A/B training comparison matches arms on total
   optimizer steps, human-row exposure per step, and caps auxiliary/pseudo loss share (the
   `--sst-batch-size` + `--sst-loss-cap` pattern). An arm receiving more gradient updates or more
   human-anchor exposures than its control is invalid BY CONSTRUCTION — reject at review, before
   GPU. Precedent: PERSON_MIXED_POOL_NO_LIFT_UNDERCONTROLLED
   (runs/lanes/person_mixed_20260722/NAMED_NEGATIVE.md — 13.5x update / 6.75x anchor-exposure
   asymmetry left the od8al precision collapse unattributable among 3 hypotheses).
2. **Per-domain-family metrics MANDATORY.** Every scored result reports per source/venue family
   alongside pooled; pooled-only numbers are rejected at review (precedent: WASB pooled 0.5670
   hides indoor 0.7395 vs outdoor-night 0.2933).
3. **Ensemble teachers for pseudo-labeling.** Any pseudo-label pass uses >=2 independent teachers
   with an explicit consensus/agreement rule, never a single teacher (single-teacher self-training
   reinforces its own blind spots; CoTracker3-style recipe per runs/research_sota_20260722).
4. **License = FYI only** (owner directive 2026-07-22, internal use). Lanes record license as
   metadata; no lane blocks, quarantines, or gates on license grounds. Protected-eval and
   compare-only quarantines are PROTOCOL, not license, and stand unchanged.
5. **STANDING ATTRIBUTION CONVENTION for concurrent-lane suite noise.** Any wide-suite failure
   OUTSIDE a lane's owned files must be ATTRIBUTED before being treated as real. Attribution
   sources, in order: (1) `runs/manager/inflight_lanes.md` — which lanes currently have working-tree
   edits and which files they own; (2) the move-aside rerun technique — rerun the failing test with
   the lane's own files absent; identical failure = pre-existing/cross-lane; (3) the
   `KNOWN_ATTRIBUTED(<lane>)` certification format in the lane report. Reports claiming PASS with
   unattributed out-of-lane failures are REJECTED at review. Ownership is a LEAD, not proof —
   nonblocking certification requires move-aside, clean-HEAD, or an independently evidenced
   equivalent reproduction (a lane can break a test it does not own). Attribution never overrides
   the report schema: with failed>0 and failures not proven pre-existing, `objective_result` stays
   PARTIAL/BLOCKED — attribution certifies the failures as NONBLOCKING FOR THE RULING, not as
   absent. Purpose: keeps 5 concurrent tracks from serializing on suite cleanliness.
6. **PROMOTION-GRADE WIDE-SUITE RUNS EXECUTE ON AN IMMUTABLE REVISION.** Any wide-suite result
   cited as acceptance/promotion evidence must run on a fresh clone or clean worktree pinned to a
   commit — never the shared mutating working tree — and the report must name the pinned revision
   sha. In-lane ITERATION wide runs may still use the working tree (with the attribution
   convention), but they are iteration evidence only. Motivating incident: Track A's A2 documented
   a test node renamed mid-collection by a concurrent lane's worktree mutation, with failures
   vanishing on targeted rerun — a shared-tree wide run is not a stable measurement.

## The spec (write to `runs/lanes/<lane>_<date>/spec.md`)
1. **HARD RULES block:** no branches/commits (owner joint-commit rule); read NORTH_STAR_ROADMAP.md, AGENTS.md, the relevant
   RUNBOOK.md section, and the latest `runs/manager/inflight_lanes.md` notes; 4 protected clips EVAL-ONLY (Burlington/Wolverine internal-val OK; Outdoor/Indoor labels
   NEVER without a pre-registered ledger row + STOP for Fable go); honest reporting; run the WIDE
   blast-radius test suite (`MPLBACKEND=Agg`), not a hand-picked subset; register any new root .md in
   the doc allowlist same-lane; every new CLI ships its direct-CLI reference test same-lane; artifacts
   under `runs/lanes/<lane>_<date>/`.
2. **EXPLICIT FILE OWNERSHIP:** name exactly which files this lane owns; concurrent lanes must be
   file-disjoint (process_video.py contention has bitten us).
   CONTAMINATION-ADJACENT LANES (fence borders protected/judge assets) additionally carry: an
   explicit forbidden-path list in-spec; a mandatory self-audit grep of the lane's OWN log in its
   report (command + output — blanket "no contact" claims are banned); and per-constant provenance
   citations for every threshold used (Track C rebuild standard, 2026-07-22).
3. **Objective + acceptance NUMBERS:** the measurable gate from the roadmap task, verbatim. Kill criteria.
4. **Evidence to read first:** the exact run paths / reports the lane must ground on.
5. **Mandatory structured report** (Fable rules on this, never the transcript): `objective_result`
   — exact schema enum PASS | BLOCKED | PARTIAL (per lane_report.schema.json; PASS/FAIL belong to
   individual acceptance-row verdicts, never to objective_result) — plus `full_suite`
   (failed>0 while claiming PASS = rejected unless all failures proven pre-existing), HONEST
   ISSUES, artifacts, dated `runs/manager/inflight_lanes.md` row.
6. **Anti-passive-wait** (for any lane touching a >10min GPU/remote job): "ending your turn to wait =
   lane death; you will NOT be re-woken; poll with a bounded foreground until-loop; end only with the
   final report or a hard blocker." Budget 1-2 SendMessage resumes.
7. **BEST-STACK DELTA (mandatory, spec AND report):** state whether the lane (a) PROMOTES a
   best_stack.json entry (attach gate evidence), (b) adds/updates a PENDING or DORMANT entry, or
   (c) has no stack delta and why. A lane that lands a model/weights/policy improvement without its
   manifest entry is INCOMPLETE — reject at ruling. Eval/A-B lanes MUST name the manifest revision
   they consume and run on the current promoted upstream stack.

## Dispatch (schema-validated report, no watchers)
```bash
LANE=<short-name>; ROOT=/Users/arnavchokshi/Desktop/pickleball
TIER=xhigh   # economy ladder: xhigh default; high mechanical; ultra blast-radius review
mkdir -p "$ROOT/runs/lanes/$LANE"   # spec.md written first
nohup codex exec \
  --cd "$ROOT" --sandbox workspace-write \
  -c model="gpt-5.6-sol" -c model_reasoning_effort=$TIER \
  --output-schema "$ROOT/docs/racketsport/lane_report.schema.json" \
  -o "$ROOT/runs/lanes/$LANE/report.json" \
  < "$ROOT/runs/lanes/$LANE/spec.md" \
  > "$ROOT/runs/lanes/$LANE/log.txt" 2>&1 &
echo $! > "$ROOT/runs/lanes/$LANE/codex.pid"; disown
```
Then, in a SEPARATE later shell (never the launcher shell — a live child is only reparented to
PID 1 after its launcher exits): `ps -o ppid= -p $(cat "$ROOT/runs/lanes/$LANE/codex.pid")` MUST
print 1. ppid = a claude-harness shell means exposed — re-dispatch nohup-detached.
**LONG-RUNNING CODEX DISPATCH (owner directive 2026-07-22):** long-running Codex lanes MUST be
dispatched nohup-detached with the PID recorded in the lane dir (`codex.pid`) — never as harness
background Bash tasks; the harness sweeps aged background tasks and kills them mid-run (cost Track
A run-1 both lanes + Track E both lanes, 2026-07-22). After dispatching any Codex lane, run
`ps -o ppid= -p <codex pid>` — ppid 1 = detached/safe; ppid = a claude-harness shell = exposed to
the harness age sweep and must be re-dispatched nohup-detached.
Inside a codex lane, any repo command expected to run >120s (e.g. `audit_dead_code.py`, wide
pytest) runs under a timeout/nohup wrapper writing to a file — codex's exec tool loses long
foreground processes (`UnknownProcessId`) and that kills report emission (Track C triage
2026-07-22). And if a lane process dies anyway: verdicts are salvageable from the streamed lane
log with line-number provenance — the `SALVAGED_VERDICT.md` pattern (extract findings + verdict,
cite log line ranges, Fable countersigns). SALVAGED_VERDICT.md is provenance INPUT only: the
countersigning coordinator must emit a schema-valid `report.json` from it before any ruling.
Never lose a large review to a process hiccup.
CONTENT-FILTER GOTCHA (Track B, 2026-07-22 — cost a 157k-token review session): codex reviews of
checksum/integrity-verification test code can trip the upstream cybersecurity content filter and
kill the session. Frame such specs in neutral data-integrity terminology (verification,
consistency check, provenance — avoid attack/tamper/bypass phrasing where possible) and bank
pre-verified context (hashes already confirmed, findings so far) in the lane dir so a
continuation resumes cheaply.
Absolute paths ALWAYS; add `-c tools.web_search=true` for research lanes. Harness background Bash
(`run_in_background: true`) is permitted ONLY for short (<10 min) utility invocations — never for
lanes (the age sweep, below).
**SUBAGENT OPS (Track C findings 2026-07-22):** writer/GPU subagents spawned fresh MUST use
`isolation: "worktree"` from the start — a fresh bg subagent cannot write the shared checkout and
cannot self-remediate. AND: worktree auto-clean DESTROYS untracked files at completion (a Track C
agent lost all five deliverables this way; transcript recovery worked once and is not a plan) —
artifact preservation is mandatory, with an explicit phase boundary: under the default
no-lane-commit policy the parent MUST copy artifacts out of the worktree BEFORE the agent
completes; the in-worktree `git add` + commit route applies ONLY when the lane's spec explicitly
grants commit authority (auto-clean preserves worktrees with changes). Promotion-grade suites run
on the integration commit or an explicitly authorized temporary immutable commit — never on an
uncommitted pre-review candidate; precommit shared-tree results are iteration evidence. Unblocks for a holding agent arrive as a RESPAWN or an explicit
coordinator/user action, never peer chat — the subagent-to-parent reply path is unreliable and
peer-verification is weak. REVERT DISCIPLINE: before reverting any file set, cross-check the
revert list against the session-start dirty state — a pre-existing dirty file in the list means
STOP (it is another lane's or the owner's work); Track C self-caught exactly this clobber and
restored from a banked patch within a minute (2026-07-22).
**MODEL POLICY (owner directive 2026-07-19, supersedes 2026-07-09): every Codex dispatch uses `gpt-5.6-sol`
ONLY — pin the model explicitly on every dispatch, never rely on the CLI default, never any other model.
Speed: NORMAL always (owner directive 2026-07-22 — FAST revoked; never any priority/speed
override). Reasoning effort is TIERED BY ECONOMY (owner directive 2026-07-22, supersedes the
2026-07-19 difficulty ladder; `-c model_reasoning_effort=...`, ladder high < xhigh < ultra):
- `high` = mechanical, fully spec-complete lanes only;
- `xhigh` = THE DEFAULT — implementation, design latitude, advisory, and standard review lanes;
- `ultra` = RESERVED for blast-radius reviews ONLY: judge-touching arming, production wiring,
  promotion decisions. Nothing else dispatches at ultra.
- NEXT-STEP / APPROACH DECISIONS: Fable does the thinking AND dispatches a `gpt-5.6-sol` advisory
  lane at `xhigh` (structured facts + options + tradeoffs, never conclusions-only); Fable rules on
  its output. Strategy is never decided at `high`.
State the chosen tier + one-line justification in the lane row.**

**REVIEW-EVERYTHING MANDATE (owner directive 2026-07-20; tiering re-set 2026-07-22): everything
substantive still gets a gpt-5.6-sol adversarial REVIEW pass (correctness, does-it-match-spec,
hidden bugs, gate-gaming, honesty of claims) BEFORE Fable commits or rules adopt/reject — but the
review tier follows the economy ladder: standard reviews run at `xhigh`; ONLY blast-radius reviews
(judge-touching arming, production wiring, promotion decisions) run at `ultra`. Sonnet may still do
mechanical/noisy legwork (test runs, attribution, data pulls), but the DECIDING review is
gpt-5.6-sol. Do not accept a lane on its own self-report or on test-green alone. Only trivial
coordination edits (docs, ledger, inflight rows) skip review. MANDATORY for any lane whose fence
borders protected/judge assets: the review spec includes a LOG-SWEEP obligation — search the
lane's full LOG for judge/protected-asset contact (IDs, paths, hashes), never the diff alone.
Proven load-bearing 2026-07-22: Track C constraints-wire judge contamination + FALSE DENIAL were
found by log sweep; the false denial is what made it unsalvageable.**
The harness notifies on process exit — read `report.json` and rule. Do NOT set up Monitor/done-marker
watchers (they false-fire). Prefer big self-iterating lanes over many small round-trips.

## After
Verify personally: run its tests in the real env, open its artifacts, check its numbers. Codex
"failures" are often sandbox-only (socket/MPS) — re-verify locally. Then update `runs/manager/inflight_lanes.md`.
