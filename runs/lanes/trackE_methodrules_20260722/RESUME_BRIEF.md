RESUME (Track E): your process was killed externally by a harness background-task sweep at ~13:24 — NOT a verdict. CRITICAL: git status shows .claude/skills/run-lane/SKILL.md and runs/lanes/person_mixed_20260722/NAMED_NEGATIVE.md are currently ABSENT/unmodified in the tree (your interim report says "final edits temporarily absent" during baseline attribution). FIRST ACTION: re-apply your final edits so both deliverables exist exactly per spec. SECOND (new requirement from the manager, add to the same SKILL.md standing-rules addition as rule/ops note): "Long-running codex lanes MUST be dispatched nohup-detached with the PID recorded in the lane dir (codex.pid) — never as harness background Bash tasks; the harness sweeps aged bg tasks and kills them mid-run (cost Track A run-1 both lanes + Track E both lanes, 2026-07-22)." Place it in the Dispatch section near run_in_background guidance so it's unmissable. THIRD: STOP the per-failure attribution excursion — it is disproportionate for a docs-only change. Bounded closure: your changes touch zero code paths, so run the wide suite ONCE post-change (MPLBACKEND=Agg, non-fail-fast), diff its failure set against your already-saved pre-change wide.junit.xml baseline; identical failure sets = all pre-existing, done. FOURTH: write report.json per schema. No commits. Spec unchanged: runs/lanes/trackE_methodrules_20260722/spec.md.
ADDENDUM (coordinator, 2026-07-22): the SKILL.md dispatch-gotcha addition must ALSO name the standard post-dispatch exposure check, verbatim concept: after dispatching any codex lane, run `ps -o ppid= -p <codex pid>` — ppid 1 = detached/safe; ppid = a claude-harness shell = exposed to the harness age sweep and must be re-dispatched nohup-detached. Include this one-line test alongside the nohup-detached rule in the Dispatch section.

ADDENDUM 2 (coordinator, 2026-07-22, from Track A's A1 landing — add to the SAME SKILL.md
standing-rules/dispatch addition as a fifth item): **STANDING ATTRIBUTION CONVENTION for
concurrent-lane suite noise.** Any wide-suite failure OUTSIDE a lane's owned files must be
ATTRIBUTED before being treated as real. Attribution sources, in order: (1)
`runs/manager/inflight_lanes.md` — which lanes currently have working-tree edits and which files
they own; (2) the move-aside rerun technique — rerun the failing test with the lane's own files
absent; identical failure = pre-existing/cross-lane; (3) the `KNOWN_ATTRIBUTED(<lane>)`
certification format in the lane report. Reports claiming PASS with unattributed out-of-lane
failures are REJECTED at review — but attributed cross-lane noise never blocks a lane's own
acceptance. Purpose: keeps 5 concurrent tracks from serializing on suite cleanliness.
APPLY IT TO YOURSELF: your own wide-suite step should use exactly this convention — best_stack
rev-15 drift failures certify as KNOWN_ATTRIBUTED(trkC_constraints_wire) per the coordinator, and
your junit-diff-vs-baseline is technique (2) in spirit; cite the convention in your report.

ADDENDUM 3 (coordinator, 2026-07-22, from Track A's A2 evidence — SIXTH item in the SAME SKILL.md
addition): **PROMOTION-GRADE WIDE-SUITE RUNS EXECUTE ON AN IMMUTABLE REVISION.** Any wide-suite
result cited as acceptance/promotion evidence must run on a fresh clone or clean worktree pinned
to a commit — never the shared mutating working tree — and the report must name the pinned
revision sha. In-lane ITERATION wide runs may still use the working tree (with the attribution
convention), but they are iteration evidence only. Motivating incident (cite it): Track A's A2
documented a test node renamed mid-collection by a concurrent lane's worktree mutation, with
failures vanishing on targeted rerun — a shared-tree wide run is not a stable measurement.
NOTE FOR YOUR OWN REPORT: your working-tree wide runs are fine as iteration evidence under the
convention; label them as such and name HEAD sha at run time — do NOT redo them on a clean clone.

ADDENDUM 4 (coordinator, 2026-07-22, from Track C triage — items 7-8 if a future resume reopens the
SKILL.md edit): (7) LONG REPO COMMANDS INSIDE CODEX LANES (>120s, e.g. audit_dead_code.py) must run
under timeout/nohup wrappers with output to a file — codex's exec tool loses long processes
(UnknownProcessId) and kills report emission. (8) SALVAGED_VERDICT.md PATTERN: verdicts are
salvageable from streamed lane logs with line-number provenance — never lose a large review to a
process hiccup; the salvage doc cites log line ranges + the ruling Fable countersigns.
