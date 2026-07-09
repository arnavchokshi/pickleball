# LANE: beststack_docs_20260708 — encode the BEST-STACK DOCTRINE into the canonical docs

## HARD RULES
- NO branches, NO commits. Read BUILD_CHECKLIST.md last ~15 bullets first.
- No new root .md files (zero doc-allowlist churn). No board files (BUILD_CHECKLIST, gpu_fleet,
  inflight_lanes, OWNER_CHECKIN*) — those are manager-owned.
- grep-verify EVERY anchor string in the SAME command before each edit (wave-1 lesson: inexact
  anchors kill doc edits); prefer small single edits over batches. Line numbers below are HINTS
  from an audit pass — re-grep at HEAD; if an anchor is absent, STOP on that item and report.
- Run the doc-consistency test files + the wide suite (MPLBACKEND=Agg, no fail-fast) at the end.
- Artifacts under runs/lanes/beststack_docs_20260708/.

## FILE OWNERSHIP
YOU OWN (surgical insertions only): NORTH_STAR_ROADMAP.md, TECH_BLUEPRINTS.md,
FABLE_OPERATING_MANUAL.md, CLAUDE.md, .claude/skills/run-lane/SKILL.md.
DO NOT TOUCH anything else. A concurrent session may append BUILD_CHECKLIST bullets — irrelevant to
your files, but `git status` before you start and report any pre-existing dirt on YOUR files.

## OBJECTIVE
Owner directive (2026-07-08): gains must always be routed into the default E2E stack, and the
future must intrinsically know this. The beststack_core lane (concurrent, file-disjoint) is building
configs/racketsport/best_stack.json + loader. YOU encode the standing rule at five anchors so every
future session/lane inherits it. Insert the texts below VERBATIM (adjust only cross-reference
numbering if you find the stated anchor numbering shifted at HEAD).

### 1. NORTH_STAR_ROADMAP.md — PART IV new rule 15
Anchor: PART IV header ("STANDING RULES FOR EVERY AGENT", ~line 1644); append AFTER the last rule
(rule 14, wave-end docs reconciliation, ~lines 1743-1747). Existing numbering is quirky (10 before
9) — do NOT renumber anything; append as rule 15:

"15. **BEST-STACK DOCTRINE (owner directive 2026-07-08).** `configs/racketsport/best_stack.json`
is the ONE default-selection surface for every stage's tech/weights/policy. (a) Every landed gain
is, in the SAME lane that lands it, either promoted to WIRED_DEFAULT in the manifest, or recorded
PENDING with its exact named gate (metric key + bar + evidence path), or DORMANT/FENCED with its
kill/ruling citation — a gain absent from the manifest is a defect, not a neutral state. (b) A
plain E2E run and the product server resolve defaults ONLY through the manifest; intentional
divergences live in its declared `server_overrides`, never in silent hardcodes. (c) The moment a
pre-registered gate passes, promotion = the manifest flip in that same lane — best-known tech is
never left opt-in. (d) GPU result runs execute the FULL E2E stack on the promoted manifest by
default; stage-isolated/instrument runs are explicit exceptions labeled non-promotable. (e)
UPSTREAM-FIRST: a promotion in an upstream stage (court calibration above all — it places everything
downstream) bumps the manifest revision; downstream banked baselines are STALE against it until
re-proven, and every downstream A/B or eval runs on the current promoted upstream stack. (f) A
manifest entry is a DEFAULT selection, never a VERIFIED claim — rule 5's VERIFIED bar is untouched.
Wave close includes a best-stack reconciliation: zero unaccounted gains (manual §14)."

(If the VERIFIED bar is a different rule number than 5 at HEAD, fix the cross-reference.)

### 2. TECH_BLUEPRINTS.md — PART A heuristic + PART B ruling B.1.6
(a) In A.2 (Judgment heuristics, ~:43), append one bullet:
"- **No gain left opt-in.** Every landed improvement gets its best_stack.json entry in the same
lane (promoted / PENDING+gate / DORMANT+ruling — NORTH_STAR Part IV rule 15). When you rule a lane
PASS, check its BEST-STACK DELTA section; a landing without one is INCOMPLETE. Downstream evals run
on the current promoted upstream (court first)."
(b) In B.1 (~:127, after B.1.5 ~:144-146), append:
"6. **The best-stack contract.** `configs/racketsport/best_stack.json` + `threed/racketsport/
best_stack.py` are the single default-selection surface (built 2026-07-08, beststack_core lane).
Interlocks: stage-insertion serialization (B.1.1) still routes any process_video/orchestrator
wiring through ONE integration lane; the ball default flip (B.1.2) is now expressed as a manifest
promotion gated on the pre-registered held-out row; paddle wiring (B.1.3) flips the manifest
paddle entry when P3-1 lands; killed approaches (Part IV rule 5 kill list) appear as DORMANT
entries citing their ruling — visible, never re-attempted without new evidence."

### 3. FABLE_OPERATING_MANUAL.md — §14 step extension + new §23
(a) §14 step 7 ("Update docs...", ~:334-335): extend the step text with:
" Wave close ALSO reconciles best_stack.json: every gain landed this wave is promoted / PENDING /
DORMANT there (Part IV rule 15) — an unaccounted gain blocks the close."
(b) Append a new section AFTER the last numbered section (§22 wave-5 field lessons, header ~:618):

"## 23. The best-stack doctrine (owner directive 2026-07-08 — standing)
Rationale: by wave 6 the repo had 8 uncoordinated default mechanisms across 9 files; the owner's
ruled PRIMARY playback fix sat opt-in; the product server silently ran a different court default
than the CLI; the strongest ball candidate was reachable only by hand-typed flag. The fix is
structural: configs/racketsport/best_stack.json is the one selection surface (loader:
threed/racketsport/best_stack.py; enforcement: manifest-integrity + resolution-contract +
CLI-vs-server parity + no-orphan audit tests). Manager duties: (1) every lane spec + report carries
a BEST-STACK DELTA section (run-lane skill); (2) ruling a lane PASS includes checking that delta;
(3) wave close runs the reconciliation — zero unaccounted gains; (4) promotions bump the manifest
revision; downstream evals name the revision they ran on (upstream-first: court calibration
promotes before any downstream placement/ball/body A/B is trusted); (5) GPU result runs are full-
stack on the promoted manifest by default — instrument/stage-isolated runs are labeled
non-promotable exceptions. A manifest entry is a DEFAULT, never a VERIFIED claim."

### 4. CLAUDE.md — hard-rules block line
Anchor: the "Hard rules (full set: NORTH_STAR Part IV ...)" block. Append one bullet:
"- Best-stack doctrine (Part IV rule 15): every landed gain is promoted/PENDING/DORMANT in
`configs/racketsport/best_stack.json` in the SAME lane; defaults resolve ONLY through that
manifest (CLI and server alike); GPU result runs = FULL promoted stack by default; downstream
evals run on the current promoted upstream (court calibration first)."

### 5. .claude/skills/run-lane/SKILL.md — mandatory spec section
After the numbered spec-section list (item 6, anti-passive-wait), append:
"7. **BEST-STACK DELTA (mandatory, spec AND report):** state whether the lane (a) PROMOTES a
best_stack.json entry (attach gate evidence), (b) adds/updates a PENDING or DORMANT entry, or
(c) has no stack delta and why. A lane that lands a model/weights/policy improvement without its
manifest entry is INCOMPLETE — reject at ruling. Eval/A-B lanes MUST name the manifest revision
they consume and run on the current promoted upstream stack."

## ACCEPTANCE
1. All five insertions present verbatim — grep-assert each sentinel: "BEST-STACK DOCTRINE",
   "No gain left opt-in", "The best-stack contract", "## 23. The best-stack doctrine",
   "Best-stack doctrine (Part IV rule 15)", "BEST-STACK DELTA (mandatory".
2. No other lines of the five files changed (git diff --stat proves surgical scope).
3. Doc-consistency tests + wide suite: full census, failures proven pre-existing only.
## KILL
Anchor absent at HEAD after re-grep -> STOP that item, report the drift; never improvise placement.

## BEST-STACK DELTA (this lane): none (doc encoding only).

## STRUCTURED REPORT: objective_result, per-item anchor-found/inserted table, full_suite census,
HONEST ISSUES, draft BUILD_CHECKLIST bullet.
