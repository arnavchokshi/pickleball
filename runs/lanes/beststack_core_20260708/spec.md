# LANE: beststack_core_20260708 — the BEST-STACK manifest + wire-now integration lane

## HARD RULES (non-negotiable)
- NO branches, NO commits (manager commits at checkpoint). Read NORTH_STAR_ROADMAP.md Part IV +
  BUILD_CHECKLIST.md last ~15 dated bullets before coding.
- 4 protected clips EVAL-ONLY: Burlington/Wolverine internal-val OK; Outdoor/Indoor labels NEVER
  touched (no ledger row exists for this lane — you need none: this lane touches NO labels).
- Honest reporting. Run the WIDE blast-radius suite (`MPLBACKEND=Agg .venv/bin/python -m pytest
  tests/racketsport tests/server -x-NEVER` — no fail-fast, census is the product), not a subset.
  `importorskip("torch")` for torch tests. Every new CLI ships its scaffold direct-reference test
  same-lane. New root .md files: NONE allowed this lane (no doc-allowlist churn).
- Artifacts under runs/lanes/beststack_core_20260708/ ONLY. Other lanes' run dirs = READ-ONLY evidence.
- NEVER a .patch file deliverable: deferred/fenced changes = inline diff hunks in the report.

## FILE OWNERSHIP (exact; violating a fence = lane rejection)
YOU OWN: configs/racketsport/best_stack.json (NEW), threed/racketsport/best_stack.py (NEW),
scripts/racketsport/process_video.py, threed/racketsport/orchestrator.py,
threed/racketsport/remote_body_dispatch.py, models/MANIFEST.json (FOV entry repair ONLY),
server/gpu_runner.py, server/worker/daemon.py, tests/racketsport/test_best_stack_manifest.py (NEW),
tests/racketsport/test_best_stack_resolution.py (NEW), tests/server/test_best_stack_parity.py (NEW),
plus minimal surgical edits to existing tests that pin the defaults you change (deterministic
fixture/contract tests) and to PIPELINE_SUMMARY schema files.
DO NOT TOUCH (live concurrent lanes / fences): threed/racketsport/court_detector_v2.py,
court_detector_v2_hypotheses.py, court_model_infer.py, tests/racketsport/test_court_e4_fusion.py,
tests/racketsport/test_court_fusion_default.py (a concurrent court session owns these NOW);
web/replay/** and ios/** (product fences); heldout_eval_ledger.md; any gate threshold or gated
metric key anywhere. Ball default checkpoint VALUE stays the raw WASB zero-shot anchor — you thread
it through the manifest but do NOT change which checkpoint is default.
SERVER CAUTION: before editing server/gpu_runner.py or server/worker/daemon.py run `git status
--porcelain server/` and `git log --oneline -5 -- server/` — if another session's uncommitted dirt
or very-fresh foreign commits conflict with your edit, emit inline diff hunks in the report for
those files instead of applying, and say so in HONEST ISSUES.

## OBJECTIVE
Owner directive (2026-07-08): every gain we land must be actually used by default; nothing orphaned.
Build the ONE default-selection surface and route the default E2E path through it, and flip the
wire-now orphans found by the 34-agent wiring audit. Evidence to read FIRST:
runs/lanes/beststack_core_20260708/audit_result_full.json (synthesis + per-area gains +
default_selection_map with file:line for everything referenced below);
runs/lanes/wiring_audit_20260705/WIRING_TRUTH_TABLE.md (historical); BUILD_CHECKLIST.md bullets
[W6 PLAYBACK RULING], [W6 MESHCAP PASS], [W6 RULINGS], [W6 GATE1B-KNOB];
runs/lanes/e2e_synergy_audit_20260705/stage_graph.json.

### 1. The manifest: configs/racketsport/best_stack.json (NEW)
Single JSON (house style: flat, schema-validated) covering EVERY default-selection point in the
audit's default_selection_map. Required per-entry fields: stage, selection (what is chosen),
value (path/constant/policy with sha256 for local checkpoint files), status
(WIRED_DEFAULT | PENDING | DORMANT | FENCED), gate (null or {name, metric_key, bar, evidence_path}),
provenance ({lane, commit, date, evidence_paths}), proven_against (upstream stage names -> manifest
revision at proof time; null ok for v1 where unknown), notes. Top-level: schema_version, revision
(int, starts at 1), updated (date), server_overrides (declared intentional CLI-vs-server divergences).
Populate ALL entries from the audit: the 13 selection surfaces in manifest_design_inputs, the
PENDING set (seed_official + stage1_official ball ckpts w/ gate = pre-registered held-out ledger
row beating the 0.7248 zero-shot anchor + owner go, evidence runs/lanes/w6_labelingest_20260708/
gpu_rescore/loso/; court_unet_v2 + E4 fusion w/ gate = PCK@5px 0.95 + calv1_fusedefault decisive
eval; mesh tier-eligibility raise w/ gate = owner product ruling per [W6 RULINGS] (d)), and the
DORMANT set (P2-2 lambda_foot smoother pending legitimate GATE-1b; Magnus fit_spin_scalar=False
kill-fired; Fast-SAM-3D-Body NOT-ADOPT; physics/2D-gated teacher killed on human GT; 
experimental_body_array_native regression-reverted; dead botsort tracker presets), each citing its
ruling (BUILD_CHECKLIST bullet name/date). FENCED set: profile-driven court/net/distortion cluster +
Wave-A auto-find UI (product-infra fence). A manifest entry is a DEFAULT selection, NEVER a VERIFIED
claim — put that sentence verbatim in a top-level "invariants" field.

### 2. The loader: threed/racketsport/best_stack.py (NEW)
Pure-stdlib module: load, schema-validate (fail-loud on missing file/field/dangling local path/sha
mismatch), and expose typed accessors. Resolution precedence everywhere: explicit CLI flag >
manifest > HARD ERROR (no silent third fallback). The current hardcoded DEFAULT_* selection
constants in process_video.py (:175-180, :252 — WASB ckpt/repo, confidence curves, mesh coverage
mode, mesh frame budget, reid model, association profile) move INTO the manifest; the constants
either disappear or become thin manifest reads (no duplicated literal values left on the decision
path — the no-orphan audit test in #8 enforces this).

### 3. Run self-documentation
PIPELINE_SUMMARY gains a best_stack block: {manifest_revision, resolved (stage->value), overrides
(flag-provided values that differed from manifest)}. Update the summary schema + any schema-sync
surface (remote runs ship code from committed HEAD; keep schema backward-tolerant for old artifacts).

### 4. SANCTIONED DEFAULT CHANGE 1 — mesh byte-budget becomes the default (owner-ruled PRIMARY fix)
Flip the no-flag default from fixed-100-frames to byte-budgeted mesh selection at 300 MiB/clip
(owner band 150-300, density prioritized). --mesh-byte-budget-mib and --target-mesh-frame-budget
remain as overrides (explicit fixed-frame flag switches policy back). Update the deterministic
no-flag fixture hash ONCE with a rationale line in the test ("sanctioned default change:
best-stack doctrine, owner PLAYBACK RULING 2026-07-08"); add an equivalence test no-flag ==
explicit --mesh-byte-budget-mib 300. Do NOT touch tier-eligibility (PENDING entry only, owner
ruling outstanding).

### 5. SANCTIONED DEFAULT CHANGE 2 — events-before-frames so ball_aware is actually contact-dense
Audit finding: on cold runs, events (contact_windows.json) execute AFTER frame scheduling, so the
default ball_aware mesh coverage silently degrades to uniform exactly at contact windows. FIRST
diagnose the true stage dependency direction at HEAD (cite stage functions + stage_graph.json).
If events genuinely need no frame-plan output: reorder events before frame scheduling. If they do:
implement a re-plan/backfill pass after events. Acceptance: on the deterministic fixture (and any
CPU-runnable real-clip path) a cold run's mesh plan is contact-dense when contacts exist —
show before/after plan-summary evidence. If BOTH variants are dependency-unsafe, STOP on this item
and report with evidence (do not force it); the rest of the lane still lands.

### 6. BODY detector/FOV default unification + stale MANIFEST path repair
Trace actual resolution on BOTH paths at HEAD (orchestrator BodyStageRunner yolo/moge2 vs
RemoteConfig ''/'' — empty-string semantics included). Unify so local and remote resolve identical
detector/FOV identifiers THROUGH the manifest; repair the models/MANIFEST.json FOV entry whose
local_path does not resolve (stale /workspace path — use the same relative-resolution/symlink
pattern the checkpoints entry uses). Fail-loud when the FOV checkpoint is absent. Unit/contract
test asserting local==remote resolved identifiers.

### 7. Remaining wire-now hygiene (all through the manifest)
(a) Confidence-gate calibration curves: manifest-owned pointer (current file stays the value;
fail-loud if missing at load). (b) Global association profile: manifest entry as the DELIBERATE
production default (same value, now conscious); mark the per-clip eval overrides eval-only in the
manifest notes. (c) Server parity: GpuRunRequest defaults + worker daemon resolve through the
loader; allow_auto_court_corners_preview=True for server uploads becomes a DECLARED
server_overrides entry (rationale: preview seed for the confirm UI), not a hardcode; parity test
asserts every shared default field resolves identically CLI-vs-server EXCEPT declared overrides.
(d) CAMERA-MOTION CONTRADICTION (audit uncertainty #1): trace the real no-flag HEAD behavior
(camera_motion auto-threshold 2.5 else-branch ~process_video.py:1256 vs enable_camera_motion=False
dataclass default) and pin the manifest entry to EXACTLY current HEAD behavior — zero behavior
change on this item this lane; report the answer with file:line proof.

### 8. Enforcement tests (NEW, the doctrine's teeth)
(i) Manifest integrity: schema-valid, every local path exists, shas match, every status enum legal,
every PENDING entry has a non-null gate, every DORMANT entry cites a ruling. (ii) Resolution
contract: a plain no-flag invocation's resolved config == the manifest's WIRED_DEFAULT set
(entry-point level, through process_video's real option-building path — not a replica). (iii)
CLI-vs-server parity (see 7c). (iv) NO-ORPHAN AUDIT (house audit style, like audit_storage_policy):
scan scripts/racketsport/process_video.py + orchestrator.py + remote_body_dispatch.py for
selection-constant patterns (DEFAULT_*model/ckpt/profile/budget/mode) and models/MANIFEST.json
entries; every hit must be represented in best_stack.json (allowlist file for sanctioned
exceptions, each with a reason).

## ACCEPTANCE (numbers, verbatim checks)
1. Deterministic no-flag fixture: byte-identical EXCEPT diffs attributable to sanctioned changes
   #4/#5 — enumerate the exact changed fields in the report; anything else = you broke a default.
2. All new tests (i)-(iv) green + equivalence test #4 green + unification test #6 green.
3. Wide blast-radius suite: report full census (passed/failed/skipped); failed>0 must be proven
   pre-existing at HEAD (show the HEAD run) or it is YOUR failure to fix. No -x/fail-fast.
4. Scaffold reference tests: 0 missing.
5. grep-proof: no remaining decision-path hardcode for the surfaces moved into the manifest
   (show the grep).
## KILL CRITERIA (pre-committed)
- If threading the manifest requires editing any fenced court file or touching a gated metric
  key/threshold: STOP, report the conflict with inline hunks, do not proceed on that item.
- If #5 is dependency-unsafe both ways: banked evidence + unchanged order (see #5).
- Never tune any threshold to make a test pass; report instead.

## BEST-STACK DELTA (this lane) 
Creates the surface itself; promotes #4 (byte-budget) + #5 (events-order) + #6/#7 hygiene as
WIRED_DEFAULT rev 1; all PENDING/DORMANT/FENCED entries recorded with gates/rulings. Ball/court
model defaults UNCHANGED.

## STRUCTURED REPORT (docs/racketsport/lane_report.schema.json via --output-schema)
objective_result PASS only if ALL acceptance items pass; full_suite census honest; HONEST ISSUES
unsoftened; CHANGES file:line; include the camera-motion adjudication answer + the #5 dependency
diagnosis + a draft BUILD_CHECKLIST bullet line.
