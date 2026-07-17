# LANE oneworld_impl_20260716 — Track K: build one_world_v1 (core + CLIs + metrics)

You are a Codex implementation lane for DinkVision (repo root /Users/arnavchokshi/Desktop/pickleball).
You implement the ADOPTED design at runs/lanes/oneworld_design_20260716/DESIGN.md — read it FIRST
and follow it exactly; it is the contract. Manager ruling context:
runs/manager/trackK_20260716/RULINGS.md. This lane = slices 1+2 of DESIGN.md §8 (core/schema +
CLI/metrics). Slice 3 (process_video.py stage wiring) is FORBIDDEN here — Track C owns it.

## 1. HARD RULES
- NO branches, NO commits (manager commits fence-only after ruling).
- Read NORTH_STAR_ROADMAP.md §3.1 + NS-04 table first; DESIGN.md quotes the binding language.
- 4 protected clips EVAL-ONLY. Wolverine/Burlington internal-val use OK.
  Outdoor/Indoor labels NEVER. SEARCH-SWEEP RULE (mandatory, new): every rg/grep/find you run
  repo-wide MUST exclude protected label dirs, e.g.
  `rg --glob '!eval_clips/ball/outdoor*' --glob '!eval_clips/ball/indoor*' ...`.
  Printing even their file paths into the transcript is a violation (it downgraded the
  design lane to PARTIAL).
- Honest reporting. Improvement claims come ONLY from the frozen metric procedure of
  DESIGN.md §5 on independent surfaces — never from a smaller optimizer residual.
- WIDE blast-radius suite mandatory (MPLBACKEND=Agg, real unpiped exit codes, no pipes around
  pytest; attribute any failure per-file; failed>0 while claiming PASS = rejected unless proven
  pre-existing). Known-benign classes: managed-sandbox socket-bind denials; concurrent lanes'
  files are volatile (see §2) — attribute, never fix others' files.
- Every new CLI ships its direct-CLI reference test same-lane (literal command paths, --help,
  synthetic run, exact exit codes) + scaffold-index coverage.
- Artifacts under runs/lanes/oneworld_impl_20260716/. Other lanes' run dirs are READ-ONLY
  evidence — clone inputs into your lane dir before any operation that writes into a run dir.
  NO .patch files: deferred fenced-file changes = inline diff hunks in the report.
- VERIFIED=0 binding: every output preview-band, render_only, not_for_detection_metrics,
  not_for_training. No promotion language anywhere.

## 2. EXPLICIT FILE OWNERSHIP
OWNED (create):
- threed/racketsport/one_world_v1.py — the module: Pydantic models (DESIGN.md §4, house style
  extra="forbid") + the deterministic pass (§3). You may split into one_world_v1.py +
  one_world_v1_metrics.py if cleaner; every new module file starts with one_world_.
- scripts/racketsport/build_one_world_v1.py, report_one_world_metrics.py,
  validate_one_world_v1.py (thin CLIs per DESIGN.md §7).
- docs/racketsport/one_world_v1.schema.json, one_world_v1_metrics.schema.json,
  one_world_v1_validation.schema.json.
- tests/racketsport/test_one_world_core.py, test_one_world_clis.py.
- runs/lanes/oneworld_impl_20260716/** (all run evidence).
OWNED (shared, ADDITIVE ONLY):
- scripts/racketsport/list_scaffold_tools.py: ONLY the three dict entries per CLI from
  DESIGN.md §8.2. The file has other tracks' uncommitted lines in the working tree —
  fresh-read it, append additively, NEVER reformat/reorder/delete anything, and list the
  exact line numbers you added in the report.
FORBIDDEN (live-owned or fenced): scripts/racketsport/process_video.py, threed/racketsport/
orchestrator.py, threed/racketsport/schemas/__init__.py (calpolicy live — your models stay in
your module; produce the ARTIFACT_MODELS + schemas/__init__.py registration as an INLINE DIFF
HUNK in the report, deferred to the integration window), ball_arc_*, placement*.py, tracks
producers, event_head/**, ios/**, web/**, RUNBOOK.md, NORTH_STAR_ROADMAP.md, eval_clips/**
(read-only), runs/manager_stage_sam3d_wolverine_v5_1_20260703T2012Z/** (READ-ONLY — clone).
VOLATILITY WARNING: process_video.py / orchestrator.py / schemas/__init__.py / ball_arc_* /
list_scaffold_tools.py carry other lanes' uncommitted edits RIGHT NOW. Import-time behavior may
shift under you. Your module must not import from process_video.py at all; prefer standalone
json reads + coordinates.py/joint_schema.py (committed, stable).

## 3. OBJECTIVE + ACCEPTANCE NUMBERS
Build the pass, prove it on real data, produce the baseline->fused table.

A1. BUILD: `.venv/bin/python scripts/racketsport/build_one_world_v1.py --run-dir <lane-dir
    clone of Wolverine v5.1> --out <lane dir>/wolverine/one_world_v1.json` EXIT 0; output
    validates (`validate_one_world_v1.py` EXIT 0) incl. schema, input sha256s, band
    inheritance, absence semantics.
A2. DETERMINISM: two consecutive builds byte-identical (sha256 equal).
A3. RAW IMMUTABILITY: sha256 of every consumed input identical before/after build (test + report).
A4. METRICS TABLE (report_one_world_metrics.py; frozen procedure DESIGN.md §5; baseline numbers
    are FIXED = design-lane values, re-derived by the same scorer):
    - M1 ball-at-contact wrist-volume residual: baseline median 7.9737 m (24 events). Fused:
      report median/p90 over SUPPORTED events + separate abstained count w/ reasons. TARGET:
      supported-event median <= 0.60 m AND every event with declared-hitter wrist >5 m while
      another player's wrist <1.5 m must NOT confirm the declared hitter (frame-78 class);
      abstentions are honest, not failures.
    - M2 bounce-to-plane |z-r_b|: baseline + fused reported (median/p90, count); NO exact-snap:
      test proves fused z != r_b identically on nonzero synthetic input; residuals nonzero honest.
    - M3 world coverage @0.5: baseline 0.39 (117/300). Fused coverage >= 0.39 (must not lose
      coverage; support never invented — coverage may only rise via placement tiers that
      genuinely qualify).
    - M4 paddle ambiguity resolution: Wolverine denominator is 0 (gen-1 wrist proxy) —
      report unsupported count + unresolved_legacy_wrist_proxy statuses; prove the resolver on
      synthetic two-hypothesis fixtures (unit tests): resolution w/ M>=0.25, tie -> unresolved,
      reprojection NEVER used to choose (assert in test).
    - M5 reprojection consistency: per-entity median/p90, fused <= baseline + max(1px, 5%);
      any per-sample kill -> refinement suppressed + reprojection_regression recorded (test this
      path synthetically too).
A5. HITTER INFERENCE AUDIT: table of all 24 Wolverine contacts — declared player_id vs fused
    hitter_id + confidence + band (resolved/too_close_to_call/unsupported), per-event wrist
    likelihoods. Include event_index 10 (frame 78) explicitly.
A6. DEMO PARTIAL (honest): clone the demo artifacts (ball_track/court_calibration/net_plane/
    timebase) into your lane dir; derive audio_onsets_v2 CPU from
    data/pbvision_11min_20260713/source_video.mp4 via the existing build_audio_onsets_v2.py CLI
    INTO YOUR LANE DIR; run build_one_world_v1 there: ball-on-court preview + soft bounce
    priors + audio soft evidence only; players/paddles/contacts honestly absent
    (world coverage 0, absence sentinels present, calibration band inherited =
    corrected_unverified/preview). EXIT 0 + validate EXIT 0. This is the demo-walkthrough
    substrate, not a coverage claim.
A7. GEN-2 ATTEMPT (non-blocking): try the DESIGN.md §6 regen recipe in a lane-dir clone to mint
    contact_windows_refined_v1.json/ball_arc_render.json (runner is mid-edit by Track C — an
    environmental failure here is a FINDING w/ exact error, not a lane failure). If minted:
    rerun build consuming gen-2 selectors and report the same metrics table for gen-2.
A8. TESTS: focused one_world suites EXIT 0 (real codes). Coverage must include (from DESIGN.md
    §8.1): both generations, same-run identity fail-closed, covariance weighting,
    repaired/approx/audio discounts, no-snap + out_of_court_bounds flag (never clamp),
    huge-outlier abstention, wrist interpolation guards, hitter ties -> too_close_to_call,
    Viterbi resolve/unresolved, camera-frame cm -> world lift (via coordinates.py typed API),
    determinism, raw-input byte hashes.
A8b. OWNER DIRECTIVE 2026-07-16 (contacts are FUSED PRODUCTS, no single signal suffices;
    neighbor-court audio bleeds): (i) every OneWorldContactRefinement emits a
    contact_evidence_vector: the upstream event sources{audio,wrist_vel,ball_inflection,
    human_review} carried verbatim + co-location likelihood components (ball term, wrist term,
    event-confidence term, audio bounded multiplier actually applied, marker discounts) so the
    combination is inspectable per hypothesis; (ii) TEST audio-only-cannot-confirm: a synthetic
    contact whose only support is audio (no wrist near ball, no ball-track support) must come
    out hitter_band=unsupported with NO refined ball position and NO confidence boost beyond the
    bounded audio multiplier on a nonexistent visual term (i.e., nothing to multiply -> nothing
    confirmed); (iii) TEST neighbor-court-bleed: a strong audio onset with zero visual proposal
    at that time must produce NO contact entry at all in one_world_v1 (fusion never creates
    events; absence sentinel only); (iv) TEST co-location-discount: a declared contact where no
    player wrist is within 1.2 m must be flagged unsupported while the raw event remains
    untouched in its source artifact.
A9. WIDE SUITE with attribution (see §1).
KILL / honesty: fabricated support, snap-to-plane, reprojection-chosen hypotheses, edits to
forbidden files, or improvement claimed from residuals alone = reject yourself and say so.

## 4. EVIDENCE TO READ FIRST
- runs/lanes/oneworld_design_20260716/DESIGN.md (THE contract) + FIELD_VERIFICATION.md +
  baseline_probe.py + baseline_probe_output.json (frozen baseline + scorer conventions).
- runs/manager/trackK_20260716/RULINGS.md (ruling + contention decisions).
- runs/lanes/trackI_placefuse_20260716/SCHEMA.md (preferred player input schema; consume as-is
  when identity gate passes — their Wolverine artifact is a DIFFERENT run, cross-run import
  FORBIDDEN; on this run you will land on the placement_fused tier and must record it).
- runs/lanes/webux3_fixes_20260716/FUSED_WORLD_VIEWER_READINESS.md (downstream schema asks —
  keep absence sentinels + per-entity confidence/band/source refs exactly as the schema draft).
- threed/racketsport/coordinates.py, joint_schema.py, virtual_world.py (baseline, immutable),
  event_fusion.py (bounded audio precedent), confidence_gate.py (ConfidenceProvenance vocab).
- Input run: runs/manager_stage_sam3d_wolverine_v5_1_20260703T2012Z/ (clone to lane dir).
- Demo inputs: runs/lanes/pbv11_headtohead_20260713/rerun_20260715/cpu_events_full/
  pbvision_11min_20260713/ (clone), data/pbvision_11min_20260713/source_video.mp4 (read-only),
  owner CAL seed at runs/lanes/pbv11_headtohead_20260713/rerun_20260715/owner_cal_seed/ (its
  court_calibration_solved.json is corrected_unverified — if you use it instead of the run's
  calibration, inherit that band and record the choice).

## 5. MANDATORY STRUCTURED REPORT (report.json via --output-schema)
- objective_result vs A1-A9 (A7 is non-blocking; PARTIAL only for real target misses).
- acceptance: one row per A1-A9 + one row per metric (baseline/after/target/verdict).
- changes: file:line per change; include the exact list_scaffold_tools.py line numbers added and
  the DEFERRED schemas/__init__.py inline hunk (registration classes/entry) verbatim.
- full_suite: wide-suite command + real counts + per-failure attribution.
- honest_issues: everything (gen-2 outcome, volatility encounters, abstention rates, demo bands).
- next: the Track C wiring request reminder (DESIGN.md §7 text is final) + any follow-ups.
- session_id: your codex session id.
- BEST-STACK DELTA: (c) no stack delta — standalone preview module, not in the default stack;
  config surface arrives with the Track C wiring lane. State this explicitly.

## 6. ANTI-PASSIVE-WAIT
All CPU-local. The full-697s demo audio derivation and builds are minutes-scale; bound any step
that exceeds ~20 min, record, and move on. Ending your turn to wait = lane death.
