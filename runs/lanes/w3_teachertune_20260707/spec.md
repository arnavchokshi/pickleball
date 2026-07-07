# LANE w3_teachertune_20260707 — 2D teacher-gate tuning + 3-point teacher production on all 40 harvest sidecars (wave-3 #5, local CPU)

## HARD RULES (violating any = lane rejected)
- You are Codex lane `w3_teachertune_20260707`. Work ONLY inside /Users/arnavchokshi/Desktop/pickleball.
- No git branches/commit/push. You own NO repo source files: repo CLIs are RUN, never edited. Everything you produce lives under runs/lanes/w3_teachertune_20260707/. If a gate CLI has a bug that blocks the sweep, STOP and report it (propose a diff; do not edit).
- Data: harvest clips + their p01b raw WASB prelabel sidecars ONLY. Held-out videos (IDs in runs/manager/heldout_eval_ledger.md — two YouTube IDs) appear NOWHERE: re-assert your 40-clip set excludes them (the p01b shard set already does; verify and record the assertion output).
- No GPU, no network. Pure local CPU. `.venv/bin/python` always.
- Honest reporting: you CANNOT measure true precision/recall (no ground truth exists yet) — never claim it. Objective proxies + rendered evidence for the manager's visual ruling is the deliverable.
- Read first: BUILD_CHECKLIST last ~15 bullets + the fleetseed dry-run artifacts (runs/lanes/w3_fleetseed_20260707/ — esp. delta_table_vs_p01b_prelabel.json, timing_summary.json, and the teacher_dry_run/ layout + metadata schema, which you must stay compatible with).

## CONTEXT (established by the fleetseed dry-run)
- The calibration-free 2D teacher chain is: `filter_ball_temporal.py --mode ballistic` → `filter_ball_ransac_arc.py` → `smooth_ball_kalman_rts.py` (+ `build_ball_2d_post_summary.py` for stats). 3D stages hard-require court calibration harvest clips don't have — OUT OF SCOPE.
- At defaults (ransac max_residual_px=5.0), coverage falls ~71-74% → ~27% visible on the 2 dry-run clips (~63% cut, suspiciously uniform). The gate demonstrably catches real false-locks (a 24-frame static-feature lock ~101px off was correctly killed). The 5.0px default may be tuned for tennis-scale imagery, not ours.
- `filter_ball_local_search.py` (image-guided refinement, calibration-optional) exists and was NOT used in the dry-run — evaluate it as an optional 4th stage if cheap; skip with one sentence if not.
- Raw WASB sidecars for ALL 40 harvest shard clips already exist locally (p01b lane). Teacher production is pure CPU (~7s/clip at dry-run rates).

## OBJECTIVE
1. **Sweep**: on ALL 40 clips' raw sidecars, sweep the two gate knobs (`filter_ball_ransac_arc --max-residual-px` over ~{5, 8, 10, 12, 15, 20} and `filter_ball_temporal` ballistic residual over a comparable grid — read the CLIs for exact flag names/defaults). For each grid point aggregate objective proxies per clip + corpus-wide: visible coverage %, mean/median kept-track arc residual, longest-gap distribution, segments count, fraction of kalman-filled (imputed) points, agreement-with-raw-at-high-confidence (kept fraction among top-confidence raw detections), false-lock canary (does the known clip-2 24-frame static false-lock stay dead at this point — it must).
2. **Pick 3 candidate operating points**: strict (≈defaults), moderate, permissive — justified by the proxy curves (e.g., knee of coverage-vs-residual). The false-lock canary must be killed at ALL 3 (any point that revives it is disqualified).
3. **Render visual evidence for the manager**: for 4 diverse clips (different channels/lighting; include one dry-run clip for continuity), render overlay evidence at each of the 3 points via `render_ball_track_overlay.py` — produce compact strips or short segments (bounded total size ≤ ~150MB): raw-vs-teacher overlaid or side-by-side, focused on 2-3 interesting windows per clip (a dense rally window, a gate-cut window, a kalman-fill window). Index them in an EVIDENCE.md with exact file paths + what to look for in each.
4. **Produce all 3 teacher sets for all 40 clips**: runs/lanes/w3_teachertune_20260707/teacher_sets/{strict,moderate,permissive}/<clip_id>/teacher_ball_track.json + metadata (fleetseed-compatible schema: teacher="chain_gated_2d", gates_applied with exact thresholds, gates_skipped {3d stages: no_court_calibration}, code_sha, operating_point). The manager blesses ONE set after visual ruling — produce, don't promote.
5. **Corpus stats table**: per operating point: total visible points, coverage % distribution (min/median/max across clips), imputed fraction, per-clip outlier list (clips whose coverage deviates >2σ — candidates for exclusion from SST seeding).

## ACCEPTANCE
- Sweep grid completed on 40/40 clips with the proxy table (machine-readable JSON + human MD).
- 3 operating points chosen with one-paragraph justifications; canary dead at all 3.
- Overlay evidence rendered + indexed (EVIDENCE.md) for 4 clips × 3 points, ≤150MB total.
- 3×40 teacher sets produced, schema-compatible with the fleetseed dry-run metadata.
- Held-out assertion output recorded. Wall-clock + disk delta reported.
- No repo file modified (git status proof scoped to your paths — concurrent lanes make the tree dirty; list only what YOU created).

## STRUCTURED REPORT
objective_result vs acceptance; the corpus stats table (compact); your recommended operating point + why (the manager makes the final visual ruling from EVIDENCE.md — say where to look); HONEST ISSUES (esp. what proxies CANNOT tell us without ground truth); NEXT; artifact index.
