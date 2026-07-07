# Court-keypoint label set — 2026-07-07 (owner-labeled, manager-reviewed)

Source: CVAT task 13 (`racketsport_metric15_court_keypoints_20260707_6frames`), exported
"CVAT for images 1.1" → `court_keypoints_metric15_20260707_annotations.zip` (+ unzipped
`annotations.xml`). Convention: metric-15 (`threed/racketsport/schemas/__init__.py:30-46`).
Manager parse-audited per frame; held-out sources excluded by construction (6 legal sources only).

| frame (source) | points | status / owner caveat |
|---|---:|---|
| 73VurrTKCZ8__rally_0002 abs_3808 | 15/15 | FULL |
| HyUqT7zFiwk__rally_0001 abs_10195 | 15/15 | FULL |
| zwCtH_i1_S4__rally_0001 abs_3636 | 15/15 | FULL |
| _L0HVmAlCQI__rally_0001 abs_509 | 7/15 | PARTIAL — **tennis court, pickleball lines don't align well**; owner labeled only confidently-placeable points (far baseline trio + near_baseline_center + all 3 net-top) |
| wBu8bC4OfUY__rally_0001 abs_10248 | 4/15 | PARTIAL — same tennis-overlay issue (near_baseline_center + all 3 net-top) |
| Ezz6HDNHlnk__rally_0004 abs_10677 | 1 (DROP) | **OWNER-DECLARED SKIP** — camera angle too low, near court barely visible. The single `near_baseline_center` point is a stray pre-skip click: **importers MUST drop this frame entirely.** |

## Import rules (binding, same pattern as harvest_review_20260707)
1. Drop the Ezz6HDNHlnk frame + its stray point entirely (owner-declared skip).
2. Partial frames import with their honest per-frame point subsets — the metric-15 loader supports
   partial sets; do NOT impute missing points.
3. Tennis-overlay caveat travels with the two partial frames as a quality flag: these are exactly
   the "tennis-overlay tolerated" hard-case class the P4 court auto-find bar names — valuable as
   HARD EVAL cases, weaker as clean calibration GT.

## Follow-ups queued for the P4 lane
- Re-select a replacement frame for source Ezz6HDNHlnk (different rally/timestamp, court-visible
  angle) — the selection heuristic picked a poor frame; one-frame re-import task.
- Consider whether _L0HVmAlCQI / wBu8bC4OfUY have cleaner-court rallies for supplementary frames;
  tennis-overlay frames stay regardless (hard-case eval value).
- Net-top points confirmed labeled at the net TOP tape (sidelines + sagging center) per convention.

Owner time: ~10 min for the pass incl. judgment calls. Labels/hour posture consistent with the
harvest-review datum (240 frames/hr for boxes; sparse points are faster per frame).
