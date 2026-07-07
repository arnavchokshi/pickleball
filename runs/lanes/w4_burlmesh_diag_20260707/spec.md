# LANE w4_burlmesh_diag_20260707 — micro READ-ONLY diagnosis: burlington virtual_world missing-mesh-vertices notice

## OBJECTIVE
The wave-3 closeout carried queue item #5: "burlington virtual_world missing-mesh-vertices notice".
Find the exact notice, root-cause it, and propose a bounded fix. **No production code changes.**

## METHOD (state each assumption as a CHECK — verify, then proceed)
1. CHECK: locate the notice. Grep the freshest burlington pipeline artifacts under `runs/` (viewer
   manifest `replay_viewer_manifest.json`, `PIPELINE_SUMMARY.json`, mesh index files, notices/QA
   fields) AND the codebase (`grep -rn "mesh" threed/racketsport/ web/replay/src/ | grep -i
   "vertices\|missing"`) for the emitting site. If you cannot find any such notice in artifacts or
   code, report that honestly with your search transcript and STOP — do not invent a defect.
2. Root-cause: why burlington specifically (mesh scheduling? mesh index build? a per-frame count
   mismatch between selected mesh frames and vertices present?). Compare against the other 3 eval
   clips' artifacts.
3. Severity: does it affect the viewer render, the mesh layer, any gate, or is it cosmetic telemetry?
4. Bounded fix proposal: files/functions, shape, blast radius, how it is proven.

## EVIDENCE POINTERS
Wave-3 decisive-run dirs under `runs/` (find the freshest 4-clip set); mesh-index tooling in
`threed/racketsport/` (grep "mesh_index"); `web/replay/src/` mesh layer; frame_rating/body_compute
mesh scheduling (contact-dense logic).

## HARD CONSTRAINTS
READ-ONLY everywhere except `runs/lanes/w4_burlmesh_diag_20260707/` (REPORT.md + any small
analysis scripts). Never touch ios/, runs/manager/, eval labels. No git ops. `.venv/bin/python`.

## STRUCTURED REPORT
objective_result PASS = notice located + root cause + severity + bounded proposal (or an honest
"notice does not exist in any artifact" with search proof). full_suite: read-only, 0 failed.
honest_issues: anything else wrong you noticed in the burlington artifacts.
