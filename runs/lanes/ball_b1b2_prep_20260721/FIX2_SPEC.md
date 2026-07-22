# FIX ROUND 2 for ball_b1b2_prep_20260721 — three remaining findings (review_r2.json)

Surgical scope; same ownership. All other checks PASSED — do not disturb them.
1. GAP SEMANTICS: the frozen temporal rule must bound the TOTAL teacher-only gap length at 2
   frames (anchors at 8 and 12 with interior frames 9-11 teacher-only = gap of 3 = REFUSED).
   Fix the per-side implementation in both the builder and the trainer's duplicate validation;
   regression test with the reviewer's exact 8/10/12 case + boundary cases (gap exactly 2 accepted,
   3 refused).
2. CONTRADICTION HANDLING: if any intermediate frame inside the candidate bridge carries a
   high-confidence WASB observation that DISAGREES spatially (> radius) with the teacher path, the
   bridge is REFUSED — the search must not skip outward past contradictory evidence. Same rule in
   builder and trainer validation; tests.
3. JUDGE-ROW ALIASING: the trainer's generic CVAT input path must apply the same canonical
   clip/media identity resolution (content SHA, not path/name) used elsewhere, so an aliased
   HyUqT7zFiwk/Ezz6HDNHlnk row cannot enter training under a renamed clip/media identity; resume
   checkpoints must carry and enforce dataset-provenance (refuse resume when the dataset identity
   set differs from the checkpoint's recorded one). Tests: aliased judge row refused; resume with
   swapped dataset refused.
Report to report_fix2.json. Suite: no NEW failures beyond the known environmental set.
