# FIX ROUND for person_mixed_20260722 — three structural findings (runs/lanes/person_mixed_20260722_review/review.json)

Same ownership. The reviewer confirmed plan/caps/determinism/teacher/caveat/bars/licensing/decode
honesty — do not disturb those. Close the three FAILs:
1. QUARANTINE BY CONTENT IDENTITY: refusal must key on canonical decoded identity — normalize/
   decode path and ID strings (percent-encoding, case, unicode forms) before matching, AND refuse
   by media content SHA against the pinned protected/compare SHA registry regardless of the
   claimed ID pairing (the reviewer paired a compare-only SHA with a permitted ID and it passed;
   an ignored manifest's ID-to-SHA association must not be trusted — recompute or require bound
   verification). Tests: percent-encoded aliases for all three quarantine classes; compare-SHA
   under permitted ID.
2. VALIDATION PURITY STRUCTURAL: validation rows must never be rewritten from injected
   pseudo/teacher provenance — refuse (never coerce) any row whose provenance fields conflict
   with human-annotation origin; add a content-identity check so the same media/frame bytes
   cannot appear on both sides under different sample IDs; ship the final-list validator as an
   EXECUTABLE check the GPU lane must run (not prose), and make the pack manifest reference it.
3. CLOSED-P1 STRUCTURAL BINDING: bind the anchor/validation inputs to the closed lane's artifact
   hashes at build time (record + verify closed_manifest/train-list/rows SHAs; refuse on drift),
   so fidelity is enforced, not coincidental.
Report to report_fix1.json. Suite: no NEW failures beyond the known environmental set.
