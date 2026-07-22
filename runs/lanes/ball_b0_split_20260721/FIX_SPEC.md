# FIX ROUND for ball_b0_split_20260721 — address ultra-review REJECT (runs/lanes/ball_b0_split_20260721_review/review.json)

Same HARD RULES + FILE OWNERSHIP as spec.md (build_ball_regroup_split.py + its test only).
Read the review JSON first. The judge CANNOT certify until these land:

- PRIMARY BLOCKER: boxless-as-negative requires attestation. Implement a required
  --negative-attestation flag consuming an owner attestation JSON (expected at
  cvat_upload/exports/w7_audit_stratum_20260709/owner_attestation.json once the owner signs;
  schema: {task_id, export_sha256, statement, attested_by, attested_utc, all_frames_inspected:
  true, boxless_means_no_ball: true}). Without a valid attestation matching the export SHA
  (fea4b9529a4020ee577c5702478753fa4ea84f99c51a9e3ef16e87e96d7fc104), boxless frames are
  UNATTESTED_NEGATIVE, evaluation_eligible=false, and the tool exits BALL_NO_CLEAN_JUDGE.
  (The orchestrator is requesting the owner attestation in parallel — build the mechanism now.)
- HIGH: bind the export to task-87 provenance: task fingerprint/import ledger, prelabel_zip must be
  null/absent (scratch-only), manual-source requirement — refuse otherwise (no unconditional
  'scratch' stamping).
- HIGH: content-bind ALL lineage inputs (selection manifests, legacy prelabel tracks, review
  exports) with pinned hashes verified at runtime; refuse on mismatch.
- HIGH: production mode must require the exact canonical protected set (all 4 protected videos +
  2 additions); a nonempty override cannot yield the clean verdict.
- Fix the lineage semantics defect: an absent original model record with final visibility
  out_of_frame must NOT classify as confirmed_prelabel via the absence shortcut (the reviewer's
  example: wBu8bC4OfUY_rally_0001:000320). Define and test explicit semantics.

Acceptance: all fixes + adversarial tests (attestation missing/mismatched-SHA/false flags; wrong
protected set; tampered lineage input; auto-sourced export refusal); re-run the real pipeline and
report the judge status honestly (expected: BALL_NO_CLEAN_JUDGE / AWAITING_ATTESTATION until the
owner file lands — if it landed during your run, re-run and report the attested judge). Wide suite
no NEW failures vs the known environmental set. Report to report_fix1.json (schema-valid).
