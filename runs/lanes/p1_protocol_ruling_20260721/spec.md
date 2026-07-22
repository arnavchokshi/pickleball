# ULTRA RULING REQUEST: amendment to P1's human box-quality CHECK recording mechanism (read-only)

Context: EXACT_PLAN §3.4 P1 CHECK requires a human-measured card: sampled box precision >=98%,
visible on-court-person recall >=95%, any source <90% quarantined. The staged audit pack has 182
frames with drawn boxes (runs/lanes/person_p1_roboflow_20260721/). The owner REFUSES per-row CSV
filling (not available as an option). PROPOSED AMENDED PROTOCOL:
- Owner performs one full pass of the 182-image gallery; reports ONLY exceptions in free text
  (frame id + defect type: box-on-non-person | on-court player unboxed), then signs a structured
  attestation binding the audit-pack manifest SHA + full image-ID list: "I viewed all 182 images;
  every unflagged image has all-correct person boxes and no unboxed on-court player."
- The coordinator mechanically converts flags+attestation into the completed review CSV
  (unflagged rows: correct/visible/complete; flagged rows: the stated defect, with per-frame
  follow-up counts only for flagged frames). Conversion is logged + hash-bound.
- Metrics computed over ALL 182 rows exactly as the CSV protocol would.
Precedent: the task-87 boxless-negative attestation (owner attests all frames inspected) was
accepted by the R2 reviewer of ball_b0_split as CLOSED.
Notes: (a) an extra box on an off-court referee is NOT a precision error under the pack's schema
(person box correct; on-court status is a separate column) — the amendment must preserve that
schema distinction in flag taxonomy; (b) the owner's informal "90%" impression is NOT a metric and
is not used.
QUESTION: is the amended protocol information-equivalent to the per-row CSV for the P1 CHECK,
with human authority preserved on every row? RULE in final JSON:
{ "ruling": "ACCEPT_AMENDMENT" | "ACCEPT_WITH_CONDITIONS" (exact conditions) | "REJECT",
  "minimal_alternative_if_reject": "<the smallest owner action that would satisfy the CHECK>" }
Consider: attention-decay risk vs CSV-filling, binding requirements, auditability, whether blanket
attestation of unflagged rows preserves box-level measurement validity.
