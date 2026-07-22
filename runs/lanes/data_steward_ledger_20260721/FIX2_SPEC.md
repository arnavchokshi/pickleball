# FIX ROUND 2 for data_steward_ledger_20260721 — close the four round-2 findings
(runs/lanes/data_steward_ledger_20260721_review/review_r2b.json)

Same ownership. Neutral implementation of the reviewer's required_fixes, verbatim intent:
1. Argv completeness: parse every token including concatenated short options (-dPATH) and
   equals-forms; any unsupported dash-prefixed form is treated as ambiguous and REFUSED; add the
   reviewer's -druns/... regression case plus equals-form variants.
2. Selector roles: a clean-subset selector counts only when it appears under a recognized
   data-bearing INPUT role; references in output/save/log/config-only roles are rejected; every
   resolved data reference in the command must use the same selector; tests for the
   --teacher-output shape.
3. Ledger correction (eval_clips_ball_protected_4): list and hash all four person_ground_truth.json
   files (2400+3379+4480+1200=11,459) as the PERSON binding; separate owner_IMG_1605/pbvision_11min
   files out of this asset's byte/file accounting into their own rows or lineage.
4. Ledger correction (roboflow_person_core_20260706): create an immutable hashed 15,312-row PERSON
   selector artifact; fix the consumer record (the 34,658-row BALL pretrain consumed the BALL
   subset, not this PERSON subset — represent both truthfully); recompute snapshot utilization.
Re-run the audit snapshot and report N assets / never-queued / state counts after correction.
Report to report_fix2.json. Suite: no NEW failures beyond the known environmental set.
