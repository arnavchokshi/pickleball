# Binding-fix ruling (Fable, 2026-07-20): NOT COMMITTED — judge contamination caught by ultra review

The projection (wolverine 0 spectFP / 0 sw / cov4 0.78) is mechanically real (UIDs one-to-one, no
fabrication) but NOT valid evidence: the lane iterated against the frozen scorer (saw 0.6767,
changed binding, re-scored to 0.7800), added an UNREGISTERED hard gate (court_presence>=0.5 vs the
registered soft fusion S>=0.5), and introduced a registered-owner rebind path that BYPASSES the
stitch veto (a vetoed stitch was rebound in the projection). Full findings:
runs/lanes/ultra_review_binding_20260720/log.txt.

DISPOSITION: work preserved at binding_fix_UNCOMMITTED.patch; tree reverted to the committed clean
selection layer (fabrication-prevention + honest export, which passed its reviews). GPU card NO-GO
on Wolverine/Burlington for this component — scorer-guided development burned them as an accuracy
judge; they can prove reproduction only.

NEXT (queued): apply the reviewer's exact must-fixes (restore/preregister the fusion rule on
non-protected data; owner-rebind must respect stitch-veto absent independent provenance; disclose
scorer-guided dev; archive/hash exact scorer inputs; add boundary/source-reuse/owner-veto/7m/s
regressions) → FREEZE code/config/scorer → ONE preregistered evaluation on an UNTOUCHED judge
(the strict-holdout protected pair via heldout_eval_ledger prereg + owner gate, or fresh
human-labeled pb.vision clips). VERIFIED=0.
