# trkL_selection_fix2_20260720 — Lane 1 round-2: independent-evidence + drop-honesty (P0-I flagship)

Codex gpt-5.6-sol ultra. The round-1 fix (in the working tree, player_selection.py) resolved 6/8 ultra findings (unbound preserved, no-enrollment no-bypass, raw UID one-to-one, Layer C invoked, interpolated survives export, enrollment ownership, f44<->f87 refused, selection-OFF byte-identical). This round-2 fixes the 2 UNRESOLVED findings from runs/lanes/ultra_review2_20260720. VERIFIED=0; will get an ULTRA re-review before commit.

## HARD RULES
- NO commits/branches/pushes. Re-reviewed by gpt-5.6-sol ULTRA before commit — fix BOTH findings or rejected. No gaming. Honest reporting.
- Selection-OFF byte-identical (keep the round-1 golden test green). Focused + wide suite (MPLBACKEND=Agg), attribute failures.
- YOUR FILES: threed/racketsport/player_selection.py, scripts/racketsport/select_players_from_pool.py, docs/racketsport/player_selection_report_schema.json, tests/racketsport/test_player_selection.py. schemas/__init__.py TrackFrame.interpolated is already landed (round-1) — leave it. Do NOT touch other lanes' files.

## THE 2 BLOCKING FINDINGS (from the joint re-review):
1. **Correlated evidence authorizes destructive drops (THE core issue).** `court_persistence` and `temporal_motion` are BOTH derived from the same `world_xy` (player_selection.py:243, :1379, :1739, :1990) — so the "two independent evidence classes required for any destructive action" rule is violated: in the fresh Wolverine run, 19/22 drop fragments (779/783 UIDs) were destroyed SOLELY by these correlated geometry signals while APPEARANCE was ACCEPT or DEFER. Dropping a real detection whose appearance says ACCEPT, on one geometry signal wearing two hats, is exactly the P0-I harm class (losing real players). FIX: treat ALL world_xy-derived court/persistence/motion signals as ONE evidence class; require a GENUINELY INDEPENDENT modality (OSNet appearance embedding) to agree before any destructive DROP. If appearance is ACCEPT/DEFER, a real detection may NOT be dropped on geometry alone — keep it unbound instead. Two-independent-evidence for destroy means {geometry-class} AND {appearance-class}, not two geometry booleans.
2. **Drop-reason honesty.** 779 dropped detections carry `reasons` with POSITIVE facts (`identity_accept`, `fusion_at_or_above_0_5`) rather than the actual destructive trigger (player_selection.py:1428). FIX: emit the ACTUAL negative trigger that caused each drop (what evidence, what threshold), constrain the allowed drop-trigger vocabulary in the report schema, and test it. A drop reason must name the destroying evidence, never a positive fact.

## ALSO (round-1 residual, finding 8 partial): repair OR retract the dependency-complete VM handoff claim (VM_EVAL_PLAN.md:76 references an unprovisioned feeder) — if you can't make the handoff dependency-complete, retract the PASS claim honestly.

## ACCEPTANCE (pre-registered): 
- New test: an identity-ACCEPT + off-court + motion-inconsistent fragment MUST remain UNBOUND (not dropped) — geometry alone cannot destroy an appearance-accepted real detection.
- On the Wolverine diagnostic, the count of geometry-only drops of appearance-accepted fragments is 0 (or each such drop now additionally requires appearance disagreement).
- Drop reasons name the destroying evidence; schema constrains the vocabulary; test enforces it.
- f44<->f87 bridge STILL refused; selection-OFF byte-identical; focused + wide suite with attribution.
- VM handoff claim honest (repaired or retracted).
