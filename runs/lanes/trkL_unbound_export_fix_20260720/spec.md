# trkL_unbound_export_fix_20260720 — surgical fix: unbound fragments must not be claimed players

Codex gpt-5.6-sol xhigh. The GPU scorecard (runs/lanes/p0i_scorecard_20260720/vm_pull/) DECISIVELY FAILED: player_selection.py:1764 returns slot_players + unbound_players as tracks.json "players" — 182 (burlington) / 38 (wolverine) unbound fragments exported as top-level claimed players → spectFP 4→651 (wolv), 0→7783 (burl). Env-fidelity was EXACT (0.000e+00 delta on all 10 frozen scalars), selection-OFF byte-identical, interpolated markers work — the defect is ONLY the enabled-path export shape. VERIFIED=0.

## THE FIX (surgical, nothing else)
1. tracks.json "players" = the 4 BOUND slots ONLY (a claimed player = a bound slot). 
2. Unbound fragments are PRESERVED (the review requirement stands — never silently deleted) in a SEPARATE ADDITIVE top-level key "unbound_observations" in tracks.json: same frame payloads, each with its selection_state/abstention reason + raw UIDs. Consumers/scorers that read "players" are untouched; audit/debug/honesty consumers read the new key. Update docs/racketsport/player_selection_report_schema.json accordingly.
3. Dropped fragments stay report-side (unchanged). Slot binding/veto/recovery logic UNTOUCHED — do not tune any threshold (the card re-run is one-shot, no tuning allowed).

## HARD RULES
- NO commits/pushes. Ultra re-review before commit. Selection-OFF must stay byte-identical (golden test green). Focused + wide suite (MPLBACKEND=Agg), attribute failures. File-disjoint lanes live (court, finetune) — only touch: threed/racketsport/player_selection.py, scripts/racketsport/select_players_from_pool.py (if needed), docs/racketsport/player_selection_report_schema.json, tests/racketsport/test_player_selection.py.

## ACCEPTANCE
- New test: enabled-path tracks.json "players" length == bound slots (<=4); unbound fragments present under "unbound_observations" with reasons; NONE of them in "players".
- Local re-run of the selection arm on the PULLED frozen artifacts (runs/lanes/p0i_scorecard_20260720/vm_pull/p0i_pull/ has the pools/calibrations/GT + the scorer) — score it LOCALLY as a smoke: expect players-count sane (4-ish per clip) and spectFP to collapse from 651/7783 (CPU association isn't score-faithful for the frozen card, so this is a SMOKE, not the card — say so honestly; the one-shot GPU card re-run is the manager's).
- Selection-OFF byte-identical; all prior 70 tests green; wide suite attributed.
