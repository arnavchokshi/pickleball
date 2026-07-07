# Harvest review label set — 2026-07-07 (owner-labeled, manager-verified)

Six review clips (one longest non-held-out rally per harvest source), ~480 shown frames, ball boxes
+ 4-level visibility. Every export parsed, counted, and concordance-checked against WASB prelabels
by the manager before filing. HELD-OUT CLEAN: pwxNwFfYQlQ / vQhtz8l6VqU appear nowhere.

| clip | boxes | med px vs machine | within25 | machine-missed caught | corrections |
|---|---|---|---|---|---|
| wBu8bC4OfUY_rally_0001 | 48 | 7.3 | 22/24 | 24 | drop 1 out_of_frame box (f320) at import |
| 73VurrTKCZ8_rally_0002 | 47 | 4.1 | 39/44 | 3 | none |
| Ezz6HDNHlnk_rally_0004 | 34 | 0.0 | 22/29 | 4 | full->clear remap; dup f1848 |
| HyUqT7zFiwk_rally_0001 | 42 | 0.0 | 27/27 | 14 | full->clear remap; dup f9100 |
| _L0HVmAlCQI_rally_0001 | 58 | 0.0 | 48/51 | 6 | full->clear remap; dup f1008 |
| zwCtH_i1_S4_rally_0001 | 45 | 0.0 | 34/34 | 9 | full->clear remap; dups f2951,f3405 |

TOTAL: ~274 human-verified ball boxes across 6 sources; 60 balls the machine missed entirely.

Corrections context: owner read `full` as "fully visible" (natural-English trap; confirmed by boxes
sitting 0.0px on machine detections, which only fire on visible balls). Deterministic full->clear
remap applied where affected; ORIGINALS preserved per-folder as annotations_raw.xml with
MANAGER_NOTE.md. Known import rules: dedupe the 5 listed dup frames preferring the owner shape;
drop the single out_of_frame-tagged box.

Honest limitation: `partial` is under-used after the remap (~21 boxes, mostly clip 1) — this pass's
4-level signal is mostly clear-vs-absent. Fine for WBCE seeding; future passes should emphasize
partial on blurred/occluded balls.
