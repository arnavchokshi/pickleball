# Manager ruling r1 (2026-07-10 ~02:1x PDT, Fable bg 03267a94)

Diagnosis ACCEPTED. All three load-bearing claims independently re-verified by the manager:
(1) b437b4118 introduced DEFAULT_MAX_SCHEDULED_FRAMES=1200 (still live at HEAD);
(2) w7_critique zwcth45s run FAILED at calibration — the spec's "rev-9 handled this clip" premise
    was WRONG (sourced from the demo session's memory; correction banked);
(3) pre_fix_repro_attempt2.log ends with the exact missing-frame signature.
Kill rule is CLEARED: its purpose (no speculative fixes) is satisfied — the cause is proven by
local repro. FIX IS AUTHORIZED per resume order r2.
