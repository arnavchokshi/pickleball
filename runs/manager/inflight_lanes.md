# In-flight lanes (write at session end, read at session start — FABLE_OPERATING_MANUAL §14 step 9)

One row per still-running lane so the next session neither double-dispatches nor loses a resume.

| lane | kind | session/task id | resume command | owned files | vm | expected done | dispatched |
|---|---|---|---|---|---|---|---|
| w7_speedgate_20260709 | sonnet-gpu (NOT DISPATCHED) | — | spec READY at runs/lanes/w7_speedgate_20260709/spec.md; dispatch as Sonnet GPU lane the moment gcloud auth is restored (owner `gcloud auth login` — typed STOP of 2026-07-09 ~05:50 in OWNER_CHECKIN_20260709.md) | NO repo edits (VM timing runs + GATE-1b arm + template re-bake) | pickleball-h100-w7speed (to create) | ~2.5-3.5h once fired | PARKED |

_(WAVE 7 CLOSED 2026-07-09 ~07:1x: ~20 lanes/rounds DONE+ruled; final clean adjudication 3315/0/26 FULLY GREEN; fleet ZERO VMs (auth-down but last list-confirmed at w7ballc teardown 12:39Z with nothing running); best_stack rev-9 reconciled zero unaccounted gains; scorecard = BUILD_CHECKLIST [WAVE-7 COMPLETE]; wave-8 marching order = runs/manager/wave8_boot_prompt.md. Only the parked speed gate above carries. Watchdog running in AUTH_DOWN mode until its 7h heartbeat.)_
