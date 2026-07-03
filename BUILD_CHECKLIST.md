# Build Checklist

Last updated: 2026-07-03.

This is the operational board. It should stay short enough that a new agent can
read it before touching code. For final goal and truth boundaries, read
`MASTER_PLAN.md`; for commands, read `RUNBOOK.md`; for tier placement, read
`CAPABILITIES.md`.

No row is `VERIFIED`.

## Status Board

| ID | Area | Status | Current blocker | Next useful action |
|---|---|---|---|---|
| DOCS-1 | Documentation | IN-PROGRESS | full cleanup proof is still incomplete | Keep docs small; continue truth/dead-code/storage audits without adding new narrative docs. |
| CAL-1 | Court calibration | SCAFFOLD/PREVIEW | no no-tap solver has passed reviewed PCK/reprojection gates | Keep v1 tap-assisted/metric seed; score any new solver fail-closed. |
| TRK-1 | Person tracking | IN-PROGRESS | pre-registered candidates still fail coverage/identity/spectator gates | Improve detector/data leverage; do not repeat exhausted association-only sweeps. |
| BALL-1 | Ball tracking/events | SCAFFOLD | reviewed F1/contact/in-out gates not passed | Use reviewed data and model-side candidates; preserve gray-zone behavior. |
| BODY-1 | 3D body | SCAFFOLD | independent-GT world-MPJPE gate missing/failing | Use external/independent GT; never promote candidate-label reviews. |
| PHYS-1 | Foot/physics | INTERNAL-VAL DONE | Wolverine internal-val proof is not protected-clip/product proof | Reverify on protected/representative clips after upstream gates improve. |
| RKT-1 | Paddle pose | SCAFFOLD | no true paddle-face corner/reference GT | Collect/consume true-corner or marker/reference data before pose claims. |
| IOS-1 | Native iOS/live tier | SCOPED PASS | full physical capture/import/live overlay/replay proof still incomplete | Run real device capture/import/live tier and report exact evidence. |
| RPL-1 | Replay/scrubber | SCOPED PASS | review viewer and scoped assets are not production replay verification | Verify native/web playback, size, FPS, and visual QA against a current bundle. |
| E2E-1 | Full pipeline | SCAFFOLD/SCOPED PASS | no clean clip meets all component gates plus replay SLA | Rerun `process_video.py` only after component gates improve. |
| DATA-1 | Data/eval policy | IN-PROGRESS | protected eval/training boundaries need constant enforcement | Keep guards/tests active; pre-register held-out evals. |

## Count Summary

| status | count |
|---|---:|
| IN-PROGRESS | 3 |
| INTERNAL-VAL DONE | 1 |
| SCAFFOLD | 3 |
| SCAFFOLD/PREVIEW | 1 |
| SCOPED PASS | 2 |
| SCAFFOLD/SCOPED PASS | 1 |

## Recent Handoffs

- [A100-SESSION-3 2026-07-03, manager-accepted] All SAM3D Phase D gates PASS on shipped defaults: steady 32.23 ms/person (≤55), first call 0.564s (≤1.0, warm-2). Wolverine ball_aware_100 dispatch succeeded, zero Skeleton3D validation errors: 4 players / 1102 annotated frames / 0 implausible / 184 mesh frames; BODY GPU 311s ≈ $0.117/clip. Artifacts: `runs/a100_sam3d_validation2_20260703T0647Z/production_remesh/wolverine_ball_aware_100/`. In flight: viewer staging of new skeletons + 4-clip wall-to-wall E2E timing with reproducibility packets.

## Rules For Updating This Board

- Keep one row per active area. Do not append chronological narratives here.
- Every status upgrade must name the command, run path, test result, device run,
  or label gate that proves it.
- If a row is scoped, include the scope in the handoff or run artifact. Do not
  let scoped evidence become a global claim.
- If a lane generates a long report, store it under `runs/` and summarize only
  the actionable result here.

## Active Priorities

1. **CAL:** maintain tap-assisted/metric seed path for v1 and fail closed on
   unverified automatic proposals.
2. **TRK:** improve detector/data and strict spectator/background handling.
3. **BALL:** pursue reviewed-label ball quality and contact/in-out gates without
   hiding uncertainty.
4. **BODY:** get independent GT for world-MPJPE and keep candidate labels out of
   promotion paths.
5. **iOS/RPL:** prove real-device capture/import/live overlay and current replay
   playback from the same artifact chain.
