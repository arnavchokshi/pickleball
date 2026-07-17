# Decision table — trk_rfdetr_prod_20260716

Status: **STOPPED AT GATE 1**. `VERIFIED=0`. No RF-DETR inference, no pooldiag execution, no flip proposal, and no stack/code/config change.

## Gate outcomes in strict order

| gate | outcome | evidence |
|---|---|---|
| 1. ENV-FIDELITY | **FAIL — STOP** | Burlington matched; Wolverine diverged by IDF1 `-0.0875705077`, cov4 `+0.0566666667`, 2 switches, and 9 spectator FPs. |
| 2. POOLDIAG Phase 1 | NOT EXECUTED | Blocked by Gate 1 STOP. |
| 3. RF-DETR-L production pool | NOT EXECUTED | Blocked by Gate 1 STOP. |
| 4. Frozen-card score | NOT EXECUTED | Blocked by Gate 1 STOP. |
| 5. Verdict | **NO-ATTEMPT** | A candidate cannot be scored honestly through a non-faithful local association environment. |

## Environment reproduction table

Detector runtime is not applicable: arm 0a consumes frozen pools and no candidate detector ran.

| clip | row | IDF1 | cov4 | switches | spectator FP | far-off-court FP | det ms/frame |
|---|---|---:|---:|---:|---:|---:|---:|
| burlington | frozen YOLO26m baseline pin | 0.8830775881 | 0.7116666667 | 0 | 0 | 0 | — |
| burlington | local arm 0a reproduction | 0.8830775880700238 | 0.7116666666666667 | 0 | 0 | 0 | — |
| burlington | RF-DETR-L production | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| wolverine | frozen YOLO26m baseline pin | 0.8515962036 | 0.7600000000 | 0 | 0 | 0 | — |
| wolverine | local arm 0a reproduction | 0.7640256959314775 | 0.8166666666666667 | 2 | 9 | 0 | — |
| wolverine | RF-DETR-L production | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |

## Failure diagnosis

All five harness MD5 pins, the OSNet SHA-256, and the frozen YOLO26m SHA-256 matched. Burlington `tracks.json` was byte-identical to detbench (`ff603695...860f`). Wolverine was not (`5069ef6e...02a` local versus `54e3e4ba...17e6` detbench).

The Wolverine inputs and association config matched, including 2,006 embeddings, 20 fragments, 15 merged fragments, and 17 selected fragments. The local CPU result synthesized 230 frames versus detbench's 213, raising exact-four-player frames from 228 to 245 while introducing two identity switches and nine true spectator FPs. Therefore this local CPU environment is not score-faithful for the frozen card.

## Verdict

**NO-ATTEMPT.** Strict STOP at Gate 1. RF-DETR-L's Apache-2.0 posture is recorded, but there is no local candidate result and no `FLIP_PROPOSAL.md`.
