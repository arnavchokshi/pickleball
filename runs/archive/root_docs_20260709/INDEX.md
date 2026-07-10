# Root Documentation Archive — 2026-07-09

This directory preserves the exact pre-consolidation narrative documents that
were active in the working tree when `NORTH_STAR_ROADMAP.md` became the single
product and execution authority.

These files are immutable historical context. They contain completed work,
superseded plans, old wave queues, intermediate measurements, and decisions
that were later corrected. They must not be used as the current queue or as
promotion evidence without reopening the linked run artifact.

Current authority: `NORTH_STAR_ROADMAP.md` at the repository root.  
Commands and actual runtime behavior: `RUNBOOK.md`.  
Repository rules and navigation: `AGENTS.md`.

## Archived files

| File | Former role | Why archived |
|---|---|---|
| `NORTH_STAR_ROADMAP_PRE_CONSOLIDATION.md` | 2,482-line roadmap, research synthesis, phase checklist, and wave history | Replaced by the concise North Star; preserved for task/evidence archaeology. |
| `MASTER_PLAN.md` | Product goal, truth boundaries, and next gates | Its live content moved into North Star. |
| `CAPABILITIES.md` | Tier split and capability matrix | Its live content moved into North Star. |
| `BUILD_CHECKLIST.md` | Status board plus append-only lane handoffs | Current queue moved into North Star; dated handoffs remain history here. |
| `TECH_STACK.md` | Stage/code/model registry | Code navigation moved to `AGENTS.md`; selected defaults remain in manifests. |
| `TIER_MAP.md` | Short mirror of the tier split | Tier policy moved into North Star. |
| `TECH_BLUEPRINTS.md` | Detailed pillar recipes and successor-manager plan | Unfinished outcomes/gates were distilled into North Star; implementation archaeology remains here. |
| `EDGE_PLAYBOOK.md` | Profile, rules, capture, and compute ideas | Adopted ideas were folded into active tasks; old roadmap deltas are historical. |
| `FABLE_OPERATING_MANUAL.md` | Wave/lane management process | Durable repo rules moved to `AGENTS.md`; transient operating ritual is retired. |
| `OVERLAPPING_COURT_CALIBRATION_GOAL.md` | Court-calibration execution goal | Current CAL decision/gate moved into North Star. |
| `RACKET_6DOF_GOAL.md` | Paddle execution goal and log | Current RKT decision/gate moved into North Star. |
| `OWNER_CHECKIN_20260707.md` | Dated owner requests | Historical. |
| `OWNER_CHECKIN_20260708.md` | Dated owner requests | Historical. |
| `OWNER_CHECKIN_20260709.md` | Dated owner requests | Unresolved owner actions were reconciled into North Star. |
| `PRODUCT_INFRA_DESIGN_20260707.md` | Superseded product-infrastructure design | Preserved because later server work cites it; not current direction. |

## Evidence hierarchy

When investigating a past decision:

1. Start with the dated lane report or scored artifact under `runs/`.
2. Use these archived documents to find the intended task, old decision, or
   evidence path.
3. Verify the referenced code, artifact, model identity, and dataset before
   repeating any claim.
4. Return to the current North Star for sequencing and promotion status.

The 2026-07-09 deep review that triggered this consolidation is
`runs/CV_PIPELINE_DEEP_REVIEW_20260709.md`.
