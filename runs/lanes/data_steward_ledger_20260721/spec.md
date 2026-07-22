# LANE data_steward_ledger_20260721 — plan §6: machine-readable data ledger + utilization audit

## HARD RULES
- No branches/commits. Read runs/regroup_20260721/EXACT_PLAN.md §6 (the authority for this
  lane) + §1 + §2.2, and runs/regroup_20260721/REGROUP_INPUTS.md §1 (inventory) first.
- Do NOT create a root DATA_LEDGER.md (doc-policy violation). Canonical JSON under
  runs/manager/, generated .md view beside it. Register nothing in the root doc allowlist.
- The ledger owns DATA lineage/utilization truth ONLY; NORTH_STAR remains product truth.
- Honest reporting; WIDE test suite (MPLBACKEND=Agg, full tests/racketsport), exact counts.
- Artifacts under runs/lanes/data_steward_ledger_20260721/.

## FILE OWNERSHIP (exclusive)
- runs/manager/data_ledger.json (new, canonical)
- runs/manager/DATA_LEDGER.md (generated view — must carry a "GENERATED, do not hand-edit" header)
- scripts/racketsport/audit_data_utilization.py (new)
- tests/racketsport/test_audit_data_utilization.py (new)
Nothing else.

## OBJECTIVE (EXACT_PLAN §6, verbatim requirements)
Every asset row: stable asset ID; paths; byte count; raw count; dedup-kept count; decoded
count; original source/game/session/channel/fork family; immutable hashes; rights posture +
the ruling allowing/forbidding each component; label authority (human GT / corrected prelabel /
confirmed prelabel / teacher / synthetic / none); protected/compare/quarantine identity +
overlap-check coverage; exact train/val/test partition; consumers (lane, command/config hash,
rows loaded, result path, metric/verdict); state READY|BLOCKED|QUARANTINED|CONSUMED|REJECTED|
DEFERRED_WITH_REASON; named owner + next check.

Seed the ledger from REGROUP_INPUTS §1 + EXACT_PLAN §1 shortlist. AT MINIMUM these assets:
pbvision_gallery_20260719 (13 videos; 3 permanent compare-only IDs 83gyqyc10y8f/iottnc0h3ekn/
o4dee9dn0ccr); pbv teacher event corpus (pbv_pickleball_corpus_20260720); owner 102 event
labels; protected 50-row event seed; event_bootstrap_20260713; event_public_20260713 (7
datasets, per-dataset rows); online_harvest_20260706 + _20260712; roboflow_universe_20260706
(core/adjacent/NC buckets); ball corpus chain 1,121->3,026 (w7_ballretrain2 contamination
finding = UNRULED, state it); w7_audit_stratum 350 scratch pack; court_diversity_20260712
100-frame pack (IYnbdRs1Jdk 3-frame permanent denial); court_keypoints_20260707 (3 usable /
3 rejected); eval_clips/ball 4 protected clips + 11,459 person boxes (eval-only); abc
experiment inputs (runs/lanes/ball_event_abc_20260720/inputs + abc_experiment_20260721/vm_pull).

`audit_data_utilization.py` must FAIL pre-dispatch when: input absent from ledger or hash
differs; train/holdout source-family overlap; teacher row represented as GT; protected asset
reachable by a trainer; baseline/check/kill threshold missing; GPU command references an
asset with zero decoded rows. It must also emit a sorted NEVER-QUEUED report (assets >24h old
with bytes/labels but no consumer/blocker/quarantine/rejection/defer ruling).

## ACCEPTANCE
- data_ledger.json validates against a schema you define in the same file or adjacent;
  DATA_LEDGER.md regenerates deterministically from it (test asserts round-trip).
- audit CLI: tests cover every failure mode above + the never-queued report.
- First real snapshot: run the audit on the seeded ledger; report N assets, M never-queued,
  and the state distribution.
- New CLI ships its direct-CLI reference test same-lane (repo rule).

## DATA CONTRACT
- Inputs: repo-on-disk inventories (read-only). No GPU. Effort cap ~8h.
- End-of-lane number: N assets registered, M never-queued, state counts.

## CROSS-SIGNAL
Consumes: all component lanes' data claims. Feeds: every lane spec's DATA CONTRACT block;
pre-dispatch gating for GPU lanes.

## BEST-STACK DELTA
None — coordination registry.

## MANDATORY STRUCTURED REPORT
objective_result; full_suite counts; HONEST ISSUES; artifacts; the snapshot numbers.
