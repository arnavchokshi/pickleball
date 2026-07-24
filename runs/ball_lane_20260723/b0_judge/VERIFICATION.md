# WS1.0 (B0) — Clean BALL Judge Verification — ball_lane_20260723

Status: TESTED-ON-REAL-DATA. Measurement-only. VERIFIED=0 remains binding: this
document certifies the DATA JUDGE, not any model. Nothing here trains anything
and nothing here is a promotion claim.

Machine-readable twin: `verification.json` (same directory). Baseline numbers:
`baseline_scores.json`. Re-score artifacts: `rescore_wasb_official_control/`.

Rerun protocol (this is frozen gate evidence): rerun `verify_b0_judge.py` from
the repo root and require BOTH exit code 0 AND a `verification.json`
byte-identical to the committed copy. A partial crash leaves the stale
committed file in place, so the exit code — not the file's presence — is the
freshness signal. The script derives the repo root from its own location and
remains runnable after this worktree is merged and removed; the only rows that
may legitimately differ across reruns are the live-ledger
`ledger_holdout_family_scan` INFO row and the worktree-conditional V1 row
(`worktree_copy_absent[informational]` replaces the second hash comparison
once no worktree checkout exists) — such a diff must be re-reviewed, not
treated as breakage.

## 1. What was verified

The frozen judge is `runs/lanes/ball_b0_split_20260721/split/` (round-1
artifacts, byte-identity pinned by `FROZEN_B0_ARTIFACT_SHA256` and
`FROZEN_B0_SOURCE_VIDEO_SHA256` in the committed
`scripts/racketsport/ball_loso_validation.py`, commit 4c27023). The fix2 round
(`split_fix2/`, accepted in `ball_b0_split_20260721_review/review_r3.json`)
added cryptographic image-byte binding; its 167 validation rows are
semantically identical to the frozen round-1 judge (verified row by row).

## 2. Row-count table

| Quantity | Expected | split | split_fix2 | Verdict |
| --- | ---: | ---: | ---: | --- |
| train rows | 2,249 | 2,249 | 2,249 | PASS |
| validation (judge) rows | 167 | 167 | 167 | PASS |
| lineage rows | 3,376 | 3,376 | 3,376 | PASS |
| scratch rows in train | 183 | 183 | 183 | PASS |
| historical (old) rows in train | 2,066 | 2,066 | 2,066 | PASS |
| historical HyU/Ezz rows excluded from every pool | 960 | 960 | 960 | PASS |
| judge rows HyUqT7zFiwk (indoor_court_level) | 100 | 100 | 100 | PASS |
| judge rows Ezz6HDNHlnk (outdoor_night_fenced) | 67 | 67 | 67 | PASS |

## 3. Judge purity (scratch-only)

All 167 validation rows in both rounds: `lineage_class == "scratch"`,
`original_prelabel == null`, `lineage_origin == "scratch_no_prelabel_package"`,
`teacher_derived == false`, `evaluation_eligible == true`. Zero violations.
The 73 negative rows are attestation-closed (review round 2 finding 1 CLOSED:
126/126 boxless export rows attested, 0 unattested negatives in validation).

Train weight policy holds: every `confirmed_prelabel` train row carries
`training_weight == 0.25` (1,546 rows in split / 1,545 in fix2); every
`corrected_prelabel` and `scratch` train row carries 1.0. Zero violations.

## 4. Family-disjointness table

Split-internal (both rounds): train ∩ validation is empty on every axis —
parent_source_id, clip_id, row_key, image_name. Train families are exactly
{73VurrTKCZ8, _L0HVmAlCQI, wBu8bC4OfUY, zwCtH_i1_S4}; validation families are
exactly {HyUqT7zFiwk, Ezz6HDNHlnk}. None of the 960 historical holdout-family
rows leaks into train. PASS.

Ledger-wide (post-refresh `runs/manager/data_ledger.json`), holdout families in
any `partitions.train`:

| Asset | State | Holdout families in train pool | Classification |
| --- | --- | --- | --- |
| ball_reviewed_corpus_chain_1121_3026 | QUARANTINED | none (refreshed this lane) | BALL pool now source-held per b0 split |
| w7_audit_stratum_scratch_350 | CONSUMED | none | judge source asset; 167 rows eval-only |
| online_harvest_20260706 | CONSUMED | HyUqT7zFiwk, Ezz6HDNHlnk | CROSS-COMPONENT CAVEAT: raw rally media registered as EVENT Stage-F training media; BALL use stays fenced (named finding BALL_JUDGE_HOLDOUT_FAMILIES added) |
| event_bootstrap_audio_20260713 | REJECTED | HyUqT7zFiwk, Ezz6HDNHlnk | rejected asset; not an active pool |

Verdict: the judge families appear in NO active BALL train pool. The EVENT
media registration is a recorded cross-component caveat, not a BALL pool.

## 5. Hash / identity table (all recomputed this lane)

| Artifact | Expected (source of truth) | Recomputed | Verdict |
| --- | --- | --- | --- |
| split/report.json | 122e6591… (code constant) | match (main + worktree) | PASS |
| split/train.jsonl | b92218d4… (code constant) | match (main + worktree) | PASS |
| split/validation.jsonl | 39a07ed6… (code constant) | match (main + worktree) | PASS |
| split/lineage_rows.jsonl | 289a46c4… (code constant) | match (main + worktree) | PASS |
| image package zip (566,890,613 B) | f1b7ba88… (fix2 report) | match | PASS |
| 167 judge image members, per-member sha256 + md5 | fix2 rows + input_contract | 167/167 match, 0 mismatch | PASS |
| scratch sampling manifest md5 | 3dc23a4e… | match | PASS |
| scratch package manifest sha256 | a04fd956… | match | PASS |
| task-87 annotation export sha256 | fea4b952… | match | PASS |
| 8 judge source videos sha256 | code constants | 8/8 match on local media | PASS |
| protected-collision guard (2,953 frames, 0 collisions, dhash ≤ 3) | fix2 report + independent review | as reported; not re-executed this lane | PASS_AS_REPORTED |

Cross-round note (informational, train-side only): exactly one row
(`wBu8bC4OfUY_rally_0001:000320`) is `confirmed_prelabel` @0.25 in round 1 but
`corrected_prelabel` @1.0 in fix2. It belongs to a train family and does not
touch the judge; the 167 judge rows are field-identical across rounds.

Scope note: the `split_fix2_hashes` block in `verification.json` is recorded,
not verified — no independent frozen reference exists for the fix2 rebuild
(only the round-1 `split/` files are pinned in committed code). The fix2-derived
rows in the table above (image zip, member digests, protected guard) are
verified against the fix2 report's own recorded values, which is exactly what
they claim.

## 6. Ledger refresh performed (this lane)

`runs/manager/data_ledger.json` (schema v3, all edits in existing entry style):

- `w7_audit_stratum_scratch_350`: BLOCKED → CONSUMED; label_count 0 → 350;
  label_authority none → human_gt; overlap check NOT_RUN → PASS (dense
  pixel-sha256 + dhash vs 2,953 protected frames); image-zip / export /
  frozen-judge hashes added; b0 consumer entry added; clean_subset (judge
  selector) added; named finding BALL_CLEAN_JUDGE_FROZEN_20260721 added.
- `ball_reviewed_corpus_chain_1121_3026`: stays QUARANTINED (74.8% finding
  root cause remains UNRULED), but partitions now record the frozen source-held
  split (four train families; 960 holdout rows in no pool), overlap check FAIL
  → PASS via the family fence, clean_subset (train selector, hash-bound), b0
  consumer entry, and named finding SOURCE_HELD_SPLIT_FROZEN_20260721 added.
- `online_harvest_20260706`: named finding BALL_JUDGE_HOLDOUT_FAMILIES added
  (EVENT registration untouched).

Validation: current `audit_data_utilization.py` → `DATA UTILIZATION AUDIT:
PASS` (34 assets, 0 queue violations). Regenerated views:
`runs/manager/DATA_LEDGER.md` and `DATA_INVENTORY.md` (`--check` clean).

## 7. Baseline on the frozen 167-row judge

Scored locally this lane (re-score of sha-verified VM predictions through the
committed frozen-identity scorer; reproduces the published VM baseline
exactly):

| Candidate | Pooled F1@20 | Indoor (HyU, 100 rows) | Outdoor-night (Ezz, 67 rows) | Hidden-FP (pooled) |
| --- | ---: | ---: | ---: | ---: |
| WASB tennis zero-shot control (9d391239…) | 0.5670 | 0.7395 | 0.2933 | 0.4932 |

Caveat on the markdown twin: the header of
`rescore_wasb_official_control/loso_report.md` carries stale internal-val
boilerplate (burlington/wolverine CVAT clip ids and a limitations line)
emitted unconditionally by the pre-existing scorer regardless of mode; the
JSON twin's `parent_source_split` block (`identity_mode=frozen_b0_20260721`)
is the authoritative record of what was actually scored.

Not locally scoreable (recorded, NON-blocking for Gate 1.0):

- w7 retrain arms `E3k_seed_official_aug` (sha 00aa8a42…) and
  `E3k_matched_seed_official_aug` (sha ae01298c…): weights ARE local under
  `runs/lanes/w7_ballretrain2_20260709/vm_pull/arm_finetunes/`, but scoring
  needs fresh full-clip inference (~48,030 frames) not run in this CPU/MPS
  lane; historical w7 predictions are disqualified while the contamination
  finding is unruled (3 of 8 judge clips are on its contaminated list). Any
  future score also carries a not-source-clean caveat (these arms trained on
  960 holdout-family rows).
- B1/B2 SST-retrained checkpoint: absent locally (no *.pt under any b1/b2
  lane; `ball_b2_realrun_20260723/` is empty); if it exists it is only on VM
  `pickleball-gpu-ball-f2` / kept disk `pickleball-gpu-ball-disk-f`.

## 8. Gate 1.0 declaration

Gate 1.0 requires (1) a clean verification table and (2) a refreshed ledger.
Both hold:

- Verification table: every substantive check PASS (one PASS_AS_REPORTED for
  the protected-collision guard, independently confirmed by the b0 review; two
  INFO rows).
- Ledger: refreshed, machine-validated PASS, views regenerated.

**Gate 1.0 (BALL_NO_CLEAN_JUDGE cleared) CAN be declared.** The BALL component
now has a frozen, scratch-only, source-held, byte-bound 167-row judge with a
committed frozen-identity scoring path and a recorded zero-shot control
baseline. VERIFIED remains 0; no model claim is made or implied.

## 9. Known caveats (do not block the gate; must travel with it)

1. The 74.8% label-provenance/control-prediction finding
   (`runs/lanes/w7_ballretrain2_20260709/control_contamination_finding.json`)
   has an unconfirmed root cause. It cannot contaminate the judge rows
   (scratch-only, attestation-closed, byte-bound), but it keeps the historical
   corpus QUARANTINED and disqualifies historical w7 predictions as scoring
   inputs.
2. Cross-component: HyU/Ezz raw rally media remain registered EVENT Stage-F
   training media (`online_harvest_20260706`). No EVENT-derived signal from
   those families may feed a BALL trainer (named finding recorded).
3. Fix2's residual assumption stands: local CVAT rendered the imported staged
   bytes for task 87; no independent job-id binding exists in the historical
   import ledger (accepted in review_r3).
4. Pre-existing, unrelated to this lane: the racketsport wide suite was
   non-green at b0 acceptance (32 failures, 2 timing flakes ruled
   LOAD_SENSITIVE); at this worktree's base commit
   `tests/racketsport/test_audit_data_utilization.py` fails at import
   (script/test version skew, present with this lane's changes stashed); the
   base-commit ledger validator reports one pre-existing missing court frame
   file (identical on the pre-edit ledger).
