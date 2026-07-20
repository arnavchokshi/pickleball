# ULTRA advisory: court + ball program decisions

Lane: `strategy_advisory_20260719`  
Role: advisory only; the manager rules  
Status: **`VERIFIED=0`** — no recommendation, experiment, scoped pass, or result below is a promotion  
Evidence cut: working tree and on-disk artifacts inspected 2026-07-19/20

## Executive ruling

| Decision | Ranked recommendation | Why now |
|---|---|---|
| D1 COURT | Finish the in-flight static lock unchanged, then run a bounded classical line-front-end hardening lane. Put guided framing in `NS-03.LIVE`. Defer tasks 88–91 and preserve them as source-grouped evaluation/gate fuel. | Clean line evidence already solves near the target; detection robustness and capture discipline, not a new solver or another learned-corner push, are the evidenced gaps. |
| D2 BALL | Keep queue #5 first. From one frozen scaled pretrain, compare RGB-only and wrist-conditioned owner fine-tunes; evaluate a default-off rally-sequence DP on dense rallies; separately re-score the existing plane hypothesis generator only as typed `BOUNCE`. | CONTACT still needs a trained head. DP and wrist have useful analog evidence. Plane geometry has a narrow bounce-only carve-out, but the prior whole-chain TT3D rejection remains binding. |
| D3 PB.VISION | **No PB gallery predictions in event-head training now.** Quarantine all 13 known PB source games and their derivatives as compare-only. A pseudo-label arm may reopen only after explicit written training rights and a frozen, source-disjoint control. | The harvest is competitor-processed; current PB terms restrict automated, commercial, and competing-product use. Public readability and `RD_ONLY` tagging do not grant training rights. Training on the same gallery destroys the intended head-to-head comparison. |
| D4 OWNER LABELS | Use the 102 banked rows now, but honor the existing source split: 61 train / 41 validation. Do not request the remaining 198 today. Ask for one 50-row uncertainty-ranked round only if the first fine-tune shows at least `+0.10` absolute source-held HIT/BOUNCE macro-F1 lift but remains below `0.80`. | More labels are rational only if the first in-domain fine-tune proves label sensitivity. A small lift means diagnose the harness/domain/features; a high absolute score means test wrist/DP before spending owner time. |

The queue-#1 ruling remains decisive: 19/24 contact refusals occur on fully real track frames, so the trained-event wall remains the binding BALL blocker and queue #5 does not move (`runs/lanes/oneworld_bridge_xref_20260719/XREF_RULING.md:14-21,31-40`; `NORTH_STAR_ROADMAP.md:419-423`).

All engineering-duration, GPU-hour, and dollar figures below are **planning estimates**, not measured commitments, unless an evidence citation explicitly says otherwise.

---

## D1 — COURT program

### Options

| Rank | Option | Evidence | Planning cost / wall | Main risk | Reversibility and standing-kill posture |
|---:|---|---|---|---|---|
| **1** | **Static lock → bounded classical line hardening; guided framing in LIVE.** Let `static_cal_firstlock_20260717` finish exactly as preregistered. Its CAL follow-on runs frozen baseline → ROI/lookalike-line rejection → line-over-point weighting → shadow-removal arm only if a shadow stratum is measurably failing. Guided framing runs file-fenced in parallel after the CAL preview interface is available. Tasks 88–91 are deferred and later frozen as source-grouped eval. | Automatic line/segment error costs 15–24 points at a fixed solver, while optimizer gains are secondary; research ranks line robustness first, pooling second, labels third, solver replacement last (`runs/research_sota_20260719/domainA_court_calibration.md:16-20,121-131,168-175`). The in-flight lane already owns pooling, k1, and the motion guard (`runs/lanes/static_cal_firstlock_20260717/spec.md:32-43,89-128`). | CAL: 2–4 engineer-days, mostly CPU. Optional shadow model: 2–6 GPU-hours / about $2–15 after license/runtime review. LIVE: 2–4 engineer-days plus 15–30 minutes device/court QA. File-fenced wall: about 3–6 days after CAL handoff. | Shadow repaint can erase paint; ROI rejection can lose partial courts; line weighting can overfit clean venues; guided framing can become a hidden recording blocker. | High: default-off arms, immutable images/evidence, no authority change. **No §2.3 conflict** because this does not promote learned auto-find or perform a synthetic-only retrain. |
| **2** | **Capture-discipline UX only.** Finish static lock, ship guided framing under LIVE, and defer line-front-end work until imported clips prove a residual gap. Keep tasks 88–91 untouched. | No fully automatic single-consumer-camera system was found near reliable ~5px performance; production systems constrain capture or add cameras (`domainA_court_calibration.md:7-20,91-102`). `NS-03.LIVE` explicitly owns capture guidance and may begin after NS-01.1/01.2a (`NORTH_STAR_ROADMAP.md:332-347`). | 2–4 engineer-days; no server GPU; 15–30 minutes device/court QA; 2–5 wall days. | Fastest for controlled captures, but arbitrary imports, shadows, fences, adjacent courts, and partial courts remain weak. | Very high; no standing-kill conflict. Ranked second because it leaves the research-ranked technical gap untested. |
| **3** | **Label-first learned-finder reopen.** Complete tasks 88–91 now, create source-separated train/eval subsets, fine-tune a learned finder, and revisit line work afterward. | Tasks 88–91 are four 25-image jobs (`cvat_upload/court_diversity_20260712/import_report_20260712_courtsession.json:5-29`), but they label points/corners, not shadow masks or lookalike-line negatives (`cvat_upload/court_diversity_20260712/OWNER_GUIDE.md:18-31`). Venue diversity is inferred, not visually confirmed (`cvat_upload/court_diversity_20260712/OWNER_GUIDE.md:40-45`). Research ranks this third (`runs/research_sota_20260719/domainA_court_calibration.md:125-131`). | Existing owner work: 45–60 minutes measured by the guide. Engineering: 3–5 days; roughly 4–12 GPU-hours / $5–25. A new semantic line/distractor pack would add about 1–3 owner-hours. | Same-source train/eval leakage, only 100 stills, wrong label schema for the top failure modes, and repetition of prior domain-transfer failure. | Artifacts are reversible; owner time and eval contamination are not. **Direct §2.3 conflict** if this becomes an authority-path push (`NORTH_STAR_ROADMAP.md:202-206`). **No new evidence currently justifies that reopen.** |

### Recommendation

Choose **Option 1**.

Why:

1. The current lane is already testing the research-ranked #2 lever and the named k1 defect; interrupting it would confound the result.
2. The best next technical question is upstream: can classical evidence extraction survive shadows, adjacent courts, fences, and partial views while the same frozen solver is held fixed?
3. Guided framing is independently useful even if the classical arm fails. It also attacks the measured sub-1px pb.vision gap without pretending that arbitrary single-camera calibration is solved (`domainA_court_calibration.md:93-102,139-143`).
4. A solver replacement attacks the wrong bottleneck and should not receive a lane.

### Tasks 88–91 ruling

Re-scope the four tasks as **frozen auto-find/front-end evaluation fuel**, not a v1 calibration lever and not immediate learned-finder training.

- They cover 100 stills; every task contains 25 images (`import_report_20260712_courtsession.json:5-29`).
- Their current schema is valuable for point/corner PCK and solve residuals, but it cannot directly supervise shadows, line masks, or lookalike rejection (`OWNER_GUIDE.md:18-31`).
- Group by original source video/channel, never by task number; the shards interleave sources.
- Visually verify venue grouping before calling the set source-disjoint because the package labels every venue group as inferred (`OWNER_GUIDE.md:40-45`).
- Defer owner time now. Once completed, freeze the rows before candidate tuning. If a factorized error audit later says semantic line/distractor labels would change the decision, make that a separate, smaller pack.

### Guided-framing placement

Put it in **`NS-03.LIVE`**, not CAL.

CAL should emit preview geometry plus typed quality/abstention reasons. iOS/LIVE should own the pre-record interaction:

- landscape, static/elevated, all-four-corners-visible guidance;
- fast advisory lock/abstain state;
- explicit override: recording always remains available;
- unmet guidance persists as typed degraded-input provenance;
- no preview CAL result becomes authority.

This matches the product contract that advisory inference may not stall recording (`NORTH_STAR_ROADMAP.md:53-56`) and the explicit LIVE ownership boundary (`NORTH_STAR_ROADMAP.md:334-347`). Add its physical check to the existing 60-second phone test or first real-game capture rather than creating a separate owner session.

### Exact learned-finder reopen conditions

The §2.3 learned-auto-find kill may reopen only when **all** of the following new evidence exists:

1. The static-lock lane completes its frozen acceptance (`static_cal_firstlock_20260717/spec.md:130-152`).
2. On untouched, source-grouped real viewpoints, the factorized classical lane still fails specifically at line extraction—not distortion, solver weighting, camera motion, or capture setup.
3. Real training sources are distinct from tasks 88–91 or another protected evaluation set; no third synthetic-only retrain.
4. A preregistered learned challenger beats the best classical front end on the same scorer and independently meets owner-viewpoint PCK@5 ≥0.95, net-height ≤2cm, distortion/handheld, completeness, and runtime gates (`NORTH_STAR_ROADMAP.md:341`).
5. Until then, every learned/automatic result remains preview-band and the owner-reviewed authority door is unchanged.

**Confidence:** high, **0.87**.

**What would change my mind:** If the in-flight lane passes all acceptance items and unseen guided captures show no meaningful front-end failures, I would select Option 2 and skip the CAL hardening lane. If the frozen error taxonomy says shadows/lookalike lines are not material, I would remove those arms. I would select Option 3 only after the five reopen conditions above exist.

---

## D2 — BALL program

### Options

| Rank | Option | Evidence | Planning cost / wall | Main risk | Reversibility and standing-kill posture |
|---:|---|---|---|---|---|
| **1** | **Queue-faithful factorized hybrid.** Finish scaled RGB pretrain → repair/freeze the fine-tune contract → run RGB owner control and wrist-conditioned arm from the same checkpoint → test event-sequence DP on dense rallies → separately re-score the existing plane generator as typed `BOUNCE` after CAL exists → only isolated survivors enter the unchanged anchor/arc gate. | CONTACT remains the trained-head wall (`XREF_RULING.md:31-40`). Current code evidence expands the former 226-row slice to 4,392 64f windows and preserves 9TP/0FP at matched context, but the GPU leg is still unattempted (`runs/lanes/event_head_corpus_20260719/LOCAL_EVIDENCE.json:28-48,62-70`). MonoTrack reports 78.1→94.3 recall with sequence constraints and +7.9pp from pose conditioning; plane geometry is evidenced only for bounce (`domainB_ball3d_labels.md:108-126,172-178`). | Existing GPU leg: ≤5h / ≤$10 cap (`VM_RUN_PLAN.md:175-230`). Harness preflight: ~0.5 day. Wrist arm: 1–2 engineer-days and 0.5–1.5 GPU-hours. DP: 0.5–1 day and <1 CPU-hour per evaluation pass. Plane rescore: 0.5–1 day **after** CAL planes exist; longer if CAL must be produced. | Tiny local judge, missing CAL on the six owner sources, incomplete wrists, and badminton constraints that may suppress fast pickleball exchanges. | High: separate checkpoints, saved raw logits, candidate sidecars, default OFF. The bounce arm is a **scoped reopen**; its exact limits are below. |
| **2** | **RGB + DP only.** Finish scaled pretrain and RGB fine-tune, validate the raw rate, then evaluate DP; defer wrist and plane bounce. | Cheapest direct answer to the trained-head wall while testing the strongest zero-label analog. | Existing GPU cap plus 0.5–1 day harness/DP work. | Leaves the quantified pose lever and bounce-only carve-out untested; DP cannot be judged on the sparse 50/102 rows. | Very high; no geometry reopen. |
| **3** | **Physics/multimodal-first reorder.** Pause the trained-head leg and build wrist/plane/DP integration before an RGB fine-tune control exists. | Contradicted by the queue-#1 ruling and by the need for a learned CONTACT head. | 3–5 engineer-days, 4–8h preprocessing, then another 2–5h GPU. | Confounds causes, delays the binding answer, and risks relitigating killed geometry. | Technically reversible but poor schedule reversibility. **Conflicts with §2.3/§5; no new evidence justifies a reorder.** |

### Recommendation and exact sequence

Choose **Option 1**, without moving queue #5:

1. Finish the scaled **RGB pretrain** and frozen ≥50-clip public evaluation.
2. Freeze the checkpoint, train manifest, source groups, and threshold-selection procedure.
3. Repair the fine-tune/protected-eval context contract before using the 102 rows:
   - `finetune_event_head.py` currently defaults to 9 frames and does not stamp `config.window_frames` into the saved checkpoint (`scripts/racketsport/finetune_event_head.py:157-168,184-192`);
   - the repaired evaluator rejects absent/mismatched checkpoint context (`scripts/racketsport/eval_event_head.py:38-63`);
   - protected evaluation still decodes anchor ±1.0s rather than the checkpoint's exact frame count (`scripts/racketsport/eval_event_head.py:134-179,199-214`).
   Derive, assert, and persist the frozen 64-frame context. Repeat the mandatory ±0.75s overlap guard before any owner row enters training. This is a required preflight, not a request to edit code in this advisory lane.
4. From the same pretrain, run:
   - H0: RGB-only owner fine-tune control;
   - H1: missingness-aware late-fusion wrist/cheap-pose arm.
   Do not fine-tune H1 from H0. Missing wrist input must reproduce H0 behavior.
5. Evaluate raw H0/H1. Wrist survives only on strict event F1/recall gain with negative-FP and timing non-regression.
6. Require a plausible raw firing rate of about 0.3–1.0 events/s before applying DP. DP may not launder another 7.16/s degenerate predictor (`NORTH_STAR_ROADMAP.md:423`; `runs/HANDOFF_20260717.md:206-220`).
7. Test event-sequence DP first on source-disjoint **densely annotated public rallies**. The protected 50 and owner 102 are sparse sampled windows, so unlabeled intervening events make them invalid judges for sequence recall/FP.
8. Test plane `BOUNCE` separately after provenance-bearing CAL planes exist. Only after raw wrist, DP, and plane arms survive in isolation may they be combined.
9. Re-run anchor fusion on the identical frozen arc gate; one candidate-attributable physics violation kills that arm.

### Event-sequence DP contract

This is **not** the killed TT3D whole-rally geometric DP (`runs/lanes/tt3d_integrate_20260712/scoring_table.md:32-36`). It is sequence selection over saved event-head logits.

Pre-register:

- rally boundaries;
- a source-faithful spacing/count prior, with 0.5s treated as an external starting hypothesis rather than a pickleball constant;
- alternation by **team/court side**, not individual player, because doubles does not alternate individuals;
- uncertain identity produces a soft penalty or abstention, never a hard veto;
- every selected timestamp must trace to a saved low-threshold raw logit; DP cannot synthesize an event;
- raw and DP outputs are always reported side by side.

It survives only if dense-rally typed F1/recall improves with precision and timing non-regression. Before product activation, require a small untouched, exhaustively typed pickleball-rally holdout. MonoTrack's result is strong analogous evidence, not permission to copy badminton grammar unchanged (`domainB_ball3d_labels.md:108-116`).

### Court-plane `BOUNCE` scoped reopen

**Conflict:** The roadmap records all geometry-only whole-solution paths as killed (`NORTH_STAR_ROADMAP.md:443-447`). The existing TT3D integration stayed at 2/3 bounce matches, reduced Wolverine coverage 300→290, and worsened fallback reprojection p50 455→571px (`runs/lanes/tt3d_integrate_20260712/scoring_table.md:24-36`).

**New evidence justifying only a narrow reopen:**

- TT3D/MonoTrack-style known-plane geometry has evidence for **bounce**, not arbitrary-height paddle contact (`domainB_ball3d_labels.md:33-41,172-178`).
- The internal failure taxonomy says the arc program needs **typed anchors**, while geometry cannot serve the whole flight/contact solution (`NORTH_STAR_ROADMAP.md:173,202-216`).
- The repo already has a pure two-piece court-plane candidate generator whose output is explicitly `hypothesis_only` and `kind=bounce`; rebuilding it would repeat work (`threed/racketsport/ball_joint_anchor_search.py:1-14,204-253`).

Allowed experiment:

1. Re-score the **existing generator only** as a typed `BOUNCE` classifier/anchor.
2. It remains default OFF, `hypothesis_only`, provenance-bearing, and raw-observation immutable.
3. It may not emit `CONTACT`, replace the solver, pin/snap the delivered trajectory, or convert preview CAL into authority.
4. Freeze CAL, ball observations, thresholds, scorer, labels, and registered CAL perturbations before scoring.
5. Require strict Pareto improvement over raw-head BOUNCE: better TP/F1 or timing, no FP/timing/tail regression, and stability under CAL uncertainty.
6. On the same frozen 188-segment arc gate, the baseline remains zero violations. **One newly introduced violation kills the entire arm; coverage never compensates.**
7. The protected seed's 11 ground rows can support a scoped paired diagnostic, not promotion certainty. Independent NS-02 bounce/contact GT remains the promotion door.

**Current prerequisite result:** all six owner-label source IDs have zero matching `court_calibration.json` artifacts under `runs/` or `data/` in the present tree. The plane arm is therefore **NO-ATTEMPT** until frozen, provenance-bearing court planes exist. Do not borrow or synthesize a plane. Preview CAL is sufficient for scoped research, never promotion.

### Confidence

- Overall Option 1: **0.82**
- Wrist conditioning: **0.77**
- Event-sequence DP: **0.74**
- Plane-bounce reopen: **0.48**

**What would change my mind:** A still-degenerate scaled raw head would stop DP/anchor work and force data/harness diagnosis. Incomplete wrist sidecars that cannot preserve RGB parity would defer H1. Dense pickleball rallies showing that DP deletes fast exchanges would reject DP. Any CAL-instability, added physics violation, or repeated 2D/tail regression would re-close the plane arm.

---

## D3 — PB.VISION gallery pseudo-labels

### Fresh inventory and governing posture

The harvest completed during this advisory. The current manifest contains **12 gallery videos plus the earlier `Demo Vid` reference: 13 source games, 164.42 minutes total**, with source-video hashes (`data/pbvision_gallery_20260719/MANIFEST.json:608-649`). The local manifest calls them `RD_ONLY competitor-processed` (`MANIFEST.json:1-5`), but that is an internal handling label, not a third-party license.

Technical precedent and rights posture are separate:

- TT4D supports physics-filtered pseudo-labeling at scale, but it does not establish a right to train on a competitor's predictions (`runs/research_sota_20260719/domainB_ball3d_labels.md:108-116`).
- Existing local event policy already says the prior PB reference is never sampled, labeled, or trained (`runs/lanes/owner_event_labels_20260715/spec.md:23-28`; `data/event_bootstrap_20260713/manifest_v0.json:257-275,1104-1110`).
- PB's current [Terms of Service](https://pb.vision/terms-of-service) restrict automated collection, non-personal/commercial use, and use toward a competing product. The [API License Agreement](https://pb.vision/api-license-agreement) requires prior written consent for vendor-interface use outside the API purpose. The [Privacy Policy](https://pb.vision/privacy-policy) describes PB's rights/use of uploader videos; it does not grant those rights downstream to us.

This is a product/research risk ruling, not legal advice. The safe engineering consequence is the same: no training without explicit written permission covering this use.

### Options

| Rank | Option | Evidence | Planning cost / wall | Main risk | Reversibility and standing-policy posture |
|---:|---|---|---|---|---|
| **1** | **Compare-only quarantine; no PB pseudo-label training.** Hash and group every source and derivative. Use PB outputs for diagnostic H2H only, never as GT. Build any pseudo-label work from rights-cleared public/owned pixels and our own frozen predictors. | Current official terms and local “never training” policy above. The manifest now makes all current PB sources explicit. | 1–2 engineer-hours to validate quarantine/lineage; $0 incremental training compute. | Gives up a potentially useful in-domain teacher signal; owner set remains small. | Very high. No policy reopen. |
| **2** | **Obtain explicit written PB permission/API training rights, then run an isolated control on future non-comparison PB source groups.** Permission must cover pixels, derived event timestamps, model training, intended commercial use, retention/deletion, and participant privacy. | API agreement supports negotiated access, but not the present unlicensed path. | External days to weeks; unknown fee. After permission: about 1 engineer-day, 6–24 GPU-hours / roughly $5–30 for three-seed controls. | Contract dependency, privacy obligations, teacher imitation, correlated false labels, and permanent benchmark exclusions. | Medium. Checkpoints/data remain isolated and deletable, but contractual and benchmark consequences persist. |
| **3** | **Immediate isolated `RD_ONLY` shadow-training arm without written permission.** Keep it out of selected/commercial lineage. | Technical feasibility only; contradicted by the current terms and local event policy. Agreement filtering does not cure rights or benchmark contamination. | 0.5–1 engineer-day; 3–8h processing/training; roughly $3–10 compute. | Terms/model-extraction exposure, privacy/ethics, correlated teacher errors, and loss of an honest PB H2H. | Technically high if isolated; governance/benchmark damage is not fully reversible. **Rejected; no new evidence justifies this reopen.** |

### Recommendation

Choose **Option 1 now**. Option 2 is the only acceptable reopen. `RD_ONLY` is not a loophole.

The proposed `PB events × our audio × our wrist` agreement rule is also weaker than it first appears:

- PB and its own trajectory/event outputs are one teacher, not independent votes.
- Wrist evidence is admissible only when it comes from real, non-interpolated observations with provenance.
- Audio previously performed at/below its time-density chance baseline; it may be a bounded non-emitting score only after per-video improvement over a time-shift null, never a decisive vote (`runs/HANDOFF_20260717.md:206-214`; `NORTH_STAR_ROADMAP.md:214`).

### Permanent compare-only quarantine

All source derivatives, excerpts, rerenders, and rally cuts inherit the original source group's quarantine.

| PB source title | Video ID |
|---|---|
| Chris Olson Match | `0tmdeghtfvjx` |
| Tustin 4.5 Match | `143sf3gdwxsa` |
| Pro Training | `98z43hspqz13` |
| Side View GoPro | `bewqc0glhgpq` |
| Wood Floor Court | `iottnc0h3ekn` |
| Outdoor Shadows | `o4dee9dn0ccr` |
| MLP Mixed Match | `pldtjpw3h0jw` |
| Lower-Level Doubles | `st0epgnab7dr` |
| Singles Match | `td2szayjwtrj` |
| Facility Camera | `tqjlrcntpjvt` |
| Multi-Line Court | `utasf5hnozwz` |
| Drill Session | `xkadsq9bli3h` |
| Demo Vid | `83gyqyc10y8f` |

The IDs, titles, and source-video SHA-256 values are pinned in `data/pbvision_gallery_20260719/MANIFEST.json:8-623`; the manifest totals are at `:631-649`. The current public gallery also exposes the featured titles observed in the research prompt ([PB Vision demo gallery](https://pb.vision/demo-gallery)).

Even if written permission later arrives, these 13 remain compare-only because they define the current competitor gallery/reference pool. Training may use only separately predeclared, separately licensed PB source groups that were frozen **before** inspecting predictions or H2H results.

### Exact permission-based reopen conditions

Every item must pass:

1. Written rights/counsel approval; public unauthenticated readability is insufficient.
2. Compare-only source groups are frozen by original-game ID and SHA before score inspection.
3. PB prediction is never accepted alone.
4. Audio exceeds a per-video time-shift null and remains non-emitting.
5. Wrist evidence excludes interpolated/stale/fabricated player positions.
6. An independently reviewed 50-row pseudo-label audit reaches at least 47/50 correct event type and within ±2 frames. Those audited rows remain out of training.
7. The controlled test below passes; any miss restores permanent compare-only posture.

### Control proving pseudo-label lift

Freeze one scaled-pretrain checkpoint, code, optimizer/update budget, human-row exposure, class weights, thresholds, source groups, and three seeds.

- **A — owner-only control:** 61 owner train rows; 41 source-held owner validation rows.
- **B — licensed pseudo arm:** identical owner exposure and optimizer budget plus agreement-filtered pseudo rows from non-quarantined, licensed PB sources, at reduced weight and a capped pseudo:human ratio.
- **C — placebo/data-exposure control:** identical gallery pixels/windows and update budget, but PB times are shuffled within rally or teacher labels are masked. This separates label information from extra in-domain pixels.

Threshold/model selection uses only the owner validation protocol. The protected 50 is touched once after the arms are frozen.

B must:

- beat A by at least **+0.10 absolute HIT/BOUNCE macro-F1 at ±2 frames**;
- have a paired-bootstrap 95% lower bound above zero;
- regress neither class F1 by more than 0.03;
- keep negative FP ≤2/21 and add no more than one FP versus A;
- keep timing p90 non-worse;
- retain a plausible full-video rate around 0.3–1.0 events/s;
- beat C, proving that teacher labels—not merely extra pixels/updates—caused the lift.

Any PB-trained model is permanently disqualified from accuracy H2H on its training source groups. PB output is never GT; a true H2H requires independent human labels on compare-only footage.

**Confidence:** high, **0.94** after the completed manifest and current terms review.

**What would change my mind:** explicit written training rights; a source/hash-complete pool distinct from all 13 compare-only games; ≥47/50 audited pseudo-label precision/timing; and an A/B/C result meeting the frozen lift and safety criteria.

---

## D4 — owner labeling budget

### Fresh data audit

- Provenance records 102 answered rows: 60 typed events and 42 hard negatives, with a mandatory zero-overlap check against the protected seed (`data/event_labels_owner_20260719/PROVENANCE.json:2-14`).
- Joining the result export to the frozen session manifest yields:
  - total: 38 `paddle`/HIT, 21 `ground`/BOUNCE, 1 `other`, 42 `none`;
  - train: **61** = 23 HIT, 17 BOUNCE, 1 other, 20 negatives;
  - source-held validation: **41** = 15 HIT, 4 BOUNCE, 22 negatives.
  Therefore “owner-102 fine-tune” must not mean training on all 102.
- The rows cover all six source groups and all three original strata.
- The provenance summary says 46 coordinate rows and 57 dt rows, while the actual `answers` objects contain x/y/dt on all 60 typed rows. The top-level legacy maps are 46/57, and ingest reads `answers` (`scripts/racketsport/ingest_event_review_results.py:51-66,94-150`). Reconcile this bookkeeping before training; do not ask the owner to relabel it.
- The protected judge is 17 HIT, 11 BOUNCE, 1 other, and 21 negatives (`runs/lanes/event_head_scaffold_20260716/smoke_evidence.md:42-55`).
- The protected 50 and owner 102 use the same six source-video families. The mandatory time-overlap exclusion prevents direct window leakage, but it does **not** make the protected seed source-independent. It remains the requested internal judge, never a promotion set.

### Options

| Rank | Option | Evidence | Planning cost / wall | Main risk | Reversibility |
|---:|---|---|---|---|---|
| **1** | **Use the banked split now; freeze owner asks until the measured gate.** Train on 61, select/report on the fixed 41, preserve the protected 50 for the final frozen arm matrix. | Pretrain + small in-domain fine-tune is the relevant evidenced pattern, while the literature gives no clean answer for ~100 typed windows (`domainB_ball3d_labels.md:96-105,165-168`). | After pretrain/harness preflight: ~1–3h incremental wall, $0.5–2 compute, plus 1–2 engineer-hours for ingest/provenance/eval. | Only 61 train rows, only four validation bounces, six harvested source groups, no owner-shot domain. | Very high. |
| **2** | **Finish the remaining ~198 rows in presentation order now.** | Produces more same-pack labels but ignores whether the first 102 are useful and ignores uncertainty. | About 79 owner-minutes at the observed ~24s/row; another 1–2h ingest/train and ~$0.5–2 compute. | Irreversible owner time; redundant same-source examples; no fresh-source diversity; may mask a harness/feature problem. | Labels remain reusable, owner time does not. |
| **3** | **Conditional active-learning round over the remaining rows.** Preserve the original pack, generate a new selection manifest, and ask only if Option 1 proves label sensitivity. | Active learning is directionally supported, but the fresh research found no event-specific quantitative 3× multiplier (`domainB_ball3d_labels.md:108-116`). | 2–4 engineer-hours to score/repack; about 20 owner-minutes per 50; 1–2h retrain/eval and ~$0.5–2 per round. | Uncertainty bias, model blind spots, same-six-source ceiling, and overfitting the small validation set across rounds. | High if capped and the protected seed stays untouched. |

### Recommendation

Choose **Option 1 now**, then Option 3 only if the concrete gate fires. Do not request the remaining 198 today.

### Concrete first-fine-tune owner-budget gate

The protected 50 cannot remain a protected judge if it is repeatedly used to choose labels. Therefore the **owner-budget decision** uses the fixed 41-row source-held validation set; the protected 50 is touched once only after RGB/wrist model selection is frozen.

Pre-register the public-eval operating threshold, then compute across three same-initialization fine-tune seeds:

`G_val = macro-F1(HIT, BOUNCE; ±2f)_owner-finetune - macro-F1(HIT, BOUNCE; ±2f)_scaled-pretrain`

on the same 41 rows, same 64-frame context, same threshold, and same scorer.

- **Ask for one 50-row active-learning batch** only if median `G_val >= +0.10`, all three seeds are non-negative, absolute validation macro-F1 remains `<0.80`, validation negative FP is `<=2/22`, and a source-disjoint full-video firing-rate check is 0.3–1.0/s.
- **Stop owner asks and proceed to wrist/DP factorization** if absolute validation macro-F1 is `>=0.80` with no class/timing/negative-FP regression.
- **Stop owner asks and diagnose** if `G_val < +0.10`, a seed goes materially negative, negative FP exceeds 2/22, timing regresses, or the firing rate is implausible. Check provenance, 64f context, domain transfer, class weighting, and wrist conditioning before buying more labels.
- After one active round, require at least another `+0.05` validation macro-F1 lift to justify a second round; cap at two rounds. Keep the original 41 fixed and never rank candidates using protected-seed features/scores.

The `+0.10`, `0.80`, and `+0.05` values are **preregistered owner-effort decision thresholds**, not literature-derived capability or promotion gates. Final internal model judgment still comes from the one-shot protected-50 comparison. Product promotion still requires fresh source-disjoint EVENTS truth and the North Star timing/type/coverage gates (`NORTH_STAR_ROADMAP.md:323-326,346,469-480`).

### Active-learning retarget if the gate fires

Do not continue rows 103–300 in their current order. Preserve the pack and make a new manifest:

- one round = 50 owner rows;
- 40 rows from the four training-source groups for training;
- 10 rows from the two validation-source groups as a separate diagnostic/audit, never training;
- within each partition: 60% head uncertainty/margin, 20% disagreement among RGB/wrist/ball/plane candidates, 20% stratified random;
- audio may affect ranking only after chance correction and may never emit a class;
- source quotas, temporal deduplication, and an explicit BOUNCE quota;
- owner UI stays blind to stratum and model prediction;
- protected seed never participates in ranking;
- fresh owner-shot footage outranks another old-pack round because the repo still has zero owned pickleball footage (`NORTH_STAR_ROADMAP.md:170,455-462`).

Separately, rally-sequence DP needs dense truth. If it first survives on densely labeled public validation, request a small untouched exhaustive typed-rally holdout rather than pretending sparse windows validate DP. That is a distinct conditional ask, not part of the first 50-row uncertainty round.

**Confidence:** medium-high, **0.80**.

**What would change my mind:** A steep first measured learning curve; fresh owner-shot/source-disjoint footage; a corrected ingest report that materially changes the 61/41 usable split; or evidence that uncertainty is dominated by systematic blind spots rather than informative boundary cases.

---

## Single proposed §5 queue edit

Do not alter or renumber current rows 1–8. Insert this **one row immediately after current row 5**:

| Order | Agent goal | Scope boundary | Required handoff |
|---|---|---|---|
| Parallel research follow-ons after rank 4 handoff; rank 5 stays in place | **COURT:** frozen classical line-front-end baseline → ROI/lookalike rejection → line-over-point weighting → shadow arm only on a measured shadow failure; guided framing is the first `NS-03.LIVE` capture-quality slice; tasks 88–91 are deferred source-grouped eval. **Inside existing rank 5:** scaled RGB pretrain → 64f fine-tune/eval preflight → owner 61-train/41-val RGB control and wrist arm → raw-rate gate → default-off dense-rally DP; separately, typed-only plane `BOUNCE` after frozen CAL exists. No PB gallery training. | File-fenced; no queue reorder; no solver replacement; no learned CAL authority; no whole-flight geometry reopen; no audio emitter; raw logits/observations immutable; all candidates default OFF. | Factorized same-scorer table; dense-rally DP report; plane arm `NO-ATTEMPT` until CAL and then 0-violation result; guided-framing device evidence; `VERIFIED=0`. |

## Owner-asks delta

| Timing | Delta |
|---|---|
| **Now** | Remove the active “finish 300 event rows” ask: 102 are banked. Do not ask for the remaining 198. Defer court tasks 88–91. Keep **record a real game** and the existing 60-second phone test as the highest-value standing asks. |
| **When LIVE guidance is ready** | Add only 5–10 minutes of guided-framing QA to the existing phone/real-game session; do not schedule a separate court-UX appointment. |
| **Only if D4 gate fires** | Ask for one 50-row uncertainty-ranked event batch (about 20 owner-minutes); at most two rounds, each requiring measured incremental lift. |
| **Only if public DP survives** | Ask for a small untouched, exhaustively typed rally holdout; planning estimate 20–40 owner-minutes. Do not substitute sparse sampled windows. |
| **Only when a CAL decision is ready** | Ask for tasks 88–91 (45–60 minutes total), frozen as source-grouped eval. Do not train on those same sources. |
| **PB posture** | No owner labeling ask. If the manager wants the pseudo-label option, the next action is a business/legal request for explicit PB training rights, not model training. |

`VERIFIED=0` remains binding. This advisory changes no selected stack, default, authority band, or promotion state.
