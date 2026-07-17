# Player-selection layer — design + ready-to-dispatch implementation spec

Track L manager, 2026-07-17. Per the owner directive ("extremely sophisticated system for
pruning… find players' feet and keep people on/close to the court, AND/OR ID people and keep
only the same consistent 4… detection finding lots of people with high confidence just means
our cleaning of that data needs to be really strong") and the TRK research build-our-own
conclusion (4-slot enrollment gallery + open-set spectator rejection + soft court-footpoint
prior — corroborated x2, no off-the-shelf tracker solves it). Grounded in the ghost forensics
(`GHOST_DIAGNOSIS.md`): the surviving wolverine failures are the association's OWN synthetic
cross-identity bridge, not detector spectators — so the layer below is designed to kill both
failure classes: real off-court people (general content) AND self-manufactured ghosts (this
card). VERIFIED=0; preview band; nothing here is an accuracy promotion.

## Architecture (three cooperating evidence layers; fusion decides, never one signal)

### Layer A — soft court-presence scoring (owner-directed)

Per track, per frame, from REAL detections only (synthetic frames contribute nothing):

- frame evidence `e(f) = exp(-max(0, footpoint_court_excess_m(f)) / SIGMA_COURT)`,
  `SIGMA_COURT = 0.5 m`; excess computed exactly as the scorer's `_off_court_excess_m`.
- track presence score = EMA of `e(f)` over real-detection frames (half-life 2.0 s), fused
  with persistence (real-detection density over the track's span) and mean visibility/conf.
- Role: replaces the binary pool-stage margin cut as the selection-time prior. The pool keeps
  EVERYTHING (min_conf 0.0, no hard margin drop) — recall is a feature; selection prunes.
  Seated-courtside people get middling court scores and are finished off by layer B; brief
  walk-through passersby die on persistence.

### Layer B — 4-slot identity enrollment + open-set rejection

- **Enrollment**: during high-confidence windows (exactly 4 tracks alive on real detections,
  all court-presence ≥ 0.8, pairwise bbox IoU ≤ 0.2, window length ≥ 1.0 s — rally-start
  shaped), enroll 4 slots: OSNet centroid (EMA over the window's crops, production checkpoint
  + 8 px padding), spatial context (side/role via existing `assign_doubles_roles`), slot
  confidence + provenance (which frames enrolled it, when refreshed).
- **Open-set decision** for any tracklet vs slot (cosine distance to centroid):
  accept ≤ 0.35; reject ≥ 0.42; defer in (0.35, 0.42) → other evidence classes decide.
  (Pre-registered from the probe: within-identity 0.10-0.16 same-window / 0.30 cross-window;
  cross-identity 0.42-0.55; the observed bad stitch scored 0.448, the legitimate re-bind 0.304.)
- **Stitch veto** (kills the ghost class): a fragment merge is REFUSED when BOTH independent
  evidence classes agree it is wrong — (i) embedding distance ≥ 0.42 (or unmeasurable on both
  endpoints), AND (ii) the implied bridge is kinematically implausible: endpoint displacement
  > 2.5 m regardless of gap length, OR net-line (y=0) crossing with zero real supporting
  detections in the gap. Defer band → refuse the merge but keep both fragments alive for
  re-bind (refusing a merge is always reversible; a synthetic bridge is not).
- **Re-bind / re-entry honesty**: a rejected-then-matching tracklet may re-bind to a slot
  (accept threshold + side/role consistency); slots may sit UNBOUND with a declared gap
  (occlusion honesty) — never synthetically filled. Slot provenance records every bind,
  unbind, and refusal.

### Layer C — gap honesty + identity-conditioned pool recovery (replaces geometric synthesis)

- Geometric interpolation across identity-ambiguous gaps is BANNED. Same-identity micro-fills
  allowed only if run ≤ 12 frames AND total displacement ≤ 2.5 m AND no net-crossing, and are
  exported with `interpolated: true` per frame (additive tracks.json field — provenance must
  survive export; today it dies in `player_id_repair` → export).
- During a bound slot's gap, search the RAW pool (already exported at min_conf 0.0) for real
  detections with embedding accept ≤ 0.35 inside the slot's motion envelope (last position +
  7 m/s cone). Recovered REAL detections re-enter the track. This is the honest cov4
  rebuilder — coverage from recall, not from fabrication (wolverine's detection-limited
  ceiling analysis shows headroom exists that the margin+NMS gate discards today).

### Fusion rule (owner's principle, mechanized)

Selection keeps a track/slot only on combined evidence S = w_A·court_presence +
w_B·identity_match + w_P·persistence (registered weights 0.4/0.4/0.2, decision S ≥ 0.5); any
DESTRUCTIVE action (merge veto, track drop, slot unbind) additionally requires two independent
evidence classes to agree. No single signal ever decides anything irreversible.

## Pre-registered acceptance (frozen 2-clip card, gate v2.1 conventions, GPU-class VM ONLY)

Declared now, before any implementation run; no threshold shopping — the numbers above
(SIGMA_COURT 0.5, EMA 2.0 s, accept 0.35 / reject 0.42, displacement 2.5 m, fill 12 f,
weights 0.4/0.4/0.2, S 0.5) are THE registered values. One evaluation run; misses are
reported as misses.

| clip | axis | variant P (baseline to beat) | FULL PASS bar | notes |
|---|---|---|---|---|
| wolverine | spectFP | 4 | **0** | hard |
| wolverine | switches | 1 | **0** | hard |
| wolverine | far-off-court FP | 0 | 0 | hard |
| wolverine | near-miss rate | 0.1244 (breach) | ≤ 0.10 | CF predicts 0.0986 |
| wolverine | IDF1 | 0.8036 | ≥ 0.8036; target ≥ 0.8516 | CF3 upper bound 0.8519 |
| wolverine | cov4 | 0.7233 (contains ~0.107 synthetic padding) | ≥ 0.7233 via layer-C recovery | 0.6167 ≤ cov4 < 0.7233 with all other axes green = PARTIAL — coordinator ruling, stated verbatim as recovery shortfall |
| burlington | all axes | 0.9220 / 0.9933 / 0 / 0 / 0 | no degradation: IDF1 ≥ 0.9200, cov4 ≥ 0.9917 (≤1 frame), FP axes stay 0 | veto rules structurally cannot fire (6 synth frames, max run 3 / 1.14 m) |

Mandatory invariants before any card run: (1) selection layer OFF ⇒ byte-identical
tracks.json; (2) env-fidelity gate — reproduce the variant-P rows within 1e-9 through the
unmodified path on the VM before the selection arm runs; (3) wolverine-bridge fixture unit
test: the f44↔f87 stitch is refused, the GT1 re-bind is accepted, burlington's 3-frame fill
is untouched.

## Implementation shape (new files only — no shared-file edits)

- `threed/racketsport/player_selection.py` — layers A/B/C as pure functions + frozen-dataclass
  config (all registered values as defaults); consumes raw pool + fragments + OSNet embeddings
  (all already produced by `raw_pool_person_authority`); emits selected tracks + a
  `selection_report` with per-track evidence vectors and every veto/bind decision w/ reasons.
- `scripts/racketsport/select_players_from_pool.py` — CLI; register in
  `scripts/racketsport/list_scaffold_tools.py` (additive dict entries only).
- `docs/racketsport/player_selection_report_schema.json` (naming per `*_schema.json` pattern).
- `tests/racketsport/test_player_selection.py` — stitch-veto fixture built from the committed
  diagnosis JSONs (real numbers, no video needed), enrollment determinism, open-set band
  semantics, no-op invariant, provenance-survives-export.
- Integration is NOT this module's job: runner wiring goes as a request to Track C
  (process_video.py owner); the seam sits AFTER association, BEFORE export, orthogonal to
  Track F's detector-injection integrate lane (`trk_rfdetr_integrate_20260717` holds
  orchestrator.py + MANIFEST + best_stack — do not touch while its ledger row is live).

## Exact next-session dispatch instruction

1. Ledger check: confirm `trk_rfdetr_integrate_20260717` state (its spec's OPERATING POINT =
   branch 2b at conf 0.18 — the conf-0.30 prereg FAILED, see
   `runs/lanes/trk_rfdetr_prod_20260716/vm_conf030/report.json`); this lane stacks on variant P
   regardless of whether the integrate lane has landed.
2. Dispatch build lane (CPU, no GPU): Codex gpt-5.6-sol high, nohup-detached, flags BEFORE
   `resume`, fence = the four new files above + lane dir
   `runs/lanes/trkL_selection_impl_<date>/` only; `process_video.py`, `orchestrator.py`,
   `models/MANIFEST.json`, `player_global_association.py`, `player_id_repair.py` READ-ONLY
   (the two association modules are diagnosed, not edited — the selection layer supersedes
   their gap synthesis downstream);
   spec = this file + `GHOST_DIAGNOSIS.md`; acceptance = invariants (1)-(3) + focused tests
   EXIT 0 + wide suite w/ attribution.
   Template: `codex exec --cd /Users/arnavchokshi/Desktop/pickleball --sandbox workspace-write
   -c model="gpt-5.6-sol" -c model_reasoning_effort=high
   --output-schema docs/racketsport/lane_report.schema.json
   -o runs/lanes/trkL_selection_impl_<date>/report.json "<mission>"` (nohup, PID file, log.txt).
3. After build passes locally: ONE micro VM evaluation session (A100 spot, ase1-c ladder per
   the conf030 precedent — $0.3-0.5, ~0.15 h, ≤$2 cap, boot-armed rail, teardown +
   list-confirm): env-fidelity gate → selection arm both clips → pull w/ two-sided sha256.
4. Score vs the acceptance table verbatim; FULL PASS ⇒ flip proposal (selection layer as
   preview default stacking on the RF-DETR-L flip, `do_not_promote`, rev bump, wolverine
   deltas stated verbatim); PARTIAL/FAIL ⇒ honest report, no proposal.
