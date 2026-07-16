# Fused-world display readiness assessment (Track H -> Track K, 2026-07-16)

Assessment only, per owner fusion-first directive relayed 2026-07-16. Nothing built;
Track K's schema is not final. Grounded in the post-webux3 viewer as committed in
88aeb1795 (web/replay/src/{viewerData.ts,App.tsx,assetRecovery.ts}).

## What the viewer already has that fusion can ride on
- Typed per-entity trust plumbing: `TrustBand` + `TrustBadge`
  ("verified"|"preview"|"low_confidence"|"too_close_to_call"), per-frame numeric
  `confidence` fields, `ConfidenceProvenance` with display_band, and
  `EvidenceProvenance` = measured | model_estimated | physics_predicted
  (viewerData.ts:8,335-401,475-638). In-pane per-entity badge dock
  (world-honesty-dock) renders label + provenance + band per player/ball/paddle.
- Layer-control dock with per-entity toggles + isolate; Ghost positioning layer is
  direct precedent for rendering an alternate hypothesis under a primary one.
- Shared timeline with provenance-banded glyph markers + Prev/Next Event nav —
  jump-to-evidence wiring exists.
- Manifest-driven artifact loading with honest-absent handling and (new today)
  status/content-type/HTML-body recovery + same-basename manifest-relative retry —
  a new `fused_world_url` asset inherits VM-manifest recovery for free.
- Opaque per-frame `source` metadata on ball frames already names "which detector/
  fuser" produced a sample (viewerData.ts:671) — a seam fusion can extend.

## What fused-world display would need (ranked)
1. Manifest + loader slot: `fused_world_url` in replay_viewer_manifest.json; typed
   decode in viewerData.ts with explicit absence sentinel (missing fused world must
   surface as a visible unavailable chip, never silently fall back to raw).
2. Per-entity confidence visualization: extend TrustBandPanel with the numeric
   confidence, plus an in-scene encoding (opacity/halo) driven by per-frame
   confidence. Constraint (trust contract §1.4): low confidence must LOOK uncertain;
   badges must not get less prominent than today's dock. Fusion likely needs a 4th
   provenance class or a "fused" origin tag added to the legend — decide with
   Track K, don't overload physics_predicted.
3. Fused-vs-raw toggle: a source switch in the layer dock (precedent: 2x FPS
   interpolated vs original). Requires both worlds resident (LRU cache exists);
   badges/legend must switch provenance labels with the source. Divergence view =
   render raw as ghost under fused (Ghost positioning precedent).
4. Contact co-location markers: fused contact events map onto the existing timeline
   glyph classes + a 3D marker at the fused location carrying its confidence;
   Prev/Next Event and jump-to-evidence work unchanged.
5. Performance guardrails learned today: keep the allocation-free per-frame buffer
   pattern (follow-FPS collapse was transition/allocation churn); the clip tail
   already dips to 31-48 fps with one world — fused layers must be toggleable and
   should not default all-on. Doubling resident world data also pressures the
   6-entry LRU.

## Schema asks for Track K (so the viewer can stay honest)
- Per entity per frame: numeric confidence + band + which sources fed the fusion.
- Explicit absence sentinels (frames where fusion abstained), never dropped rows.
- trust_band per entity at artifact level (viewer fails safe to preview otherwise).
- Manifest-relative asset URLs (VM-absolute paths only survive via today's recovery
  banner — relative is clean).
