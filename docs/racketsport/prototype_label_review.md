# Prototype Label Review

This workflow is for the 5-clip `PROTOTYPE-GATE` only. It creates draft
labels and correction bundles; it does not mark anything `VERIFIED`.

Prototype clips:

- `ppa_austin_md_qf_1200_high_baseline`
- `ppa_singles_0500_high_baseline`
- `gear360_0200_high_near_overhead`
- `burlington_gold_0300_low_steep_corner`
- `side_view_game5_0100_high_side_fence`

Run the review export after draft labels exist:

```bash
python scripts/racketsport/export_review_frames.py \
  --drafts-root runs/eval0/prototype_gate \
  --frames-root runs/label_frames \
  --out runs/eval0/prototype_gate/review_bundle
```

For each clip, open `review_bundle/images/<clip>/`. For calibration, click or
enter four court corners in this order: `far_left`, `far_right`, `near_right`,
`near_left`. Put corrected items into
`review_bundle/corrections/<clip>/<target_file>.json`.

To create CVAT-style task folders:

```bash
python scripts/racketsport/export_cvat_tasks.py \
  --review-manifest runs/eval0/prototype_gate/review_bundle/review_manifest.json \
  --out runs/eval0/prototype_gate/cvat_tasks
```

After corrections are saved:

```bash
python scripts/racketsport/import_cvat_labels.py \
  --drafts-root runs/eval0/prototype_gate \
  --corrections-root runs/eval0/prototype_gate/review_bundle/corrections
```

Imported corrections stay `corrected_unverified` until the lead runs the
prototype acceptance checks and overlays.
