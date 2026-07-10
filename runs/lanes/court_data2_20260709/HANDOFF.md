# COURT-DATA-2 training handoff

- Default exact-loader corpus: `real_court_corpus/`; 35 rows from `chetan-rajagiri-9abfm__pickleball-court-v2__v1` (train 25, valid 7, test 3), all with 15 directly mapped keypoints and `label_status=reviewed`.
- The Chetan README states that its 35 images are annotated in COCO format with no image augmentation. Its source `far_*` rows are image-near and `near_*` rows are image-far; the committed mapping resolves that naming reversal using court topology plus five overlays.
- Do not random-split frames. This exact corpus has only one source dataset, so it cannot support a leakage-safe two-dataset validation holdout.
- Large partial-map inventory is preserved in `keypoint_mappings.json`: Xuann/Testworkspace/Stump/Necromancer/Nigh/Syncz provide 12 planar points; n-do-tran/ping-pong provide 12 planar points plus two net endpoints. The current loader requires exactly all 15, so these were not padded or guessed into the default corpus.
- Audit: 23 court/theme-related directories from all 65 locally downloaded manifest datasets; 15 usable direct/partial/corners-only datasets have five rendered overlays each.
- Exact-dedup and leakage guard: 82 eval/harvest image files hashed (70 unique), zero corpus matches; `pwxNwFfYQlQ` and `vQhtz8l6VqU` are explicit denylisted source IDs.
- License: all 23 audited court/theme datasets record CC BY 4.0; preserve project attribution. Noncommercial/unknown licenses fail closed and the direct-CLI test covers that quarantine path.
- Best-stack delta: none. This is data preparation only; `VERIFIED=0` is unchanged.
