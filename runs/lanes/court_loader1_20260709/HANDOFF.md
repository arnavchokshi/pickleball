# COURT-LOADER-1 handoff

- Exact partial-row schema for the corpus-builder lane: every item `keypoints` object MUST
  contain all 15 canonical keypoint names. A labeled entry is the existing two-number JSON
  array `[x, y]`; an unlabeled entry is JSON `null`. Missing names, extra names,
  `{"labeled": false}`, and all other markers are invalid and fail loudly. At least one point
  must be labeled. `null` means no supervision, not occlusion; an occluded-but-known point keeps
  its `[x, y]` coordinate.
- External human-annotated rows MUST use item
  `"status": "reviewed_external_dataset"`. They train normally, count only in
  `labels_external_dataset_frame_count`, and never enter the owner-independent-human bucket.
- Best-stack delta: NO stack delta. This is training infrastructure only; no default model,
  checkpoint, runtime route, or policy changed.
