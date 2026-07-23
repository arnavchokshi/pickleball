# Roboflow court-keypoint schema map

Canonical output is the trainer's 15-key dictionary order. `null` channels are unsupervised. The current owner act permits only the 12 floor channels from approved external workspaces.

| Workspace | Raw images | Source keypoints | Schema/mapping | Confidence | Approved-mode yield | Disposition |
|---|---:|---|---|---|---:|---|
| `acmai__pickleball-courts-emwra__v3` | 120 | none (boxes/masks only) | no_usable_keypoint_mapping | none | 0 | skipped: no COCO keypoint schema; boxes/masks are not guessed into intersections |
| `chetan-rajagiri-9abfm__pickleball-court-v2__v1` | 35 | 15: far_left_baseline, far_center_baseline, far_right_baseline, far_left_kitchen, far_center_kitchen, far_right_kitchen, net_left, net_center, net_right, near_left_kitchen, near_center_kitchen, near_right_kitchen, near_left_baseline, near_center_baseline, near_right_baseline | semantic_direct_with_source_depth_reversal | high | 28 | mapped and emitted |
| `gideons__pickleball-court__v1` | 120 | none (boxes/masks only) | no_usable_keypoint_mapping | none | 0 | skipped: no COCO keypoint schema; boxes/masks are not guessed into intersections |
| `hughs-workspace-plw3g__pickleball-court-cfyv4__v1` | 3616 | none (boxes/masks only) | no_usable_keypoint_mapping | none | 0 | skipped: no COCO keypoint schema; boxes/masks are not guessed into intersections |
| `luiss-workspace-99bfi__pickleball-court-detection-o8i4o__v1` | 62 | none (boxes/masks only) | no_usable_keypoint_mapping | none | 0 | skipped: no COCO keypoint schema; boxes/masks are not guessed into intersections |
| `n-do-tran__pickleball-court-p3chl__v4` | 120 | 14: new-point-0, new-point-1, new-point-2, new-point-3, new-point-4, new-point-5, new-point-6, new-point-7, new-point-8, new-point-9, new-point-10, new-point-11, new-point-12, new-point-13 | generic_index_geometric_inferred_static | high | 25 | mapped and emitted |
| `necromancer__pickleball-court-vbmkq__v2` | 1772 | 12: SB1, BS1, SB2, SNZ2, SNZ3, SB3, BS2, SB4, SNZ4, SNZ1, NS1, NS2 | semantic_abbreviation_direct | high | 47 | mapped and emitted |
| `nigh-workspace__pickleball-court-vhpgp__v11` | 295 | 12: l-n-b, l-m-b, l-f-b, l-f-k, r-f-k, r-f-b, r-m-b, r-n-b, r-n-k, l-n-k, l-m-k, r-m-k | semantic_abbreviation_direct | high | 100 | mapped and emitted |
| `pickleball-ball-detection__pickleball-court-keypoints-syncz__v6` | 41 | 12: new-point-0, new-point-1, new-point-2, new-point-3, new-point-4, new-point-5, new-point-6, new-point-7, new-point-8, new-point-9, new-point-10, new-point-11 | generic_index_geometric_inferred_static | high | 13 | mapped and emitted |
| `pickleball-dl6zm__pickleball-courts-emwra-w8dsr__v1` | 119 | none (boxes/masks only) | no_usable_keypoint_mapping | none | 0 | skipped: no COCO keypoint schema; boxes/masks are not guessed into intersections |
| `ping-pong-paddle-ai-with-images__pickleball-court-p3chl-7tufp__v3` | 288 | 14: new-point-0, new-point-1, new-point-2, new-point-3, new-point-4, new-point-5, new-point-6, new-point-7, new-point-8, new-point-9, new-point-10, new-point-11, new-point-12, new-point-13 | generic_index_geometric_inferred_static | high | 44 | mapped and emitted |
| `stump-detection-front-view-mj39q__pickle-ball-court-keypoints__v1` | 296 | 12: A, B, C, D, E, F, G, H, I, J, K, L | generic_letter_geometric_inferred_static | high | 7 | mapped and emitted |
| `testworkspace-i8nb1__pickle-court-keypoints__v2` | 76 | 12: new-point-0, new-point-1, new-point-2, new-point-3, new-point-4, new-point-5, new-point-6, new-point-7, new-point-8, new-point-9, new-point-10, new-point-11 | generic_index_geometric_inferred_static | high | 0 | schema mapped but skipped: outside current immutable owner act |
| `xuann-bacc-ujr91__pickle-court-keypoints-nluo7__v10` | 1034 | 12: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 | generic_numeric_geometric_inferred_static | high | 0 | schema mapped but skipped: outside current immutable owner act |
