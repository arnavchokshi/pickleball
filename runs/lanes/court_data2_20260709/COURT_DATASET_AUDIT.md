# Roboflow real court dataset audit

Scanned 65 locally downloaded manifest datasets and found 23 court/theme-related directories.

| Dataset | Task | Split images | Resolution | License | Viewpoint | Verdict | Mapped | Confidence |
|---|---:|---:|---:|---|---|---|---:|---|
| `acmai__pickleball-courts-emwra__v3` | segmentation | train:84, valid:24, test:12 | 640-640x640-640 | CC BY 4.0 | elevated, broadcast | corners-only | 0 | medium |
| `chetan-rajagiri-9abfm__pickleball-court-v2__v1` | keypoints | train:25, valid:7, test:3 | 640-640x640-640 | CC BY 4.0 | elevated, steep | direct-map | 15 | high |
| `dians-workspace-qq6mg__pickle-net__v3` | segmentation | train:135, valid:9, test:8 | 384-384x384-384 | CC BY 4.0 | elevated, steep | unusable | 0 | none |
| `gideons__pickleball-court__v1` | segmentation | train:84, valid:24, test:12 | 640-640x640-640 | CC BY 4.0 | elevated, broadcast | corners-only | 0 | medium |
| `hilab__pickleball-lkbro__v1` | segmentation | train:8, valid:3, test:1 | 640-640x640-640 | CC BY 4.0 | broadcast | unusable | 0 | none |
| `hughs-workspace-plw3g__pickleball-court-cfyv4__v1` | segmentation | train:3438, valid:86, test:92 | 640-640x640-640 | CC BY 4.0 | broadcast, elevated | corners-only | 0 | medium |
| `liberin-technologies__pickleball-vision__v9` | segmentation | train:4483, valid:737, test:170 | 640-640x640-640 | CC BY 4.0 | broadcast, elevated | unusable | 0 | none |
| `luiss-workspace-99bfi__pickleball-court-detection-o8i4o__v1` | bbox | train:53, valid:6, test:3 | 512-512x512-512 | CC BY 4.0 | low | unusable_ambiguous | 0 | low |
| `n-do-tran__pickleball-court-p3chl__v4` | keypoints | train:84, valid:24, test:12 | 640-640x640-640 | CC BY 4.0 | elevated, steep, broadcast | partial-map | 14 | high |
| `necromancer__pickleball-court-vbmkq__v2` | keypoints | train:1584, valid:127, test:61 | 640-640x640-640 | CC BY 4.0 | broadcast, elevated, steep | partial-map | 12 | high |
| `nigh-workspace__pickleball-court-vhpgp__v11` | keypoints | train:241, valid:38, test:16 | 640-640x640-640 | CC BY 4.0 | broadcast, elevated | partial-map | 12 | high |
| `pickle-es3fs__pickleball-video__v10` | segmentation | train:185, valid:52, test:26 | 1280-2552x720-1206 | CC BY 4.0 | broadcast | corners-only | 0 | low |
| `pickleball-ball-detection__pickleball-court-keypoints-syncz__v6` | keypoints | train:29, valid:8, test:4 | 1280-1280x720-720 | CC BY 4.0 | broadcast, elevated | partial-map | 12 | high |
| `pickleball-dl6zm__pickleball-courts-emwra-w8dsr__v1` | segmentation | train:84, valid:24, test:11 | 640-640x640-640 | CC BY 4.0 | elevated, broadcast | corners-only | 0 | medium |
| `pickleball-od8al__pickleball-seg__v20` | segmentation | train:868, valid:228, test:110 | 1280-1280x720-720 | CC BY 4.0 | broadcast, elevated | unusable_ambiguous | 0 | low |
| `pickleball-tsibp__paddle-4point-aljfe__v1` | keypoints | train:480, valid:46, test:23 | 640-640x640-640 | CC BY 4.0 | low, portrait | unusable | 0 | none |
| `ping-pong-paddle-ai-with-images__pickleball-court-p3chl-7tufp__v3` | keypoints | train:252, valid:24, test:12 | 640-640x640-640 | CC BY 4.0 | elevated, steep, broadcast | partial-map | 14 | high |
| `stump-detection-front-view-mj39q__pickle-ball-court-keypoints__v1` | keypoints | train:296 | 640-640x640-640 | CC BY 4.0 | broadcast, elevated | partial-map | 12 | high |
| `testworkspace-i8nb1__pickle-court-keypoints__v2` | keypoints | train:53, valid:15, test:8 | 640-640x640-640 | CC BY 4.0 | broadcast, low, steep | partial-map | 12 | high |
| `vit-bdraw__pickle-ball-vka5h__v7` | segmentation | train:291, valid:20, test:10 | 1280-1920x720-1080 | CC BY 4.0 | low | unusable | 0 | none |
| `xas-workspace-j20pu__pickleball-net-corners__v1` | keypoints | train:1008, valid:95, test:31 | 640-640x360-360 | CC BY 4.0 | low, broadcast | partial-map | 2 | medium |
| `xas-workspace-j20pu__pickleball-paddle-backhand-dink__v1` | keypoints | train:82, valid:8, test:2 | 640-640x360-360 | CC BY 4.0 | low, portrait | unusable | 0 | none |
| `xuann-bacc-ujr91__pickle-court-keypoints-nluo7__v10` | keypoints | train:683, valid:195, test:156 | 640-1280x360-720 | CC BY 4.0 | broadcast, elevated | partial-map | 12 | high |
