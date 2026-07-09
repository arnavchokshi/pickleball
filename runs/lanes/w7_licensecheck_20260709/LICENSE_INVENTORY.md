# P7-4d Licensing Inventory vs Future Monetization

Lane: `w7_licensecheck_20260709`  
Scope: read-only inventory for Stripe/future monetization. Private/internal use remains owner-approved, but that is not commercial clearance.  
Method: local-only evidence from repo docs, manifests, vendored LICENSE/README files, package locks, and dataset READMEs/manifests. No network was used. Anything not locally determinable is marked `UNRESOLVED-needs-network`.

## Gate Verdict

**P7-4d commercial launch verdict: BLOCKED before Stripe flips on.**

The product can continue private/internal experimentation under the owner ruling, but monetized use needs resolution or replacement for:

1. **Roboflow Universe ToS/model-commercialization terms**: local dataset licenses are mostly CC BY 4.0, but local files do not contain Roboflow platform ToS terms for commercial training, model resale, hosted inference, or redistribution. Treat the Roboflow-trained model path as `UNRESOLVED-needs-network`.
2. **One Roboflow NC dataset**: `testing-esifc/pickle-ball-labeling-mff1d` is recorded as `BY-NC-SA 4.0`; exclude it from any commercial training set and rebuild derived checkpoints without it.
3. **PnLCalib / No-Bells-Just-Whistles**: local roadmap/blueprint evidence says GPL-2.0. Do not ship GPL court-calibration code/weights in a proprietary monetized service without counsel-approved isolation/replacement.
4. **SAM-3D-Body / MHR / SMPL-family BODY assets**: active BODY relies on Fast-SAM-3D-Body/SAM-3D-Body/MHR-family assets. The local model manifest records SAM License / verify-commercial posture; local README also requires SMPL assets for MHR-to-SMPL/SAT-HMR paths. Commercial terms must be verified or replaced.
5. **SMPL-family body model files**: `third_party/SAT-HMR/weights/smpl_data/smpl/*.pkl` are present, but no local license text accompanies them. Treat as restricted/unknown for commercial launch until licensed.
6. **AGPL YOLO weights**: model manifest records YOLO11/YOLO26 detector weights as AGPL-3.0 / `agpl_caveat`. Replace with permissive detectors or buy/verify a commercial license before product use.
7. **Harvested public YouTube footage**: owner waived copyright concerns for private use only. Commercial training/testing from harvested clips needs rights review; never redistribute clips or customer-facing retained artifacts derived from them without clearance.
8. **RacketVision**: local docs reference MIT and downloadable checkpoints, but the repo/checkpoints are not vendored locally. Before adopting, fetch and archive LICENSE/model-card terms.
9. **iOS brand image assets**: local files exist with no local license/provenance note. If they are first-party generated, record that; otherwise resolve before App Store/commercial distribution.

## Evidence Pointers

- Repo truth says `VERIFIED=0` and no promotion from smoke/internal artifacts: `README.md:20-27`, `MASTER_PLAN.md:29-72`.
- P7-4d is explicitly a monetization gate: `NORTH_STAR_ROADMAP.md:1750-1758`; private-use freedom is not commercial clearance: `NORTH_STAR_ROADMAP.md:1793-1800`.
- Roboflow download manifest records 75 enumerated / 65 downloaded datasets and per-project `license_as_recorded`: `data/roboflow_universe_20260706/manifest.json:1-19`, first entry shape at `data/roboflow_universe_20260706/manifest.json:51-80`.
- Aggregated corpus has 61,260 kept samples and index-only policy: `data/roboflow_universe_20260706/aggregated/corpus_card.json:1-34`, `data/roboflow_universe_20260706/aggregated/corpus_card.json:111-115`.
- Vendored code pins and roles: `third_party/VENDOR_PINS.md:8-17`.
- Model/checkpoint license registry: `models/MANIFEST.json:5-69`, `models/MANIFEST.json:143-225`, `models/MANIFEST.json:388-456`, `models/MANIFEST.json:459-470`.
- Web package dependencies: `web/replay/package.json:12-25`; lockfile license examples: `web/replay/package-lock.json:26-40`, `web/replay/package-lock.json:1286-1291`, `web/replay/package-lock.json:1442-1447`, `web/replay/package-lock.json:1578-1590`, `web/replay/package-lock.json:1698-1703`.
- iOS package has local targets and no external Swift package dependency list: `ios/Package.swift:4-52`.
- Harvest private-use note is encoded in code/artifacts: `NORTH_STAR_ROADMAP.md:67-72`, `NORTH_STAR_ROADMAP.md:693-708`, `threed/racketsport/online_harvest_ingest.py:35-44`, `threed/racketsport/online_harvest_ingest.py:512-525`.

## Main Inventory

| Component | Where used | Local license evidence | Commercial-use verdict | Load-bearing or swappable | Remediation option |
|---|---|---|---|---|---|
| Roboflow Universe pickleball corpus, 65 downloaded datasets | `configs/racketsport/ball_pretrain_roboflow_wasb.json`; aggregation under `data/roboflow_universe_20260706/aggregated/`; BALL pretrain/aux data | Manifest records `downloaded=65`; license distribution from local `license_as_recorded`: 63 `CC BY 4.0`, 1 `Public Domain`, 1 `BY-NC-SA 4.0`; dataset README files have no license line, so the manifest is the local license evidence | Mixed: CC BY/Public Domain are commercially usable in principle with attribution/provenance; NC dataset is restricted; Roboflow platform ToS is `UNRESOLVED-needs-network` | Load-bearing for public pretrain/aux, swappable with owner-labeled data or licensed datasets | Exclude NC row, preserve attribution/provenance for CC BY, resolve Roboflow ToS, rebuild any model trained on disallowed data |
| Roboflow NC dataset `testing-esifc/pickle-ball-labeling-mff1d` | Same corpus | `data/roboflow_universe_20260706/manifest.json` entry records `license_as_recorded=BY-NC-SA 4.0` | **NC/restricted** | Swappable | Remove from commercial corpus and retrain/re-score without it |
| Roboflow ToS constraints on trained-model commercialization | Any model trained with Roboflow Universe data | No local ToS file found; roadmap explicitly flags "Roboflow Universe ToS vs commercial training and redistribution" | **unknown / UNRESOLVED-needs-network** | Load-bearing if public pretrain remains in commercial model lineage | Obtain current Roboflow ToS/export/model commercialization terms; record allowed uses; rebuild without Roboflow if terms block monetization |
| PnLCalib / No-Bells-Just-Whistles court calibration lineage | P4 court auto-find research/solver candidates; `court_keypoint_geometric_loss.py` is PnLCalib-style, not vendored code | `NORTH_STAR_ROADMAP.md:593-595` says GPL-2.0 code; `TECH_BLUEPRINTS.md:2290-2295` says GPL-2.0 and weights were previously local; current `models/checkpoints/court_external/` is absent | **viral-GPL** for code/weights if adopted; current in-repo "style" loss is own implementation but any direct GPL code/weights must be isolated | Swappable; current product has manual/profile court path | Do not ship GPL code/weights in proprietary product path; use own/permissive model, manual/profile court flow, or counsel-approved service isolation |
| WASB-SBDT vendored code | BALL verifier/inference path; `scripts/racketsport/run_wasb_ball.py`, `threed/racketsport/wasb_adapter.py` | Vendored LICENSE is MIT (`third_party/WASB-SBDT/LICENSE.md:1-13`); README advertises MIT badge and model zoo (`third_party/WASB-SBDT/README.md:1-26`) | OK for code/checkpoint license per local evidence | Load-bearing today for BALL zero-shot/prelabel seed, but swappable | Keep license notice; verify downloaded checkpoint provenance remains official; preserve attribution |
| WASB tennis checkpoint | `models/checkpoints/wasb/wasb_tennis_best.pth.tar`; BALL verifier seed | Model manifest records MIT, `commercial_posture=ok`, official model zoo source, sha256 (`models/MANIFEST.json:427-443`) | OK per local manifest | Load-bearing as current BALL anchor, swappable with trained owner model | Keep manifest/sha; retrain a commercial-clean owner-data model before promotion |
| TrackNetV3 vendored code/checkpoints | BALL candidate/pretrain; `scripts/racketsport/run_tracknet_ball.py` | Vendored LICENSE is MIT (`third_party/TrackNetV3/LICENSE:1-13`); README points to checkpoints (`third_party/TrackNetV3/README.md:58-60`); model manifest records MIT and official seed (`models/MANIFEST.json:388-424`) | OK per local evidence | Swappable; not current promoted default | Keep notices; if fine-tuned, record training data licenses |
| TrackNetV4 vendored code | BALL candidate; `scripts/racketsport/run_tracknetv4_ball.py` | Vendored LICENSE is MIT (`third_party/TrackNetV4/LICENSE:1-13`); vendor pin says blocked-no-usable-weights (`third_party/VENDOR_PINS.md:12`) | OK for code; weights not available locally | Swappable / currently not load-bearing | No launch blocker if unused; verify any future checkpoint license before use |
| TOTNet vendored code/checkpoint | BALL candidate; `scripts/racketsport/run_totnet_ball.py`; `third_party/TOTNet/weights/...best.pth` | Vendored LICENSE is MIT (`third_party/TOTNet/LICENSE:1-13`); README says TTA dataset is research-only (`third_party/TOTNet/README.md:68-71`) | Code OK; bundled tennis checkpoint terms are not independently documented beyond repo; TTA data path restricted if used | Swappable / measured-dead candidate per vendor pins | Keep out of commercial default unless checkpoint/data license is documented; retrain on clean data if needed |
| blurball vendored code/weights links | WASB-family training fork / blur sidecar lineage | Vendored LICENSE is MIT (`third_party/blurball/LICENSE.md:1-13`); README describes dataset/weights links (`third_party/blurball/README.md:30-55`) | Code OK; external dataset/weight terms `UNRESOLVED-needs-network` if adopted | Swappable | Use code with notice; do not commercialize external dataset/weights until license/model-card terms are archived |
| RacketVision code/checkpoints | Planned P1-8/P3-4 ball/racket/paddle auxiliary source; not vendored under `third_party/` | No vendored local repo/checkpoint. Local planning docs say MIT/downloadable (`TECH_BLUEPRINTS.md:1058-1059`, `TECH_BLUEPRINTS.md:2002-2005`) | **unknown until vendored LICENSE/model-card verified locally** | Swappable / not currently load-bearing | Before use: fetch repo/checkpoints, archive LICENSE/model-card, record dataset and weight terms |
| Fast-SAM-3D-Body vendored code | Active BODY runtime wrapper / remote body path | Vendored LICENSE is MIT (`third_party/Fast-SAM-3D-Body/LICENSE:1-13`); README says it builds on SAM 3D Body and MHR (`third_party/Fast-SAM-3D-Body/README.md:118-121`) | Code OK, but upstream SAM/MHR/checkpoint licenses still matter | Load-bearing for active BODY, swappable only with major BODY replacement | Keep vendored code notice; resolve underlying SAM-3D-Body/MHR weight terms before monetization |
| SAM-3D-Body DINOv3 checkpoint | Active BODY deep-tier backbone | Model manifest records `SAM License`, `commercial_posture=research_ok_verify_commercial`, remote/local path and sha (`models/MANIFEST.json:5-23`) | **unknown / verify-commercial** | Load-bearing | Obtain SAM License text/current model-card terms; if commercial use not allowed, replace BODY backbone or get license |
| MHR model / MHR-to-SMPL mapping assets | BODY mesh/joint pipeline and MHR latent smoothing/decode | Model manifest records MHR asset as `SAM License` and verify-commercial (`models/MANIFEST.json:26-36`); roadmap says MHR code is Apache-2.0 but SAM-3D-Body custom license (`NORTH_STAR_ROADMAP.md:413-415`) | **unknown / verify-commercial** for weights/assets; code may be OK but asset terms govern deployment | Load-bearing for current BODY artifacts | Archive actual license text for MHR weights/assets; keep only MHR70 outputs if allowed; otherwise replace |
| SMPL / SMPL-family body model files | SAT-HMR fallback; MHR2SMPL conversion; local files under `third_party/SAT-HMR/weights/smpl_data/smpl/` | SAT-HMR README instructs downloading SMPL weights from SMPL/SMPLify (`third_party/SAT-HMR/README.md:83-91`); Fast-SAM mhr2smpl README says `SMPL_NEUTRAL.pkl` must be downloaded separately from smplx (`third_party/Fast-SAM-3D-Body/mhr2smpl/README.md:121-128`, `:160-164`) | **restricted/unknown; treat as commercial blocker until licensed** | Swappable if BODY avoids SMPL output; load-bearing for SMPL export/fallback | Do not ship SMPL-derived export/fallback commercially until commercial SMPL terms are obtained; prefer MHR/native skeleton if cleared |
| SAT-HMR code/checkpoint | BODY fallback/fast mesh preview candidate | Vendored LICENSE is Apache-2.0 (`third_party/SAT-HMR/LICENSE:1-5`, grant at `:66-70`); model manifest says checkpoint/data terms need verification (`models/MANIFEST.json:39-53`, `:446-456`) | Code OK; checkpoint/data/SMPL terms **UNRESOLVED-needs-network** | Swappable / fallback, not active default | Keep out of monetized default until checkpoint and SMPL terms are cleared |
| MultiHMR2 checkpoint | Fast mesh preview candidate | Model manifest: "Internal use only per upstream pyproject metadata; user-approved for personal testing only", `commercial_posture=avoid` (`models/MANIFEST.json:459-470`) | **restricted / avoid** | Swappable / not default | Exclude from commercial product; replace with permissive model |
| MoGe/FOV depth prior | Fast-SAM-3D-Body FOV estimator path | Model manifest says "See upstream MoGe license", `commercial_posture=unknown`, and "license must be verified before commercialization" (`models/MANIFEST.json:56-69`) | **unknown / UNRESOLVED-needs-network** | Swappable / feature-dependent | Disable or replace before launch unless license cleared |
| YOLO11/YOLO26 Ultralytics weights | Racket failed probes; person detector candidate (`yolo26m`) | Model manifest records AGPL-3.0 via YOLO11/YOLO26 and AGPL caveat (`models/MANIFEST.json:143-149`, `:163-168`, `:182-187`, `:214-225`) | **viral-GPL/AGPL caveat** | Swappable; `yolo26m` may be load-bearing for TRK/person detect candidate | Replace with Apache/MIT/commercial detector or obtain commercial Ultralytics license before monetized use |
| Harvested public YouTube/online footage | P0-1b harvest corpus, prelabels, broad tests, potential training | Owner private-use approval encoded in roadmap and ingest artifacts (`NORTH_STAR_ROADMAP.md:67-72`, `:693-708`; `threed/racketsport/online_harvest_ingest.py:35-44`, `:512-525`) | Private use OK by owner; **commercial rights unknown** | Useful training/test fuel, swappable with owner/licensed footage | For monetization, either use only licensed/owned footage, obtain rights, or exclude harvested footage from commercial model lineage |
| Web replay NPM libraries | `web/replay` React/Three/Vite/Vitest app | `package.json` deps are React, React-DOM, Three, @react-three/fiber; lockfile license distribution: 121 MIT, 3 Apache-2.0, 2 BSD-3-Clause, 2 ISC | OK | Load-bearing for web viewer but dependencies are permissive | Keep package-lock, reproduce third-party notices in distributed web/app bundle |
| Web replay fonts/assets | `web/replay` | Local scan found no committed font or raster/SVG media assets outside source/fixtures | OK / none found | Not applicable | If assets are added later, record source/license |
| iOS Swift package/code | `ios/` capture/live/upload/replay app | `ios/Package.swift` defines local targets only, with no external package dependencies (`ios/Package.swift:4-52`) | OK for first-party code; Apple SDK/App Store terms not locally inventoried | Load-bearing | For launch, separately handle Apple Developer/App Store terms outside this repo inventory |
| iOS app PNG assets | App icon, DinkVision mark/lockup under `ios/App/Assets.xcassets` | Local files exist; no local license/provenance note found | **unknown unless first-party** | Load-bearing for app branding | Add a short provenance/license note before commercial distribution; replace if not first-party |
| iOS replay fixtures (`USDZ`, `virtual_world.json`) | Swift replay tests/resources | `ios/Package.swift` copies `Resources/WorldFixture` and `Resources/RealityReplayFixture` (`ios/Package.swift:28-35`); fixture license follows generated pipeline/source assets, not a separate local license | Internal/test OK; distribution depends on underlying model/data/video licenses | Swappable test fixture | Keep fixtures test-only or regenerate from commercial-clean sources before public app distribution |

## Roboflow Dataset License Distribution

Downloaded dataset count: 65.

| License as recorded | Count | Commercial posture |
|---|---:|---|
| CC BY 4.0 | 63 | OK with attribution/provenance, subject to Roboflow ToS and source-media rights |
| Public Domain | 1 | OK, subject to Roboflow ToS/source-media verification |
| BY-NC-SA 4.0 | 1 | NC/restricted; exclude from commercial training |

Non-permissive Roboflow dataset:

| Dataset | License | Verdict | Evidence |
|---|---|---|---|
| `testing-esifc/pickle-ball-labeling-mff1d` v1 | BY-NC-SA 4.0 | NC/restricted | `data/roboflow_universe_20260706/manifest.json` `license_as_recorded`; local README path `data/roboflow_universe_20260706/testing-esifc__pickle-ball-labeling-mff1d__v1/README.roboflow.txt` |

## Roboflow Downloaded Dataset Appendix

The dataset README files record Roboflow export metadata but no license line; the license column below comes from `data/roboflow_universe_20260706/manifest.json` `license_as_recorded`.

| Dataset | License | Verdict | Local evidence |
|---|---|---|---|
| `acmai/pickleball-courts-emwra` v3 | CC BY 4.0 | OK-with-attribution | `acmai__pickleball-courts-emwra__v3/README.roboflow.txt`; manifest `license_as_recorded` |
| `aipickleballref/ai-pickleball-referee` v5 | CC BY 4.0 | OK-with-attribution | `aipickleballref__ai-pickleball-referee__v5/README.roboflow.txt`; manifest `license_as_recorded` |
| `ak-zcxgt/pickleball-uninu-suhi2` v1 | CC BY 4.0 | OK-with-attribution | `ak-zcxgt__pickleball-uninu-suhi2__v1/README.roboflow.txt`; manifest `license_as_recorded` |
| `ball-th4g9/pickle-ball-ncgxh` v1 | CC BY 4.0 | OK-with-attribution | `ball-th4g9__pickle-ball-ncgxh__v1/README.roboflow.txt`; manifest `license_as_recorded` |
| `balthasars-workspace/pickle-ball-detection-sample` v8 | Public Domain | OK | `balthasars-workspace__pickle-ball-detection-sample__v8/README.roboflow.txt`; manifest `license_as_recorded` |
| `chetan-rajagiri-9abfm/pickleball-court-v2` v1 | CC BY 4.0 | OK-with-attribution | `chetan-rajagiri-9abfm__pickleball-court-v2__v1/README.roboflow.txt`; manifest `license_as_recorded` |
| `connor-mckinnis-t2mcr/pickleball-dibs-rack` v1 | CC BY 4.0 | OK-with-attribution | `connor-mckinnis-t2mcr__pickleball-dibs-rack__v1/README.roboflow.txt`; manifest `license_as_recorded` |
| `dataset-w4lqc/pickle-ball` v3 | CC BY 4.0 | OK-with-attribution | `dataset-w4lqc__pickle-ball__v3/README.roboflow.txt`; manifest `license_as_recorded` |
| `dians-workspace-qq6mg/pickle-net` v3 | CC BY 4.0 | OK-with-attribution | `dians-workspace-qq6mg__pickle-net__v3/README.roboflow.txt`; manifest `license_as_recorded` |
| `gamechangerv1/pickleball-detection-1oqlw` v3 | CC BY 4.0 | OK-with-attribution | `gamechangerv1__pickleball-detection-1oqlw__v3/README.roboflow.txt`; manifest `license_as_recorded` |
| `gideons/pickleball-court` v1 | CC BY 4.0 | OK-with-attribution | `gideons__pickleball-court__v1/README.roboflow.txt`; manifest `license_as_recorded` |
| `harshita-wafmq/pickleball-ball-tracking-9s7d6` v1 | CC BY 4.0 | OK-with-attribution | `harshita-wafmq__pickleball-ball-tracking-9s7d6__v1/README.roboflow.txt`; manifest `license_as_recorded` |
| `hemel/pickleball-cedmo` v13 | CC BY 4.0 | OK-with-attribution | `hemel__pickleball-cedmo__v13/README.roboflow.txt`; manifest `license_as_recorded` |
| `hilab/pickleball-lkbro` v1 | CC BY 4.0 | OK-with-attribution | `hilab__pickleball-lkbro__v1/README.roboflow.txt`; manifest `license_as_recorded` |
| `hughs-workspace-plw3g/pickleball-court-cfyv4` v1 | CC BY 4.0 | OK-with-attribution | `hughs-workspace-plw3g__pickleball-court-cfyv4__v1/README.roboflow.txt`; manifest `license_as_recorded` |
| `hughs-workspace-plw3g/pickleball-with-players-topw1` v1 | CC BY 4.0 | OK-with-attribution | `hughs-workspace-plw3g__pickleball-with-players-topw1__v1/README.roboflow.txt`; manifest `license_as_recorded` |
| `hughs-workspace-qflbd/pickleball-6ijze` v2 | CC BY 4.0 | OK-with-attribution | `hughs-workspace-qflbd__pickleball-6ijze__v2/README.roboflow.txt`; manifest `license_as_recorded` |
| `jeff-xqthf/pickleball-wyhqe` v1 | CC BY 4.0 | OK-with-attribution | `jeff-xqthf__pickleball-wyhqe__v1/README.roboflow.txt`; manifest `license_as_recorded` |
| `khangnguyen-thqz2/pickleball-zwqih` v2 | CC BY 4.0 | OK-with-attribution | `khangnguyen-thqz2__pickleball-zwqih__v2/README.roboflow.txt`; manifest `license_as_recorded` |
| `liberin-technologies/pickleball-vision` v9 | CC BY 4.0 | OK-with-attribution | `liberin-technologies__pickleball-vision__v9/README.roboflow.txt`; manifest `license_as_recorded` |
| `luiss-workspace-99bfi/pickleball-court-detection-o8i4o` v1 | CC BY 4.0 | OK-with-attribution | `luiss-workspace-99bfi__pickleball-court-detection-o8i4o__v1/README.roboflow.txt`; manifest `license_as_recorded` |
| `meesoc/pickleball-detection-crqer` v1 | CC BY 4.0 | OK-with-attribution | `meesoc__pickleball-detection-crqer__v1/README.roboflow.txt`; manifest `license_as_recorded` |
| `michael-chen-debbx/pickleball-6douw` v5 | CC BY 4.0 | OK-with-attribution | `michael-chen-debbx__pickleball-6douw__v5/README.roboflow.txt`; manifest `license_as_recorded` |
| `mostafas-workspace-8icqn/pickleball-ball-detection-ngaan` v1 | CC BY 4.0 | OK-with-attribution | `mostafas-workspace-8icqn__pickleball-ball-detection-ngaan__v1/README.roboflow.txt`; manifest `license_as_recorded` |
| `n-do-tran/pickleball-court-p3chl` v4 | CC BY 4.0 | OK-with-attribution | `n-do-tran__pickleball-court-p3chl__v4/README.roboflow.txt`; manifest `license_as_recorded` |
| `narens/racket-cnl5d` v4 | CC BY 4.0 | OK-with-attribution | `narens__racket-cnl5d__v4/README.roboflow.txt`; manifest `license_as_recorded` |
| `necromancer/pickleball-court-vbmkq` v2 | CC BY 4.0 | OK-with-attribution | `necromancer__pickleball-court-vbmkq__v2/README.roboflow.txt`; manifest `license_as_recorded` |
| `nigh-workspace/pickleball-court-vhpgp` v11 | CC BY 4.0 | OK-with-attribution | `nigh-workspace__pickleball-court-vhpgp__v11/README.roboflow.txt`; manifest `license_as_recorded` |
| `nigh-workspace/pickleball-player-object-detection-cc2sw` v19 | CC BY 4.0 | OK-with-attribution | `nigh-workspace__pickleball-player-object-detection-cc2sw__v19/README.roboflow.txt`; manifest `license_as_recorded` |
| `object-detection-sb2zh/pickleball-detze` v1 | CC BY 4.0 | OK-with-attribution | `object-detection-sb2zh__pickleball-detze__v1/README.roboflow.txt`; manifest `license_as_recorded` |
| `object-detection-sb2zh/pickleball-qcata` v2 | CC BY 4.0 | OK-with-attribution | `object-detection-sb2zh__pickleball-qcata__v2/README.roboflow.txt`; manifest `license_as_recorded` |
| `personal-projj/pickleball-player-detection` v1 | CC BY 4.0 | OK-with-attribution | `personal-projj__pickleball-player-detection__v1/README.roboflow.txt`; manifest `license_as_recorded` |
| `pickle-2dggt/pickle-citxv` v2 | CC BY 4.0 | OK-with-attribution | `pickle-2dggt__pickle-citxv__v2/README.roboflow.txt`; manifest `license_as_recorded` |
| `pickle-es3fs/pickleball-video` v10 | CC BY 4.0 | OK-with-attribution | `pickle-es3fs__pickleball-video__v10/README.roboflow.txt`; manifest `license_as_recorded` |
| `pickleball-1uztf/pickleball-uninu` v1 | CC BY 4.0 | OK-with-attribution | `pickleball-1uztf__pickleball-uninu__v1/README.roboflow.txt`; manifest `license_as_recorded` |
| `pickleball-ball-detection/pickleball-court-keypoints-syncz` v6 | CC BY 4.0 | OK-with-attribution | `pickleball-ball-detection__pickleball-court-keypoints-syncz__v6/README.roboflow.txt`; manifest `license_as_recorded` |
| `pickleball-ball-detector/pickleball-ball-detector-4hus7` v2 | CC BY 4.0 | OK-with-attribution | `pickleball-ball-detector__pickleball-ball-detector-4hus7__v2/README.roboflow.txt`; manifest `license_as_recorded` |
| `pickleball-detection/pickleball-5pshr` v2 | CC BY 4.0 | OK-with-attribution | `pickleball-detection__pickleball-5pshr__v2/README.roboflow.txt`; manifest `license_as_recorded` |
| `pickleball-dl6zm/pickleball-courts-emwra-w8dsr` v1 | CC BY 4.0 | OK-with-attribution | `pickleball-dl6zm__pickleball-courts-emwra-w8dsr__v1/README.roboflow.txt`; manifest `license_as_recorded` |
| `pickleball-kjawm/tennis-ball-detection-sxi3e-inzuo` v1 | CC BY 4.0 | OK-with-attribution | `pickleball-kjawm__tennis-ball-detection-sxi3e-inzuo__v1/README.roboflow.txt`; manifest `license_as_recorded` |
| `pickleball-od8al/pickleball-seg` v20 | CC BY 4.0 | OK-with-attribution | `pickleball-od8al__pickleball-seg__v20/README.roboflow.txt`; manifest `license_as_recorded` |
| `pickleball-od8al/pickleball-tsgju` v4 | CC BY 4.0 | OK-with-attribution | `pickleball-od8al__pickleball-tsgju__v4/README.roboflow.txt`; manifest `license_as_recorded` |
| `pickleball-od8al/pickleball-version2` v3 | CC BY 4.0 | OK-with-attribution | `pickleball-od8al__pickleball-version2__v3/README.roboflow.txt`; manifest `license_as_recorded` |
| `pickleball-paddle-trainer/pickleball-paddle-detection` v1 | CC BY 4.0 | OK-with-attribution | `pickleball-paddle-trainer__pickleball-paddle-detection__v1/README.roboflow.txt`; manifest `license_as_recorded` |
| `pickleball-paddle-trainer/pickleball-paddle-detection-v2` v2 | CC BY 4.0 | OK-with-attribution | `pickleball-paddle-trainer__pickleball-paddle-detection-v2__v2/README.roboflow.txt`; manifest `license_as_recorded` |
| `pickleball-tsibp/paddle-4point-aljfe` v1 | CC BY 4.0 | OK-with-attribution | `pickleball-tsibp__paddle-4point-aljfe__v1/README.roboflow.txt`; manifest `license_as_recorded` |
| `pickleball-tsibp/paddle-detect-xsccm` v3 | CC BY 4.0 | OK-with-attribution | `pickleball-tsibp__paddle-detect-xsccm__v3/README.roboflow.txt`; manifest `license_as_recorded` |
| `pickleballballdetection/pickleball-ball-detection-uamje` v2 | CC BY 4.0 | OK-with-attribution | `pickleballballdetection__pickleball-ball-detection-uamje__v2/README.roboflow.txt`; manifest `license_as_recorded` |
| `ping-pong-paddle-ai-with-images/pickleball-court-p3chl-7tufp` v3 | CC BY 4.0 | OK-with-attribution | `ping-pong-paddle-ai-with-images__pickleball-court-p3chl-7tufp__v3/README.roboflow.txt`; manifest `license_as_recorded` |
| `racket-ai/pickleball-iiv9m` v5 | CC BY 4.0 | OK-with-attribution | `racket-ai__pickleball-iiv9m__v5/README.roboflow.txt`; manifest `license_as_recorded` |
| `s-workspace-zbpen/pickleball-7kgpr` v3 | CC BY 4.0 | OK-with-attribution | `s-workspace-zbpen__pickleball-7kgpr__v3/README.roboflow.txt`; manifest `license_as_recorded` |
| `salo-levy-nlqrn/pickle-ball-3fl0e` v12 | CC BY 4.0 | OK-with-attribution | `salo-levy-nlqrn__pickle-ball-3fl0e__v12/README.roboflow.txt`; manifest `license_as_recorded` |
| `seniordesignproject/pickleball-n86ai` v1 | CC BY 4.0 | OK-with-attribution | `seniordesignproject__pickleball-n86ai__v1/README.roboflow.txt`; manifest `license_as_recorded` |
| `stump-detection-front-view-mj39q/pickle-ball-court-keypoints` v1 | CC BY 4.0 | OK-with-attribution | `stump-detection-front-view-mj39q__pickle-ball-court-keypoints__v1/README.roboflow.txt`; manifest `license_as_recorded` |
| `test-hbuaz/pickleball-8anzg` v1 | CC BY 4.0 | OK-with-attribution | `test-hbuaz__pickleball-8anzg__v1/README.roboflow.txt`; manifest `license_as_recorded` |
| `testing-esifc/pickle-ball-labeling-mff1d` v1 | BY-NC-SA 4.0 | NC/restricted | `testing-esifc__pickle-ball-labeling-mff1d__v1/README.roboflow.txt`; manifest `license_as_recorded` |
| `testworkspace-i8nb1/pickle-court-keypoints` v2 | CC BY 4.0 | OK-with-attribution | `testworkspace-i8nb1__pickle-court-keypoints__v2/README.roboflow.txt`; manifest `license_as_recorded` |
| `thys-workspace-pfaeb/pickleball-ball-9lznj` v1 | CC BY 4.0 | OK-with-attribution | `thys-workspace-pfaeb__pickleball-ball-9lznj__v1/README.roboflow.txt`; manifest `license_as_recorded` |
| `vit-bdraw/pickle-ball-vka5h` v7 | CC BY 4.0 | OK-with-attribution | `vit-bdraw__pickle-ball-vka5h__v7/README.roboflow.txt`; manifest `license_as_recorded` |
| `xas-workspace-j20pu/pickleball-net-corners` v1 | CC BY 4.0 | OK-with-attribution | `xas-workspace-j20pu__pickleball-net-corners__v1/README.roboflow.txt`; manifest `license_as_recorded` |
| `xas-workspace-j20pu/pickleball-paddle-backhand-dink` v1 | CC BY 4.0 | OK-with-attribution | `xas-workspace-j20pu__pickleball-paddle-backhand-dink__v1/README.roboflow.txt`; manifest `license_as_recorded` |
| `xuann-bacc-ujr91/pickle-court-keypoints-nluo7` v10 | CC BY 4.0 | OK-with-attribution | `xuann-bacc-ujr91__pickle-court-keypoints-nluo7__v10/README.roboflow.txt`; manifest `license_as_recorded` |
| `xuann-bacc-ujr91/player-pickleball` v2 | CC BY 4.0 | OK-with-attribution | `xuann-bacc-ujr91__player-pickleball__v2/README.roboflow.txt`; manifest `license_as_recorded` |
| `yashkpro/pickleball-valmb` v5 | CC BY 4.0 | OK-with-attribution | `yashkpro__pickleball-valmb__v5/README.roboflow.txt`; manifest `license_as_recorded` |
| `yolo-na8ch/pickleball-akmms` v1 | CC BY 4.0 | OK-with-attribution | `yolo-na8ch__pickleball-akmms__v1/README.roboflow.txt`; manifest `license_as_recorded` |

## Counts

- Main inventory rows: 25.
- Roboflow downloaded datasets inventoried: 65.
- Locally non-permissive/restricted categories: 6 (`BY-NC-SA` dataset, PnLCalib GPL, YOLO AGPL, SMPL-family assets, MultiHMR2 internal-only, harvested footage for commercial use).
- `UNRESOLVED-needs-network` categories: 9 (Roboflow ToS, SAM License text, MHR/SAM weights, SMPL commercial terms, SAT-HMR checkpoint/data terms, MoGe, RacketVision actual LICENSE/model-card, blurball external weights/data, harvested footage commercial rights).

## Required Before Monetization

1. Produce a commercial-clean training-data ledger: source, license, ToS, attribution, role, and whether any trained checkpoint consumed it.
2. Rebuild or prove clean any checkpoint that consumed the `BY-NC-SA` Roboflow dataset or harvested public-video labels.
3. Decide the BODY commercial path: either clear SAM-3D-Body/MHR/SMPL terms, avoid SMPL-family outputs entirely, or replace with a permissive/commercial model.
4. Replace AGPL YOLO detector dependencies or document a commercial license.
5. Keep PnLCalib GPL code/weights out of the proprietary product path unless counsel approves an isolation strategy.
6. Add third-party notices for web NPM packages and vendored MIT/Apache components.
7. Record provenance/license for iOS brand PNGs and generated replay fixtures before App Store/commercial distribution.
