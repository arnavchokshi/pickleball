# RKT refutation lane — 2nd vote on load-bearing single-source claims (adversarial verification)

You are the REFUTATION lane in a 2-vote primary-source verification pattern. Two independent
surveys ran; the claims below are load-bearing for an adoption report + benchmark/GT-capture spec
and currently carry ONE vote. Independently CONFIRM or REFUTE each from primary sources you fetch
yourself. Be adversarial. Do NOT read `runs/research_trk_rkt_20260716/rkt_survey_A_20260716/` or
`rkt_survey_B_20260716/` (claims restated fully below; independent fetching is the point).

For EACH claim output: verdict CONFIRM / REFUTE / PARTIAL / UNRESOLVED, primary URL(s) fetched,
short evidence quote, fetch date, corrected fact if any.

## Claims

C1. RacketVision bundle (AAAI-26 paper, repo github.com/OrcustD/RacketVision, HF dataset
    linfeng302/RacketVision): (a) dataset = ~1,672 clips / ~435,179 frames / ~24,621 racket
    annotations, five 2D keypoints (top, bottom, handle, left, right) + box — NO 3D/6DoF pose,
    NO camera intrinsics, NO contact GT; (b) racket-pose baseline is single-frame RTMDet-M +
    RTMPose-M; PCK@0.2 / MPJPE: table tennis 81.8/9.71px, tennis 89.6/5.34px, badminton
    88.5/5.00px; side keypoints (left/right) drop to ~64.8-80.1 PCK; (c) source material is
    professional YouTube broadcast video (~942 videos); (d) code MIT, dataset card MIT-labeled;
    (e) HF model folder ~617MB with pose weights ~411MB and ~107MB, NO model card/license, and
    HF flags the pickle files "unsafe". HEAD the weight files.

C2. RACE-6D (CVPR 2026 Findings, Ha et al.; repo github.com/Yoonwoo-Ha/RACE-6D): joint
    detection+6D-pose RT-DETR extension for known objects; RGB model ~76.7% BOP AR on YCB-Video
    at ~16.6 FPS; mean rotation error on YCB-V ~10.3° RGB / ~8.0° RGB-D; repo live, Apache-2.0,
    code but NO released checkpoints as of 2026-07-16.

C3. TT4D (arXiv 2605.01234): paddle/racket state inferred by inverse control from ball
    trajectories; against 92 mocap strokes with IR racket markers: mean orientation error
    26.4 ± 4.4°, velocity error 0.58 ± 0.40 m/s; no public release artifact w/ license as of
    2026-07-16.

C4. Event-camera contact evidence: (a) single-event-camera tennis paper (arXiv 2506.08327):
    impact-position error <15mm when contours recovered, but contour success 24/26 without
    direct sunlight vs ~3/20 in direct sunlight; (b) dual-event-camera badminton study
    (arXiv 2605.28011): 116/124 (~93.5%) impact-localization success, bias ~1.84ms timing /
    ~3.45mm and ~-1.92mm on face axes, 95% LoA within ~±10mm.

C5. Sim2real numbers bundle: (a) Self6D (arXiv 2004.06468): LineMOD ADD(-S) recall
    synthetic-only ~40.1% → +unlabeled-real ~58.9% vs real-labeled ~86.9%; Occluded-LineMOD
    15.1%/32.1%/70.2%; (b) DOPE (Tremblay et al., CoRL 2018): DR+photoreal synthetic-only
    sugar-box ~77.0 ADD AUC vs 66.64 DR-only / 62.94 photoreal-only; (c) MegaPose: ModelNet
    5°/5cm refinement accuracy ~88.6% RGB — a refinement-from-noisy-init number, NOT end-to-end;
    BOP mean AR ~54.5 RGB. (d) ROCK (zhongcl-thu.github.io/rock): synthetic-only keypoint model
    ~59.4% avg recall on YCB-Video 6DoF eval, above Self6D variants.

C6. GT-capture metrology precedents: (a) Garon et al. (arXiv 1803.10075): 8-camera Vicon
    MX-T40, 3mm retroreflective markers, cited accuracy ~0.15mm static / ~2mm dynamic;
    (b) PhoCaL (CVPR 2022): robot + hand-eye + ICP GT with ~0.20mm / ~0.38° RMSE refinement;
    (c) Imitrob: two RealSense 848x480@60Hz + HTC Vive 30Hz tracker on the tool, ~184k images,
    CC BY-NC-SA 4.0; (d) Anipose ChArUco multi-camera validation: >90% of board pose estimates
    <1° angular error in its six-camera test.

C7. BlenderProc: current code license is GPL-3.0, and an official example exists enabling
    motion blur + rolling shutter (`enable_motion_blur(..., rolling_shutter_type="TOP", ...)`)
    — confirm the exact API/example exists in the live repo. Kubric = Apache-2.0, archived or
    weak maintenance signal. Isaac Sim Replicator object-SDG tutorial exists w/ subframe
    motion blur; platform under NVIDIA Omniverse EULA.

C8. License/elimination bundle: (a) FoundationPose license = NVIDIA non-commercial (research/
    evaluation only) AND its released/evaluated path is RGB-D — fetch the LICENSE and README;
    (b) FoundPose repo = CC BY-NC 4.0 and releases COARSE pose only (no featuremetric
    refinement); (c) GigaPose = MIT code, ~3.81GB checkpoint live on HF, paper documents a
    small-visible-segment failure mode; (d) KV-Tracker (CVPR 2026, repo Marwan99/kv_tracker)
    exists, ~30FPS monocular-RGB online tracking claim, custom Imperial non-commercial license;
    (e) GRAB = custom non-commercial; GraspXL data = CC BY-NC 4.0; (f) SAM-6D repo has NO
    LICENSE file; (g) IPPE reference implementation (tobycollins/IPPE) = BSD-3-Clause and the
    method returns exactly two planar solutions.

C9. Blur-direction evidence: BlurHandNet (CVPR 2023) recovers a temporal hand-mesh sequence
    from ONE blurry image and beats sharp-trained baselines on blurry input; "Human from Blur"
    (ICCV 2023) fits sub-frame pose sequences via differentiable blur rendering. Confirm both
    exist as described and that no equivalent released package exists for rigid sports
    equipment.

C10. Sanity-check the sync math used in the GT spec: at 60fps, 0.5 frame = ~8.33ms; at
     10-20 m/s paddle/ball relative speed that is ~8-17cm of motion — i.e. NS-02-style
     "≤0.5 frame" sync is insufficient for 3cm contact GT, and ~≤1ms (or ≤0.25 of a 240fps
     frame) is the right target. Also: for a line of projected length L px with endpoint error
     sigma px, in-plane angle noise ≈ sqrt(2)*sigma/L rad — check this formula's validity as a
     first-order estimate (σ=5px, L=40px → ~10°; L=80px → ~5°).

## Deliverables (write ONLY into /Users/arnavchokshi/Desktop/pickleball/runs/research_trk_rkt_20260716/rkt_refute_20260716/)

1. `REFUTATION.md` — one section per claim C1-C10: verdict, evidence, URL, date, corrections.
2. `livechecks.md` — every URL fetched: HTTP status/bytes, date.
3. Final message: ≤25 lines — verdict list + anything materially changing the picture.

Rules: no GPU work, no multi-GB downloads (HEAD/partial fetch), no pipeline/config edits, no
other runs/ dirs. Published numbers ≠ pickleball accuracy. VERIFIED=0 stands.
