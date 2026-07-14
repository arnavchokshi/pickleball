# Event-data dual-survey cross-check ruling — 2026-07-13 (Fable)

Sources: ADOPTION_REPORT.md (Fable 32-agent fanout, 2-vote refuted) x SOL_SURVEY.md (independent
sol xhigh web survey). Owner hypothesis (public bootstrap data exists) VINDICATED by both.

## Convergent (adopt with confidence)
- OpenTTGames + Extended OpenTT: frame-exact BOUNCE (+HIT via extension), fixed camera, CC BY-NC-SA
  (R&D-ok, NC-flagged). BOTH rank it top-2. [CORROBORATED x2 + sol primary-fetch]
- ShuttleSet: ~36.5k timestamp-anchored HIT strokes, MIT annotations, broadcast video self-sourced.

## sol-only adds (fold in)
- Shuttlecock Hitting Event Detection (Zenodo 14677727): 1,199 clips POINT HitFrame + trained
  models, CC BY 4.0 = cleanest license of ANY point-label package. ADOPT (acquisition updated).
- AUDIO PILLAR the fanout under-covered: TT Sounds (5,702 impact snippets racket/table/floor,
  CC BY-NC), squash audio (5,791 events incl. racket-vs-floor, CC BY), padel CVPRW24 hit sounds
  (2,377; access-blocked, license unknown — owner-optional email). Pairs directly with our
  audio-first bootstrap; a small audio contact-classifier can be pretrained BEFORE any video head.
- GolfDB Impact-frame + SwingNet ckpt (CC BY-NC): timing-head init option.

## Disagreements resolved
- PadelTracker100: fanout rank-6 w/ open semantics question -> sol ANSWERED: shot labels are motion
  WINDOWS, not contact points => DEMOTED to weak-window supervision + ball/context only.
- jhong93/spot tennis (fanout rank-1, 33,791 HIT+BOUNCE single-frame, BSD-3 labels): NOT
  independently surfaced by sol => rank-1 CONDITIONAL on the acquisition lane's empirical JSON
  inspection (in flight, high priority). If confirmed, it stays rank-1; recipe unchanged either way.
- FineBadminton: interval labels (sol) / model-generated hit frames (fanout) => weak supervision
  only, both agree not point-GT.

## Ruling: bootstrap recipe v1 (supersedes report §2 recipe where they differ)
1. Video event head: pretrain 2-class(+bg) spotting head on union {jhong93/spot IF confirmed,
   OpenTTGames+Ext, ShuttleSet, Shuttlecock-HitFrame}, loss-masked per available class; condition on
   our WASB track + wrist channels.
2. Audio contact classifier: pretrain on TT Sounds + squash audio; fine-tune on our audio-bootstrap
   tier-A labels; fuse w/ the video head (product captures always have audio).
3. Fine-tune BOTH on pickleball: event_bootstrap tier-A labels (audio x track agreement) + owner
   spot-check gate (>=90% label precision on the 50-sample pack) BEFORE GPU training spend.
4. License ledger: NC-flagged sets = R&D/bootstrap only; commercially-clean spine exists (Zenodo
   shuttlecock CC BY 4.0 + jhong93 BSD-3 labels + ShuttleSet MIT + our own labels) for the eventual
   NS-07.3 relicense pass.

## ACQUISITION CORRECTIONS (Stage-0 empirical, 2026-07-13 — supersede where conflicting)
- jhong93/spot RANK-1 CONFIRMED empirically: 33,791 events reproduce exactly, +-frame window eval
  verified in util/eval.py, all 28 source videos live. Rank-1 stands unconditionally.
- Extended OpenTT stroke frame = CONTACT frame (evidence in data/event_public_20260713/ANSWERS.md)
  -> direct HIT supervision, not onset-shifted.
- Shuttlecock Zenodo (14677727) DEMOTED: test split ships NO ground-truth HitFrame (model-generated
  per its own README); part1.zip follow-up pending. The commercially-clean spine = jhong93 BSD-3
  labels + ShuttleSet MIT + our own labels (PadelTracker100 windows = context only).
- Squash audio: 0 confirmed structured labels on the fetched assets — dropped from the audio pillar
  unless figshare assets prove otherwise; TT Sounds remains the audio-pretrain anchor (NC-flagged).
- ON DISK NOW (data/event_public_20260713, 6.1GB, manifests+ledger): ~130k labeled events
  (jhong93 33,791 / F3Set 43,655 / ShuttleSet+22 88,840 partial-overlap / OpenTTGames 4,271 /
  TT Sounds 5,702 audio / GolfDB 11,200).
