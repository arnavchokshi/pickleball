# Third-party vendor pins (ball tracking additions, 2026-07-05)

These checkouts are tracked as pinned gitlink entries (same pattern as TrackNetV3/SAT-HMR).
To restore on a fresh machine: git clone <url> third_party/<name> && git -C third_party/<name> checkout <sha>.
Local-only additions inside them are marked "pickleball addition, not upstream" in file headers
(pickleball dataset classes/configs in WASB-SBDT + blurball).

| dir | pinned sha | origin | role | local-only large files (never commit) |
|---|---|---|---|---|
| third_party/SAT-HMR | 920b1380e1d4967e47b1de1782408ac30f4f44da | https://github.com/ChiSu001/SAT-HMR.git | vendored-only BODY candidate/provenance; not live default | weights/* |
| third_party/TOTNet | 8a757f63391b262c14d18b4095486336852dbeef | https://github.com/AugustRushG/TOTNet.git | measured-dead zero-shot candidate (kept for provenance) | weights/*best.pth (~94MB) |
| third_party/TrackNetV4 | cb7eea7988474771ceac7e880bbffc35bfa87bca | https://github.com/TrackNetV4/TrackNetV4.git | blocked-no-usable-weights (motion-fusion ckpt undeserializable upstream) | none |
| third_party/WASB-SBDT | 923462cacdeb3353b84ddebdedb3f4b7a8553b0f | https://github.com/nttcom/WASB-SBDT.git | WASB inference (default ball stage) | none |
| third_party/blurball | 2f0f5496f7ba4b5b1a36790749935121b2ce972d | https://github.com/cogsys-tuebingen/blurball.git | WASB-family TRAINING fork (train_blur) + blur sidecar lineage | demo.gif (5.9MB) |
| third_party/spot | edec4201471beed631bed374bd0b95fcdc8a2f4f | https://github.com/jhong93/spot.git | E2E-Spot reference for event spotting - eval protocol + architecture reference; labels BSD-3 | none |

Ball model weights live under models/checkpoints/ (gitignored; sha256s in runs/manager/heldout_eval_ledger.md
row 22 and runs/lanes/ball_t6_gpu2_train_20260704/eval_results/).
