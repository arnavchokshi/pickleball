# VM run plan — manager GPU leg only

Status: handoff plan, not executed in the local code lane. `VERIFIED=0` stays
binding. The protected four clips and the protected 50-row owner seed are eval
only and must never be copied into the training tree.

## 0. Provisioning and cost rails

Use the accelerator ladder in this order: A100-40GB, L4, T4. Reject any quote
whose five-hour worst case plus disk exceeds $10. Put the shutdown command in
the VM startup script so it is armed at boot, before SSH or driver setup can
race it:

```bash
#!/usr/bin/env bash
set -euo pipefail
sudo shutdown -P +300 "event_head_corpus hard five-hour wall rail"
date -u +%FT%TZ > /var/tmp/event_head_boot_rail_armed.txt
cat /run/systemd/shutdown/scheduled >> /var/tmp/event_head_boot_rail_armed.txt
```

Fail closed if `/run/systemd/shutdown/scheduled` is absent. Also run an idle
watchdog that powers off after 25 minutes without acquisition, train, eval, or
artifact-transfer processes. Delete the VM immediately if projected total cost
crosses $10; do not continue merely because a phase is already running.

## 1. Ship code and labels without AppleDouble files

On macOS, set `COPYFILE_DISABLE=1` on every tar creation and exclude existing
sidecars explicitly:

```bash
export COPYFILE_DISABLE=1
tar --exclude='._*' --exclude='.DS_Store' -czf event_head_code.tar.gz \
  threed/racketsport/event_head scripts/racketsport/build_event_head_dataset.py \
  scripts/racketsport/train_event_head.py scripts/racketsport/eval_event_head.py
tar --exclude='._*' --exclude='.DS_Store' -czf event_head_labels.tar.gz \
  data/event_public_20260713/jhong93_spot/data/tennis \
  data/event_public_20260713/jhong93_spot/manifest.json \
  data/event_public_20260713/openttgames/markup/extracted \
  data/event_public_20260713/openttgames/manifest.json \
  data/event_public_20260713/extended_openttgames/data \
  data/event_public_20260713/extended_openttgames/manifest.json \
  data/event_public_20260713/coachai_shuttleset/ShuttleSet/set
md5 event_head_code.tar.gz event_head_labels.tar.gz
```

After upload, compare the VM `md5sum` values to those Mac values before
extracting. Then strip any residual sidecars and prove none remain:

```bash
find ~/pickleball_g -name '._*' -delete
test -z "$(find ~/pickleball_g -name '._*' -print -quit)"
```

## 2. Re-probe and fetch all jhong93 sources as H.264 on the VM

Copy the current `jhong93_probe.tsv` into the VM, but re-probe every ID because
the 2026-07-13 LIVE result is stale. Fail if any source is unavailable; record
the fresh probe table. Fetch only missing parents, directly as AVC/H.264 at at
most 360p:

```bash
set -euo pipefail
ROOT=~/pickleball_g
PUB=$ROOT/data/event_public_20260713
VID=$PUB/jhong93_spot/videos_pilot
PROBE=$ROOT/runs/vm_out/jhong93_reprobe_$(date -u +%Y%m%dT%H%M%SZ).tsv
mkdir -p "$VID" "$ROOT/runs/vm_out"
printf 'yt_id\tname\tstatus\n' > "$PROBE"
tail -n +2 "$ROOT/runs/vm_in/jhong93_probe.tsv" | while IFS=$'\t' read -r yt_id name _; do
  if yt-dlp --simulate --no-playlist "https://www.youtube.com/watch?v=$yt_id" >/dev/null; then
    printf '%s\t%s\tLIVE\n' "$yt_id" "$name" >> "$PROBE"
  else
    printf '%s\t%s\tUNAVAILABLE\n' "$yt_id" "$name" >> "$PROBE"
    exit 20
  fi
done
tail -n +2 "$PROBE" | while IFS=$'\t' read -r yt_id name status; do
  find "$VID" -maxdepth 1 -type f -name "$name.*" ! -name '._*' | grep -q . && continue
  yt-dlp --no-playlist \
    -f 'bv*[vcodec^=avc1][height<=360]+ba/b[vcodec^=avc1][height<=360]' \
    --merge-output-format mp4 -o "$VID/$name.%(ext)s" \
    "https://www.youtube.com/watch?v=$yt_id"
done
```

Do not fall back to AV1. If a source lacks the declared AVC format, record the
miss and stop for a manager ruling instead of silently changing codecs.

## 3. Fetch the ten remaining OpenTT games

Use the 12 markup directory names as the authoritative list. Fetch missing
archives VM-side, extract exactly one MP4, and name it for the loader:

```bash
set -euo pipefail
PUB=~/pickleball_g/data/event_public_20260713
mkdir -p "$PUB/openttgames/videos"
for event_json in "$PUB"/openttgames/markup/extracted/*/events_markup.json; do
  name=$(basename "$(dirname "$event_json")")
  out="$PUB/openttgames/videos/$name.mp4"
  test -f "$out" && continue
  tmp=$(mktemp -d)
  curl -fL --retry 3 -o "$tmp/$name.zip" \
    "https://lab.osai.ai/datasets/openttgames/data/$name.zip"
  unzip -q "$tmp/$name.zip" -d "$tmp/unpacked"
  mapfile -t videos < <(find "$tmp/unpacked" -type f -iname '*.mp4' ! -name '._*')
  test "${#videos[@]}" -eq 1
  mv "${videos[0]}" "$out"
  rm -rf "$tmp"
done
```

## 4. Decode-verify every staged video before building

This is mandatory and precedes training. Open every file, require H.264/AVC,
positive FPS/frame count, and decode ten evenly spaced frames:

```bash
python3 - <<'PY'
from pathlib import Path
import cv2

roots = [
    Path.home()/"pickleball_g/data/event_public_20260713/jhong93_spot/videos_pilot",
    Path.home()/"pickleball_g/data/event_public_20260713/openttgames/videos",
]
videos = sorted(p for root in roots for p in root.glob("*.mp4") if not p.name.startswith("._"))
assert len(videos) == 40, f"expected 28 jhong93 + 12 OpenTT videos, got {len(videos)}"
for path in videos:
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), path
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = ''.join(chr((fourcc >> (8*i)) & 0xff) for i in range(4)).lower()
    assert frames > 0 and fps > 0, (path, frames, fps)
    assert codec in {"avc1", "h264"}, (path, codec)
    for index in [round(i*(frames-1)/9) for i in range(10)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        assert ok and frame is not None and frame.size, (path, index)
    cap.release()
print({"videos": len(videos), "decode_verified": True})
PY
```

## 5. Build twice and enforce corpus/split/window gates

```bash
cd ~/pickleball_g
python3 scripts/racketsport/build_event_head_dataset.py \
  --public-root data/event_public_20260713 --seed 20260716 \
  --out runs/vm_out/manifest_a.json
python3 scripts/racketsport/build_event_head_dataset.py \
  --public-root data/event_public_20260713 --seed 20260716 \
  --out runs/vm_out/manifest_b.json
cmp -s runs/vm_out/manifest_a.json runs/vm_out/manifest_b.json
python3 - <<'PY'
import json
from threed.racketsport.event_head.datasets import EXPECTED_UNIVERSE, manifest_windows
m=json.load(open("runs/vm_out/manifest_a.json"))
assert {k:v["inventory_events"] for k,v in m["totals"].items()} == EXPECTED_UNIVERSE
w=manifest_windows(m, split="train", limit=len(m["rows"]), window_frames=64, stride_frames=32)
assert len(w) >= 10_000, len(w)
print({"train_windows":len(w), "inventory":EXPECTED_UNIVERSE})
PY
```

The required totals are exactly jhong93 33,791, OpenTT 4,271, ShuttleSet
36,484. Preserve the manifest and its hash; do not alter the split seed after
seeing scores.

## 6. Probe workers, then train inside the five-hour VM wall

Use on-the-fly decode only; do not add a frame cache. Start with workers 8 and
prefetch 2. If host RAM or decoder stability fails, one predeclared fallback to
workers 4 is allowed; no other throughput tuning precedes the frozen run.

```bash
python3 scripts/racketsport/train_event_head.py --full \
  --manifest runs/vm_out/manifest_a.json --device cuda --weights imagenet \
  --image-size 224 --window-frames 64 --stride-frames 32 \
  --batch-size 4 --num-workers 8 --prefetch-factor 2 --lr 1e-3 \
  --steps 200 --val-every 200 --seed 20260716 \
  --out runs/vm_out/probe
```

Record sustained GPU utilization over the last 100 probe steps. The target is
at least 60%; a lower value is an honest miss, not permission to raise batch
size or change the gate. Then run 15 epochs, with the exact step count computed
from the measured train-window count and batch size. Reserve one hour of the
five-hour VM wall for staging/eval/pull, so training itself is capped at four
hours:

```bash
TRAIN_WINDOWS=$(python3 -c 'import json;from threed.racketsport.event_head.datasets import manifest_windows;m=json.load(open("runs/vm_out/manifest_a.json"));print(len(manifest_windows(m,split="train",limit=len(m["rows"]),window_frames=64,stride_frames=32)))')
STEPS=$(( (TRAIN_WINDOWS + 3) / 4 * 15 ))
VAL_EVERY=$(( (TRAIN_WINDOWS + 3) / 4 ))
python3 scripts/racketsport/train_event_head.py --full \
  --manifest runs/vm_out/manifest_a.json --device cuda --weights imagenet \
  --image-size 224 --window-frames 64 --stride-frames 32 \
  --batch-size 4 --num-workers 8 --prefetch-factor 2 --lr 1e-3 \
  --steps "$STEPS" --val-every "$VAL_EVERY" --seed 20260716 \
  --max-wall-minutes 240 --out runs/vm_out/train
```

If the 200-step probe required the declared workers-4 fallback, use workers 4
for the frozen run and stamp that change in the report.

## 7. Matched-window public eval, at least 50 windows, frozen sweep

The CLI defaults its window to the checkpoint's
`config.window_frames` and rejects mismatches. Run the preregistered threshold
sweep without changing values after viewing results:

```bash
for threshold in 0.5 0.3 0.2 0.1 0.05; do
  python3 scripts/racketsport/eval_event_head.py \
    --checkpoint runs/vm_out/train/best_event_head.pt --mode public \
    --manifest runs/vm_out/manifest_a.json --max-clips 50 \
    --threshold "$threshold" \
    --out "runs/vm_out/eval_threshold_${threshold}.json"
done
```

Each artifact must report `clip_count >= 50`, the checkpoint-matched 64-frame
window, and tolerances 1/2/5. These are public held-out labels only. Do not run
Outdoor/Indoor labels and do not expose the protected owner seed to training.

## 8. Pull, verify, and tear down

Hash every manifest, checkpoint, eval file, log, and probe artifact on the VM;
after pull, compare all hashes on the Mac before accepting the transfer. The
manager may then add a PENDING checkpoint entry in `models/MANIFEST.json`; this
code lane makes no best-stack change. Finally delete the VM, then prove both
the instance list and attached-disk list contain no lane resource. Record the
provider invoice/cost, total wall time, GPU utilization, artifact hashes, and
deletion evidence in the GPU-leg report.
