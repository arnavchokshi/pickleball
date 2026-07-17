#!/usr/bin/env bash
# trk_detbench_20260716 VM bootstrap — run ON THE VM as the first command batch.
# Usage: bash vm_bootstrap.sh <mac_head_sha> <bundle_path_on_vm> <payload_tar_gz_on_vm>
# Emits PROOF lines the Mac-side lane log captures. Idempotent where possible.
set -uo pipefail
MAC_HEAD_SHA="${1:?mac head sha}"
BUNDLE="${2:?bundle path}"
PAYLOAD="${3:?payload tar.gz path}"

echo "=== RAIL: scheduling sudo shutdown -P +210 (hard teardown rail) ==="
sudo shutdown -P +210 2>&1 | tee /tmp/shutdown_rail_proof.txt
echo "RAIL_PROOF_BEGIN"
cat /tmp/shutdown_rail_proof.txt
# systemd writes the scheduled time to /run/systemd/shutdown/scheduled
if [ -f /run/systemd/shutdown/scheduled ]; then
  echo "-- /run/systemd/shutdown/scheduled --"
  cat /run/systemd/shutdown/scheduled
fi
echo "RAIL_PROOF_END"

echo "=== RAIL: 60-min no-heartbeat self-stop watcher ==="
touch /tmp/lane_heartbeat
sudo bash -c 'cat > /usr/local/bin/heartbeat_watch.sh <<"EOF"
#!/usr/bin/env bash
while true; do
  sleep 300
  if [ ! -f /tmp/lane_heartbeat ]; then continue; fi
  age=$(( $(date +%s) - $(stat -c %Y /tmp/lane_heartbeat) ))
  if [ "$age" -gt 3600 ]; then
    echo "no heartbeat for ${age}s — self-stopping" | systemd-cat -t heartbeat_watch
    shutdown -P now
  fi
done
EOF
chmod +x /usr/local/bin/heartbeat_watch.sh'
sudo systemd-run --unit=lane-heartbeat-watch /usr/local/bin/heartbeat_watch.sh 2>&1 || \
  (nohup sudo /usr/local/bin/heartbeat_watch.sh >/tmp/heartbeat_watch.log 2>&1 & echo "fallback nohup watcher started")
echo "HEARTBEAT_WATCHER_ARMED"

echo "=== GPU + compute mode (standing policy: DEFAULT for single self-dispatch lane) ==="
sudo nvidia-smi -c DEFAULT || true
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "=== repo discovery ==="
REPO=""
for cand in /home/arnavchokshi/pickleball_git /home/arnavchokshi/pickleball /opt/pickleball; do
  if [ -d "$cand/.git" ]; then REPO="$cand"; break; fi
done
if [ -z "$REPO" ]; then
  echo "REPO_NOT_FOUND: searching..."
  REPO=$(find /home -maxdepth 3 -name ".git" -type d 2>/dev/null | head -1 | xargs -r dirname)
fi
echo "REPO=$REPO"
[ -z "$REPO" ] && { echo "FATAL: no repo on snapshot"; exit 2; }
cd "$REPO"

echo "=== boot ritual: reset --hard if dirty beyond the 2 vendor-submodule lines ==="
git status --short | head -20
DIRTY_NONVENDOR=$(git status --short | grep -cv "third_party/WASB-SBDT\|third_party/blurball" || true)
if [ "$DIRTY_NONVENDOR" -gt 0 ]; then
  echo "dirty beyond vendor pins -> git reset --hard"
  git reset --hard
fi

echo "=== code identity: fetch bundle + checkout pinned Mac HEAD ==="
echo "bundle sha256 (VM side):"; sha256sum "$BUNDLE"
git bundle verify "$BUNDLE" 2>&1 | tail -2
git fetch "$BUNDLE" "HEAD:refs/heads/detbench_pin" 2>&1 | tail -1 || git fetch "$BUNDLE" 2>&1 | tail -1
git checkout --detach "$MAC_HEAD_SHA"
echo "VM HEAD now: $(git rev-parse HEAD)"

echo "=== boot ritual: fresh ssh-keyscan SELF entry after checkout ==="
SELF_IP=$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip || echo "")
mkdir -p configs/ssh
ssh-keyscan -T 5 localhost 2>/dev/null >> configs/ssh/a100_known_hosts || true
[ -n "$SELF_IP" ] && ssh-keyscan -T 5 "$SELF_IP" 2>/dev/null >> configs/ssh/a100_known_hosts || true
echo "keyscan appended (self_ip=$SELF_IP)"

echo "=== unpack payload tar ==="
echo "payload md5 (VM side):"; md5sum "$PAYLOAD"
tar -xzf "$PAYLOAD" -C "$REPO"
echo "payload unpacked into repo tree (relative paths preserved)"

echo "=== PREFLIGHT: byte-verify pinned files (MUST match Mac md5s) ==="
md5sum \
  scripts/racketsport/benchmark_person_trackers.py \
  scripts/racketsport/score_person_track_sources.py \
  threed/racketsport/person_track_gt_scoring.py \
  threed/racketsport/raw_pool_person_authority.py \
  threed/racketsport/player_global_association.py
echo "expected:"
echo "07deba04bc00f9eaff9670676ac3ec45  scripts/racketsport/benchmark_person_trackers.py"
echo "cd7ae4891c482a257807761f3b934a90  scripts/racketsport/score_person_track_sources.py"
echo "be38f76547d05d8ac7b12274de5b659d  threed/racketsport/person_track_gt_scoring.py"
echo "ea30bfdf3a57bf7e2fff06476ec6295c  threed/racketsport/raw_pool_person_authority.py"
echo "5e761c5db3327a1841fc0e54281bb9d7  threed/racketsport/player_global_association.py"

echo "=== checkpoint sha256s (VM side) ==="
sha256sum models/checkpoints/yolo26m.pt models/checkpoints/osnet_x1_0_market1501.pt 2>&1

echo "=== venv discovery + torchreid gap fill ==="
VENV=""
for v in "$REPO/.venv" /home/arnavchokshi/pickleball_git/.venv; do
  [ -x "$v/bin/python3" ] && VENV="$v" && break
done
echo "VENV=$VENV"
if [ -n "$VENV" ]; then
  "$VENV/bin/python3" -c "import ultralytics, torch; print('ultralytics', ultralytics.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())" 2>&1
  "$VENV/bin/python3" -c "import torchreid; print('torchreid OK', torchreid.__version__)" 2>&1 || {
    echo "installing torchreid (known snapshot gap)"
    "$VENV/bin/pip" install torchreid 2>&1 | tail -2
  }
fi

echo "BOOTSTRAP_COMPLETE repo=$REPO venv=$VENV head=$(git rev-parse HEAD)"
