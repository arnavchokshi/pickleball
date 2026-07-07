#!/bin/bash
# w4_freshproof teardown: md5 cross-check remote BODY outputs vs the locally pulled copies,
# then DELETE all 3 fan VMs and list-confirm zero w4-freshproof VMs remain.
set -uo pipefail
REPO=/Users/arnavchokshi/Desktop/pickleball
LANE="$REPO/runs/lanes/w4_freshproof_20260707"
KH="$REPO/configs/ssh/a100_known_hosts"
SA=pickleball-fleet@gifted-electron-498923-h1.iam.gserviceaccount.com
SSH="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=yes -o UserKnownHostsFile=$KH"

md5_crosscheck() { # $1=ip $2=remote_clip_dir_glob $3=local_clip_dir
  ip=$1; rglob=$2; ldir=$3
  echo "--- md5 crosscheck $ip $rglob"
  $SSH arnavchokshi@$ip "d=\$(ls -dt $rglob 2>/dev/null | head -1); [ -n \"\$d\" ] && cd \$d && find . -maxdepth 1 -name '*.json' -newer /tmp -type f 2>/dev/null; cd \$d && md5sum *.json 2>/dev/null" > /tmp/w4_remote_md5.txt || { echo "remote md5 FAILED"; return 1; }
  fails=0; checked=0
  while read -r md5 name; do
    name=${name#./}
    [ -f "$ldir/$name" ] || continue
    lmd5=$(md5 -q "$ldir/$name" 2>/dev/null)
    checked=$((checked+1))
    if [ "$lmd5" != "$md5" ]; then echo "MISMATCH $name remote=$md5 local=$lmd5"; fails=$((fails+1)); fi
  done < /tmp/w4_remote_md5.txt
  echo "checked=$checked mismatches=$fails"
}

echo "=== teardown start $(date -u +%FT%TZ) ===" | tee -a "$LANE/logs/teardown.log"
# VMs + their clips (fan3 hosted wolverine AND img1605)
md5_crosscheck 34.126.94.44 '~/coldstart_20260706/repo/runs/process_video_body_dispatch/outdoor_*_2026*' "$LANE/outdoor/outdoor_webcam_iynbd_1500_long_high_baseline/outdoor_webcam_iynbd_1500_long_high_baseline" 2>&1 | tee -a "$LANE/logs/teardown.log"
md5_crosscheck 34.87.20.4 '~/coldstart_20260706/repo/runs/process_video_body_dispatch/burlington_*_2026*' "$LANE/burlington/burlington_gold_0300_low_steep_corner/burlington_gold_0300_low_steep_corner" 2>&1 | tee -a "$LANE/logs/teardown.log"
md5_crosscheck 34.21.239.35 '~/coldstart_20260706/repo/runs/process_video_body_dispatch/wolverine_*_2026*' "$LANE/wolverine/wolverine_mixed_0200_mid_steep_corner/wolverine_mixed_0200_mid_steep_corner" 2>&1 | tee -a "$LANE/logs/teardown.log"
md5_crosscheck 34.21.239.35 '~/coldstart_20260706/repo/runs/process_video_body_dispatch/owner_IMG_1605_*_2026*' "$LANE/img1605/owner_IMG_1605_8a193402780b/owner_IMG_1605_8a193402780b" 2>&1 | tee -a "$LANE/logs/teardown.log"

echo "=== deleting fan VMs $(date -u +%FT%TZ) ===" | tee -a "$LANE/logs/teardown.log"
gcloud compute instances delete pickleball-a100-w4fan1 pickleball-a100-w4fan2 pickleball-a100-w4fan3 \
  --zone=asia-southeast1-a --quiet --impersonate-service-account=$SA 2>&1 | tee -a "$LANE/logs/teardown.log"
echo "=== list-confirm $(date -u +%FT%TZ) ===" | tee -a "$LANE/logs/teardown.log"
gcloud compute instances list --filter=labels.fable-fleet=pickleball --impersonate-service-account=$SA 2>&1 | tee -a "$LANE/logs/teardown.log"
echo "=== teardown done $(date -u +%FT%TZ) ===" | tee -a "$LANE/logs/teardown.log"
