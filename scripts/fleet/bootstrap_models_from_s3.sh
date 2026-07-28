#!/usr/bin/env bash
# Restore pipeline model weights from the durable S3 store.
#
# Primary fast-boot path for GPU VMs remains the GCP snapshot
# (pickleball-court23-warm-20260728); this script is the durable fallback and
# the laptop-side restore path. See scripts/fleet/MODEL_STORE.md for the
# inventory and sha256 table.
#
# Usage:
#   scripts/fleet/bootstrap_models_from_s3.sh [--dest <dir>] [--prefix <s3-prefix>]
#
# Requires AWS credentials able to read s3://sway-videos/pickleball-models/.
# On credential-less VMs, generate presigned GET URLs on a credentialed host
# (aws s3 presign s3://sway-videos/pickleball-models/<ver>/<file>) and fetch
# with curl -o instead; verify the sha256 table from MODEL_STORE.md either way.
set -euo pipefail

DEST="models/checkpoints"
PREFIX="s3://sway-videos/pickleball-models/20260728"

while (($#)); do
  case "$1" in
    --dest) DEST="${2:?}"; shift 2 ;;
    --prefix) PREFIX="${2:?}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# file|relative dest|sha256
TABLE=$(cat <<'EOF'
yolo26m.pt|yolo26m.pt|401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7
yolo26n-cls.pt|yolo26n-cls.pt|0dd6f8dbc448870ac98a3cbb7156f923f7ce21fed3755d4019169ffffd279e81
osnet_x1_0_market1501.pt|osnet_x1_0_market1501.pt|2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154
court_model_v2.pt|court_unet_v2/court_model_v2.pt|31da51630b82a85ee39384e65eb705b045adcdda900dd025ca15784a2edd3ffe
resnet34-b627a593.pth|court_external/torchvision/resnet34-b627a593.pth|b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f
model_tennis_court_det.pt|court_external/TennisCourtDetector/weights/model_tennis_court_det.pt|09aa8c4338459ba1d643f2dc329f45f464dedec3720fccc1a4abfd1f7b464d04
wasb_tennis_best.pth.tar|wasb/wasb_tennis_best.pth.tar|9d391239ab10c733f8e5bfadf16ab72838e7a8ebc88e8ae2038501c03d42b4bb
sam-3d-body-dinov3_model.ckpt|body_runtime/sam-3d-body-dinov3/model.ckpt|b5a2f9d305dd02626b967aa2e86021fba07065df66ce7a7e00ffb9664f150abf
mhr_model.pt|body_runtime/sam-3d-body-dinov3/assets/mhr_model.pt|352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc
EOF
)

sha_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}';
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}

while IFS='|' read -r src rel want; do
  [[ -z "$src" ]] && continue
  out="$DEST/$rel"
  if [[ -f "$out" ]] && [[ "$(sha_of "$out")" == "$want" ]]; then
    echo "OK (cached) $rel"
    continue
  fi
  mkdir -p "$(dirname "$out")"
  echo "FETCH $src -> $out"
  aws s3 cp "$PREFIX/$src" "$out"
  got="$(sha_of "$out")"
  if [[ "$got" != "$want" ]]; then
    echo "SHA MISMATCH for $rel: got $got want $want" >&2
    exit 1
  fi
  echo "OK (fetched) $rel"
done <<<"$TABLE"

echo "bootstrap_models_from_s3: all models present and sha-verified in $DEST"
