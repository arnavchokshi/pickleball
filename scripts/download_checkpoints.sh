#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="${1:-"${ROOT_DIR}/models/checkpoints/tracknetv3"}"
TRACKNETV3_FILE_ID="1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA"
TRACKNET_SHA256="df867641a02712b021f04548ff4b1208ddfdb47f629ab2094ceb978667e83b1a"
INPAINTNET_SHA256="5749b66b8002f3ad9e0af841604004706fc796df30599e6bf01952696009688c"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

verify_file() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256_file "$path")"
  if [[ "$actual" != "$expected" ]]; then
    echo "sha256 mismatch for ${path}: expected ${expected}, got ${actual}" >&2
    return 1
  fi
}

install_from_dir() {
  local source_dir="$1"
  local tracknet
  local inpaintnet
  tracknet="$(find "$source_dir" -name TrackNet_best.pt -type f | head -n 1)"
  inpaintnet="$(find "$source_dir" -name InpaintNet_best.pt -type f | head -n 1)"
  if [[ -z "$tracknet" || -z "$inpaintnet" ]]; then
    echo "missing TrackNet_best.pt or InpaintNet_best.pt under ${source_dir}" >&2
    return 1
  fi

  mkdir -p "$DEST_DIR"
  cp "$tracknet" "${DEST_DIR}/TrackNet_best.pt"
  cp "$inpaintnet" "${DEST_DIR}/InpaintNet_best.pt"
}

if [[ -f "${DEST_DIR}/TrackNet_best.pt" && -f "${DEST_DIR}/InpaintNet_best.pt" ]]; then
  verify_file "${DEST_DIR}/TrackNet_best.pt" "$TRACKNET_SHA256"
  verify_file "${DEST_DIR}/InpaintNet_best.pt" "$INPAINTNET_SHA256"
  echo "TrackNetV3 checkpoints already present and verified in ${DEST_DIR}"
  exit 0
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

if [[ -n "${TRACKNETV3_CKPT_DIR:-}" ]]; then
  install_from_dir "$TRACKNETV3_CKPT_DIR"
elif [[ -n "${TRACKNETV3_CKPT_ZIP:-}" ]]; then
  unzip -q "$TRACKNETV3_CKPT_ZIP" -d "${tmp_dir}/ckpts"
  install_from_dir "${tmp_dir}/ckpts"
else
  zip_path="${tmp_dir}/TrackNetV3_ckpts.zip"
  if python -m gdown --help >/dev/null 2>&1; then
    python -m gdown --id "$TRACKNETV3_FILE_ID" -O "$zip_path"
  else
    curl -L "https://drive.google.com/uc?export=download&id=${TRACKNETV3_FILE_ID}" -o "$zip_path"
  fi
  unzip -q "$zip_path" -d "${tmp_dir}/ckpts"
  install_from_dir "${tmp_dir}/ckpts"
fi

verify_file "${DEST_DIR}/TrackNet_best.pt" "$TRACKNET_SHA256"
verify_file "${DEST_DIR}/InpaintNet_best.pt" "$INPAINTNET_SHA256"
echo "Installed and verified TrackNetV3 checkpoints in ${DEST_DIR}"
