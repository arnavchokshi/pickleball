#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_ROOT="${ROOT_DIR}/models/checkpoints"
LEGACY_TRACKNET_DEST=""
VERIFY_ONLY=0
TRACKNETV3_FILE_ID="1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA"
TRACKNET_SHA256="df867641a02712b021f04548ff4b1208ddfdb47f629ab2094ceb978667e83b1a"
INPAINTNET_SHA256="5749b66b8002f3ad9e0af841604004706fc796df30599e6bf01952696009688c"
YOLO26N_SHA256="9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
YOLO26M_SHA256="401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7"

usage() {
  cat <<'USAGE'
Usage: scripts/download_checkpoints.sh [--verify-only] [--dest-root PATH] [TRACKNET_DEST_DIR]

Downloads or installs local, non-H100 checkpoints used by the racketsport gates.

Options:
  --verify-only      Only verify existing files by sha256; do not download or copy.
  --dest-root PATH   Checkpoint root directory. Default: models/checkpoints.
  -h, --help         Show this help text.

Environment overrides:
  TRACKNETV3_CKPT_DIR   Directory containing TrackNet_best.pt and InpaintNet_best.pt.
  TRACKNETV3_CKPT_ZIP   Zip file containing TrackNetV3 checkpoint files.
  YOLO26_CKPT_DIR       Directory containing yolo26n.pt and yolo26m.pt.

The optional TRACKNET_DEST_DIR positional argument is kept for the old script
interface and only changes where TrackNetV3 files are installed.
USAGE
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --verify-only)
      VERIFY_ONLY=1
      shift
      ;;
    --dest-root)
      if [[ "$#" -lt 2 ]]; then
        echo "missing value for --dest-root" >&2
        usage >&2
        exit 64
      fi
      DEST_ROOT="$2"
      shift 2
      ;;
    --dest-root=*)
      DEST_ROOT="${1#--dest-root=}"
      shift
      ;;
    --*)
      echo "unknown option: $1" >&2
      usage >&2
      exit 64
      ;;
    *)
      if [[ -n "$LEGACY_TRACKNET_DEST" ]]; then
        echo "unexpected argument: $1" >&2
        usage >&2
        exit 64
      fi
      LEGACY_TRACKNET_DEST="$1"
      shift
      ;;
  esac
done

TRACKNET_DEST_DIR="${LEGACY_TRACKNET_DEST:-"${DEST_ROOT}/tracknetv3"}"
YOLO_DEST_DIR="$DEST_ROOT"

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
  if [[ ! -f "$path" ]]; then
    echo "missing checkpoint: ${path}" >&2
    return 1
  fi
  actual="$(sha256_file "$path")"
  if [[ "$actual" != "$expected" ]]; then
    echo "sha256 mismatch for ${path}: expected ${expected}, got ${actual}" >&2
    return 1
  fi
}

verify_tracknetv3() {
  verify_file "${TRACKNET_DEST_DIR}/TrackNet_best.pt" "$TRACKNET_SHA256"
  verify_file "${TRACKNET_DEST_DIR}/InpaintNet_best.pt" "$INPAINTNET_SHA256"
  echo "TrackNetV3 checkpoints verified in ${TRACKNET_DEST_DIR}"
}

verify_yolo26() {
  verify_file "${YOLO_DEST_DIR}/yolo26n.pt" "$YOLO26N_SHA256"
  verify_file "${YOLO_DEST_DIR}/yolo26m.pt" "$YOLO26M_SHA256"
  echo "YOLO detector checkpoints verified in ${YOLO_DEST_DIR}"
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

  mkdir -p "$TRACKNET_DEST_DIR"
  cp "$tracknet" "${TRACKNET_DEST_DIR}/TrackNet_best.pt"
  cp "$inpaintnet" "${TRACKNET_DEST_DIR}/InpaintNet_best.pt"
}

install_yolo_from_dir() {
  local source_dir="$1"
  local yolo26n
  local yolo26m
  yolo26n="$(find "$source_dir" -name yolo26n.pt -type f | head -n 1)"
  yolo26m="$(find "$source_dir" -name yolo26m.pt -type f | head -n 1)"
  if [[ -z "$yolo26n" || -z "$yolo26m" ]]; then
    echo "missing yolo26n.pt or yolo26m.pt under ${source_dir}" >&2
    return 1
  fi

  mkdir -p "$YOLO_DEST_DIR"
  cp "$yolo26n" "${YOLO_DEST_DIR}/yolo26n.pt"
  cp "$yolo26m" "${YOLO_DEST_DIR}/yolo26m.pt"
}

if [[ "$VERIFY_ONLY" -eq 1 ]]; then
  verify_tracknetv3
  verify_yolo26
  exit 0
fi

if [[ -f "${TRACKNET_DEST_DIR}/TrackNet_best.pt" && -f "${TRACKNET_DEST_DIR}/InpaintNet_best.pt" ]]; then
  verify_tracknetv3
else
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

  verify_tracknetv3
fi

if [[ -f "${YOLO_DEST_DIR}/yolo26n.pt" && -f "${YOLO_DEST_DIR}/yolo26m.pt" ]]; then
  verify_yolo26
elif [[ -n "${YOLO26_CKPT_DIR:-}" ]]; then
  install_yolo_from_dir "$YOLO26_CKPT_DIR"
  verify_yolo26
else
  echo "missing YOLO detector checkpoints in ${YOLO_DEST_DIR}" >&2
  echo "Provide YOLO26_CKPT_DIR or place yolo26n.pt and yolo26m.pt there." >&2
  exit 1
fi
