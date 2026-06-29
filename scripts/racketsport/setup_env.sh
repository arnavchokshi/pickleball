#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/racketsport/setup_env.sh

Creates the local Python virtual environment and expected Phase 0 directories.
EOF
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
esac
if [ "$#" -gt 0 ]; then
  usage >&2
  exit 64
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-racketsport.txt

mkdir -p models/checkpoints runs/phase0 data/testclips
echo "local Phase 0 environment ready: $ROOT/.venv"
