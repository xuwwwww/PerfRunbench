#!/usr/bin/env bash
set -uo pipefail

ENV_NAME="${AUTOTUNEAI_ENV_NAME:-autotuneai-benchmark}"
CONDA_BIN="${CONDA_EXE:-}"
MINIFORGE_PREFIX="${AUTOTUNEAI_MINIFORGE_PREFIX:-$HOME/miniforge3}"

usage() {
  cat <<'EOF'
Usage: bash scripts/diagnose_server_env.sh [options]

Run read-only diagnostics for a Linux/WSL/server AutoTuneAI benchmark environment.
This script does not create, update, or delete conda environments.

Options:
  --env-name NAME          Conda env name. Default: autotuneai-benchmark
  --conda PATH             Explicit conda or mamba executable.
  -h, --help               Show this help.

Examples:
  bash scripts/diagnose_server_env.sh
  bash scripts/diagnose_server_env.sh --env-name autotuneai-benchmark
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="$2"
      shift 2
      ;;
    --conda)
      CONDA_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

find_conda() {
  if [[ -n "$CONDA_BIN" && -x "$CONDA_BIN" ]]; then
    printf '%s\n' "$CONDA_BIN"
    return 0
  fi
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return 0
  fi
  if command -v mamba >/dev/null 2>&1; then
    command -v mamba
    return 0
  fi
  if [[ -x "$MINIFORGE_PREFIX/bin/conda" ]]; then
    printf '%s\n' "$MINIFORGE_PREFIX/bin/conda"
    return 0
  fi
  return 1
}

run_step() {
  local label="$1"
  shift
  echo
  echo "== $label =="
  "$@"
  local code=$?
  if [[ "$code" -eq 0 ]]; then
    echo "PASS: $label"
  else
    echo "FAIL: $label (exit $code)"
  fi
  return "$code"
}

failures=0

echo "AutoTuneAI server environment diagnostics"
echo "cwd: $(pwd)"
echo "uname: $(uname -a 2>/dev/null || true)"

if [[ ! -f "pyproject.toml" ]]; then
  echo "FAIL: pyproject.toml not found. Run from the repository root." >&2
  exit 2
fi

if ! CONDA_BIN="$(find_conda)"; then
  echo "FAIL: no conda/mamba executable found."
  echo "Try: bash scripts/bootstrap_server_env.sh --install-miniforge"
  exit 2
fi

echo "conda: $CONDA_BIN"
echo "env: $ENV_NAME"

run_step "conda sees environment" "$CONDA_BIN" run -n "$ENV_NAME" python --version || failures=$((failures + 1))

run_step "editable package and CLI import" "$CONDA_BIN" run -n "$ENV_NAME" python - <<'PY' || failures=$((failures + 1))
import autotune
from autotune.cli import main
print("autotune package:", getattr(autotune, "__file__", "unknown"))
print("cli main:", callable(main))
PY

run_step "torch/protobuf imports" "$CONDA_BIN" run -n "$ENV_NAME" python - <<'PY' || failures=$((failures + 1))
import google.protobuf
import torch
print("protobuf:", google.protobuf.__version__)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY

run_step "autotuneai CLI" "$CONDA_BIN" run -n "$ENV_NAME" autotuneai --help >/dev/null || failures=$((failures + 1))

if command -v nvidia-smi >/dev/null 2>&1; then
  run_step "nvidia-smi visible" nvidia-smi -L || failures=$((failures + 1))
else
  echo
  echo "FAIL: nvidia-smi not found on PATH."
  failures=$((failures + 1))
fi

run_step "executor capability probe" "$CONDA_BIN" run -n "$ENV_NAME" autotuneai executors --probe-systemd --check-sudo-cache || true

echo
if [[ "$failures" -eq 0 ]]; then
  echo "Diagnostics completed: no blocking failures detected."
  exit 0
fi

echo "Diagnostics completed with $failures blocking failure(s)."
echo "Common fixes:"
echo "  bash scripts/bootstrap_server_env.sh --update"
echo "  bash scripts/bootstrap_server_env.sh --pytorch-index-url https://download.pytorch.org/whl/cu121"
echo "  $CONDA_BIN run -n $ENV_NAME python -m pip install -e ."
exit 1
