#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${AUTOTUNEAI_ENV_NAME:-autotuneai-benchmark}"
ENV_FILE="${AUTOTUNEAI_ENV_FILE:-environment-benchmark.yml}"
CONDA_BIN="${CONDA_EXE:-}"
INSTALL_MINIFORGE=0
MINIFORGE_PREFIX="${AUTOTUNEAI_MINIFORGE_PREFIX:-$HOME/miniforge3}"
UPDATE_ENV=0
RUN_TESTS=1
PYTORCH_INDEX_URL="${AUTOTUNEAI_PYTORCH_INDEX_URL:-}"

usage() {
  cat <<'EOF'
Usage: bash scripts/bootstrap_server_env.sh [options]

Create or repair the PerfRunbench benchmark environment on a Linux/WSL/server host.
The script does not require `conda activate`; it uses `conda run`.

Options:
  --env-name NAME          Conda env name. Default: autotuneai-benchmark
  --env-file PATH          Environment file. Default: environment-benchmark.yml
  --conda PATH             Explicit conda or mamba executable.
  --install-miniforge      Install Miniforge under --miniforge-prefix if conda is missing.
  --miniforge-prefix PATH  Miniforge install prefix. Default: $HOME/miniforge3
  --update                 Update an existing env from --env-file.
  --skip-tests             Skip scripts/run_tests.py --fast.
  --pytorch-index-url URL  Reinstall torch/torchvision from a specific PyTorch wheel index.
  -h, --help               Show this help.

Examples:
  bash scripts/bootstrap_server_env.sh
  bash scripts/bootstrap_server_env.sh --install-miniforge
  bash scripts/bootstrap_server_env.sh --pytorch-index-url https://download.pytorch.org/whl/cu121
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="$2"
      shift 2
      ;;
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --conda)
      CONDA_BIN="$2"
      shift 2
      ;;
    --install-miniforge)
      INSTALL_MINIFORGE=1
      shift
      ;;
    --miniforge-prefix)
      MINIFORGE_PREFIX="$2"
      shift 2
      ;;
    --update)
      UPDATE_ENV=1
      shift
      ;;
    --skip-tests)
      RUN_TESTS=0
      shift
      ;;
    --pytorch-index-url)
      PYTORCH_INDEX_URL="$2"
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

if [[ ! -f "$ENV_FILE" || ! -f "pyproject.toml" ]]; then
  echo "Run this script from the repository root. Missing $ENV_FILE or pyproject.toml." >&2
  exit 2
fi

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

install_miniforge() {
  local os arch url installer
  os="$(uname -s)"
  arch="$(uname -m)"
  if [[ "$os" != "Linux" ]]; then
    echo "--install-miniforge currently supports Linux/WSL only. Install conda manually or pass --conda." >&2
    exit 2
  fi
  case "$arch" in
    x86_64|aarch64) ;;
    *)
      echo "Unsupported architecture for automatic Miniforge install: $arch" >&2
      exit 2
      ;;
  esac
  if [[ -x "$MINIFORGE_PREFIX/bin/conda" ]]; then
    return 0
  fi
  if [[ -e "$MINIFORGE_PREFIX" ]]; then
    echo "Miniforge prefix already exists but conda was not found: $MINIFORGE_PREFIX" >&2
    exit 2
  fi
  url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${arch}.sh"
  installer="$(mktemp)"
  echo "Downloading Miniforge: $url"
  if command -v curl >/dev/null 2>&1; then
    curl -L "$url" -o "$installer"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$installer" "$url"
  else
    echo "curl or wget is required to install Miniforge automatically." >&2
    exit 2
  fi
  bash "$installer" -b -p "$MINIFORGE_PREFIX"
  rm -f "$installer"
}

if ! CONDA_BIN="$(find_conda)"; then
  if [[ "$INSTALL_MINIFORGE" -eq 1 ]]; then
    install_miniforge
    CONDA_BIN="$(find_conda)"
  else
    cat >&2 <<EOF
No conda/mamba executable found.
Install Miniforge manually, activate conda in the shell, pass --conda PATH, or rerun:
  bash scripts/bootstrap_server_env.sh --install-miniforge
EOF
    exit 2
  fi
fi

echo "Using conda executable: $CONDA_BIN"
echo "Environment name: $ENV_NAME"
echo "Environment file: $ENV_FILE"

if "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda env exists: $ENV_NAME"
  if [[ "$UPDATE_ENV" -eq 1 ]]; then
    "$CONDA_BIN" env update -n "$ENV_NAME" -f "$ENV_FILE"
  fi
else
  "$CONDA_BIN" env create -n "$ENV_NAME" -f "$ENV_FILE"
fi

if [[ -n "$PYTORCH_INDEX_URL" ]]; then
  "$CONDA_BIN" run -n "$ENV_NAME" python -m pip install --upgrade --index-url "$PYTORCH_INDEX_URL" torch torchvision
fi

"$CONDA_BIN" run -n "$ENV_NAME" python -m pip install -e .

"$CONDA_BIN" run -n "$ENV_NAME" python -c 'import google.protobuf, torch; print("protobuf", google.protobuf.__version__); print("torch", torch.__version__); print("cuda", torch.cuda.is_available()); print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'
"$CONDA_BIN" run -n "$ENV_NAME" autotuneai --help >/dev/null

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
else
  echo "nvidia-smi was not found on PATH; GPU tuning/pressure tests may not be available."
fi

"$CONDA_BIN" run -n "$ENV_NAME" autotuneai executors --probe-systemd --check-sudo-cache || true

if [[ "$RUN_TESTS" -eq 1 ]]; then
  "$CONDA_BIN" run -n "$ENV_NAME" python scripts/run_tests.py --fast
fi

CONDA_BASE="$("$CONDA_BIN" info --base 2>/dev/null || true)"
if [[ -n "$CONDA_BASE" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  ACTIVATE_SETUP="source \"$CONDA_BASE/etc/profile.d/conda.sh\""
else
  ACTIVATE_SETUP="# initialize conda for your shell first if needed"
fi

cat <<EOF

Server environment is ready.

Use without activating the shell:
  $CONDA_BIN run -n $ENV_NAME autotuneai inspect

Or activate manually:
  $ACTIVATE_SETUP
  conda activate $ENV_NAME

Suggested GPU performance smoke:
  sudo -v
  autotuneai optimize-performance --target gpu --executor systemd --sudo --gpu-tuning-sudo --monitor-mode minimal --time-budget-hours 0.2 --max-candidates 4 --repeat 2 --warmup-runs 1 --cooldown-seconds 2 -- python examples/gpu_training_pressure.py --config examples/gpu_training_pressure_sweep_config.yaml
EOF
