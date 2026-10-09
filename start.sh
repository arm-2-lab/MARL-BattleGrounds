#!/bin/sh
# Install MARL-BattleGrounds and open the BattleClient with one command.
#
# Usage, from the repository folder (Linux, an Apple Silicon Mac, or WSL2 on
# Windows):
#
#   sh start.sh [play.py options]
#
# What it does:
#   1. Installs uv, the Python package manager (https://docs.astral.sh/uv/),
#      with its official installer when uv is missing. The installer adds
#      ~/.local/bin to your shell's startup files, so `uv` works in new
#      terminals.
#   2. Picks the JAX build for this machine (see "The build" below).
#   3. Runs `uv sync --locked --extra all --group jupyter [--extra BUILD]`.
#      The first time, uv downloads Python 3.14 and the exact locked packages
#      (up to 3.5 GB with a GPU build), including JupyterLab for the example
#      notebooks. This sync is exact: it removes any other extra or build
#      installed in this folder before, so rerun start.sh, or use `uv run`,
#      never a plain `uv sync`.
#   4. Runs `uv run --no-sync python play.py [options]`: the BattleClient opens
#      in your browser. By default you play Team A against the trained
#      Recurrent MAPPO team on map 47. See `sh start.sh --help` for options.
#
# The build. MARL_BGS_ACCELERATOR=cpu, cuda13, cuda12, rocm or tpu picks it
# yourself; rocm and tpu (Linux x86_64 only, not tested yet) are only ever
# chosen this way. Otherwise start.sh reads nvidia-smi (on PATH, or
# /usr/lib/wsl/lib/nvidia-smi inside WSL2) and uses the lowest driver version
# and compute capability across your NVIDIA GPUs:
#   driver 580 or newer and compute capability 7.5 or higher -> cuda13
#   driver 525 or newer and compute capability 5.2 or higher -> cuda12
#   anything else, or no NVIDIA GPU                          -> cpu
#
# Option handled here, not passed to play.py (anywhere on the line):
#   --dry-run   Print the chosen build and the two commands, run nothing,
#               and do not install uv.
# Every other option goes to play.py, for example:
#   sh start.sh --team-b qmix --map 12
#   sh start.sh --port 8765 --replay-port 8767 --no-browser
#                                   (on a remote machine; see docs/install.md)
#
# Exit status: 0 for --dry-run; otherwise play.py's status. It stops with an
# error message when MARL_BGS_ACCELERATOR has another value, when uv cannot
# be installed, or when the sync fails.
set -eu

cd "$(dirname "$0")"

# Take --dry-run out of the options, wherever it is; keep the rest in order.
dry_run=0
for argument in "$@"; do
  shift
  if [ "$argument" = "--dry-run" ]; then
    dry_run=1
  else
    set -- "$@" "$argument"
  fi
done

# Print "BUILD REASON" for this machine.
pick_build() {
  if [ -n "${MARL_BGS_ACCELERATOR-}" ]; then
    case "$MARL_BGS_ACCELERATOR" in
      cpu | cuda13 | cuda12 | rocm | tpu)
        echo "$MARL_BGS_ACCELERATOR set by MARL_BGS_ACCELERATOR"
        return 0
        ;;
      *)
        echo "MARL_BGS_ACCELERATOR must be cpu, cuda13, cuda12, rocm or tpu, not '$MARL_BGS_ACCELERATOR'." >&2
        return 1
        ;;
    esac
  fi
  smi=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    smi=nvidia-smi
  elif [ -x /usr/lib/wsl/lib/nvidia-smi ]; then
    smi=/usr/lib/wsl/lib/nvidia-smi
  fi
  if [ -z "$smi" ]; then
    echo "cpu no NVIDIA GPU found"
    return 0
  fi
  gpus=$("$smi" --query-gpu=driver_version,compute_cap --format=csv,noheader 2>/dev/null || true)
  echo "$gpus" | awk -F', *' '
    NF >= 2 && $1 ~ /^[0-9]/ && $2 ~ /^[0-9]/ {
      split($1, d, "."); driver = d[1] + 0
      split($2, c, "."); cap = c[1] * 10 + c[2]
      if (n == 0 || driver < low_driver) low_driver = driver
      if (n == 0 || cap < low_cap) low_cap = cap
      n++
    }
    END {
      if (n == 0) { print "cpu nvidia-smi found no usable GPU"; exit }
      why = sprintf("NVIDIA driver %d, compute capability %d.%d", low_driver, int(low_cap / 10), low_cap % 10)
      if (low_driver >= 580 && low_cap >= 75) print "cuda13 " why
      else if (low_driver >= 525 && low_cap >= 52) print "cuda12 " why
      else print "cpu " why " is too old for JAX GPU builds"
    }'
}

choice=$(pick_build)
build=${choice%% *}
reason=${choice#* }
extras="--extra all --group jupyter"
if [ "$build" != "cpu" ]; then
  extras="$extras --extra $build"
fi

echo "MARL-BattleGrounds: using the $build build ($reason)."
if [ "$dry_run" = 1 ]; then
  echo "uv sync --locked $extras"
  echo "uv run --no-sync python play.py $*"
  exit 0
fi

if ! command -v uv >/dev/null 2>&1; then
  PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv, the Python package manager (https://docs.astral.sh/uv/) ..."
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  else
    echo "Please install curl or wget, or install uv yourself: https://docs.astral.sh/uv/" >&2
    exit 1
  fi
  PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  if ! command -v uv >/dev/null 2>&1; then
    echo "Could not install uv. Check your internet connection, or install uv yourself: https://docs.astral.sh/uv/" >&2
    exit 1
  fi
fi

# shellcheck disable=SC2086  # $extras is a list of flags.
if ! uv sync --locked $extras; then
  echo "The install failed. Intel Macs and Linux older than glibc 2.27 need Docker (see docs/install.md). If uv is old, update it the way you installed it, for example 'uv self update'." >&2
  exit 1
fi
exec uv run --no-sync python play.py "$@"
