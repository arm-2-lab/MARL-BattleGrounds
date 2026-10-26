#!/bin/sh
# Check the published install routes exactly as a user runs them, so every
# "Tested" claim in the README and the install guide has a real run behind it.
# The release workflow (.github/workflows/release.yml) calls this script from
# the repository root. Every check stops at the first failure (exit status 1)
# and prints what it was doing. It needs curl; the pip check also needs uv.
#
#   check_install.sh pip PYTHON [constraints]
#       In a fresh virtual environment with Python PYTHON (3.12, 3.13 or 3.14),
#       run the README's pip line, pip install "marl-battlegrounds[all]", with
#       the wheel in dist/ standing in for PyPI. Then run the quick start and
#       evaluate a released model by name, with no token. With "constraints",
#       repeat it in a second environment with constraints/constraints-cpu.txt.
#   check_install.sh start
#       Run the one command, sh start.sh --no-browser --port 8765
#       --replay-port 8767, in this folder, with a new computer's PATH: it
#       installs everything and opens the BattleClient. Check that the page
#       answers and that a localhost address is refused, then stop it. When
#       start.sh installed uv (always on CI, where uv is absent), check that
#       uv's installer added uv to the shell startup files a new terminal
#       reads. Then evaluate a released model and run every notebook under
#       examples/ from top to bottom on the CPU.
#   check_install.sh start-readonly-config
#       Run the one command again as on a computer whose ~/.config this user
#       cannot write. First remove uv (~/.local/bin/uv and uvx) and the
#       ~/.config/fish and ~/.config/uv folders an earlier install made, then
#       make ~/.config read-only (writable again on exit). uv's installer
#       then puts uv in place and fails creating ~/.config/fish; start.sh
#       must say so, carry on and open the BattleClient on ports 8775 and
#       8777, so a BattleClient left from "start" cannot answer for it.
#   check_install.sh oldest IMAGE
#       Unpack this commit as the release tarball holds it, then run the install
#       guide's sh start.sh --help and its evaluate command inside the Linux
#       container IMAGE (an old distribution that has curl).
#   check_install.sh container ENGINE IMAGE
#       With ENGINE (docker or podman), run IMAGE as the README does: the
#       BattleClient must answer and refuse localhost; then evaluate the
#       baked-in Season 0 model offline, and check the results belong to the
#       owner of the mounted folder.
#   check_install.sh apptainer IMAGE
#       Turn the Docker image IMAGE into an Apptainer file and run the install
#       guide's evaluate command with apptainer exec, on the CPU. As the guide
#       says, Apptainer's cache and temporary files go to roomy storage (beside
#       the Apptainer file), because /tmp may be too small for the image.
#
# Files go under $RUNNER_TEMP (a new temporary folder when it is not set).
set -eu

work="${RUNNER_TEMP:-$(mktemp -d)}"
released="MARL-BattleGrounds/tdm-season-0-recurrent-ippo@final"

# Run the quick start and one evaluation by name with the Python in $1;
# results go to the folder $2 under $work.
quick_start() {
  JAX_PLATFORMS=cpu "$1" - <<'PY'
import jax
import marl_battlegrounds as marl_bgs

env = marl_bgs.make("tdm", map_id=0, num_envs=2)
observation, state = env.reset(jax.random.key(0))
actions = env.sample_actions(jax.random.key(1), state)
observation, state, reward, done, info = env.step(jax.random.key(2), state, actions)
print("quick start ok")
PY
  (cd "$work" && JAX_PLATFORMS=cpu "$1" -m marl_battlegrounds evaluate \
    --system "$released" --opponent tdm-alpha \
    --episodes 2 --maps 47 --num-envs 2 --output-dir "$2")
}

# Wait for the BattleClient link in the output that command $1 prints, while
# the check $2 (its process, or its container, is still running) passes. Then
# check the page on port $3 (default 8765).
wait_for_link() {
  port="${3:-8765}"
  until $1 2>&1 | grep -q '#token='; do
    if ! eval "$2"; then
      $1 2>&1 || true
      echo "check_install.sh: the BattleClient stopped before it printed its link." >&2
      exit 1
    fi
    sleep 2
  done
  test "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/")" = 200 ||
    { echo "check_install.sh: the BattleClient page did not answer." >&2; exit 1; }
  test "$(curl -s -o /dev/null -w '%{http_code}' -H "Host: localhost:$port" "http://127.0.0.1:$port/")" = 403 ||
    { echo "check_install.sh: the BattleClient did not refuse a localhost address." >&2; exit 1; }
  echo "BattleClient ok"
}

pip_route() {
  wheel="$(ls "$PWD"/dist/*.whl)"
  env="$work/pip-$1"
  uv venv --seed --python "$1" "$env"
  "$env/bin/pip" install "${wheel}[all]"
  quick_start "$env/bin/python" "eval-pip-$1"
  if [ "${2:-}" = constraints ]; then
    env="$work/pip-$1-constraints"
    uv venv --seed --python "$1" "$env"
    "$env/bin/pip" install "${wheel}[all]" -c "$PWD/constraints/constraints-cpu.txt"
    quick_start "$env/bin/python" "eval-pip-$1-constraints"
  fi
}

start_route() {
  log="$work/start.log"
  # A new computer's PATH. GitHub's runners already have uv's folder,
  # ~/.local/bin, on PATH, and uv's installer then leaves the shell startup
  # files alone, which would hide whether a new terminal can find uv.
  env PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin \
    sh start.sh --no-browser --port 8765 --replay-port 8767 >"$log" 2>&1 &
  pid=$!
  wait_for_link "cat $log" "kill -0 $pid 2>/dev/null"
  kill "$pid" 2>/dev/null || true
  if grep -q "Installing uv" "$log"; then
    # The files a new terminal reads: zsh on a Mac, sh and bash on Linux.
    case "$(uname -s)" in
      Darwin) startup=".zshrc .zshenv" ;;
      *) startup=".profile .bashrc .bash_profile .bash_login" ;;
    esac
    found=no
    for file in $startup; do
      if grep -qs 'local/bin/env' "$HOME/$file"; then found=yes; fi
    done
    test "$found" = yes ||
      { echo "check_install.sh: uv's installer added uv to none of $startup, so a new terminal would not find uv." >&2; exit 1; }
    echo "new terminal ok: uv is in the shell startup files"
  elif [ "${CI:-}" = true ]; then
    echo "check_install.sh: start.sh did not install uv on this runner, so the new terminal check could not run." >&2
    exit 1
  fi
  PATH="$HOME/.local/bin:$PATH"
  export PATH
  JAX_PLATFORMS=cpu uv run --no-sync python -m marl_battlegrounds evaluate \
    --system "$released" --opponent tdm-alpha \
    --episodes 2 --maps 47 --num-envs 2 --output-dir "$work/start-evaluation"
  find examples -name '*.ipynb' | sort | while read -r notebook; do
    echo "Running $notebook"
    JAX_PLATFORMS=cpu uv run --no-sync jupyter nbconvert --to notebook --execute \
      --output-dir "$work/notebooks" "$notebook"
  done
}

readonly_config_route() {
  log="$work/start-readonly-config.log"
  # Undo the earlier install, so start.sh installs uv again and the installer
  # must create ~/.config/fish (a folder already there would hide the failure).
  rm -f "$HOME/.local/bin/uv" "$HOME/.local/bin/uvx"
  rm -rf "$HOME/.config/fish" "$HOME/.config/uv"
  mkdir -p "$HOME/.config"
  chmod a-w "$HOME/.config"
  trap 'chmod u+w "$HOME/.config"' EXIT
  env PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin \
    sh start.sh --no-browser --port 8775 --replay-port 8777 >"$log" 2>&1 &
  pid=$!
  wait_for_link "cat $log" "kill -0 $pid 2>/dev/null" 8775
  kill "$pid" 2>/dev/null || true
  grep -q "Installing uv" "$log" ||
    { echo "check_install.sh: start.sh did not install uv again." >&2; exit 1; }
  grep -q '\.config/fish' "$log" ||
    { echo "check_install.sh: uv's installer never met the read-only ~/.config, so this run tested nothing." >&2; exit 1; }
  grep -q "so start.sh carries on" "$log" ||
    { echo "check_install.sh: start.sh did not say the installer stopped late." >&2; exit 1; }
  echo "read-only ~/.config ok: uv installed, and start.sh carried on to the BattleClient"
}

oldest_route() {
  version="$(sed -n 's/^version = "\(.*\)"$/\1/p' pyproject.toml | head -n 1)"
  folder="MARL-BattleGrounds-$version"
  place="$work/oldest-$(echo "$1" | tr -c 'A-Za-z0-9\n' '-')"
  mkdir -p "$place"
  git archive --prefix="$folder/" HEAD | tar -x -C "$place"
  docker run --rm -v "$place/$folder:/root/$folder" -w "/root/$folder" "$1" sh -ec '
    sh start.sh --help
    "$HOME/.local/bin/uv" run --no-sync python -m marl_battlegrounds evaluate \
      --system random --opponent tdm-alpha --episodes 32 --output-dir runs/check'
  test -d "$place/$folder/runs/check"
  echo "oldest ok: $1"
}

container_route() {
  engine="$1"
  mount="$work/$engine-work"
  mkdir -p "$mount"
  "$engine" run -d --name play -p 127.0.0.1:8765:8766 -p 127.0.0.1:8767:8768 \
    -v "$mount:/work" "$2"
  wait_for_link "$engine logs play" "[ -n \"\$($engine ps -q -f name=play)\" ]"
  "$engine" stop play
  "$engine" rm play
  "$engine" run --rm -e HF_HUB_OFFLINE=1 -v "$mount:/work" "$2" evaluate \
    --system MARL-BattleGrounds/tdm-season-0-recurrent-mappo --opponent tdm-alpha \
    --episodes 2 --maps 47 --num-envs 2 --output-dir runs/check
  test "$(stat -c %u "$mount/runs/check")" = "$(id -u)" ||
    { echo "check_install.sh: the results do not belong to the folder's owner." >&2; exit 1; }
  echo "$engine ok: $2"
}

apptainer_route() {
  place="$work/apptainer"
  export APPTAINER_CACHEDIR="$place/cache" APPTAINER_TMPDIR="$place/tmp"
  mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
  cd "$place"
  apptainer build marl-bgs.sif "docker-daemon://$1"
  apptainer exec marl-bgs.sif python -m marl_battlegrounds evaluate \
    --system random --opponent tdm-alpha --episodes 32 --output-dir runs/check
  test -d runs/check
  echo "apptainer ok"
}

case "${1:-}" in
  pip) pip_route "$2" "${3:-}" ;;
  start) start_route ;;
  start-readonly-config) readonly_config_route ;;
  oldest) oldest_route "$2" ;;
  container) container_route "$2" "$3" ;;
  apptainer) apptainer_route "$2" ;;
  *)
    echo "usage: check_install.sh pip PYTHON [constraints] | start | start-readonly-config | oldest IMAGE | container ENGINE IMAGE | apptainer IMAGE" >&2
    exit 2
    ;;
esac
