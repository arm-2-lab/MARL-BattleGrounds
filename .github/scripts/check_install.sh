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
#       --replay-port 8767, in this folder: it installs everything and opens
#       the BattleClient. Check that the page answers and that a localhost
#       address is refused, then stop it. Then evaluate a released model and run
#       every notebook under examples/ from top to bottom on the CPU.
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
#       guide's evaluate command with apptainer exec, on the CPU.
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
# process $2 (or, for a container, the name check in $3) is still running.
wait_for_link() {
  until $1 2>&1 | grep -q '#token='; do
    if ! eval "$2"; then
      $1 2>&1 || true
      echo "check_install.sh: the BattleClient stopped before it printed its link." >&2
      exit 1
    fi
    sleep 2
  done
  test "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8765/)" = 200
  test "$(curl -s -o /dev/null -w '%{http_code}' -H 'Host: localhost:8765' http://127.0.0.1:8765/)" = 403
  echo "BattleClient ok"
}

pip_route() {
  wheel="$(ls "$PWD"/dist/*.whl)"
  env="$work/pip-$1"
  uv venv --seed --python "$1" "$env"
  "$env/bin/pip" install "${wheel}[all]"
  quick_start "$env/bin/python" "pip-$1"
  if [ "${2:-}" = constraints ]; then
    env="$work/pip-$1-constraints"
    uv venv --seed --python "$1" "$env"
    "$env/bin/pip" install "${wheel}[all]" -c "$PWD/constraints/constraints-cpu.txt"
    quick_start "$env/bin/python" "pip-$1-constraints"
  fi
}

start_route() {
  log="$work/start.log"
  sh start.sh --no-browser --port 8765 --replay-port 8767 >"$log" 2>&1 &
  pid=$!
  wait_for_link "cat $log" "kill -0 $pid 2>/dev/null"
  kill "$pid" 2>/dev/null || true
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

oldest_route() {
  version="$(sed -n 's/^version = "\(.*\)"$/\1/p' pyproject.toml | head -n 1)"
  folder="MARL-BattleGrounds-$version"
  place="$work/oldest-$(echo "$1" | tr -c 'A-Za-z0-9\n' '-')"
  mkdir -p "$place"
  git archive --prefix="$folder/" HEAD | tar -x -C "$place"
  docker run --rm -v "$place/$folder:/root/$folder" -w "/root/$folder" "$1" sh -c '
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
  test "$(stat -c %u "$mount/runs/check")" = "$(id -u)"
  echo "$engine ok: $2"
}

apptainer_route() {
  place="$work/apptainer"
  mkdir -p "$place"
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
  oldest) oldest_route "$2" ;;
  container) container_route "$2" "$3" ;;
  apptainer) apptainer_route "$2" ;;
  *)
    echo "usage: check_install.sh pip PYTHON [constraints] | start | oldest IMAGE | container ENGINE IMAGE | apptainer IMAGE" >&2
    exit 2
    ;;
esac
