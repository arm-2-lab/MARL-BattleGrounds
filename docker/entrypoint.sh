#!/bin/sh
# Start a command inside the MARL-BattleGrounds image (docker/Dockerfile).
#
# Commands (the first argument):
#   play [play.py options]    The default. Opens the BattleClient: you against
#                             the trained Recurrent MAPPO team on map 47.
#                             Options alone (--team-b qmix) also mean play.
#   replay FILE [options]     Opens a saved replay in the Replay Viewer.
#   evaluate, train, tournament, models, ... (any marl_battlegrounds command)
#                             Runs `python -m marl_battlegrounds COMMAND ...`.
#   anything else             Runs as given, for example python, bash or
#                             jupyter lab --ip 0.0.0.0 --no-browser.
#
# The browser clients listen only on 127.0.0.1 inside the container and check
# that the browser asks for 127.0.0.1 with the same port. So for play and
# replay, inside Docker or Podman, this script starts socat relays:
#   0.0.0.0:8766 -> 127.0.0.1:$MARL_BGS_PORT        (the page; default 8765)
#   0.0.0.0:8768 -> 127.0.0.1:$MARL_BGS_REPLAY_PORT (Open Replay; default 8767)
# Publish them with -p 127.0.0.1:8765:8766 -p 127.0.0.1:8767:8768. The host
# ports must equal MARL_BGS_PORT and MARL_BGS_REPLAY_PORT. Apptainer shares the
# host's network, so no relay starts there (APPTAINER_CONTAINER or
# SINGULARITY_CONTAINER set).
#
# Run as root with a host folder mounted at /work, it continues as that
# folder's owner, so saved replays and results stay readable on the host.
# HOME is then /tmp/home and USER is "me". Docker's stop sends Ctrl-C
# (STOPSIGNAL in the Dockerfile), so a game being played saves its replay.
#
# Exit status: the command's own. 2 when MARL_BGS_PORT or MARL_BGS_REPLAY_PORT
# is not a whole number from 1 to 65535, when the four ports are not all
# different, or when play gets --port or --replay-port (use the two settings).
set -eu

port=${MARL_BGS_PORT:-8765}
replay_port=${MARL_BGS_REPLAY_PORT:-8767}
for value in "$port" "$replay_port"; do
  case "$value" in
    '' | *[!0-9]*)
      echo "MARL_BGS_PORT and MARL_BGS_REPLAY_PORT must be whole numbers, not '$value'." >&2
      exit 2
      ;;
  esac
  if [ "$value" -lt 1 ] || [ "$value" -gt 65535 ]; then
    echo "MARL_BGS_PORT and MARL_BGS_REPLAY_PORT must be from 1 to 65535, not $value." >&2
    exit 2
  fi
done
if [ "$(printf '%s\n' "$port" 8766 "$replay_port" 8768 | sort -u | wc -l)" -ne 4 ]; then
  echo "MARL_BGS_PORT ($port), MARL_BGS_REPLAY_PORT ($replay_port) and the relay ports 8766 and 8768 must all differ." >&2
  exit 2
fi

# Continue as the owner of a mounted /work folder instead of root.
if [ "$(id -u)" = 0 ] && [ -d /work ]; then
  owner=$(stat -c %u /work)
  group=$(stat -c %g /work)
  if [ "$owner" != 0 ]; then
    mkdir -p /tmp/home
    chown "$owner:$group" /tmp/home
    export HOME=/tmp/home
    # The host user has no name in the image; Python's getpass needs one.
    # A two-letter name stays out of the anonymous-export name check.
    export USER=me LOGNAME=me
    exec setpriv --reuid="$owner" --regid="$group" --clear-groups "$0" "$@"
  fi
fi
if [ ! -w "${HOME:-/}" ]; then
  mkdir -p /tmp/home
  export HOME=/tmp/home
fi

relay() {
  if { [ -e /.dockerenv ] || [ -e /run/.containerenv ]; } \
    && [ -z "${APPTAINER_CONTAINER-}" ] && [ -z "${SINGULARITY_CONTAINER-}" ]; then
    # Quiet: a browser that connects before the page is ready is not an error.
    socat "TCP-LISTEN:$1,fork,reuseaddr" "TCP:127.0.0.1:$2" 2>/dev/null &
  fi
}

command=${1:-play}
case "$command" in
  -*) command=play ;; # Options alone mean play.
  *) if [ "$#" -gt 0 ]; then shift; fi ;;
esac
case "$command" in
  play)
    for argument in "$@"; do
      case "$argument" in
        --port | --port=* | --replay-port | --replay-port=*)
          echo "In Docker, set the ports with -e MARL_BGS_PORT and -e MARL_BGS_REPLAY_PORT, not $argument." >&2
          exit 2
          ;;
      esac
    done
    relay 8766 "$port"
    relay 8768 "$replay_port"
    exec python /opt/marl-bgs/play.py --port "$port" --replay-port "$replay_port" --no-browser "$@"
    ;;
  replay)
    relay 8766 "$port"
    echo "Open the http://127.0.0.1 link printed below in your browser (use 127.0.0.1, not localhost)."
    exec python -m marl_battlegrounds replay "$@" --port "$port" --no-open
    ;;
esac
if command -v "$command" >/dev/null 2>&1; then
  exec "$command" "$@"
fi
exec python -m marl_battlegrounds "$command" "$@"
