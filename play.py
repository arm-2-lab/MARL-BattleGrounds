"""Play MARL-BattleGrounds in your browser, or watch two teams play.

Run it from the repository folder::

    sh start.sh [options]                     # installs everything, then runs this
    uv run --no-sync python play.py [options] # after start.sh has run once

By default you control Team A (Mage, Warrior, Hunter, Rogue, Priest) against
the trained Recurrent MAPPO team on map 47. A trained team's model downloads
from the MARL-BattleGrounds organization on Hugging Face the first time you
use it and is kept in your Hugging Face cache. Press ``?`` in the page for How
To Play. Press Exit in the page, or Ctrl-C here, to stop. Each game's replay
is saved under ``recordings/`` in the folder you run from.

Examples::

    sh start.sh                               # you against Recurrent MAPPO
    sh start.sh --team-b qmix --map 12        # another opponent and map
    sh start.sh --team-a ippo --team-b mappo  # watch two trained teams
    sh start.sh --port 8765 --replay-port 8767 --no-browser  # remote (SSH tunnel)

Team names: ``manual`` (you); the six trained Season 0 teams ``mappo``,
``ippo``, ``ff-mappo``, ``ff-ippo``, ``qmix`` and ``pqn-vdn``; and the built-in
``tdm-alpha``, ``tdm-beta``, ``tdm-gamma`` and ``random``. Any other value goes
to ``marl_battlegrounds.battle_client`` unchanged, so a saved model folder, a
released model name or a trusted ``module:function`` works too.
"""

from __future__ import annotations

import argparse
import os
import sys

RELEASED_PREFIX = "MARL-BattleGrounds/tdm-season-0-"
TRAINED_TEAMS = {
    "mappo": "recurrent-mappo",
    "ippo": "recurrent-ippo",
    "ff-mappo": "feedforward-mappo",
    "ff-ippo": "feedforward-ippo",
    "qmix": "qmix",
    "pqn-vdn": "pqn-vdn",
}


def team(name: str) -> str:
    """Turn a team name from the command line into what the BattleClient takes.

    Parameters
    ----------
    name : str
        A trained team's short name (``mappo``, ``ippo``, ``ff-mappo``,
        ``ff-ippo``, ``qmix`` or ``pqn-vdn``), or anything
        ``marl_battlegrounds.battle_client`` accepts: ``manual``, a built-in
        name, a saved model folder, a released model name or a trusted
        ``module:function`` factory.

    Returns
    -------
    str
        The released Hugging Face name for a short name, for example
        ``"MARL-BattleGrounds/tdm-season-0-recurrent-mappo"`` for ``"mappo"``;
        any other value unchanged. The BattleClient checks it and explains a
        value it cannot use.
    """
    if name in TRAINED_TEAMS:
        return RELEASED_PREFIX + TRAINED_TEAMS[name]
    return name


def parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    """Read the command line.

    Parameters
    ----------
    argv : list of str or None
        Arguments without the program name; None reads ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        ``team_a`` (default ``"manual"``), ``team_b`` (default ``"mappo"``),
        ``map`` (default 47), ``port`` and ``replay_port`` (default 0, a free
        port) and ``no_browser`` (default False).

    Raises
    ------
    SystemExit
        For ``--help`` (status 0) or an option argparse cannot read (status 2).
    """
    parser = argparse.ArgumentParser(
        prog="play.py",
        description=(
            "Play MARL-BattleGrounds in your browser, or watch two teams play. "
            "Team names: manual (you); the trained teams "
            + ", ".join(TRAINED_TEAMS)
            + "; the built-in tdm-alpha, tdm-beta, tdm-gamma and random; or "
            "anything marl_battlegrounds.battle_client accepts."
        ),
        epilog=(
            "start.sh also takes --dry-run (show the install, run nothing), and "
            "MARL_BGS_ACCELERATOR=cpu, cuda13, cuda12, rocm or tpu picks the JAX "
            "build. See docs/install.md."
        ),
    )
    parser.add_argument(
        "--team-a", default="manual", help="Team A (default: manual, you)"
    )
    parser.add_argument(
        "--team-b", default="mappo", help="Team B (default: mappo, Recurrent MAPPO)"
    )
    parser.add_argument(
        "--map", type=int, default=47, help="Map 0 to 51 (default: 47, a test map)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Port of the game page on 127.0.0.1 (default: 0, any free port)",
    )
    parser.add_argument(
        "--replay-port",
        type=int,
        default=0,
        help=(
            "Port of the Replay Viewer that Open Replay starts "
            "(default: 0, any free port)"
        ),
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Print the page's address instead of opening a browser",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Open the BattleClient until Exit or Ctrl-C.

    Parameters
    ----------
    argv : list of str or None, default=None
        Arguments without the program name; None reads ``sys.argv``. See
        ``parse_arguments`` for the options.

    Returns
    -------
    int
        The BattleClient's exit status; 2 when it refuses a setting (an unknown
        team, a map outside 0 to 51, two equal ports), and 1 when the system
        refuses (for example a port already in use). Then a plain message is
        printed instead of a traceback, with the team names for a setting.

    Notes
    -----
    Sets ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` unless it is already set, so
    JAX takes GPU memory as it needs it instead of most of the card at start.
    Other errors reach the terminal unchanged, with their traceback.
    """
    arguments = parse_arguments(argv)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import marl_battlegrounds as marl_bgs

    if arguments.no_browser:
        print(
            "Open the http://127.0.0.1 link printed below in your browser "
            "(use 127.0.0.1, not localhost).",
            flush=True,
        )
    try:
        return marl_bgs.battle_client(
            map_id=arguments.map,
            team_a=team(arguments.team_a),
            team_b=team(arguments.team_b),
            port=arguments.port,
            replay_port=arguments.replay_port,
            open_browser=not arguments.no_browser,
        )
    except ValueError as error:
        print(f"play.py: {error}", file=sys.stderr)
        print(
            "Teams: manual, " + ", ".join(TRAINED_TEAMS) + ", tdm-alpha, tdm-beta, "
            "tdm-gamma, random, or a saved model folder. Maps: 0 to 51.",
            file=sys.stderr,
        )
        return 2
    except OSError as error:
        print(f"play.py: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
