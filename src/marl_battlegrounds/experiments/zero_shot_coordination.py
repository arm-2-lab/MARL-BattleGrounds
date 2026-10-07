"""Test whether a team still plays well with a teammate it never trained with.

Use XLA_FLAGS=--xla_gpu_deterministic_ops=true python -m
marl_battlegrounds.experiments.zero_shot_coordination models_dir=/path/models
output_dir=/path/run.

A team trained by self-play learns habits that only its own teammates share,
for example who starts a fight or where the healer stands. This stage checks
whether those habits break when one teammate is a stranger:

1. Load each method once with marl_bgs.load_method.
2. Build the teams with marl_bgs.team. The host team (recurrent IPPO by
   default) plays alone as the control. Then, one class at a time, one of its
   five agents is replaced by the same class's agent from a newcomer method:
   every other Season 0 baseline, and Random as a floor.
3. Play every team against every opponent (the Season 0 field by default)
   with marl_bgs.evaluate and full metrics. All teams use the same seed, so
   game N of every team has the same map, spawn end and random draws as game N
   of the host team alone, and the analysis can compare them game by game.

The settings come from configs/zero_shot_coordination.yaml, which takes the
method paths from configs/field/season_0.yaml. Run the same command again to
continue: finished games are kept and skipped. Outputs in output_dir:
settings.json (the settings of the first launch; a later launch with other
settings is refused), and teams/<team>/<opponent>/ with each evaluate run,
indexed by teams/<team>/runs.json. This stage trains nothing.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Import the optional adapter first for its installation error and parser fix.
# isort: off
from marl_battlegrounds.experiments.hydra import prepare_hydra

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
# isort: on

if TYPE_CHECKING:
    from marl_battlegrounds import Policy, System

CLASSES = ("Mage", "Warrior", "Hunter", "Rogue", "Priest")


def folder_name(name: str) -> str:
    """Turn a display name into a folder name.

    Parameters
    ----------
    name : str
        Display name, for example ``"Recurrent IPPO with a QMIX Priest"``.

    Returns
    -------
    str
        Lowercase letters, digits and single hyphens, for example
        ``"recurrent-ippo-with-a-qmix-priest"``.
    """
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def team_plan(
    host: str, newcomers: Sequence[str], classes: Sequence[str]
) -> list[dict[str, str | None]]:
    """List the teams to play: the host alone, then each newcomer in each class.

    Parameters
    ----------
    host : str
        Display name of the host method, which controls every slot it keeps.
    newcomers : sequence of str
        Display names of the methods that lend one agent each, in table order.
        The host may not be one of them.
    classes : sequence of str
        Classes to replace, in table order, from ``CLASSES``.

    Returns
    -------
    list of dict
        One entry per team, the control first: ``name`` (for example
        ``"Recurrent IPPO with a QMIX Priest"``), ``newcomer`` and
        ``newcomer_class`` (both None for the control).

    Raises
    ------
    ValueError
        No newcomer or class is listed, a name repeats, the host is listed as a
        newcomer, or a class is not one of ``CLASSES``.
    """
    if not newcomers or not classes:
        raise ValueError("list at least one newcomer and one class")
    if len(set(newcomers)) != len(newcomers) or len(set(classes)) != len(classes):
        raise ValueError("newcomers and classes must not repeat")
    if host in newcomers:
        raise ValueError("the host cannot also be a newcomer")
    unknown = sorted(set(classes) - set(CLASSES))
    if unknown:
        raise ValueError(f"unknown classes {unknown}; use {list(CLASSES)}")
    teams: list[dict[str, str | None]] = [
        {"name": host, "newcomer": None, "newcomer_class": None}
    ]
    for newcomer in newcomers:
        for agent_class in classes:
            teams.append(
                {
                    "name": f"{host} with a {newcomer} {agent_class}",
                    "newcomer": newcomer,
                    "newcomer_class": agent_class,
                }
            )
    return teams


def build_team(
    host: System | Policy,
    newcomer: System | Policy | None,
    newcomer_class: str | None,
) -> System | Policy:
    """Build one team: the host alone, or the host with one newcomer agent.

    Parameters
    ----------
    host : System or Policy
        Loaded host method.
    newcomer : System, Policy or None
        Loaded newcomer method, or None for the host alone.
    newcomer_class : str or None
        The class the newcomer plays, from ``CLASSES``; None with no newcomer.

    Returns
    -------
    System or Policy
        ``host`` itself, or ``marl_bgs.team(host, newcomer, slots=...)`` where
        the newcomer controls the agent of ``newcomer_class`` and the host
        controls the other four. Each member keeps its own memory and sees only
        what its agents may see.
    """
    import marl_battlegrounds as marl_bgs

    if newcomer is None or newcomer_class is None:
        return host
    kept = [agent_class for agent_class in CLASSES if agent_class != newcomer_class]
    return marl_bgs.team(host, newcomer, slots=[kept, [newcomer_class]])


def play_team(
    team: System | Policy,
    opponents: Mapping[str, System | Policy],
    *,
    team_dir: Path,
    maps: Sequence[int],
    num_episodes: int,
    seed: int,
    num_envs: int,
    metrics: str,
) -> dict[str, Any]:
    """Play one team against every opponent, skipping opponents already played.

    Parameters
    ----------
    team : System or Policy
        The team, always Team A.
    opponents : mapping of str to System or Policy
        Display name to loaded opponent, always Team B, in play order.
    team_dir : pathlib.Path
        The team's folder. Each opponent's run goes in its own subfolder.
    maps : sequence of int
        Map IDs; paired games cycle through them.
    num_episodes : int
        Games per opponent, an even number: pairs of games share a seed and
        swap spawn ends.
    seed : int
        Root of the evaluator's random draws, the same for every team.
    num_envs : int
        Games played side by side. GPU runs use 32, 64, 128, 512 or 1024.
    metrics : {'priority', 'full', 'none'}
        Per-game metrics saved with each game.

    Returns
    -------
    dict
        The record also written to ``team_dir / "runs.json"``: ``runs``
        (opponent name to its exact run folder) and ``seconds`` (opponent name
        to the wall-clock seconds of its ``evaluate`` call, which loads,
        compiles and plays).

    Raises
    ------
    ValueError, TypeError, OSError
        Raised by the evaluator for invalid settings or a file error.
        Opponents finished before the error stay recorded in ``runs.json``.

    Notes
    -----
    ``runs.json`` is written after each opponent, so a stopped run continues
    with the first opponent it does not list. An opponent stopped part way
    is played again in a new folder; its unfinished folder is left as it is.
    """
    import marl_battlegrounds as marl_bgs

    index = team_dir / "runs.json"
    record: dict[str, Any] = (
        json.loads(index.read_text(encoding="utf-8"))
        if index.exists()
        else {"runs": {}, "seconds": {}}
    )
    for name, opponent in opponents.items():
        if name in record["runs"]:
            continue
        started = time.perf_counter()
        result = marl_bgs.evaluate(
            team,
            opponent,
            num_episodes=num_episodes,
            maps=list(maps),
            seed=seed,
            num_envs=num_envs,
            metrics=metrics,  # pyright: ignore[reportArgumentType] - the evaluator checks it.
            output_dir=team_dir / folder_name(name),
        )
        record["runs"][name] = str(result.run_dir)
        record["seconds"][name] = round(time.perf_counter() - started, 1)
        index.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"  {name}: {record['seconds'][name]} s")
    return record


@hydra.main(
    version_base="1.3", config_path="configs", config_name="zero_shot_coordination"
)
def main(config: DictConfig) -> dict[str, Any]:
    """Play every planned team against every opponent.

    Parameters
    ----------
    config : omegaconf.DictConfig
        ``systems`` (name and method of every method, from the field
        settings), ``host`` and ``newcomers`` and ``opponents`` (names from
        ``systems``), ``classes``, ``maps``, ``num_episodes`` (per opponent),
        ``seed``, ``num_envs``, ``metrics`` and the required ``output_dir``;
        ``models_dir`` fills ``${models_dir}`` in method paths. Relative paths
        use the folder the command was started from.

    Returns
    -------
    dict
        ``settings`` (as saved in ``settings.json``) and ``teams`` (team name
        to its ``play_team`` record).

    Raises
    ------
    ValueError
        A name is not in ``systems``, ``team_plan`` refuses the lists, the
        output folder was started with other settings, or a GPU run lacks
        XLA's deterministic mode.

    Notes
    -----
    Each method is loaded once and reused by every team and opponent. Every
    ``evaluate`` call compiles its own program for its pair of methods. For
    Season 0 on one RTX 5090 (128 games side by side, full metrics), a call
    of 200 games took 8.4 seconds on average (4.6 to 25.9), compiling
    included; all 279 calls took 39 minutes.
    """
    import jax

    import marl_battlegrounds as marl_bgs
    from marl_battlegrounds.evaluation.runtime_provenance import (
        declared_gpu_deterministic_ops,
        read_declared_xla_flags,
    )

    settings: dict[str, Any] = OmegaConf.to_container(config, resolve=True)  # pyright: ignore[reportAssignmentType] - the root is a mapping.
    methods = {
        str(entry["name"]): str(entry["method"]) for entry in settings["systems"]
    }
    host, newcomers = str(settings["host"]), [str(n) for n in settings["newcomers"]]
    opponent_names = [str(name) for name in settings["opponents"]]
    missing = sorted({host, *newcomers, *opponent_names} - set(methods))
    if missing:
        raise ValueError(f"not in systems: {missing}")
    teams = team_plan(host, newcomers, [str(c) for c in settings["classes"]])
    if jax.default_backend() != "cpu" and not declared_gpu_deterministic_ops(
        read_declared_xla_flags()
    ):
        raise ValueError(
            "set XLA_FLAGS=--xla_gpu_deterministic_ops=true before Python starts"
        )

    output_dir = Path(to_absolute_path(str(Path(settings["output_dir"]).expanduser())))
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = {key: value for key, value in settings.items() if key != "output_dir"}
    settings_path = output_dir / "settings.json"
    if settings_path.exists():
        if json.loads(settings_path.read_text(encoding="utf-8")) != saved:
            raise ValueError(
                f"{output_dir} was started with other settings; use a new output_dir"
            )
    else:
        settings_path.write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8")

    loaded = {
        name: marl_bgs.load_method(methods[name])
        for name in dict.fromkeys([host, *newcomers, *opponent_names])
    }
    opponents = {name: loaded[name] for name in opponent_names}
    records: dict[str, Any] = {}
    for entry in teams:
        name = str(entry["name"])
        print(name)
        newcomer = entry["newcomer"]
        records[name] = play_team(
            build_team(
                loaded[host],
                None if newcomer is None else loaded[newcomer],
                entry["newcomer_class"],
            ),
            opponents,
            team_dir=output_dir / "teams" / folder_name(name),
            maps=settings["maps"],
            num_episodes=settings["num_episodes"],
            seed=settings["seed"],
            num_envs=settings["num_envs"],
            metrics=settings["metrics"],
        )
    return {"settings": saved, "teams": records}


if __name__ == "__main__":
    prepare_hydra()
    main()
