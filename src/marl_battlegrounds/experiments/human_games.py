"""Play recorded games against, and alongside, trained Systems in the BattleClient.

Use python -m marl_battlegrounds.experiments.human_games output_dir=/path/series
(add field=season_0_local models_dir=/path/models for local actor folders
instead of the released models). A person plays Team A in the BattleClient
(marl_bgs.battle_client), in Agent POV; the opponent System plays Team B. The
series is a list of parts, played in order, each with its own settings:

- ``kind='against'``: the person controls all five Team A agents.
- ``kind='alongside'``: the person controls one class (``player_class``) and the
  partner System controls the other four Team A agents.

A part plays one game per entry in its ``maps``. Team A starts on the left in
the part's odd-numbered games and on the right in its even-numbered games
(``spawn_mode``), and the part's game N uses the seed ``seed + N - 1``. Games
are numbered through the whole series. Before each game the stage waits for
Enter and then opens the BattleClient; play the game to its end and press Exit.
The stage checks the saved replay against the schedule (check_replay), measures
it with marl_bgs.replay_metrics, which gives the same full metrics evaluate
saves, and writes game.json and full_metrics.csv beside the replay, then
results.csv and results.md for the series.

Every game counts: the first replay in a game's folder that was played to its
end is the game, and a later replay never replaces it. Run the same command
again to continue a series: recorded games are skipped. settings.json keeps the
first launch's settings, and a launch with other settings is refused. The
settings come from configs/human_games.yaml, which names Systems from
configs/field/season_0.yaml.
"""

from __future__ import annotations

import csv
import io
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
    from marl_battlegrounds.evaluation.replay_io import LoadedReplay

KINDS = ("against", "alongside")
# Catalog columns copied into each game's record, from Team A's side.
SUMMARY_COLUMNS = (
    "team_a_score",
    "team_b_score",
    "score_difference",
    "team_a_kills",
    "team_a_deaths",
    "episode_length",
)


@dataclass(frozen=True, slots=True)
class Game:
    """One scheduled game of a series.

    Attributes
    ----------
    number : int
        Position in the whole series, from 1.
    kind : {'against', 'alongside'}
        ``'against'``: the person controls all five Team A agents.
        ``'alongside'``: the person controls ``player_class`` and the partner
        the other four Team A agents.
    map_id : int
        Packaged map.
    swapped : bool
        False: Team A starts on the left (``spawn_mode='default'``). True:
        Team A starts on the right (``spawn_mode='swapped'``).
    seed : int
        The game's own seed, from 0 through 2**32 - 1.
    opponent : str
        Name of the System that plays Team B.
    partner : str or None
        Name of the System beside the person in alongside games; None in
        against games.
    player_class : str or None
        The class the person plays in alongside games, for example
        ``'mage'``; None in against games.
    """

    number: int
    kind: str
    map_id: int
    swapped: bool
    seed: int
    opponent: str
    partner: str | None = None
    player_class: str | None = None

    @property
    def side(self) -> str:
        """Return where Team A starts: ``'left'`` or ``'right'``."""
        return "right" if self.swapped else "left"

    @property
    def folder(self) -> str:
        """Return the game's folder name.

        For example ``'game-04-alongside-map-47-left'``.
        """
        return f"game-{self.number:02d}-{self.kind}-map-{self.map_id}-{self.side}"


def schedule(parts: Sequence[Mapping[str, Any]]) -> list[Game]:
    """Put every part's games in playing order.

    Parameters
    ----------
    parts : sequence of mapping
        The parts in playing order. Each has ``kind`` (``'against'`` or
        ``'alongside'``), ``opponent`` (a System name), ``maps`` (one map ID
        per game, repeats allowed) and ``seed`` (the part's first seed);
        alongside parts also have ``partner`` and ``player_class``.

    Returns
    -------
    list of Game
        Numbered from 1 through the whole series. Within a part, game N plays
        ``maps[N - 1]`` with seed ``seed + N - 1``, Team A on the left when N
        is odd.

    Raises
    ------
    ValueError
        No part or no map is given, a kind is unknown, an alongside part lacks
        its partner or class, or a seed falls outside 0 through 2**32 - 1.

    Examples
    --------
    >>> games = schedule([{"kind": "against", "opponent": "A", "maps": [47, 48],
    ...                    "seed": 7}])
    >>> [(g.number, g.map_id, g.side, g.seed) for g in games]
    [(1, 47, 'left', 7), (2, 48, 'right', 8)]
    """
    if not parts:
        raise ValueError("parts must list at least one part")
    games: list[Game] = []
    for part in parts:
        kind = str(part["kind"])
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {list(KINDS)}")
        if not part["maps"]:
            raise ValueError("every part must list at least one map")
        alongside = kind == "alongside"
        if alongside and not (part.get("partner") and part.get("player_class")):
            raise ValueError("an alongside part needs a partner and a player_class")
        for index, map_id in enumerate(part["maps"], start=1):
            seed = int(part["seed"]) + index - 1
            if not 0 <= seed < 2**32:
                raise ValueError(
                    f"Seed {seed} is outside 0 through 2**32 - 1; choose another seed."
                )
            games.append(
                Game(
                    number=len(games) + 1,
                    kind=kind,
                    map_id=int(map_id),
                    swapped=index % 2 == 0,
                    seed=seed,
                    opponent=str(part["opponent"]),
                    partner=str(part["partner"]) if alongside else None,
                    player_class=str(part["player_class"]) if alongside else None,
                )
            )
    return games


def person_slots(game: Game) -> tuple[int, ...]:
    """Return the Team A slots the person controls in a game.

    Parameters
    ----------
    game : Game
        The scheduled game.

    Returns
    -------
    tuple of int
        ``(0, 1, 2, 3, 4)`` in against games, or the canonical roster slot of
        the game's ``player_class`` in alongside games, for example ``(0,)``
        for the Mage.

    Raises
    ------
    ValueError
        The class is not in the canonical roster: ``'mage'``, ``'warrior'``,
        ``'hunter'``, ``'rogue'`` or ``'priest'``.
    """
    import marl_battlegrounds as marl_bgs

    if game.kind == "against":
        return tuple(range(5))
    roster = marl_bgs.canonical_tournament_rosters()[0]
    if game.player_class not in roster:
        raise ValueError(f"player_class must be one of {list(roster)}")
    return (roster.index(game.player_class),)


def team_a_controllers(
    game: Game, partner: Policy | System | None
) -> str | list[str | Policy | System]:
    """Return Team A's controllers for ``marl_bgs.battle_client``.

    Parameters
    ----------
    game : Game
        The scheduled game.
    partner : Policy, System or None
        The loaded partner System for alongside games; None for against games.

    Returns
    -------
    str or list
        ``'manual'`` (all five slots) in against games, or one controller per
        roster slot in alongside games: ``'manual'`` at the person's slot and
        ``partner`` elsewhere.

    Raises
    ------
    ValueError
        The class is unknown (``person_slots``), or an alongside game has no
        partner.
    """
    slots = person_slots(game)
    if game.kind == "against":
        return "manual"
    if partner is None:
        raise ValueError(f"Game {game.number} needs a partner System")
    return ["manual" if slot in slots else partner for slot in range(5)]


def read_attempts(folder: Path) -> list[tuple[Path, LoadedReplay]]:
    """Load every replay saved in a game's folder, oldest first.

    The BattleClient saves a replay when a game ends, and also when a person
    saves or exits a game early. Each file is one attempt at the game.

    Parameters
    ----------
    folder : pathlib.Path
        The game's folder. A missing folder has no attempts.

    Returns
    -------
    list of (pathlib.Path, LoadedReplay)
        Each ``*.marlbg-replay.json`` file and its loaded content, ordered by
        file time. Each file is read once.

    Raises
    ------
    marl_battlegrounds.evaluation.replay_io.ReplayLoadError
        A saved file is not a valid replay.
    """
    from marl_battlegrounds.evaluation import load_replay

    paths = sorted(folder.glob("*.marlbg-replay.json"), key=lambda p: p.stat().st_mtime)
    return [(path, load_replay(path)) for path in paths]


def counted_attempt(
    attempts: Sequence[tuple[Path, LoadedReplay]],
) -> tuple[Path, LoadedReplay] | None:
    """Return the attempt that counts: the first one played to its end.

    Parameters
    ----------
    attempts : sequence of (pathlib.Path, LoadedReplay)
        From ``read_attempts``.

    Returns
    -------
    (pathlib.Path, LoadedReplay) or None
        None when no attempt was played to its end.
    """
    for path, loaded in attempts:
        if loaded.replay.completion.completion_state == "complete":
            return path, loaded
    return None


def check_replay(game: Game, loaded: LoadedReplay) -> None:
    """Refuse a replay that is not this game, played as scheduled.

    Parameters
    ----------
    game : Game
        The scheduled game.
    loaded : LoadedReplay
        Its counted replay.

    Raises
    ------
    ValueError
        The game did not end, or the replay records another map, other spawn
        ends, another seed than ``game.seed``, or a person ("Manual") in other
        slots than ``person_slots(game)``.
    """
    from marl_battlegrounds.evaluation._recording.map_identity import recorded_map
    from marl_battlegrounds.evaluation._replays.controller_names import (
        recorded_slot_controller_names,
    )

    replay = loaded.replay
    context = replay.header.context
    keys = {row.name: row.value for row in context.aggregation_keys}
    names = recorded_slot_controller_names(context) or (None,) * 10
    manual = tuple(slot for slot, name in enumerate(names) if name == "Manual")
    slots = person_slots(game)
    problems = {
        "was not played to its end": replay.completion.completion_state != "complete",
        f"is not map {game.map_id}": recorded_map(context).map_id != game.map_id,
        f"does not start Team A on the {game.side}": keys.get(
            "marl_bgs.spawn_locations"
        )
        != str(int(game.swapped)),
        f"did not use seed {game.seed}": context.seed_protocol.root_seed != game.seed,
        f"does not have the person in slots {slots}": manual != slots,
    }
    failed = [problem for problem, found in problems.items() if found]
    if failed:
        raise ValueError(f"Game {game.number}'s replay " + "; ".join(failed) + ".")


def record_game(
    game: Game,
    attempts: Sequence[tuple[Path, LoadedReplay]],
    *,
    played: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Measure a game's counted attempt and save its full metrics and record.

    Writes ``full_metrics.csv`` (one row: the game and its replay, then every
    full metric column; an empty cell is a metric that does not apply) and
    then ``game.json`` (the returned record) in the replay's folder.
    ``game.json`` marks the game as done.

    Parameters
    ----------
    game : Game
        The scheduled game. Call ``check_replay`` on its counted attempt first.
    attempts : sequence of (pathlib.Path, LoadedReplay)
        Every replay of the game, from ``read_attempts``; the first finished
        one counts.
    played : (float, float) or None, default=None
        Wall-clock start and end of the BattleClient launch, in Unix seconds,
        or None when unknown (a game measured on a later run).

    Returns
    -------
    dict
        ``number``, ``kind``, ``map_id``, ``side``, ``replay`` (file name),
        ``replay_digest``; the game's ``seed``, ``opponent``, ``partner`` and
        ``player_class``; ``controllers`` and ``classes`` (one entry per global
        slot, as recorded; ``"Manual"`` marks the person); ``attempts`` (every
        replay file and its completion state, oldest first); ``wall_seconds``
        (how long the BattleClient was open, or None); ``outcome``
        (``'win'``, ``'draw'`` or ``'loss'``, from Team A's side); and the
        ``SUMMARY_COLUMNS`` values.

    Raises
    ------
    ValueError
        No attempt was played to its end or its result is unknown. Nothing is
        written then.
    """
    import marl_battlegrounds as marl_bgs
    from marl_battlegrounds.evaluation._replays.controller_names import (
        recorded_slot_controller_names,
    )

    counted = counted_attempt(attempts)
    if counted is None:
        raise ValueError(f"Game {game.number} has no attempt played to its end.")
    path, loaded = counted
    replay = loaded.replay
    context = replay.header.context
    values = marl_bgs.replay_metrics(loaded, metrics="full")
    outcomes = [o for o in ("win", "draw", "loss") if values[f"team_a_{o}"] == 1]
    if len(outcomes) != 1:
        raise ValueError(f"Game {game.number}'s result is unknown.")
    identity = {
        "number": game.number,
        "kind": game.kind,
        "map_id": game.map_id,
        "side": game.side,
        "replay": path.name,
        "replay_digest": replay.canonical_digest_sha256,
    }
    table = io.StringIO()
    writer = csv.DictWriter(table, fieldnames=[*identity, *values], lineterminator="\n")
    writer.writeheader()
    writer.writerow(
        identity
        | {name: "" if math.isnan(value) else value for name, value in values.items()}
    )
    (path.parent / "full_metrics.csv").write_text(table.getvalue(), encoding="utf-8")
    catalog = context.static_mechanics_catalog.class_name_by_id
    record = identity | {
        "seed": game.seed,
        "opponent": game.opponent,
        "partner": game.partner,
        "player_class": game.player_class,
        "controllers": list(recorded_slot_controller_names(context) or [None] * 10),
        "classes": [catalog[row.class_id] for row in context.roster],
        "attempts": [
            {"replay": p.name, "completion": a.replay.completion.completion_state}
            for p, a in attempts
        ],
        "wall_seconds": None if played is None else round(played[1] - played[0]),
        "outcome": outcomes[0],
        **{name: int(values[name]) for name in SUMMARY_COLUMNS},
    }
    (path.parent / "game.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return record


def write_tables(records: Sequence[dict[str, Any]], output_dir: Path) -> None:
    """Write ``results.csv`` and ``results.md`` for the recorded games.

    Parameters
    ----------
    records : sequence of dict
        ``record_game`` records in playing order.
    output_dir : pathlib.Path
        Folder for the two tables. Existing tables are replaced.

    Notes
    -----
    ``results.csv`` has one row per game, with the catalog's column names.
    ``results.md`` gives, for each kind, wins · draws · losses from Team A's
    side and the mean point margin, kills and deaths per game, then one line
    per game.
    """
    fields = [
        "number", "kind", "map_id", "side", "opponent", "player_class", "outcome",
        *SUMMARY_COLUMNS, "wall_seconds", "replay",
    ]  # fmt: skip
    table = io.StringIO()
    writer = csv.DictWriter(
        table, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(records)
    (output_dir / "results.csv").write_text(table.getvalue(), encoding="utf-8")
    lines = [
        "| Kind | Games | Wins · Draws · Losses | Point Margin | Kills | Deaths |",
        "| --- | ---: | :---: | ---: | ---: | ---: |",
    ]
    for kind in KINDS:
        games = [record for record in records if record["kind"] == kind]
        if not games:
            continue
        counts = [
            sum(r["outcome"] == o for r in games) for o in ("win", "draw", "loss")
        ]
        means = [
            sum(r[name] for r in games) / len(games)
            for name in ("score_difference", "team_a_kills", "team_a_deaths")
        ]
        lines.append(
            f"| {kind.capitalize()} | {len(games)} | "
            + " · ".join(map(str, counts))
            + " | {:+.1f} | {:.1f} | {:.1f} |".format(*means)
        )
    lines += [
        "",
        "| Game | Kind | Opponent | Map | Team A Starts | Result | Points "
        "| Kills · Deaths |",
        "| ---: | --- | --- | ---: | --- | --- | :---: | :---: |",
    ]
    lines += [
        f"| {r['number']} | {r['kind']} | {r['opponent']} | {r['map_id']} | "
        f"{r['side']} | {r['outcome']} | {r['team_a_score']} to {r['team_b_score']} | "
        f"{r['team_a_kills']} · {r['team_a_deaths']} |"
        for r in records
    ]
    note = (
        "Team A is the person's team. Point margin, kills and deaths are means "
        "per game."
    )
    (output_dir / "results.md").write_text(
        note + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8"
    )


@hydra.main(version_base="1.3", config_path="configs", config_name="human_games")
def main(config: DictConfig) -> list[dict[str, Any]]:
    """Play the scheduled games one by one, measuring each finished game.

    Recorded games are skipped, and a game played to its end but not yet
    measured is measured without being played again, so running the command
    again continues the series. Before each new game the stage prints the
    game and waits for Enter; it stops without error at Ctrl-C there, or after
    a game exited before its end.

    Parameters
    ----------
    config : omegaconf.DictConfig
        ``systems`` (name and method of every System, from the field
        settings); ``parts`` (see ``schedule``; ``opponent`` and ``partner``
        are names from ``systems``) and the required ``output_dir``.
        ``models_dir`` fills ``${models_dir}`` in method paths (the
        ``season_0_local`` field). Relative paths use the folder the command
        was started from.

    Returns
    -------
    list of dict
        The ``record_game`` records of every recorded game, in playing order.

    Raises
    ------
    ValueError
        A setting is invalid (``schedule``, ``person_slots``), a name is not in
        ``systems``, the folder holds a series with other settings, or a
        replay fails ``check_replay``.

    Notes
    -----
    Each System is loaded once. The BattleClient plays the Systems on the
    default JAX device; set ``CUDA_VISIBLE_DEVICES`` to choose a GPU. It needs
    a browser.
    """
    import marl_battlegrounds as marl_bgs

    settings: dict[str, Any] = OmegaConf.to_container(config, resolve=True)  # pyright: ignore[reportAssignmentType] - the root is a mapping.
    methods = {
        str(entry["name"]): str(entry["method"]) for entry in settings["systems"]
    }
    games = schedule(settings["parts"])
    names = {name for g in games for name in (g.opponent, g.partner) if name}
    missing = sorted(names - set(methods))
    if missing:
        raise ValueError(f"not in systems: {missing}")
    for game in games:
        person_slots(game)
    output_dir = Path(to_absolute_path(str(Path(settings["output_dir"]).expanduser())))
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = {key: value for key, value in settings.items() if key != "output_dir"}
    settings_path = output_dir / "settings.json"
    if settings_path.exists():
        if json.loads(settings_path.read_text(encoding="utf-8")) != saved:
            raise ValueError(
                f"{output_dir} holds a series with other settings; "
                "continue it with its own settings or choose another output_dir"
            )
    else:
        settings_path.write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8")

    loaded: dict[str, Policy | System] = {}

    def system(name: str) -> Policy | System:
        """Load a System once, on its first use."""
        if name not in loaded:
            loaded[name] = marl_bgs.load_method(methods[name])
        return loaded[name]

    records: list[dict[str, Any]] = []
    for game in games:
        folder = output_dir / game.folder
        if (folder / "game.json").is_file():
            records.append(json.loads((folder / "game.json").read_text("utf-8")))
            continue
        attempts, played = read_attempts(folder), None
        if counted_attempt(attempts) is None:
            who = (
                "all of Team A"
                if game.kind == "against"
                else f"the {game.player_class}"
            )
            print(
                f"Game {game.number} of {len(games)} ({game.kind}): you play {who} "
                f"against {game.opponent} on map {game.map_id}, "
                f"Team A starts on the {game.side}."
            )
            try:
                input("Press Enter to open it, or Ctrl-C to stop. ")
            except (KeyboardInterrupt, EOFError):
                print("\nStopped. Run the same command to continue.")
                break
            folder.mkdir(parents=True, exist_ok=True)
            started = time.time()
            marl_bgs.battle_client(
                map_id=game.map_id,
                team_a=team_a_controllers(
                    game, None if game.partner is None else system(game.partner)
                ),
                team_b=system(game.opponent),
                seed=game.seed,
                spawn_mode="swapped" if game.swapped else "default",
                output_dir=folder,
            )
            played = (started, time.time())
            attempts = read_attempts(folder)
        counted = counted_attempt(attempts)
        if counted is None:
            print(
                f"Game {game.number} was not played to its end. Run again to replay it."
            )
            break
        check_replay(game, counted[1])
        records.append(record_game(game, attempts, played=played))
        print(f"Game {game.number}: {records[-1]['outcome']}, saved in {folder}")
    if records:
        write_tables(records, output_dir)
        print(output_dir / "results.md")
    return records


if __name__ == "__main__":
    prepare_hydra()
    main()
