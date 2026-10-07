"""Play recorded games against, and alongside, a trained System in the BattleClient.

Use python -m marl_battlegrounds.experiments.human_games models_dir=/path/models
output_dir=/path/run. A person plays Team A in the BattleClient
(marl_bgs.battle_client); the opponent System, by default the Season 0 winner,
recurrent IPPO, plays Team B. In "against" games the person controls all five
Team A agents. In "alongside" games the person controls one class, the Priest
by default, and the partner System, by default the opponent itself, controls
the other four. The games visit the test maps 47 to 51, once from each spawn
end (spawn_mode). After each finished game the stage checks its replay against
the schedule, measures it with marl_bgs.replay_metrics, saves the full metrics
and a record beside it, and rewrites results.csv and results.md. Run the same
command again to continue where you stopped; a different series needs its own
output_dir. A person's moves cannot be repeated, so the saved replays are the
record of the games.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
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
# Bumped when game.json gains fields; older records are measured again.
RECORD_VERSION = 2
# Settings a series keeps. games_per_kind may also change, but only between
# whole rounds (multiples of the number of maps), so earlier games keep their
# numbers.
SERIES_SETTINGS = ("opponent", "partner", "kinds", "maps", "player_class", "seed")
# Subfolder for a redone game's earlier files: still listed, never counted.
REDONE = "redone"
# Catalog columns summarized for every game, from Team A's side.
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
    """One scheduled game of the series.

    Attributes
    ----------
    number : int
        Position in the whole series, from 1.
    kind : {'against', 'alongside'}
        ``'against'``: the person controls all five Team A agents.
        ``'alongside'``: the person controls one class and the partner System
        the other four Team A agents.
    index : int
        Position among the games of its kind, from 1.
    map_id : int
        Packaged map.
    swapped : bool
        False: Team A starts on the left (``spawn_mode='default'``). True:
        Team A starts on the right (``spawn_mode='swapped'``).
    """

    number: int
    kind: str
    index: int
    map_id: int
    swapped: bool

    @property
    def side(self) -> str:
        """Return where Team A starts: ``'left'`` or ``'right'``."""
        return "right" if self.swapped else "left"

    @property
    def folder(self) -> str:
        """Return the game's folder, relative to the output folder.

        For example ``'alongside/game-03-map-49-left'``.
        """
        return f"{self.kind}/game-{self.index:02d}-map-{self.map_id}-{self.side}"


def schedule(
    kinds: Sequence[str], maps: Sequence[int], games_per_kind: int
) -> list[Game]:
    """Put the games of every kind in playing order.

    Each kind plays its games in rounds of one game per map. In the first round
    the ends alternate map by map, starting with Team A on the left; each later
    round swaps every map's end. With five maps and ten games per kind, games 1
    to 5 visit every map once and games 6 to 10 visit them again from the
    other end. Each round plays every kind in turn, so stopping when a round
    of the last kind ends leaves the same number of games of each kind. When
    both counts are whole rounds (multiples of the number of maps), the
    schedule for fewer games per kind is the start of the schedule for more.

    Parameters
    ----------
    kinds : sequence of {'against', 'alongside'}
        Distinct kinds, in the order each round plays them.
    maps : sequence of int
        Distinct packaged map IDs, in the order each round plays them.
    games_per_kind : int
        Games of each kind, at least 1. When games_per_kind is not a multiple
        of the number of maps, the last round is cut short.

    Returns
    -------
    list of Game
        The series in playing order, numbered from 1.

    Raises
    ------
    ValueError
        A kind is unknown or repeated, no kind or map is given, a map repeats,
        or games_per_kind is below 1.

    Examples
    --------
    >>> [(g.index, g.map_id, g.side) for g in schedule(["against"], [47, 48], 4)]
    [(1, 47, 'left'), (2, 48, 'right'), (3, 47, 'right'), (4, 48, 'left')]
    """
    if not kinds or len(set(kinds)) != len(kinds) or not set(kinds) <= set(KINDS):
        raise ValueError(f"kinds must be distinct values from {list(KINDS)}")
    if not maps or len(set(maps)) != len(maps):
        raise ValueError("maps must list at least one map, without repeats")
    if type(games_per_kind) is not int or games_per_kind < 1:
        raise ValueError("games_per_kind must be an integer of at least 1")
    games: list[Game] = []
    for round_index in range(math.ceil(games_per_kind / len(maps))):
        for kind in kinds:
            for position, map_id in enumerate(maps):
                index = round_index * len(maps) + position + 1
                if index > games_per_kind:
                    break
                swapped = (position + round_index) % 2 == 1
                games.append(Game(len(games) + 1, kind, index, map_id, swapped))
    return games


def person_slots(game: Game, player_class: str) -> tuple[int, ...]:
    """Return the global slots the person controls in a game.

    Parameters
    ----------
    game : Game
        The game to play.
    player_class : str
        The class the person plays in alongside games, one of the canonical
        roster's classes: ``'mage'``, ``'warrior'``, ``'hunter'``, ``'rogue'``
        or ``'priest'``.

    Returns
    -------
    tuple of int
        ``(0, 1, 2, 3, 4)`` (all of Team A) for an against game, or the one
        Team A slot of player_class for an alongside game, for example ``(4,)``
        for the Priest.

    Raises
    ------
    ValueError
        player_class is not in the canonical roster.
    """
    import marl_battlegrounds as marl_bgs

    roster = marl_bgs.canonical_tournament_rosters()[0]
    if player_class not in roster:
        raise ValueError(f"player_class must be one of {list(roster)}")
    return tuple(range(5)) if game.kind == "against" else (roster.index(player_class),)


def team_a_controllers(
    game: Game, partner: Policy | System, player_class: str
) -> str | list[str | Policy | System]:
    """Return Team A's controllers for ``marl_bgs.battle_client``.

    Parameters
    ----------
    game : Game
        The game to play.
    partner : Policy or System
        The loaded System that fills Team A's other slots in alongside games.
    player_class : str
        The class the person plays in alongside games (see ``person_slots``).

    Returns
    -------
    str or list
        ``'manual'`` (all five slots) for an against game, or one controller per
        canonical roster slot for an alongside game: ``'manual'`` at the
        player's class and ``partner`` elsewhere.

    Raises
    ------
    ValueError
        player_class is not in the canonical roster.
    """
    slots = person_slots(game, player_class)
    if game.kind == "against":
        return "manual"
    return ["manual" if slot in slots else partner for slot in range(5)]


def read_attempts(folder: Path) -> list[tuple[Path, LoadedReplay]]:
    """Load every replay saved in a game's folder, oldest first.

    The BattleClient saves a replay when a game ends, and also when a person
    finishes, saves or exits a game early, or restarts after saving. Each
    file is one attempt at the game. A Restart whose recording is discarded
    leaves no file. Replays moved into the folder's ``redone`` subfolder, to
    redo a recorded game, are attempts too, but never count.

    Parameters
    ----------
    folder : pathlib.Path
        The game's folder. A missing folder has no attempts.

    Returns
    -------
    list of (pathlib.Path, LoadedReplay)
        Each replay file, in the folder or its ``redone`` subfolder, and its
        loaded content, ordered by file time. Each file is read once.

    Raises
    ------
    marl_battlegrounds.evaluation.replay_io.ReplayLoadError
        A saved file is not a valid replay.
    """
    from marl_battlegrounds.evaluation import load_replay

    if not folder.is_dir():
        return []
    paths = sorted(
        [
            *folder.glob(f"{REDONE}/*.marlbg-replay.json"),
            *folder.glob("*.marlbg-replay.json"),
        ],
        key=lambda path: path.stat().st_mtime,
    )
    return [(path, load_replay(path)) for path in paths]


def counted_attempt(
    attempts: Sequence[tuple[Path, LoadedReplay]],
) -> tuple[Path, LoadedReplay] | None:
    """Return the attempt that counts: the first one played to its end.

    A replayed game never replaces an earlier finished one. To redo a
    recorded game, move its ``game.json`` and its finished replay into the
    game folder's ``redone`` subfolder; they stay listed as attempts, and a
    redo should be reported with the results.

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
        if (
            path.parent.name != REDONE
            and loaded.replay.completion.completion_state == "complete"
        ):
            return path, loaded
    return None


def check_replay(
    game: Game, loaded: LoadedReplay, *, seed: int, player_class: str
) -> None:
    """Refuse a replay that is not this game, played as scheduled.

    Parameters
    ----------
    game : Game
        The scheduled game.
    loaded : LoadedReplay
        Its counted replay.
    seed : int
        The seed every game of the series uses.
    player_class : str
        The class the person plays in alongside games.

    Raises
    ------
    ValueError
        The game did not end, or the replay records another map, other spawn
        ends, another seed, or a person ("Manual") in other slots than the
        game gives the person.
    """
    from marl_battlegrounds.evaluation.controller_names import (
        recorded_slot_controller_names,
    )
    from marl_battlegrounds.evaluation.map_identity import recorded_map

    replay = loaded.replay
    context = replay.header.context
    keys = {row.name: row.value for row in context.aggregation_keys}
    names = recorded_slot_controller_names(context) or (None,) * 10
    manual = tuple(slot for slot, name in enumerate(names) if name == "Manual")
    problems = {
        "was not played to its end": replay.completion.completion_state != "complete",
        f"is not map {game.map_id}": recorded_map(context).map_id != game.map_id,
        f"does not start Team A on the {game.side}": keys.get(
            "marl_bgs.spawn_locations"
        )
        != str(int(game.swapped)),
        f"did not use seed {seed}": context.seed_protocol.root_seed != seed,
        f"does not have the person in slots {person_slots(game, player_class)}": (
            manual != person_slots(game, player_class)
        ),
    }
    failed = [problem for problem, found in problems.items() if found]
    if failed:
        raise ValueError(f"Game {game.number}'s replay " + "; ".join(failed) + ".")


def _write_text(path: Path, text: str) -> None:
    """Replace a file in one step, so a stopped run never leaves half a file."""
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def record_game(
    game: Game,
    attempts: Sequence[tuple[Path, LoadedReplay]],
    *,
    settings: Mapping[str, Any],
    played: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Measure a game's counted attempt and save its full metrics and record.

    Writes ``full_metrics.csv`` (one row: the game, its replay and its digest,
    then every full metric column; an empty cell is a metric that does not
    apply) and ``game.json`` (the returned record) in the game's folder.
    ``game.json`` is written last and marks the game as done.

    Parameters
    ----------
    game : Game
        The scheduled game.
    attempts : sequence of (pathlib.Path, LoadedReplay)
        Every replay of the game, from ``read_attempts``; the first finished
        one counts. Call ``check_replay`` on it first.
    settings : mapping
        The series settings: ``opponent`` and ``partner`` (``name`` and
        ``method``), ``player_class`` and ``seed``.
    played : (float, float) or None, default=None
        Wall-clock start and end of the BattleClient launch, in Unix seconds,
        or None when unknown (for example a game measured on a later run).

    Returns
    -------
    dict
        ``record_version``; the game (``asdict(game)`` plus ``side``); the
        settings above; ``replay`` (file name), ``replay_digest``,
        ``episode_id``; ``controllers`` and ``classes`` (one entry per global
        slot, as recorded; ``"Manual"`` marks the person); ``attempts`` (every
        replay file of the game, relative to its folder, with its completion
        state, oldest first); ``started_at``, ``ended_at`` and ``wall_seconds``
        (how long the BattleClient was open for the launch that saved the
        counted game, including about 12 s of setup and any time in an opened
        Replay Viewer; None when unknown);
        ``metric_schema_version``; ``measured_with`` (``XLA_FLAGS`` and the
        JAX backend that measured it); ``outcome`` (``'win'``, ``'draw'`` or
        ``'loss'``, from Team A's side); and the SUMMARY_COLUMNS values.

    Raises
    ------
    ValueError
        No attempt was played to its end, or its result is unknown.
    """
    import jax

    import marl_battlegrounds as marl_bgs
    from marl_battlegrounds.evaluation.controller_names import (
        recorded_slot_controller_names,
    )
    from marl_battlegrounds.evaluation.metric_catalog import METRIC_SCHEMA_VERSION

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
    identity = asdict(game) | {"side": game.side, "replay": path.name}
    identity["replay_digest"] = replay.canonical_digest_sha256
    table = io.StringIO()
    writer = csv.DictWriter(table, fieldnames=[*identity, *values], lineterminator="\n")
    writer.writeheader()
    writer.writerow(
        identity
        | {name: "" if math.isnan(value) else value for name, value in values.items()}
    )
    _write_text(path.parent / "full_metrics.csv", table.getvalue())
    catalog = context.static_mechanics_catalog.class_name_by_id
    record = identity | {
        "record_version": RECORD_VERSION,
        **{name: settings[name] for name in ("opponent", "partner", "player_class")},
        "seed": int(settings["seed"]),
        "episode_id": context.identity.episode_id,
        "controllers": list(recorded_slot_controller_names(context) or [None] * 10),
        "classes": [catalog[row.class_id] for row in context.roster],
        "attempts": [
            {
                "replay": str(p.relative_to(path.parent)),
                "completion": a.replay.completion.completion_state,
            }
            for p, a in attempts
        ],
        "started_at": None if played is None else played[0],
        "ended_at": None if played is None else played[1],
        "wall_seconds": None if played is None else round(played[1] - played[0]),
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "measured_with": {
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "jax_backend": jax.default_backend(),
        },
        "outcome": outcomes[0],
        **{name: int(values[name]) for name in SUMMARY_COLUMNS},
    }
    _write_text(path.parent / "game.json", json.dumps(record, indent=2) + "\n")
    return record


def method_reference(reference: str) -> str:
    """Return a method reference with a folder path made absolute.

    The same actor folder then compares equal whether a run named it with a
    relative or an absolute path. ``marl_bgs.load_method``'s own rule decides
    what a reference is: a built-in name (for example ``'tdm-alpha'``) or a
    ``module:function`` factory is returned unchanged, even when a folder of
    that name exists.

    Parameters
    ----------
    reference : str
        Any reference ``marl_bgs.load_method`` accepts.

    Returns
    -------
    str
        The absolute path, relative to the folder the command was started
        from, when ``load_method`` would read the reference as a saved folder;
        else ``reference``.

    Raises
    ------
    TypeError, ValueError
        reference is not text, or is blank.
    """
    from marl_battlegrounds._method_loading import (
        _reference_kind,  # pyright: ignore[reportPrivateUsage] - load_method's own rule.
    )

    if _reference_kind(reference) != "artifact":
        return reference
    return to_absolute_path(str(Path(reference).expanduser()))


def check_series(output_dir: Path, settings: Mapping[str, Any]) -> None:
    """Refuse settings that differ from the series already in a folder.

    Parameters
    ----------
    output_dir : pathlib.Path
        The series folder. ``series.json`` there, written by ``save_series``
        when the first game starts, holds the series settings.
    settings : mapping
        The current run's settings, with method references made absolute by
        ``method_reference``.

    Raises
    ------
    ValueError
        A saved setting differs, or games_per_kind changes by a part of a
        round. Use another output_dir for another series.
    """
    saved_path = output_dir / "series.json"
    if not saved_path.is_file():
        return
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    changed = [name for name in SERIES_SETTINGS if saved.get(name) != settings[name]]
    rounds = len(settings["maps"])
    if saved.get("games_per_kind") != settings["games_per_kind"] and (
        saved.get("games_per_kind", 0) % rounds or settings["games_per_kind"] % rounds
    ):
        changed.append(f"games_per_kind (change it only by whole rounds of {rounds})")
    if changed:
        raise ValueError(
            f"{output_dir} holds a series with other {', '.join(changed)}; "
            "continue it with its own settings or choose another output_dir."
        )


def save_series(output_dir: Path, settings: Mapping[str, Any]) -> None:
    """Save the series settings once, when the series' first game starts.

    Parameters
    ----------
    output_dir : pathlib.Path
        The series folder.
    settings : mapping
        The settings checked by ``check_series``. An existing ``series.json``
        is kept.
    """
    saved_path = output_dir / "series.json"
    if not saved_path.is_file():
        names = (*SERIES_SETTINGS, "games_per_kind")
        _write_text(
            saved_path, json.dumps({n: settings[n] for n in names}, indent=2) + "\n"
        )


def recorded_games(
    games: Sequence[Game], output_dir: Path, settings: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Read the record of every scheduled game that has one.

    A record from an older version of this stage is measured again from its
    replay, which is the record of the game.

    Parameters
    ----------
    games : sequence of Game
        The schedule.
    output_dir : pathlib.Path
        The series folder.
    settings : mapping
        The series settings, as for ``record_game``.

    Returns
    -------
    list of dict
        The records in playing order, numbered by the current schedule.

    Raises
    ------
    ValueError
        A record names a replay that no longer exists, or an older record's
        replay fails ``check_replay``.
    """
    records: list[dict[str, Any]] = []
    for game in games:
        record_path = output_dir / game.folder / "game.json"
        if not record_path.is_file():
            continue
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if not (record_path.parent / record["replay"]).is_file():
            raise ValueError(
                f"{record_path} names a missing replay; to redo the game, move "
                "game.json into the game folder's redone/ subfolder."
            )
        if record.get("record_version") != RECORD_VERSION:
            print(f"Measuring game {game.number} again (an older record)...")
            attempts = read_attempts(record_path.parent)
            counted = counted_attempt(attempts)
            if counted is not None:
                check_replay(
                    game,
                    counted[1],
                    seed=int(settings["seed"]),
                    player_class=settings["player_class"],
                )
            record = record_game(game, attempts, settings=settings)
        records.append(record | {"number": game.number})
    return records


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
        "number", "kind", "index", "map_id", "side", "outcome", *SUMMARY_COLUMNS,
        "wall_seconds", "replay",
    ]  # fmt: skip
    table = io.StringIO()
    writer = csv.DictWriter(
        table, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(records)
    _write_text(output_dir / "results.csv", table.getvalue())
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
        "| Game | Kind | Map | Team A Starts | Result | Points | Kills · Deaths |",
        "| ---: | --- | ---: | --- | --- | :---: | :---: |",
    ]
    lines += [
        f"| {r['number']} | {r['kind']} {r['index']} | {r['map_id']} | {r['side']} | "
        f"{r['outcome']} | {r['team_a_score']} to {r['team_b_score']} | "
        f"{r['team_a_kills']} · {r['team_a_deaths']} |"
        for r in records
    ]
    note = (
        "Team A is the person's team. Point margin, kills and deaths are means "
        "per game."
    )
    _write_text(output_dir / "results.md", note + "\n\n" + "\n".join(lines) + "\n")


@hydra.main(version_base="1.3", config_path="configs", config_name="human_games")
def main(config: DictConfig) -> list[dict[str, Any]]:
    """Play the scheduled games one by one, measuring each finished game.

    First it loads the methods and checks every recorded game, measuring an
    older record again. Before each game it prints which game comes next and
    waits for Enter. It then saves ``series.json`` (once) and opens the
    BattleClient, which returns when the person presses Exit. Recorded games
    are skipped, and a game that was played to its end but not yet measured
    is measured without being played again, so running the command again
    continues the series. The stage stops, without error, when the person
    presses Ctrl-C at the prompt or exits a game before its end. Method
    references that name folders are made absolute (``method_reference``)
    before they are saved or compared.

    Parameters
    ----------
    config : omegaconf.DictConfig
        ``opponent`` and ``partner`` (each a ``name`` and a ``method``, any
        reference ``marl_bgs.load_method`` accepts; the partner defaults to
        the opponent), ``kinds``, ``maps``, ``games_per_kind``,
        ``player_class``, ``seed`` and the required ``output_dir``;
        ``models_dir`` fills ``${models_dir}`` in the default paths. Relative
        paths use the folder the command was started from.

    Returns
    -------
    list of dict
        The ``record_game`` records of every recorded game, in playing order.

    Raises
    ------
    ValueError
        The schedule settings or player_class are invalid, output_dir holds a
        series with other settings (see ``check_series``), or a replay fails
        ``check_replay``.
    """
    import marl_battlegrounds as marl_bgs

    settings: dict[str, Any] = OmegaConf.to_container(config, resolve=True)  # pyright: ignore[reportAssignmentType] - the root is a mapping.
    for role in ("opponent", "partner"):
        settings[role]["method"] = method_reference(settings[role]["method"])
    games = schedule(settings["kinds"], settings["maps"], settings["games_per_kind"])
    player_class, seed = settings["player_class"], int(settings["seed"])
    person_slots(games[0], player_class)  # Check the class before any game.
    output_dir = Path(to_absolute_path(str(Path(settings["output_dir"]).expanduser())))
    output_dir.mkdir(parents=True, exist_ok=True)
    check_series(output_dir, settings)
    # Load once: every game reuses the same frozen weights.
    opponent = marl_bgs.load_method(settings["opponent"]["method"])
    partner = (
        opponent
        if settings["partner"]["method"] == settings["opponent"]["method"]
        else marl_bgs.load_method(settings["partner"]["method"])
    )
    # Check earlier games now, not after the next game is played.
    if recorded_games(games, output_dir, settings):
        save_series(output_dir, settings)
    for game in games:
        folder = output_dir / game.folder
        if (folder / "game.json").is_file():
            continue
        attempts, played = read_attempts(folder), None
        if counted_attempt(attempts) is None:
            print(
                f"Game {game.number} of {len(games)}: {game.kind} {game.index}, "
                f"map {game.map_id}, Team A starts on the {game.side}."
            )
            try:
                input("Press Enter to open it, or Ctrl-C to stop. ")
            except KeyboardInterrupt, EOFError:
                print("\nStopped. Run the same command to continue.")
                break
            save_series(output_dir, settings)
            folder.mkdir(parents=True, exist_ok=True)
            started = time.time()
            marl_bgs.battle_client(
                map_id=game.map_id,
                team_a=team_a_controllers(game, partner, player_class),
                team_b=opponent,
                seed=seed,
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
        print(f"Measuring game {game.number}...")
        check_replay(game, counted[1], seed=seed, player_class=player_class)
        save_series(output_dir, settings)
        record = record_game(game, attempts, settings=settings, played=played)
        write_tables(recorded_games(games, output_dir, settings), output_dir)
        print(f"Game {game.number}: {record['outcome']}, saved in {folder}")
    records = recorded_games(games, output_dir, settings)
    if records:
        write_tables(records, output_dir)
        print(output_dir / "results.md")
    return records


if __name__ == "__main__":
    prepare_hydra()
    main()
