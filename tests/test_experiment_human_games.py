"""Check the human_games experiment stage without a person or a browser.

Contracts: the packaged settings compose with models_dir filling the opponent's
path (recurrent IPPO, the Season 0 winner), the partner defaulting to the
opponent, and ten games of each kind on maps 47 to 51 with the Priest;
schedule() alternates kinds round by round, visits every map once per round,
alternates the ends map by map and swaps them each round, makes a shorter
series of whole rounds a prefix of a longer one, and rejects bad settings;
against games give the person all of Team A and alongside games only the
player's class, with the partner elsewhere; method_reference() makes a folder
path absolute and keeps a built-in name or factory even beside a same-named
folder; check_replay() names every way a replay differs from its scheduled
game; record_game() counts the first finished attempt outside the redone
subfolder, refuses a game with no finished attempt or an unknown result, and
writes full_metrics.csv and game.json with every attempt, Team A's result and
summary columns; check_series() refuses a folder holding a series with other
settings or a games_per_kind change by part of a round, and save_series()
writes series.json only once; main() checks player_class before any game, saves
series.json only when a game starts, checks recorded games before any prompt,
measures a finished but unmeasured game without playing it again, opens the
BattleClient with each game's map, seed and spawn_mode, stops cleanly at Ctrl-C
or after an unfinished game, and writes results.csv and results.md from every
recorded game; recorded_games() measures records of an older version again and
refuses a record whose replay is gone. Fakes replace the BattleClient, replays
and metrics, so no JAX program is compiled; tests/test_replay_metrics.py runs
check_replay and record_game on real replays.
"""

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from hydra import compose, initialize_config_module
from omegaconf import DictConfig

import marl_battlegrounds as marl_bgs
from marl_battlegrounds.experiments import human_games
from marl_battlegrounds.experiments.human_games import Game, schedule

SETTINGS: dict[str, Any] = {
    "opponent": {"name": "Recurrent IPPO", "method": "/m/ippo"},
    "partner": {"name": "Recurrent IPPO", "method": "/m/ippo"},
    "player_class": "priest",
    "seed": 7,
}


def composed(*overrides: str) -> DictConfig:
    with initialize_config_module(
        config_module="marl_battlegrounds.experiments.configs", version_base="1.3"
    ):
        return compose(config_name="human_games", overrides=list(overrides))


def series(tmp_path: Path, *extra: str) -> DictConfig:
    return composed(
        "models_dir=/models", f"output_dir={tmp_path}", "kinds=[alongside]",
        "maps=[47,48]", "games_per_kind=3", "seed=11", *extra,
    )  # fmt: skip


def test_settings_compose_with_the_season_0_winner() -> None:
    config = composed("models_dir=/models", "output_dir=out")
    assert config.opponent.name == "Recurrent IPPO"
    assert str(config.opponent.method).startswith("/models/250m/study/cases/ippo-")
    assert config.partner == config.opponent
    assert list(config.kinds) == ["against", "alongside"]
    assert list(config.maps) == [47, 48, 49, 50, 51]
    assert (config.games_per_kind, config.player_class) == (10, "priest")
    mine = composed(
        "models_dir=/m", "output_dir=o", "partner.name=Mine", "partner.method=/mine"
    )
    assert (mine.partner.name, mine.partner.method) == ("Mine", "/mine")
    assert mine.opponent.name == "Recurrent IPPO"


def test_schedule_alternates_kinds_maps_and_ends() -> None:
    games = schedule(["against", "alongside"], [47, 48, 49, 50, 51], 10)
    assert [g.number for g in games] == list(range(1, 21))
    assert [(g.kind, g.index) for g in games[:10]] == [
        *(("against", i) for i in range(1, 6)),
        *(("alongside", i) for i in range(1, 6)),
    ]
    first = [(g.map_id, g.side) for g in games[:5]]
    assert first == [
        (47, "left"), (48, "right"), (49, "left"), (50, "right"), (51, "left"),
    ]  # fmt: skip
    assert [g.side for g in games[5:10]] == [side for _, side in first]
    # The second round of each kind plays every map from its other end.
    assert [(g.map_id, g.side) for g in games[10:15]] == [
        (47, "right"), (48, "left"), (49, "right"), (50, "left"), (51, "right"),
    ]  # fmt: skip
    for kind in ("against", "alongside"):
        ends = {(g.map_id, g.side) for g in games if g.kind == kind}
        assert len(ends) == 10
    assert games[13].folder == "against/game-09-map-50-left"
    assert schedule(["against", "alongside"], [47, 48, 49, 50, 51], 5) == games[:10]
    assert len(schedule(["alongside"], [47, 48, 49, 50, 51], 7)) == 7


@pytest.mark.parametrize(
    ("kinds", "maps", "count"),
    [([], [47], 1), (["against", "against"], [47], 1), (["solo"], [47], 1),
     (["against"], [], 1), (["against"], [47, 47], 1), (["against"], [47], 0)],
)  # fmt: skip
def test_schedule_rejects_bad_settings(
    kinds: list[str], maps: list[int], count: int
) -> None:
    with pytest.raises(ValueError):
        schedule(kinds, maps, count)


def test_team_a_gives_the_person_all_or_one_slot() -> None:
    partner: Any = object()
    against = Game(1, "against", 1, 47, False)
    alongside = Game(2, "alongside", 1, 47, False)
    assert human_games.team_a_controllers(against, partner, "priest") == "manual"
    assert human_games.team_a_controllers(alongside, partner, "priest") == [
        partner, partner, partner, partner, "manual",
    ]  # fmt: skip
    assert human_games.person_slots(alongside, "mage") == (0,)
    assert human_games.person_slots(against, "mage") == (0, 1, 2, 3, 4)
    with pytest.raises(ValueError, match="player_class"):
        human_games.team_a_controllers(alongside, partner, "paladin")


def test_method_reference_makes_folders_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("actor", "tdm-alpha", "pkg.mod:make"):
        (tmp_path / name).mkdir()
    monkeypatch.chdir(tmp_path)
    assert human_games.method_reference("actor") == str(tmp_path / "actor")
    assert human_games.method_reference(str(tmp_path / "actor")) == str(
        tmp_path / "actor"
    )
    # load_method's rule wins over a folder of the same name.
    assert human_games.method_reference("tdm-alpha") == "tdm-alpha"
    assert human_games.method_reference("pkg.mod:make") == "pkg.mod:make"


def _replay(
    state: str = "complete",
    *,
    map_id: str = "48",
    spawn: str = "1",
    seed: int = 7,
    manual: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> Any:  # noqa: ANN401
    keys = {"map_id": map_id, "marl_bgs.spawn_locations": spawn}
    keys |= {
        f"marl_bgs.controller_name.slot_{s}": json.dumps(
            "Manual" if s in manual else "IPPO"
        )
        for s in range(10)
    }
    context = SimpleNamespace(
        aggregation_keys=[SimpleNamespace(name=k, value=v) for k, v in keys.items()],
        identity=SimpleNamespace(episode_id="e-1"),
        seed_protocol=SimpleNamespace(root_seed=seed),
        roster=[SimpleNamespace(class_id=1 + s % 5) for s in range(10)],
        static_mechanics_catalog=SimpleNamespace(
            class_name_by_id=("Neutral", "Mage", "Warrior", "Hunter", "Rogue", "Priest")
        ),
    )
    return SimpleNamespace(
        replay=SimpleNamespace(
            completion=SimpleNamespace(completion_state=state),
            header=SimpleNamespace(context=context),
            canonical_digest_sha256="d" * 64,
        )
    )


def _recorded_map(context: Any) -> SimpleNamespace:  # noqa: ANN401
    keys = {row.name: row.value for row in context.aggregation_keys}
    return SimpleNamespace(map_id=int(keys["map_id"]))


def test_check_replay_names_every_difference(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "marl_battlegrounds.evaluation.map_identity.recorded_map", _recorded_map
    )
    game = Game(2, "against", 2, 48, True)
    human_games.check_replay(game, _replay(), seed=7, player_class="priest")
    wrong = _replay("partial", map_id="47", spawn="0", seed=8, manual=(4,))
    with pytest.raises(ValueError) as caught:
        human_games.check_replay(game, wrong, seed=7, player_class="priest")
    for part in ("not played to its end", "is not map 48", "on the right",
                 "did not use seed 7", "person in slots (0, 1, 2, 3, 4)"):  # fmt: skip
        assert part in str(caught.value)


def _metrics(**values: float) -> dict[str, float]:
    names = ("team_a_win", "team_a_draw", "team_a_loss", *human_games.SUMMARY_COLUMNS)
    base = dict.fromkeys(names, 0.0) | {"agent_9_healing_done": math.nan}
    return base | values


def test_record_game_saves_every_attempt_and_the_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _metrics(
        team_a_win=1.0, team_a_score=20.0, team_b_score=7.0, score_difference=13.0,
        team_a_kills=18.0, team_a_deaths=7.0, episode_length=212.0,
    )  # fmt: skip

    def replay_metrics(replay: object, metrics: str) -> dict[str, float]:
        return values

    monkeypatch.setattr(marl_bgs, "replay_metrics", replay_metrics)
    game = Game(2, "against", 2, 48, True)
    attempts = [
        (tmp_path / "redone/z.marlbg-replay.json", _replay()),
        (tmp_path / "a.marlbg-replay.json", _replay("partial")),
        (tmp_path / "b.marlbg-replay.json", _replay()),
        (tmp_path / "c.marlbg-replay.json", _replay()),
    ]
    record = human_games.record_game(game, attempts, settings=SETTINGS)
    summary = ("outcome", "score_difference", "episode_length", "team_a_kills")
    assert [record[name] for name in summary] == ["win", 13, 212, 18]
    assert record["replay"] == "b.marlbg-replay.json"
    assert [a["replay"] for a in record["attempts"]] == [
        "redone/z.marlbg-replay.json", "a.marlbg-replay.json",
        "b.marlbg-replay.json", "c.marlbg-replay.json",
    ]  # fmt: skip
    assert [a["completion"] for a in record["attempts"]][1:] == [
        "partial", "complete", "complete",
    ]  # fmt: skip
    assert record["controllers"] == ["Manual"] * 5 + ["IPPO"] * 5
    assert record["classes"][4] == "Priest" and record["wall_seconds"] is None
    assert (record["seed"], record["player_class"]) == (7, "priest")
    assert json.loads((tmp_path / "game.json").read_text("utf-8")) == record
    header, row = (tmp_path / "full_metrics.csv").read_text("utf-8").splitlines()
    assert header.startswith("number,kind,index,map_id,swapped,side,replay,")
    assert header.endswith(",episode_length,agent_9_healing_done")
    # A metric that does not apply (NaN) is an empty cell, as in the Viewer's CSV.
    assert row.endswith(",212.0,")
    with pytest.raises(ValueError, match="no attempt played to its end"):
        human_games.record_game(game, attempts[:2], settings=SETTINGS)

    def unknown(replay: object, metrics: str) -> dict[str, float]:
        return _metrics()

    monkeypatch.setattr(marl_bgs, "replay_metrics", unknown)
    with pytest.raises(ValueError, match="result is unknown"):
        human_games.record_game(game, attempts, settings=SETTINGS)


def test_check_series_refuses_other_settings(tmp_path: Path) -> None:
    maps = [47, 48, 49, 50, 51]
    settings = SETTINGS | {"kinds": ["against"], "maps": maps, "games_per_kind": 10}
    human_games.check_series(tmp_path, settings | {"seed": 1})  # Nothing saved yet.
    human_games.save_series(tmp_path, settings)
    human_games.save_series(tmp_path, settings | {"seed": 1})  # Kept, not replaced.
    human_games.check_series(tmp_path, settings | {"games_per_kind": 5})
    other = settings | {"player_class": "mage", "seed": 1}
    with pytest.raises(ValueError, match="other player_class, seed"):
        human_games.check_series(tmp_path, other)
    with pytest.raises(ValueError, match="whole rounds of 5"):
        human_games.check_series(tmp_path, settings | {"games_per_kind": 7})
    # A part round would renumber games: 7 per kind is not the start of 10.
    assert (
        schedule(["against", "alongside"], maps, 7)
        != (schedule(["against", "alongside"], maps, 10)[:14])
    )


def _record(game: Game, **extra: Any) -> dict[str, Any]:  # noqa: ANN401
    return {
        "record_version": human_games.RECORD_VERSION, "number": game.number,
        "kind": game.kind, "index": game.index, "map_id": game.map_id,
        "side": game.side, "outcome": "win", "team_a_score": 20,
        "team_b_score": 5, "score_difference": 15, "team_a_kills": 18,
        "team_a_deaths": 5, "episode_length": 200, "wall_seconds": 60,
        "replay": "r.marlbg-replay.json", **extra,
    }  # fmt: skip


def _save(folder: Path, record: dict[str, Any]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "r.marlbg-replay.json").write_text("{}", "utf-8")
    (folder / "game.json").write_text(json.dumps(record), "utf-8")


def test_main_plays_measures_resumes_and_stops_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    games = schedule(["alongside"], [47, 48], 3)
    opponent = object()
    launches: list[dict[str, Any]] = []
    # Game 1 was played to its end on an earlier run but never measured.
    finished: dict[str, list[Any]] = {games[0].folder: [(tmp_path / "p1", _replay())]}

    def load_method(reference: str) -> object:
        return opponent

    def battle_client(**options: Any) -> int:  # noqa: ANN401
        launches.append(options)
        if len(launches) == 1:  # Game 2 is played to its end; game 3 is not.
            finished[games[1].folder] = [(tmp_path / "p2", _replay())]
        return 0

    def read_attempts(folder: Path) -> list[Any]:
        return finished.get(str(folder.relative_to(tmp_path)), [])

    def record_game(game: Game, attempts: object, **options: Any) -> dict[str, Any]:  # noqa: ANN401
        record = _record(game, played=options["played"])
        _save(tmp_path / game.folder, record)
        return record

    def check_replay(game: Game, loaded: object, **options: Any) -> None:  # noqa: ANN401
        assert options == {"seed": 11, "player_class": "priest"}

    def enter(prompt: str) -> str:
        return ""

    monkeypatch.setattr(marl_bgs, "load_method", load_method)
    monkeypatch.setattr(marl_bgs, "battle_client", battle_client)
    monkeypatch.setattr(human_games, "read_attempts", read_attempts)
    monkeypatch.setattr(human_games, "record_game", record_game)
    monkeypatch.setattr(human_games, "check_replay", check_replay)
    monkeypatch.setattr("builtins.input", enter)
    first = human_games.main.__wrapped__(series(tmp_path))
    # Game 1 was measured without a launch; game 3 ended early, so the stage
    # stopped after recording games 1 and 2.
    assert [r["number"] for r in first] == [1, 2]
    assert first[0]["played"] is None and first[1]["played"] is not None
    assert [(k["map_id"], k["spawn_mode"], k["seed"]) for k in launches] == [
        (48, "swapped", 11), (47, "swapped", 11),
    ]  # fmt: skip
    assert launches[0]["team_b"] is opponent and launches[0]["team_a"][4] == "manual"
    assert launches[0]["output_dir"] == tmp_path / "alongside/game-02-map-48-right"
    table = (tmp_path / "results.md").read_text(encoding="utf-8")
    assert "| Alongside | 2 | 2 · 0 · 0 | +15.0 | 18.0 | 5.0 |" in table
    assert len((tmp_path / "results.csv").read_text("utf-8").splitlines()) == 3

    # A later run reads every recorded game, even past a stop, and stops at Ctrl-C.
    _save(tmp_path / games[2].folder, _record(games[2]))
    (tmp_path / games[1].folder / "game.json").unlink()
    finished.pop(games[1].folder)

    def interrupt(prompt: str) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupt)
    launches.clear()
    again = human_games.main.__wrapped__(series(tmp_path))
    assert [r["number"] for r in again] == [1, 3] and launches == []
    with pytest.raises(ValueError, match="other seed"):
        human_games.main.__wrapped__(series(tmp_path, "seed=12"))
    with pytest.raises(ValueError, match="player_class must be"):
        human_games.main.__wrapped__(series(tmp_path, "player_class=paladin"))


def _loaded(reference: str) -> object:
    return object()


def test_main_saves_the_series_only_when_a_game_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(reference: str) -> object:
        raise FileNotFoundError(reference)

    def stop(prompt: str) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr(marl_bgs, "load_method", missing)
    with pytest.raises(FileNotFoundError):
        human_games.main.__wrapped__(series(tmp_path, "opponent.method=/typo"))
    monkeypatch.setattr(marl_bgs, "load_method", _loaded)
    monkeypatch.setattr("builtins.input", stop)
    # A wrong first try leaves no series.json, so the corrected run is accepted.
    assert human_games.main.__wrapped__(series(tmp_path, "seed=99")) == []
    assert human_games.main.__wrapped__(series(tmp_path)) == []
    assert not (tmp_path / "series.json").exists()


def test_main_checks_recorded_games_before_any_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    game = schedule(["alongside"], [47, 48], 3)[0]
    _save(tmp_path / game.folder, {"replay": "r.marlbg-replay.json"})
    events: list[str] = []

    def check_replay(game: Game, loaded: object, **options: Any) -> None:  # noqa: ANN401
        raise ValueError("did not use seed 11")

    def prompt(text: str) -> str:
        events.append("prompt")
        return ""

    def read_attempts(folder: Path) -> list[tuple[Path, Any]]:
        return [(folder / "r", _replay())]

    monkeypatch.setattr(marl_bgs, "load_method", _loaded)
    monkeypatch.setattr(human_games, "read_attempts", read_attempts)
    monkeypatch.setattr(human_games, "check_replay", check_replay)
    monkeypatch.setattr("builtins.input", prompt)
    with pytest.raises(ValueError, match="did not use seed 11"):
        human_games.main.__wrapped__(series(tmp_path))
    assert events == []


def test_older_records_are_measured_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    game = Game(1, "against", 1, 47, False)
    _save(tmp_path / game.folder, {"replay": "r.marlbg-replay.json", "outcome": "win"})
    calls: list[Game] = []

    def read_attempts(folder: Path) -> list[Any]:
        return [(folder / "r.marlbg-replay.json", _replay())]

    def check_replay(game: Game, loaded: object, **options: Any) -> None:  # noqa: ANN401
        calls.append(game)

    def record_game(game: Game, attempts: object, **options: Any) -> dict[str, Any]:  # noqa: ANN401
        calls.append(game)
        return _record(game)

    monkeypatch.setattr(human_games, "read_attempts", read_attempts)
    monkeypatch.setattr(human_games, "check_replay", check_replay)
    monkeypatch.setattr(human_games, "record_game", record_game)
    records = human_games.recorded_games([game], tmp_path, SETTINGS)
    assert calls == [game, game] and records[0]["score_difference"] == 15
    (tmp_path / game.folder / "r.marlbg-replay.json").unlink()
    with pytest.raises(ValueError, match="names a missing replay"):
        human_games.recorded_games([game], tmp_path, SETTINGS)
