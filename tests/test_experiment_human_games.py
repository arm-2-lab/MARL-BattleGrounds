"""Check the human_games experiment stage without a person or a browser.

Contracts: the packaged settings compose to two parts, three games against
recurrent IPPO (the Season 0 winner) on maps 47 to 49 from seed 20261007, then
three games as the Mage beside recurrent IPPO against PQN-VDN on the same maps
from seed 20261013, with models_dir filling the method paths; schedule()
numbers games through the whole series, gives each part's game N the part's
seed plus N - 1 and Team A on the left when N is odd, and refuses no parts, no
maps, an unknown kind, an alongside part without partner or class, and a seed
outside 0 through 2**32 - 1; against games give the person all of Team A and
alongside games only the player's class, with the partner elsewhere, and an
unknown class fails; check_replay() names every way a replay differs from its
scheduled game; record_game() counts the first finished attempt, refuses a
game with no finished attempt or an unknown result, and writes
full_metrics.csv and game.json with the game's seed and Systems, every
attempt, Team A's result and summary columns; main() refuses names missing
from systems and a folder started with other settings, measures a finished
but unmeasured game without playing it again, opens the BattleClient with each
game's map, seed, spawn_mode and controllers, stops cleanly at Ctrl-C or after
an unfinished game, reads recorded games on a later run, and writes
results.csv and results.md. Fakes replace the BattleClient, replays and
metrics, so no JAX program is compiled; tests/test_replay_metrics.py runs
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

AGAINST: dict[str, Any] = {
    "kind": "against", "opponent": "Recurrent IPPO", "maps": [47, 48], "seed": 7,
}  # fmt: skip
ALONGSIDE: dict[str, Any] = {
    "kind": "alongside", "opponent": "PQN-VDN", "partner": "Recurrent IPPO",
    "player_class": "mage", "maps": [47], "seed": 20,
}  # fmt: skip


def composed(*overrides: str) -> DictConfig:
    with initialize_config_module(
        config_module="marl_battlegrounds.experiments.configs", version_base="1.3"
    ):
        return compose(config_name="human_games", overrides=list(overrides))


def test_settings_compose_the_two_parts() -> None:
    config = composed("models_dir=/models", "output_dir=out")
    against, alongside = config.parts
    assert (against.kind, against.opponent, list(against.maps), against.seed) == (
        "against", "Recurrent IPPO", [47, 48, 49], 20261007,
    )  # fmt: skip
    assert (alongside.kind, alongside.opponent, alongside.partner) == (
        "alongside", "PQN-VDN", "Recurrent IPPO",
    )  # fmt: skip
    assert (alongside.player_class, list(alongside.maps), alongside.seed) == (
        "mage", [47, 48, 49], 20261013,
    )  # fmt: skip
    methods = {entry.name: str(entry.method) for entry in config.systems}
    assert methods["Recurrent IPPO"].startswith("/models/250m/study/cases/ippo-")
    assert methods["PQN-VDN"].startswith("/models/250m/study/cases/pqn_vdn-")


def test_schedule_numbers_parts_alternates_ends_and_seeds() -> None:
    games = schedule([AGAINST, ALONGSIDE | {"maps": [47, 48, 49]}])
    assert [(g.number, g.kind, g.map_id, g.side, g.seed) for g in games] == [
        (1, "against", 47, "left", 7), (2, "against", 48, "right", 8),
        (3, "alongside", 47, "left", 20), (4, "alongside", 48, "right", 21),
        (5, "alongside", 49, "left", 22),
    ]  # fmt: skip
    assert games[0].partner is None and games[2].partner == "Recurrent IPPO"
    assert games[3].folder == "game-04-alongside-map-48-right"
    bad: tuple[tuple[list[dict[str, Any]], str], ...] = (
        ([], "at least one part"),
        ([AGAINST | {"maps": []}], "at least one map"),
        ([AGAINST | {"kind": "beside"}], "kind must be"),
        ([ALONGSIDE | {"partner": None}], "needs a partner"),
        ([AGAINST | {"seed": 2**32 - 1}], "outside 0 through 2\\*\\*32 - 1"),
    )
    for parts, message in bad:
        with pytest.raises(ValueError, match=message):
            schedule(parts)


def test_team_a_gives_the_person_all_or_one_slot() -> None:
    partner: Any = object()
    against, alongside = schedule([AGAINST, ALONGSIDE])[1:]
    assert human_games.team_a_controllers(against, None) == "manual"
    assert human_games.team_a_controllers(alongside, partner) == [
        "manual", partner, partner, partner, partner,
    ]  # fmt: skip
    assert human_games.person_slots(alongside) == (0,)
    paladin = schedule([ALONGSIDE | {"player_class": "paladin"}])[0]
    with pytest.raises(ValueError, match="player_class must be"):
        human_games.person_slots(paladin)


def _replay(
    state: str = "complete",
    *,
    map_id: str = "48",
    spawn: str = "1",
    seed: int = 8,
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
    game = schedule([AGAINST])[1]  # Map 48, Team A on the right, seed 8.
    human_games.check_replay(game, _replay())
    wrong = _replay("partial", map_id="47", spawn="0", seed=7, manual=(1,))
    with pytest.raises(ValueError) as caught:
        human_games.check_replay(game, wrong)
    for part in ("not played to its end", "is not map 48", "on the right",
                 "did not use seed 8", "person in slots (0, 1, 2, 3, 4)"):  # fmt: skip
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
    game = schedule([AGAINST])[1]
    attempts = [
        (tmp_path / "a.marlbg-replay.json", _replay("partial")),
        (tmp_path / "b.marlbg-replay.json", _replay()),
        (tmp_path / "c.marlbg-replay.json", _replay()),
    ]
    record = human_games.record_game(game, attempts)
    summary = ("outcome", "score_difference", "episode_length", "team_a_kills")
    assert [record[name] for name in summary] == ["win", 13, 212, 18]
    assert record["replay"] == "b.marlbg-replay.json"
    assert [a["completion"] for a in record["attempts"]] == [
        "partial", "complete", "complete",
    ]  # fmt: skip
    assert record["controllers"] == ["Manual"] * 5 + ["IPPO"] * 5
    assert record["classes"][0] == "Mage" and record["wall_seconds"] is None
    assert (record["seed"], record["kind"], record["opponent"]) == (
        8, "against", "Recurrent IPPO",
    )  # fmt: skip
    assert json.loads((tmp_path / "game.json").read_text("utf-8")) == record
    header, row = (tmp_path / "full_metrics.csv").read_text("utf-8").splitlines()
    assert header.startswith("number,kind,map_id,side,replay,replay_digest,team_a_win,")
    # A metric that does not apply (NaN) is an empty cell, as in the Viewer's CSV.
    assert row.endswith(",212.0,")
    with pytest.raises(ValueError, match="no attempt played to its end"):
        human_games.record_game(game, attempts[:1])

    def unknown(replay: object, metrics: str) -> dict[str, float]:
        return _metrics()

    monkeypatch.setattr(marl_bgs, "replay_metrics", unknown)
    with pytest.raises(ValueError, match="result is unknown"):
        human_games.record_game(game, attempts)


def _record(game: Game) -> dict[str, Any]:
    return {
        "number": game.number, "kind": game.kind, "map_id": game.map_id,
        "side": game.side, "opponent": game.opponent, "outcome": "win",
        "team_a_score": 20, "team_b_score": 5, "score_difference": 15,
        "team_a_kills": 18, "team_a_deaths": 5, "episode_length": 200,
        "wall_seconds": 60, "replay": "r.marlbg-replay.json",
    }  # fmt: skip


def series(tmp_path: Path, *extra: str) -> DictConfig:
    return composed(
        "models_dir=/models", f"output_dir={tmp_path}",
        "parts.0.maps=[47]", "parts.0.seed=11", "parts.1.maps=[48,49]",
        "parts.1.seed=30", *extra,
    )  # fmt: skip


def test_main_plays_measures_continues_and_stops_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    games = schedule(series(tmp_path).parts)  # pyright: ignore[reportArgumentType]
    loaded: dict[str, object] = {}
    launches: list[dict[str, Any]] = []
    # Game 1 was played to its end on an earlier run but never measured.
    finished: dict[str, list[Any]] = {games[0].folder: [(tmp_path / "p1", _replay())]}

    def load_method(reference: str) -> object:
        return loaded.setdefault(reference, object())

    def battle_client(**options: Any) -> int:  # noqa: ANN401
        launches.append(options)
        if len(launches) == 1:  # Game 2 is played to its end; game 3 is not.
            finished[games[1].folder] = [(tmp_path / "p2", _replay())]
        return 0

    def read_attempts(folder: Path) -> list[Any]:
        return finished.get(folder.name, [])

    def record_game(game: Game, attempts: object, **options: Any) -> dict[str, Any]:  # noqa: ANN401
        record = _record(game) | {"played": options["played"]}
        (tmp_path / game.folder).mkdir(exist_ok=True)
        (tmp_path / game.folder / "game.json").write_text(json.dumps(record), "utf-8")
        return record

    def check_replay(game: Game, replay: object) -> None:
        assert game.number in (1, 2)

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
        (48, "default", 30), (49, "swapped", 31),
    ]  # fmt: skip
    ippo = next(value for key, value in loaded.items() if "/ippo-" in key)
    pqn = next(value for key, value in loaded.items() if "/pqn_vdn-" in key)
    assert launches[0]["team_b"] is pqn
    assert launches[0]["team_a"] == ["manual", ippo, ippo, ippo, ippo]
    assert launches[0]["output_dir"] == tmp_path / "game-02-alongside-map-48-left"
    table = (tmp_path / "results.md").read_text(encoding="utf-8")
    assert "| Against | 1 | 1 · 0 · 0 | +15.0 | 18.0 | 5.0 |" in table
    assert "| Alongside | 1 | 1 · 0 · 0 |" in table
    assert len((tmp_path / "results.csv").read_text("utf-8").splitlines()) == 3

    # A later run reads the recorded games and stops at Ctrl-C.
    def interrupt(prompt: str) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupt)
    launches.clear()
    again = human_games.main.__wrapped__(series(tmp_path))
    assert [r["number"] for r in again] == [1, 2] and launches == []
    with pytest.raises(ValueError, match="other settings"):
        human_games.main.__wrapped__(series(tmp_path, "parts.1.seed=12"))
    with pytest.raises(ValueError, match="not in systems"):
        human_games.main.__wrapped__(series(tmp_path, "parts.1.opponent=Mine"))
