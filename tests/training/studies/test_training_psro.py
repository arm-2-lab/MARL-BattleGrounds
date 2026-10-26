"""Check PSRO's bundle lookup, round sizing, score table and Nash meta-solver.

Contracts: bundle_references maps a pool.json name to its absolute actor
folder or a scripted name to itself, and refuses a missing pool.json, another
schema, an unknown name and an actor folder that is missing or that exists
outside the bundle; train_psro refuses a negative shortlist, empty
payoff_maps, a PQN-VDN oracle whose learning rate decays and a budget with no
room for one round, before it writes a file or plays a game; _round_steps
gives a full round, all that remains below 1.5 rounds, rounded down to whole
environment batches, and zero when nothing fits.
score_table fills each played pair both ways (score of j against i is 1 minus
i against j, draws count 1/2) with 1/2 on the diagonal, accepts a pair in
either order, and refuses no members, repeated names, unknown members, a
member against itself, negative counts, a pair with no games, a pair reported
twice and a missing pair. nash_mixture gives weight 1 to a single member,
equal weights in rock-paper-scissors, all weight to a member that beats every
other, weight 0 to a dominated member, and on random fair tables weights that
score at least 1/2 against every member (the Nash property, checked
independently of the solver); with a preference it picks the preferred member
of two that always draw, leaves a unique Nash mixture unchanged, and on random
fair tables still meets the Nash property with a preference value at least
that of the mixture found without one; it refuses tables that are empty, not
square, outside [0, 1] or not filled both ways, and a preference of the wrong
length or with a value that is not finite. worst_case_score reports the
mixture's lowest expected score over the members and refuses weights of the
wrong shape, negative or not summing to 1. Plain NumPy and small files; no
game is played.
"""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from marl_battlegrounds.training import TrainConfig, psro
from marl_battlegrounds.training.psro import (
    PairResult,
    _round_steps,  # pyright: ignore[reportPrivateUsage]
    bundle_references,
    nash_mixture,
    score_table,
    worst_case_score,
)

ROCK_PAPER_SCISSORS = [[0.5, 0.0, 1.0], [1.0, 0.5, 0.0], [0.0, 1.0, 0.5]]


def test_score_table_fills_both_ways() -> None:
    table = score_table(
        ["a", "b", "c"],
        [
            PairResult("a", "b", wins=3, draws=2, losses=5),
            PairResult("c", "a", wins=0, draws=4, losses=0),
            PairResult("b", "c", wins=10, draws=0, losses=0),
        ],
    )
    expected = np.array([[0.5, 0.4, 0.5], [0.6, 0.5, 1.0], [0.5, 0.0, 0.5]])
    np.testing.assert_allclose(table, expected)
    np.testing.assert_array_equal(score_table(["solo"], []), [[0.5]])


@pytest.mark.parametrize(
    ("members", "results", "message"),
    [
        ([], [], "at least one member"),
        (["a", "a"], [], "distinct"),
        (["a", "b"], [PairResult("a", "x", 1, 0, 0)], "unknown member"),
        (["a", "b"], [PairResult("a", "a", 1, 0, 0)], "cannot play itself"),
        (["a", "b"], [PairResult("a", "b", -1, 2, 0)], "nonnegative"),
        (["a", "b"], [PairResult("a", "b", 0, 0, 0)], "at least one game"),
        (
            ["a", "b"],
            [PairResult("a", "b", 1, 0, 0), PairResult("b", "a", 1, 0, 0)],
            "reported twice",
        ),
        (["a", "b", "c"], [PairResult("a", "b", 1, 0, 0)], "no result"),
    ],
)
def test_score_table_refuses_bad_results(
    members: list[str], results: list[PairResult], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        score_table(members, results)


def test_nash_mixture_known_games() -> None:
    np.testing.assert_array_equal(nash_mixture([[0.5]]), [1.0])
    np.testing.assert_allclose(
        nash_mixture(ROCK_PAPER_SCISSORS), [1 / 3] * 3, atol=1e-9
    )
    # Member 1 beats everyone, so it takes all the weight.
    dominant = [[0.5, 0.2, 0.6], [0.8, 0.5, 0.9], [0.4, 0.1, 0.5]]
    np.testing.assert_allclose(nash_mixture(dominant), [0.0, 1.0, 0.0], atol=1e-9)
    # A fourth member that loses to rock, paper and scissors alike gets no weight.
    weak = [
        [0.5, 0.0, 1.0, 0.9],
        [1.0, 0.5, 0.0, 0.9],
        [0.0, 1.0, 0.5, 0.9],
        [0.1, 0.1, 0.1, 0.5],
    ]
    weights = nash_mixture(weak)
    assert weights[3] == 0.0
    np.testing.assert_allclose(weights[:3], [1 / 3] * 3, atol=1e-9)


def test_nash_mixture_meets_the_nash_property_on_random_fair_tables() -> None:
    generator = np.random.default_rng(20261020)
    for size in (2, 3, 5, 8, 12):
        for _ in range(20):
            upper = generator.uniform(size=(size, size))
            # Rounding to 1/48 mimics 48-game pairs, with ties and repeated rows.
            table = np.triu(np.round(upper * 48) / 48, k=1)
            table = table + np.tril(1 - table.T, k=-1)
            np.fill_diagonal(table, 0.5)
            weights = nash_mixture(table)
            assert weights.shape == (size,)
            assert weights.min() >= 0.0
            assert abs(weights.sum() - 1.0) < 1e-12
            assert (weights @ table).min() >= 0.5 - 1e-9
            assert worst_case_score(table, weights) == pytest.approx(
                (weights @ table).min()
            )
            preference = generator.normal(size=size)
            preferred = nash_mixture(table, preference=preference)
            assert preferred.min() >= 0.0
            assert abs(preferred.sum() - 1.0) < 1e-12
            assert (preferred @ table).min() >= 0.5 - 1e-8
            assert preferred @ preference >= weights @ preference - 1e-7


def test_nash_mixture_preference_picks_among_tied_mixtures() -> None:
    draws = [[0.5, 0.5], [0.5, 0.5]]
    np.testing.assert_allclose(
        nash_mixture(draws, preference=[-19.2, -8.0]), [0.0, 1.0], atol=1e-9
    )
    np.testing.assert_allclose(
        nash_mixture(draws, preference=[-8.0, -19.2]), [1.0, 0.0], atol=1e-9
    )
    np.testing.assert_allclose(
        nash_mixture(ROCK_PAPER_SCISSORS, preference=[9.0, 0.0, -9.0]),
        [1 / 3] * 3,
        atol=1e-9,
    )
    for preference in ([1.0], [0.0, np.inf]):
        with pytest.raises(ValueError, match="2 finite values"):
            nash_mixture(draws, preference=preference)


@pytest.mark.parametrize(
    ("scores", "message"),
    [
        (np.zeros((0, 0)), "nonempty square"),
        (np.full((2, 3), 0.5), "nonempty square"),
        ([[0.5, 1.5], [-0.5, 0.5]], "between 0 and 1"),
        ([[0.5, np.nan], [0.5, 0.5]], "finite"),
        ([[0.5, 0.7], [0.7, 0.5]], "must be 1"),
    ],
)
def test_nash_mixture_refuses_bad_tables(scores: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        nash_mixture(scores)  # pyright: ignore[reportArgumentType]


def test_worst_case_score() -> None:
    assert worst_case_score(ROCK_PAPER_SCISSORS, [1.0, 0.0, 0.0]) == 0.0
    assert worst_case_score(ROCK_PAPER_SCISSORS, [1 / 3] * 3) == pytest.approx(0.5)
    for weights, message in (
        ([1.0, 0.0], "3 nonnegative"),
        ([1.5, -0.5, 0.0], "3 nonnegative"),
        ([0.5, 0.2, 0.2], "sum to 1"),
    ):
        with pytest.raises(ValueError, match=message):
            worst_case_score(ROCK_PAPER_SCISSORS, weights)


def test_bundle_references(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    (bundle / "models" / "fixed").mkdir(parents=True)
    (tmp_path / "elsewhere").mkdir()
    pool = {
        "schema_version": 1,
        "members": [
            {"name": "fixed", "kind": "actor", "path": "models/fixed"},
            {"name": "outside", "kind": "actor", "path": "../elsewhere"},
            {"name": "gone", "kind": "actor", "path": "models/gone"},
            {"name": "tdm-alpha", "kind": "scripted"},
        ],
    }
    (bundle / "pool.json").write_text(json.dumps(pool), encoding="utf-8")
    assert bundle_references(bundle, ["tdm-alpha", "fixed"]) == {
        "tdm-alpha": "tdm-alpha",
        "fixed": str((bundle / "models" / "fixed").resolve()),
    }
    for name, message in (
        ("missing", "is not in"),
        ("outside", "no actor folder"),
        ("gone", "no actor folder"),
    ):
        with pytest.raises(ValueError, match=message):
            bundle_references(bundle, [name])
    (bundle / "pool.json").write_text('{"schema_version": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="schema 1"):
        bundle_references(bundle, ["fixed"])
    with pytest.raises(ValueError, match=r"no pool\.json"):
        bundle_references(bundle / "models", ["fixed"])


def test_train_psro_refuses_before_any_file_or_game(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_games(*args: object, **kwargs: object) -> None:
        raise AssertionError("a game was played")

    monkeypatch.setattr(psro, "play_pair", no_games)
    oracle = TrainConfig(method="ippo", total_env_steps=20_000_000)
    cases: list[tuple[TrainConfig, dict[str, Any], str]] = [
        (oracle, {"shortlist": -1}, "shortlist"),
        (oracle, {"payoff_maps": ()}, "payoff_maps"),
        (TrainConfig(method="pqn_vdn"), {}, "lr_linear_decay"),
        (replace(oracle, total_env_steps=100_000), {}, "no PSRO round fits"),
    ]
    for config, extra, message in cases:
        with pytest.raises(ValueError, match=message):
            psro.train_psro(
                config,
                output_dir=tmp_path / "run",
                seed_members={"random": "random", "tdm-alpha": "tdm-alpha"},
                panel={"tdm-beta": "tdm-beta"},
                **extra,
            )
    assert not (tmp_path / "run").exists()


def test_round_steps() -> None:
    assert _round_steps(remaining=10_000, round_env_steps=4096, num_envs=32) == 4096
    assert _round_steps(remaining=6000, round_env_steps=4096, num_envs=32) == 5984
    assert _round_steps(remaining=31, round_env_steps=4096, num_envs=32) == 0
    assert _round_steps(remaining=-5, round_env_steps=4096, num_envs=32) == 0
