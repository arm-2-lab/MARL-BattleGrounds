"""Train a PSRO population with the public training tools and choose its entrant.

``train_psro`` runs PSRO from Python with a built-in training method as its
oracle (tested with recurrent IPPO; the others are accepted but untested);
``load_psro_entrant`` loads the chosen mixture as one System. The
optional Hydra launcher ``python -m marl_battlegrounds.experiments.psro``
calls the same function with the Season 0 recipe and opponents.

PSRO (Policy-Space Response Oracles, Lanctot et al., NeurIPS 2017) grows a
population of team policies. Each new member is trained against a mixture of
the earlier ones, and the mixture's weights come from a Nash equilibrium of
the game in which each side picks one member. A run does this, in order:

1. **Seeded members.** The given members (the Season 0 training opponents in
   the launcher) play every pair once, so round 1 trains against a real Nash
   mixture.
2. **A round.** The best-response learner trains for ``round_env_steps``
   against the Nash mixture of the whole population: round 1 with ``train``,
   later rounds with ``extend_training``, which keeps the whole learner
   (critic and optimizer included) and adds the newest member as an opponent.
3. **Payoff games.** The round's final actor joins as a member and plays every
   other member: ``payoff_games`` games per pair on the training maps, spawn
   ends exchanged, scored win 1, draw 1/2, loss 0.
4. **Candidate.** The Nash mixture over PSRO's own members only (no seeded
   member, so the entrant holds no outside model or bot) is this round's
   candidate entrant. The newest member plays the validation panel (routine
   games). The candidate's routine score is computed from its members' own
   routine results, weighted by the Nash weights: the exact expected score of
   a mixture that picks one member per game, with no extra games.
5. **Entrant.** After the last round, the best candidates by routine point
   margin and the last round's candidate play confirmation games; the
   highest confirmed point margin wins, ties to the earlier round (the
   trainer's point-margin rule).

One budget, the oracle's ``total_env_steps``, counts every game step:
training, payoff and validation games (Rule 2.1). Rounds shrink so the run
ends inside it, and the last round takes what is left.

Files in ``output_dir``: ``settings.json`` (a later call with other settings
is refused); ``rounds/round-NN/`` (each round's training run); ``payoff.csv``
(one row per played pair); ``rounds.json`` (each round's weights, results and
step ledger); ``entrant.json`` (the chosen mixture). The pure parts,
``PairResult``, ``score_table``, ``nash_mixture`` and ``worst_case_score``, use
plain NumPy and no files. ``bundle_references`` reads an opponent bundle's
``pool.json``. No part of this module needs Hydra.
"""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize  # pyright: ignore[reportMissingTypeStubs]

from marl_battlegrounds.training._run_io import atomic_json

if TYPE_CHECKING:
    from marl_battlegrounds import Policy, System
    from marl_battlegrounds.training import TrainConfig

_TOLERANCE = 1e-9
# Every game ends by its 300th tick (make's max_steps), so this bounds the steps
# a planned game can still use when rounds are sized against the budget.
_MAX_GAME_STEPS = 300
_PAYOFF_FIELDS: tuple[str, ...] = (
    "member",
    "opponent",
    "wins",
    "draws",
    "losses",
    "game_steps",
    "seed",
)


class PairResult(NamedTuple):
    """The games one pair of members played against each other.

    Attributes
    ----------
    member : str
        Name of the first member, whose wins are counted.
    opponent : str
        Name of the second member. It must differ from ``member``.
    wins : int
        Games ``member`` won. Nonnegative.
    draws : int
        Games that ended in a draw (a timeout with neither team at the
        threshold). Nonnegative.
    losses : int
        Games ``member`` lost. Nonnegative. ``wins + draws + losses`` must be
        positive.
    """

    member: str
    opponent: str
    wins: int
    draws: int
    losses: int


def score_table(
    members: Sequence[str], results: Iterable[PairResult]
) -> NDArray[np.float64]:
    """Build the game-score table of a population from its played pairs.

    Parameters
    ----------
    members : sequence of str
        Member names in table order. Names must be distinct; at least one.
    results : iterable of PairResult
        Exactly one result for every unordered pair of different members, in
        either order. A single member needs none.

    Returns
    -------
    numpy.ndarray
        Shape ``(n, n)``, float64, where ``n = len(members)``. Entry ``[i, j]``
        is member i's mean game score against member j: (wins + draws / 2) /
        games, between 0 and 1. The diagonal is 1/2, and ``[j, i]`` is
        ``1 - [i, j]``.

    Raises
    ------
    ValueError
        No members, a repeated member name, a result naming an unknown member
        or a member against itself, negative counts or no games, a pair
        reported twice, or a pair with no result.
    """
    names = list(members)
    if not names:
        raise ValueError("score_table needs at least one member")
    if len(set(names)) != len(names):
        raise ValueError(f"member names must be distinct: {names}")
    index = {name: position for position, name in enumerate(names)}
    table = np.full((len(names), len(names)), np.nan)
    np.fill_diagonal(table, 0.5)
    for result in results:
        if result.member not in index or result.opponent not in index:
            raise ValueError(f"unknown member in {result}")
        if result.member == result.opponent:
            raise ValueError(f"a member cannot play itself: {result}")
        counts = (result.wins, result.draws, result.losses)
        if min(counts) < 0 or sum(counts) == 0:
            raise ValueError(
                f"counts must be nonnegative with at least one game: {result}"
            )
        row, column = index[result.member], index[result.opponent]
        if not np.isnan(table[row, column]):
            raise ValueError(
                f"pair reported twice: {result.member} and {result.opponent}"
            )
        score = (result.wins + result.draws / 2) / sum(counts)
        table[row, column], table[column, row] = score, 1.0 - score
    missing = np.argwhere(np.isnan(table))
    if missing.size:
        row, column = missing[0]
        raise ValueError(f"no result for {names[row]} against {names[column]}")
    return table


def _checked_table(scores: ArrayLike) -> NDArray[np.float64]:
    """Return a copy of a score table after checking the shape score_table makes."""
    table = np.array(scores, dtype=np.float64)
    if table.ndim != 2 or table.shape[0] != table.shape[1] or table.shape[0] == 0:
        raise ValueError(
            f"scores must be a nonempty square table, got shape {table.shape}"
        )
    if not np.all(np.isfinite(table)) or table.min() < 0 or table.max() > 1:
        raise ValueError("scores must be finite and between 0 and 1")
    if np.abs(table + table.T - 1).max() > _TOLERANCE:
        raise ValueError(
            "scores[i, j] + scores[j, i] must be 1, with 1/2 on the diagonal"
        )
    return table


def nash_mixture(scores: ArrayLike) -> NDArray[np.float64]:
    """Find member weights that no single member beats on average.

    The game: each team picks one member, and member i scores ``scores[i, j]``
    against member j. Because the table is symmetric in this sense (the two
    scores of a pair add up to 1), the game is fair: its value is exactly 1/2.
    A Nash mixture is a set of weights whose expected score is at least 1/2
    against every member.

    Parameters
    ----------
    scores : array_like
        Square table of shape ``(n, n)`` as returned by ``score_table``:
        finite values between 0 and 1, ``scores[i, j] + scores[j, i] == 1``
        within 1e-9, so 1/2 on the diagonal.

    Returns
    -------
    numpy.ndarray
        Shape ``(n,)``, float64, nonnegative weights summing to 1. A single
        member gets weight 1.

    Raises
    ------
    ValueError
        The table is empty, not square, not finite, outside [0, 1] or not
        filled both ways as described above.
    RuntimeError
        The linear program fails, which a valid table should never cause.

    Notes
    -----
    One linear program, solved by SciPy's HiGHS: maximize v subject to
    ``weights @ scores[:, j] >= v`` for every member j, weights nonnegative and
    summing to 1. When several Nash mixtures exist (for example two identical
    members), the solver returns one of them; the same table gives the same
    answer. Weights below 1e-12 become 0 before the final normalization.

    Examples
    --------
    Rock, paper, scissors gives every member the same weight:

    >>> from marl_battlegrounds.training.psro import nash_mixture
    >>> nash_mixture([[0.5, 0.0, 1.0], [1.0, 0.5, 0.0], [0.0, 1.0, 0.5]]).round(3)
    array([0.333, 0.333, 0.333])
    """
    table = _checked_table(scores)
    size = table.shape[0]
    if size == 1:
        return np.ones(1)
    # Variables: the n weights, then the value v. Minimize -v.
    objective = np.zeros(size + 1)
    objective[-1] = -1.0
    # For every column j: v - weights @ scores[:, j] <= 0.
    upper = np.hstack([-table.T, np.ones((size, 1))])
    equal = np.hstack([np.ones((1, size)), np.zeros((1, 1))])
    solved: Any = optimize.linprog(  # pyright: ignore[reportUnknownMemberType]
        objective,
        A_ub=upper,
        b_ub=np.zeros(size),
        A_eq=equal,
        b_eq=np.ones(1),
        bounds=[(0.0, None)] * size + [(None, None)],
        method="highs",
    )
    if solved.status != 0:
        raise RuntimeError(f"the Nash linear program failed: {solved.message}")
    weights = np.asarray(solved.x[:size], dtype=np.float64)
    weights[weights < 1e-12] = 0.0
    return weights / weights.sum()


def worst_case_score(scores: ArrayLike, weights: ArrayLike) -> float:
    """Return a mixture's expected game score against its hardest member.

    Parameters
    ----------
    scores : array_like
        Square table of shape ``(n, n)``, with the same rules as for
        ``nash_mixture``.
    weights : array_like
        Shape ``(n,)``, nonnegative, summing to 1 within 1e-9.

    Returns
    -------
    float
        ``min over j of weights @ scores[:, j]``, between 0 and 1. A Nash
        mixture scores 1/2; less means some member beats the mixture.

    Raises
    ------
    ValueError
        The table breaks ``nash_mixture``'s rules, or the weights have the
        wrong shape, a negative entry or a sum other than 1.
    """
    table = _checked_table(scores)
    mixture = np.asarray(weights, dtype=np.float64)
    if mixture.shape != (table.shape[0],) or mixture.min() < 0:
        raise ValueError(f"weights must be {table.shape[0]} nonnegative numbers")
    if abs(mixture.sum() - 1) > _TOLERANCE:
        raise ValueError("weights must sum to 1")
    return float((mixture @ table).min())


def bundle_references(bundle: Path, names: Sequence[str]) -> dict[str, str]:
    """Look up members of an opponent bundle by name.

    Parameters
    ----------
    bundle : Path
        Folder holding ``pool.json`` (schema 1: a ``members`` list of objects
        with ``name``, ``kind`` ``"actor"`` or ``"scripted"``, and for an actor
        a ``path`` relative to the bundle), as in the released Season 0
        training opponents.
    names : sequence of str
        Member names to look up, in the order wanted.

    Returns
    -------
    dict of str to str
        Each name, in the given order, mapped to what ``load_method`` takes:
        the absolute actor folder for an actor, the name itself for a scripted
        bot.

    Raises
    ------
    ValueError
        ``pool.json`` is missing or not schema 1, a name is not in it, or an
        actor's folder is missing or outside the bundle.
    """
    source = bundle / "pool.json"
    if not source.is_file():
        raise ValueError(f"no pool.json in the opponent bundle {bundle}")
    pool = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(pool, dict) or pool.get("schema_version") != 1:
        raise ValueError(f"{source} is not a schema 1 opponent pool")
    members = {
        str(member["name"]): member
        for member in cast(list[dict[str, Any]], pool.get("members", []))
    }
    root = bundle.resolve()
    references: dict[str, str] = {}
    for name in names:
        if name not in members:
            raise ValueError(f"{name!r} is not in {source}")
        member = members[name]
        if member.get("kind") == "scripted":
            references[name] = name
            continue
        folder = (root / str(member["path"])).resolve()
        if not folder.is_relative_to(root) or not folder.is_dir():
            raise ValueError(f"{name!r} has no actor folder inside {root}")
        references[name] = str(folder)
    return references


def _read_payoff(path: Path) -> list[dict[str, Any]]:
    """Read payoff.csv rows with integer counts; a missing file has none."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            {
                **row,
                **{key: int(row[key]) for key in _PAYOFF_FIELDS[2:]},
            }
            for row in csv.DictReader(handle)
        ]


def _write_payoff(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Rewrite payoff.csv whole; an interrupted write leaves the old file."""
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_PAYOFF_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def play_pair(
    system: System | Policy,
    opponent: System | Policy,
    *,
    games: int,
    maps: Sequence[int],
    seed: int,
    num_envs: int,
) -> tuple[int, int, int, int]:
    """Play one payoff pair and count the first method's results.

    Parameters
    ----------
    system, opponent : System or Policy
        Loaded methods; ``system`` plays Team A.
    games : int
        Positive even number of games; spawn ends are exchanged between the
        two games of each pair (``evaluate``'s paired mode).
    maps : sequence of int
        Map IDs, used in order, one map per pair of games, repeating.
    seed : int
        Evaluation seed for this pair.
    num_envs : int
        Lanes per batch. The batch keeps this size (``keep_batch_size``), so
        every pair of methods with the same two structures reuses one
        compiled program.

    Returns
    -------
    tuple of int
        ``(wins, draws, losses, game_steps)`` for ``system``; game_steps is the
        total length of the games in environment steps.

    Notes
    -----
    Plays games on the default JAX device with no metrics and writes no file.
    """
    from marl_battlegrounds import evaluate

    result = evaluate(
        system,
        opponent,
        num_episodes=games,
        maps=list(maps),
        spawn_mode="paired",
        seed=seed,
        num_envs=num_envs,
        keep_batch_size=True,
        metrics="none",
    )
    outcomes = [episode.outcome for episode in result.episodes]
    return (
        outcomes.count(1),
        outcomes.count(3),
        outcomes.count(2),
        sum(episode.episode_length for episode in result.episodes),
    )


def validate(
    system: System | Policy,
    panel: Mapping[str, System | Policy],
    *,
    maps: Sequence[int],
    seed_pairs: int,
    seed: int,
    num_envs: int,
) -> dict[str, Any]:
    """Play a validation panel and report the point margin with kills beside it.

    Parameters
    ----------
    system : System or Policy
        The method being validated, as Team A.
    panel : mapping of str to System or Policy
        Panel members by name, each played as Team B; at least one.
    maps : sequence of int
        Validation map IDs. Each panel member plays ``2 * seed_pairs`` games
        on every map, spawn ends exchanged.
    seed_pairs : int
        Positive pairs of games per map and panel member.
    seed : int
        Evaluation seed, the same for every member, so two validated methods
        meet the same games.
    num_envs : int
        Lanes per batch, kept fixed across calls.

    Returns
    -------
    dict
        ``games``; ``point_margin`` (mean Team A points minus Team B points per
        game, over the whole panel); ``points_for`` and ``points_against``;
        ``kills_for`` and ``kills_against`` (mean kills per game, from the
        priority metrics); ``game_steps``; and ``by_opponent``, the same means
        for each panel member.
    """
    from marl_battlegrounds import evaluate

    sums: dict[str, dict[str, float]] = {}
    steps = 0
    for name, opponent in panel.items():
        result = evaluate(
            system,
            opponent,
            num_episodes=2 * seed_pairs * len(maps),
            maps=list(maps),
            spawn_mode="paired",
            seed=seed,
            num_envs=num_envs,
            keep_batch_size=True,
            metrics="priority",
        )
        episodes = result.episodes
        sums[name] = {
            "games": len(episodes),
            "points_for": sum(e.team_a_score for e in episodes),
            "points_against": sum(e.team_b_score for e in episodes),
            "kills_for": float(np.sum(result.priority_metrics["team_a_kills"])),
            "kills_against": float(np.sum(result.priority_metrics["team_b_kills"])),
        }
        steps += sum(e.episode_length for e in episodes)

    def means(row: Mapping[str, float]) -> dict[str, Any]:
        """Turn summed counts into per-game means beside the point margin."""
        mean = {key: value / row["games"] for key, value in row.items()}
        mean["games"] = int(row["games"])
        return {"point_margin": mean["points_for"] - mean["points_against"], **mean}

    first = next(iter(sums.values()))
    total = {key: sum(row[key] for row in sums.values()) for key in first}
    return {
        **means(total),
        "game_steps": steps,
        "by_opponent": {name: means(row) for name, row in sums.items()},
    }


def _pairs_to_play(
    members: Sequence[str], done: Iterable[Mapping[str, Any]]
) -> list[tuple[str, str]]:
    """List the unordered member pairs with no payoff row, in member order."""
    played = {frozenset((row["member"], row["opponent"])) for row in done}
    return [
        (first, second)
        for index, first in enumerate(members)
        for second in members[index + 1 :]
        if frozenset((first, second)) not in played
    ]


def _round_steps(*, remaining: int, round_env_steps: int, num_envs: int) -> int:
    """Size the next round: the full round, or all that remains near the end.

    remaining is the budget left after reserving every later game. A round is
    round_env_steps; when less than 1.5 rounds remain, this round takes all of
    it, so no short final round is left over. The result is a multiple of
    num_envs; zero means no round fits.
    """
    steps = remaining if remaining < round_env_steps * 3 // 2 else round_env_steps
    return max(0, steps // num_envs * num_envs)


def train_psro(
    config: TrainConfig,
    *,
    output_dir: str | Path,
    seed_members: Mapping[str, str],
    panel: Mapping[str, str],
    round_env_steps: int = 20_054_016,
    payoff_games: int = 64,
    payoff_maps: Sequence[int] = tuple(range(42)),
    payoff_seed: int = 20261021,
    validation_seed: int = 20261022,
    confirmation_seed: int = 20261023,
    eval_num_envs: int = 64,
    shortlist: int = 2,
) -> dict[str, Any]:
    """Run or continue one PSRO population and choose its entrant.

    Parameters
    ----------
    config : TrainConfig
        The oracle: method, learner settings, reward, seed and ``num_envs``
        for every round, for example a Season 0 recipe. ``total_env_steps`` is
        the whole budget in game steps (training, payoff and validation
        games, Rule 2.1). ``validation_maps`` (all five validation maps when
        None), ``routine_seed_pairs`` and ``confirmation_seed_pairs`` set the
        validation games. Its own opponents, history, explicit saves and
        validation panel are replaced in every round, and its Random
        diagnostic games and extra actor exports are turned off, so every
        game step is counted and only each round's final actor is saved. A
        PQN-VDN oracle needs ``pqn.lr_linear_decay=False``: a decaying rate
        would end inside round 1, and later rounds could not continue it.
    output_dir : str or Path
        The run folder. A new or existing folder; an existing one must hold
        the same settings.
    seed_members : mapping of str to str
        Seeded population, name to any reference ``load_method`` takes (a
        built-in name, an actor folder, a model file or a released name). At
        least one. Names must not start with ``psro-``, which names PSRO's
        own members.
    panel : mapping of str to str
        Validation panel, name to reference, at least one member.
    round_env_steps : int, default=20_054_016
        Training transitions per round. A round shrinks to fit the budget, and
        the last round takes what is left once less than 1.5 rounds remain.
    payoff_games : int, default=64
        Positive even number of games per member pair, spawn ends exchanged.
    payoff_maps : sequence of int, default=range(42)
        Maps for payoff games (the training maps 0-41), at least one. Pair k
        starts where the pairs before it stopped (map index
        ``k * payoff_games / 2``, wrapping round), so the games spread evenly
        over the maps.
    payoff_seed : int, default=20261021
        Pair k (in the order played) uses ``payoff_seed + k``.
    validation_seed : int, default=20261022
        Seed of every routine validation, so every round meets the same games.
    confirmation_seed : int, default=20261023
        Seed of the confirmation games, which differ from the routine ones.
    eval_num_envs : int, default=64
        Lanes for payoff and validation games, kept fixed, so every pair of
        methods with the same two structures reuses one compiled program.
    shortlist : int, default=2
        Rounds confirmed besides the last one: the best by routine candidate
        point margin. Zero or more.

    Returns
    -------
    dict
        ``settings`` (as saved in ``settings.json``), ``rounds`` (the records
        in ``rounds.json``) and ``entrant`` (the record in ``entrant.json``:
        ``round``, ``members`` with their weights and actor folders relative
        to ``output_dir``, ``confirmation``, ``shortlist``,
        ``ledger_env_steps`` and ``budget_env_steps``).

    Raises
    ------
    ValueError
        The folder was started with other settings, a count is out of range,
        ``payoff_maps`` is empty, a seeded name starts with ``psro-``, the
        oracle is PQN-VDN with a decaying learning rate, no round fits the
        budget, or a round's folder stopped before its first save (delete
        that folder and call again). All but the last are raised before any
        file is written or game played.

    Notes
    -----
    Trains and plays games on the default JAX device. Each round is one
    training run in ``rounds/round-NN``: round 1 by ``train``, later rounds by
    ``extend_training`` from the round before, adding the newest member to
    the opponents. Every finished step is written before the next starts, so
    calling again with the same arguments continues a stopped run: recorded
    payoff pairs and validations are kept, an interrupted round resumes
    through the trainer's own resume, and a finished run plays no game.
    Member actors stay where the trainer saved them; seeded references are
    used as given.

    See Also
    --------
    load_psro_entrant : Load the chosen mixture as one System.
    """
    import marl_battlegrounds as marl_bgs
    from marl_battlegrounds import training
    from marl_battlegrounds.training.config import VALIDATION_MAPS, config_to_dict

    if payoff_games <= 0 or payoff_games % 2:
        raise ValueError("payoff_games must be a positive even number")
    if not seed_members or not panel:
        raise ValueError("seed_members and panel need at least one member each")
    if any(name.startswith("psro-") for name in seed_members):
        raise ValueError("seeded member names must not start with 'psro-'")
    if round_env_steps <= 0 or eval_num_envs <= 0:
        raise ValueError("round_env_steps and eval_num_envs must be positive")
    if shortlist < 0:
        raise ValueError("shortlist must be zero or more")
    if not payoff_maps:
        raise ValueError("payoff_maps needs at least one map")
    if config.pqn is not None and config.pqn.lr_linear_decay:
        raise ValueError(
            "A PQN-VDN oracle's decaying learning rate would end inside round 1, "
            "so later rounds could not continue it; set pqn.lr_linear_decay=False"
        )
    validation_maps = list(config.validation_maps or VALIDATION_MAPS)
    panel_games = 2 * len(validation_maps) * len(panel)

    def round_size(used: int, members: int, unplayed_pairs: int = 0) -> int:
        """Size the next round after reserving every later game at full length.

        used is the steps already spent; the new member will play members
        pairs, after unplayed_pairs recorded pairs (the seeded ones) are played.
        """
        later = (
            (unplayed_pairs + members) * payoff_games
            + config.routine_seed_pairs * panel_games
            + (shortlist + 1) * config.confirmation_seed_pairs * panel_games
        )
        return _round_steps(
            remaining=config.total_env_steps - used - later * _MAX_GAME_STEPS,
            round_env_steps=round_env_steps,
            num_envs=config.num_envs,
        )

    seeded_pairs = len(seed_members) * (len(seed_members) - 1) // 2
    if round_size(0, len(seed_members), seeded_pairs) == 0:
        raise ValueError("no PSRO round fits in total_env_steps")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Any] = {
        "config": config_to_dict(config),
        "seed_members": dict(seed_members),
        "panel": dict(panel),
        "round_env_steps": round_env_steps,
        "payoff_games": payoff_games,
        "payoff_maps": [int(map_id) for map_id in payoff_maps],
        "payoff_seed": payoff_seed,
        "validation_seed": validation_seed,
        "confirmation_seed": confirmation_seed,
        "eval_num_envs": eval_num_envs,
        "shortlist": shortlist,
    }
    settings_path = output_dir / "settings.json"
    if settings_path.exists():
        if json.loads(settings_path.read_text(encoding="utf-8")) != saved:
            raise ValueError(
                f"{output_dir} was started with other settings; use a new output_dir"
            )
    else:
        atomic_json(settings_path, saved)

    loaded: dict[str, System | Policy] = {}

    def method(reference: str) -> System | Policy:
        """Load a member once per process and reuse it for every game."""
        if reference not in loaded:
            loaded[reference] = marl_bgs.load_method(reference)
        return loaded[reference]

    payoff_path, rounds_path = output_dir / "payoff.csv", output_dir / "rounds.json"
    payoff = _read_payoff(payoff_path)
    rounds: list[dict[str, Any]] = (
        json.loads(rounds_path.read_text(encoding="utf-8"))
        if rounds_path.exists()
        else []
    )
    seeded = dict(seed_members)
    references = {
        **seeded,
        **{
            entry["member"]: str(output_dir / entry["actor"])
            for entry in rounds
            if "actor" in entry
        },
    }
    maps_ = saved["payoff_maps"]
    panel_systems = {name: method(reference) for name, reference in panel.items()}

    def play_missing(members: Sequence[str]) -> None:
        """Play and record every pair among members that has no payoff row."""
        for first, second in _pairs_to_play(members, payoff):
            seed = payoff_seed + len(payoff)
            start = len(payoff) * payoff_games // 2 % len(maps_)
            wins, draws, losses, steps = play_pair(
                method(references[first]),
                method(references[second]),
                games=payoff_games,
                maps=maps_[start:] + maps_[:start],
                seed=seed,
                num_envs=eval_num_envs,
            )
            payoff.append(
                dict(
                    zip(
                        _PAYOFF_FIELDS,
                        (first, second, wins, draws, losses, steps, seed),
                        strict=True,
                    )
                )
            )
            _write_payoff(payoff_path, payoff)

    def results() -> list[PairResult]:
        """Return the recorded payoff rows as PairResult values."""
        return [
            PairResult(r["member"], r["opponent"], r["wins"], r["draws"], r["losses"])
            for r in payoff
        ]

    def used_steps() -> int:
        """Count every game step so far: training, payoff and validation."""
        trained = max(
            (entry.get("training_env_steps", 0) for entry in rounds), default=0
        )
        validated = sum(
            record.get("game_steps", 0)
            for entry in rounds
            for record in entry.get("validation", {}).values()
        )
        return trained + sum(row["game_steps"] for row in payoff) + validated

    def save_rounds() -> None:
        """Write rounds.json with every round record."""
        atomic_json(rounds_path, rounds)

    play_missing(list(seeded))
    while True:
        entry: dict[str, Any]
        population = [*seeded, *(record["member"] for record in rounds)]
        if rounds and rounds[-1]["status"] in ("training", "trained"):
            entry = rounds[-1]
        else:
            if rounds and rounds[-1].get("last"):
                break
            # Round 1 always fits: the check before the seeded games reserved them.
            steps = round_size(used_steps(), len(population))
            if steps == 0:
                rounds[-1]["last"] = True
                save_rounds()
                break
            weights = nash_mixture(score_table(population, results()))
            number = len(rounds) + 1
            entry = {
                "round": number,
                "member": f"psro-{number:02d}",
                "run_dir": f"rounds/round-{number:02d}",
                "round_env_steps": steps,
                "opponent_weights": dict(
                    zip(population, weights.tolist(), strict=True)
                ),
                "status": "training",
            }
            rounds.append(entry)
            save_rounds()
        if entry["status"] == "training":
            run_dir = output_dir / entry["run_dir"]
            bindings = {name: references[name] for name in entry["opponent_weights"]}
            if (run_dir / "latest_checkpoint.json").exists():
                result = training.train(resume_from=run_dir, opponents=bindings)
            elif run_dir.exists() and any(run_dir.iterdir()):
                raise ValueError(
                    f"{run_dir} stopped before its first save; delete that "
                    "folder and call again"
                )
            elif entry["round"] == 1:
                round_config = replace(
                    config,
                    opponents=entry["opponent_weights"],
                    keep_past=0,
                    past_capture_interval=None,
                    history_capture_env_steps=(entry["round_env_steps"],),
                    checkpoint_env_steps=(),
                    total_env_steps=entry["round_env_steps"],
                    validation_panel=None,
                    validation_opponents=None,
                    random_diagnostic_seed_pairs=None,
                    random_initialization_result=None,
                    retain_all_actors=False,
                )
                result = training.train(
                    round_config, output_dir=run_dir, opponents=bindings
                )
            else:
                result = training.extend_training(
                    output_dir / rounds[-2]["checkpoint"],
                    additional_env_steps=entry["round_env_steps"],
                    output_dir=run_dir,
                    changes={"opponents": entry["opponent_weights"]},
                    opponents=bindings,
                )
            entry.update(
                status="trained",
                actor=os.path.relpath(result.final_actor, output_dir),
                checkpoint=os.path.relpath(result.final_checkpoint, output_dir),
                training_env_steps=result.completed_env_steps,
            )
            save_rounds()
        references[entry["member"]] = str(output_dir / entry["actor"])
        play_missing([*seeded, *(e["member"] for e in rounds)])
        own = [e["member"] for e in rounds]
        table = score_table(
            own, [r for r in results() if {r.member, r.opponent} <= set(own)]
        )
        candidate = dict(zip(own, nash_mixture(table).tolist(), strict=True))
        entry["candidate_weights"] = candidate
        validation = entry.setdefault("validation", {})
        if "member" not in validation:
            validation["member"] = validate(
                method(references[entry["member"]]),
                panel_systems,
                maps=validation_maps,
                seed_pairs=config.routine_seed_pairs,
                seed=validation_seed,
                num_envs=eval_num_envs,
            )
            save_rounds()
        validation["candidate"] = _mixture_record(
            candidate, {e["member"]: e["validation"]["member"] for e in rounds}
        )
        entry["status"] = "complete"
        entry["ledger_env_steps"] = used_steps()
        save_rounds()

    entrant = _choose_entrant(
        rounds,
        references,
        method,
        panel=panel_systems,
        maps=validation_maps,
        seed_pairs=config.confirmation_seed_pairs,
        seed=confirmation_seed,
        num_envs=eval_num_envs,
        shortlist=shortlist,
        save=save_rounds,
    )
    entrant["ledger_env_steps"] = used_steps()
    entrant["budget_env_steps"] = config.total_env_steps
    atomic_json(output_dir / "entrant.json", entrant)
    return {"settings": saved, "rounds": rounds, "entrant": entrant}


def _mixture_record(
    weights: Mapping[str, float], records: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Compute a mixture's validation record from its members' records.

    Parameters
    ----------
    weights : mapping of str to float
        Member name to Nash weight; nonnegative, summing to 1.
    records : mapping of str to mapping
        Member name to that member's own ``validate`` record. Every weighted
        member needs one, and all come from the same panel, maps, seed pairs
        and seed.

    Returns
    -------
    dict
        ``validate``'s per-game means (``point_margin``, ``points_for``,
        ``points_against``, ``kills_for``, ``kills_against``), overall and in
        ``by_opponent``, each the weighted mean of the members' values: the
        exact expected score, on the same games, of a System that picks one
        member per game with these weights (``pool``). ``games`` and
        ``game_steps`` are 0, because no game is played.
    """
    fields = ("point_margin", "points_for", "points_against")
    fields += ("kills_for", "kills_against")

    def mix(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
        """Weight one set of member rows field by field."""
        return {
            key: sum(weight * rows[name][key] for name, weight in weights.items())
            for key in fields
        }

    first = records[next(iter(weights))]["by_opponent"]
    return {
        "games": 0,
        **mix(records),
        "game_steps": 0,
        "by_opponent": {
            opponent: mix(
                {name: records[name]["by_opponent"][opponent] for name in weights}
            )
            for opponent in first
        },
    }


def _mixture(
    weights: Mapping[str, float],
    references: Mapping[str, str],
    method: Callable[[str], System | Policy],
) -> System:
    """Build a pool() System of the members with positive weight."""
    import marl_battlegrounds as marl_bgs

    return marl_bgs.pool(
        {
            method(references[name]): weight
            for name, weight in weights.items()
            if weight > 0
        }
    )


def _choose_entrant(
    rounds: list[dict[str, Any]],
    references: Mapping[str, str],
    method: Callable[[str], System | Policy],
    *,
    panel: Mapping[str, System | Policy],
    maps: Sequence[int],
    seed_pairs: int,
    seed: int,
    num_envs: int,
    shortlist: int,
    save: Callable[[], None],
) -> dict[str, Any]:
    """Confirm the best candidates and return the entrant record.

    The shortlist is the ``shortlist`` best rounds by routine candidate point
    margin (ties to the earlier round) plus the last round. Each plays the
    panel again with confirmation games, recorded in its round's
    ``validation["confirmation"]``. The highest confirmed point margin wins;
    ties go to the earlier round, as in the trainer's point-margin rule.
    """
    ranked = sorted(
        rounds,
        key=lambda e: (-e["validation"]["candidate"]["point_margin"], e["round"]),
    )
    chosen = {e["round"] for e in ranked[:shortlist]} | {rounds[-1]["round"]}
    for entry in rounds:
        if entry["round"] in chosen and "confirmation" not in entry["validation"]:
            entry["validation"]["confirmation"] = validate(
                _mixture(entry["candidate_weights"], references, method),
                panel,
                maps=maps,
                seed_pairs=seed_pairs,
                seed=seed,
                num_envs=num_envs,
            )
            save()
    best = min(
        (e for e in rounds if e["round"] in chosen),
        key=lambda e: (-e["validation"]["confirmation"]["point_margin"], e["round"]),
    )
    return {
        "round": best["round"],
        "members": {
            name: {
                "weight": weight,
                "actor": next(e["actor"] for e in rounds if e["member"] == name),
            }
            for name, weight in best["candidate_weights"].items()
            if weight > 0
        },
        "confirmation": best["validation"]["confirmation"],
        "shortlist": sorted(chosen),
    }


def load_psro_entrant(output_dir: str | Path) -> System:
    """Load a finished PSRO run's entrant as one System.

    Parameters
    ----------
    output_dir : str or Path
        The run folder holding ``entrant.json``.

    Returns
    -------
    System
        A ``pool()`` of the entrant's members with their Nash weights: each
        game draws one member and keeps it until the game resets. A single
        member comes back inside a one-member pool, so the result is always a
        System.

    Raises
    ------
    ValueError
        The folder has no ``entrant.json`` (the run has not finished).

    Notes
    -----
    An entrant is a weighted set of saved actors, not one actor folder, so it
    loads through this function: ``load_method`` and the command-line tools
    do not read ``entrant.json``.
    """
    import marl_battlegrounds as marl_bgs

    folder = Path(output_dir).expanduser().resolve()
    path = folder / "entrant.json"
    if not path.is_file():
        raise ValueError(f"{folder} has no entrant.json; the PSRO run has not finished")
    entrant = json.loads(path.read_text(encoding="utf-8"))
    return marl_bgs.pool(
        {
            str(folder / member["actor"]): float(member["weight"])
            for member in entrant["members"].values()
        }
    )
