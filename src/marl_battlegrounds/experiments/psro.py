"""PSRO for two-team games: the score table and the Nash meta-solver.

PSRO (Policy-Space Response Oracles, Lanctot et al., NeurIPS 2017) grows a
population of team policies. Each new member is trained against a mixture of
the earlier ones, and the mixture's weights come from a Nash equilibrium of
the game in which each side picks one member. This module owns that game:

- ``PairResult`` is one played pair: how often one member beat, drew with
  and lost to another.
- ``score_table`` turns the played pairs into a square table of game scores
  (win 1, draw 1/2, loss 0). A member's score against itself is 1/2, and the
  table is filled both ways from the same games, so the score of j against i
  is 1 minus the score of i against j.
- ``nash_mixture`` finds weights over the members that no single member can
  beat on average, by one linear program (``scipy.optimize.linprog``).
- ``worst_case_score`` reports how a mixture does against its hardest member.

Everything here is plain NumPy on the host, with no JAX and no files.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize  # pyright: ignore[reportMissingTypeStubs]

_TOLERANCE = 1e-9


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

    >>> from marl_battlegrounds.experiments.psro import nash_mixture
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
