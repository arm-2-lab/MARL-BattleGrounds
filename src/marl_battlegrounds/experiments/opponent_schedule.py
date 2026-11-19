"""Switch the training opponents' shares at fixed training steps.

make_opponent_schedule returns a pure matchmaking rule for train,
extend_training and declared studies (factory
``marl_battlegrounds.experiments.opponent_schedule:make_opponent_schedule``).
For example, a run can train against later saves of the same opponents as it
trains longer. The rule has its own file on purpose: a declared study's resume
receipt records the SHA-256 of the file that defines its matchmaking factory,
so any edit to this file stops ordinary resume of a run that uses it. Keeping
it apart lets other callbacks change while such runs are live. Importing this
module starts no numerical backend.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast


def make_opponent_schedule(
    stages: Sequence[Mapping[str, Any]],
) -> Callable[[dict[str, Any], int], dict[str, object]]:
    """Return a matchmaking rule that switches the opponents' shares at set steps.

    Pass it as ``train(..., matchmaking=rule)``, or in a declared study as
    ``runtime.matchmaking`` with ``factory:
    marl_battlegrounds.experiments.opponent_schedule:make_opponent_schedule``
    and ``kwargs: {stages: [...]}``.

    Parameters
    ----------
    stages : sequence of mapping
        In order, each ``{"from_env_steps": int, "opponents": {name: share}}``.
        The first starts at 0 and each later one at a larger step, counted in
        completed training transitions (``env_steps``). The shares follow
        ``TrainConfig.opponents``: names are ``"self"``, ``"past"`` and declared
        members; values are finite, nonnegative and relative, with a positive
        total. A name a stage leaves out has share 0 in that stage, so each
        stage repeats ``"self"`` and ``"past"`` if it wants them.

    Returns
    -------
    callable
        ``rule(stats, env_steps)``, returning ``{"opponents": shares}`` for the
        last stage whose ``from_env_steps`` is at most ``env_steps``. The shares
        name every member any stage names, with 0 for those outside the
        current stage, so the trainer refuses an unknown name at its first
        call, at step 0, not when that name's stage starts. It ignores
        ``stats``, keeps no state and returns a new dict each call.

    Raises
    ------
    ValueError
        No stage; a stage that is not a mapping with exactly the two keys; a
        first stage that does not start at 0, or later steps that are not ints
        in increasing order; shares that are not a mapping or that
        ``TrainConfig.opponents`` refuses (empty, negative, not finite, zero
        total, blank name).
    TypeError
        A share is not a real number.

    Notes
    -----
    The trainer calls the rule before every update, including the first and
    the first after a resume, so a stage starts at the first update that
    begins at or after its step, and a resumed run gets the same shares as an
    uninterrupted one. A change affects new games only: a game under way
    finishes against its old opponent. The first ``num_envs`` games start
    before the first call, from ``TrainConfig.opponents``, so set that to the
    first stage's shares. The rule cannot add members: declare every member of
    every stage in ``TrainConfig.opponents`` (0 outside the first stage) and
    bind it through ``train``'s ``opponents`` or the study's preparation.
    Resume a run through the same route it started on. A declared study saves
    a receipt and refuses a missing or changed rule; a plain Python resume
    without the rule keeps the last shares it saved.

    Examples
    --------
    >>> from marl_battlegrounds.experiments.opponent_schedule import (
    ...     make_opponent_schedule,
    ... )
    >>> rule = make_opponent_schedule(
    ...     [
    ...         {"from_env_steps": 0, "opponents": {"self": 0.8, "early": 0.2}},
    ...         {"from_env_steps": 40_108_032, "opponents": {"self": 0.8, "late": 0.2}},
    ...     ]
    ... )
    >>> rule({}, 0)
    {'opponents': {'self': 0.8, 'early': 0.2, 'late': 0.0}}
    >>> rule({}, 40_108_032)
    {'opponents': {'self': 0.8, 'early': 0.0, 'late': 0.2}}
    """
    from marl_battlegrounds.training.config import opponent_selection_names

    schedule: list[tuple[int, Mapping[str, float]]] = []
    for stage in stages:
        if not isinstance(cast(object, stage), Mapping) or set(stage) != {
            "from_env_steps",
            "opponents",
        }:
            raise ValueError("Each stage needs exactly from_env_steps and opponents")
        start, shares = stage["from_env_steps"], stage["opponents"]
        if type(start) is not int or start <= (schedule[-1][0] if schedule else -1):
            raise ValueError("Stage steps must be ints in increasing order")
        if not schedule and start != 0:
            raise ValueError("The first stage must start at from_env_steps 0")
        if not isinstance(shares, Mapping):
            raise ValueError("A stage's opponents must map names to shares")
        checked = cast(Mapping[str, float], shares)
        opponent_selection_names(checked)
        schedule.append((start, checked))
    if not schedule:
        raise ValueError("An opponent schedule needs at least one stage")
    names = list(dict.fromkeys(name for _, shares in schedule for name in shares))
    full = [
        (start, {name: float(shares.get(name, 0.0)) for name in names})
        for start, shares in schedule
    ]

    def rule(stats: dict[str, Any], env_steps: int) -> dict[str, object]:
        """Return every scheduled name's share in the stage env_steps reached."""
        del stats
        shares = next(s for start, s in reversed(full) if start <= env_steps)
        return {"opponents": dict(shares)}

    return rule
