"""Evaluate any valid System at exact packaged scenario starts.

Install the hydra extra, then run::

    python examples/evaluate/scenario_evaluation.py system=tdm-alpha scenario_ids=[1] \
        output_dir=artifacts/scenario-evaluation metrics=full replay_episodes=[1]

Python callers can pass live Systems or Policies to run(). The public scenario
loader owns starts; the public evaluator owns actions, memory, results and
recovery. No training, horizon changes or client application is involved.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from marl_battlegrounds import Policy, System
    from marl_battlegrounds.environment import MetricMode
    from marl_battlegrounds.evaluation.results import EvaluationResult


def run(
    *,
    system: str | System | Policy = "tdm-alpha",
    scenario_ids: tuple[int, ...] | list[int] = (1,),
    opponent: str | System | Policy | None = None,
    num_episodes: int = 1,
    seed: int | None = None,
    num_envs: int = 32,
    metrics: MetricMode | None = None,
    replay_episodes: tuple[int, ...] | list[int] | None = None,
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
) -> tuple[EvaluationResult, ...]:
    """Run each exact scenario and return its ordinary evaluation result.

    Parameters
    ----------
    system : str or System or Policy, default='tdm-alpha'
        Team A method. Strings use load_method: a built-in policy, saved actor
        directory or trusted module:function factory. Live methods keep their
        normal information limits; caller-owned external sessions stay open.
    scenario_ids : tuple or list of int, default=(1,)
        Distinct packaged scenario IDs from 1 through 8, in execution order.
        Preserve all five configured slots, including dead agents, and every
        authored state field. Do not reset, mirror or shorten these starts.
    opponent : str or System or Policy or None, default=None
        None uses each scenario's prescribed pressure. An explicit Team B method
        makes this a custom comparison. It does not change the authored start.
    num_episodes : int, default=1
        Positive trial count per scenario, at most 2**31 - 1. IDs and seed IDs
        are 1 through this count. Each trial uses the authored state and fresh
        method memory. Repeat this count when resuming an exact saved schedule.
    seed : int or None, default=None
        Evaluator random root. None uses its default on a fresh run and inherits
        the saved root on resume. An explicit value asserts equality on resume.
    num_envs : int, default=32
        Parallel worker count. Keep the batch shape when there are fewer trials;
        padded lanes perform no games. This changes no transition budget.
    metrics : {'priority', 'full', 'none'} or None, default=None
        None uses priority on a fresh run and inherits the saved choice on resume.
        Full metrics are for later analysis, not training.
    replay_episodes : tuple or list of int or None, default=None
        Trial IDs to record. None records none on a fresh run and inherits the
        saved selection on resume. IDs are local to each scenario result.
    output_dir : str or Path or None, default=None
        Optional parent for scenario-<id>/<generated-run> outputs. None creates
        no result files. Print each actual run path; never guess the child name.
    resume_from : str or Path or None, default=None
        Exact saved result folder for one scenario. Omit output_dir and pass
        that same scenario, methods and trial count. Completed games stay saved;
        an unfinished game restarts at the authored state with fresh memory.

    Returns
    -------
    tuple of EvaluationResult
        One result per scenario, in requested order. Use result.run_dir and
        load_results to read all saved rows, including earlier resumed episodes.

    Raises
    ------
    ValueError, TypeError, OSError
        Invalid IDs, conflicting paths, method/schedule mismatch or input/output
        errors. Shared owners reject invalid inputs before their result writes.
        Execution errors retain completed result paths as exception notes.

    Notes
    -----
    Methods receive no invented history before the starting tick. Death and
    respawn do not start a new episode. Normal result files describe behavior;
    they do not certify protected evaluation, skill, or unseen training content.
    Each distinct string method is loaded once. No method-family restriction or
    alternate evaluator is added. Python paths use the caller's working folder;
    Hydra preserves its original launch folder.
    """
    import marl_battlegrounds as marl_bgs

    results = marl_bgs.evaluate_scenarios(
        system,
        scenario_ids=scenario_ids,
        opponent=opponent,
        num_episodes=num_episodes,
        seed=seed,
        num_envs=num_envs,
        metrics=metrics,
        replay_episodes=replay_episodes,
        output_dir=output_dir,
        resume_from=resume_from,
    )
    for identifier, result in zip(scenario_ids, results, strict=True):
        print(f"Scenario {identifier}: {result.run_dir or 'Results kept in memory'}")
    return results


def main() -> int:
    """Run the same callable lesson with packaged Hydra settings and overrides."""
    from marl_battlegrounds.experiments.hydra import example_main

    return example_main("scenario_evaluation", run)


if __name__ == "__main__":
    raise SystemExit(main())
