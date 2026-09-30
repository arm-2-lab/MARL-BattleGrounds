"""Resolve public live-match inputs through the existing packaged task owners.

Setup validates explicit fixed-scenario overrides before reading content or
loading a method. It keeps all ten physical slots and the exact packaged start.
Nothing here starts a server, runs a controller or writes a recording.
"""

from __future__ import annotations

# Reuse the existing verified-content builders; do not duplicate their loaders.
# pyright: reportPrivateUsage=false
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import jax

from marl_battlegrounds._battle_client.model import (
    SCENARIO_OPPONENT_ERROR,
    DebuggerScenario,
)
from marl_battlegrounds._tdm_assets import scenario_content
from marl_battlegrounds.core.env import reset
from marl_battlegrounds.evaluation.evaluation_conditions import OMITTED, Omitted
from marl_battlegrounds.evaluation.policy_execution import Policy, System
from marl_battlegrounds.evaluation.tdm_scenarios import (
    TDM_BETA_SCENARIO_IDS,
    _prepared_scenario_specification,
    build_tdm_qualification_seed_schedule,
)
from marl_battlegrounds.tasks import (
    DEFAULT_TDM_RED_ZONE_DEPTH,
    AgentClassName,
    _load_tdm_scenario,
    canonical_tournament_rosters,
    list_tdm_maps,
    list_tdm_scenarios,
    make_standard_team_deathmatch_config,
)

type Controller = str | Path | Policy | System
type TeamControllers = Controller | Sequence[Controller]


@dataclass(frozen=True, slots=True)
class BattleSetup:
    """Keep one validated source and its still-unloaded controller assignments.

    The scenario owns a prepared config/start and its public content identity.
    Team sources are a single controller or an immutable roster-length tuple.
    Method loading, memory, resources and output paths belong to the launch.
    """

    scenario: DebuggerScenario
    seed: int
    team_a: Controller | tuple[Controller, ...]
    team_b: Controller | tuple[Controller, ...]


def _controllers(
    value: TeamControllers, *, size: int, name: str
) -> Controller | tuple[Controller, ...]:
    """Validate sources without running factories or inspecting method internals."""
    if isinstance(value, (str, Path, Policy, System)):
        members = (value,)
        result = value
    elif isinstance(cast(object, value), Sequence):
        members = tuple(value)
        if len(members) != size:
            raise ValueError(
                f"{name} needs exactly {size} controllers in roster order."
            )
        result = members
    else:
        raise TypeError(f"{name} must be a controller or an ordered controller list.")
    for member in members:
        if not isinstance(cast(object, member), (str, Path, Policy, System)):
            raise TypeError(
                f"Each {name} controller must be text, a path, Policy or System."
            )
        if isinstance(member, str) and not member.strip():
            raise ValueError(f"Each {name} controller must be nonempty.")
    return result


def prepare_battle_setup(
    *,
    map_id: int | Omitted = OMITTED,
    scenario_id: int | Omitted = OMITTED,
    team_a: TeamControllers = "manual",
    team_b: TeamControllers | Omitted = OMITTED,
    team_a_roster: Sequence[AgentClassName] | Omitted = OMITTED,
    team_b_roster: Sequence[AgentClassName] | Omitted = OMITTED,
    seed: int | Omitted = OMITTED,
    max_steps: int | Omitted = OMITTED,
    score_threshold: int | Omitted = OMITTED,
    red_zone_depth: float | Omitted = OMITTED,
) -> BattleSetup:
    """Prepare one ordinary map or exact fixed scenario without loading methods.

    Parameters
    ----------
    map_id, scenario_id : int or Omitted
        Exactly one packaged source. Maps are 0 through 51; scenarios 1 through 8.
    team_a, team_b : controller or ordered sequence
        Text, Path, Policy or System for a team, or one per configured roster slot.
        Ordinary defaults are Manual. Fixed scenarios accept only Team A choices.
    team_a_roster, team_b_roster : sequence of class names or Omitted
        Ordinary one-to-five rosters; defaults are the shared canonical five.
    seed : int or Omitted
        Ordinary unsigned 32-bit seed, default 0. Fixed scenarios use their
        existing qualification schedule's first case.
    max_steps, score_threshold : int or Omitted
        Ordinary defaults are 300 and 20, checked by the existing task builder.
    red_zone_depth : float or Omitted
        Ordinary default 5.0 map units, checked by the existing task builder.

    Returns
    -------
    BattleSetup
        Prepared content and unloaded sources. Restart reuses this exact content.

    Raises
    ------
    TypeError, ValueError
        Source selection, controller shape or a task rule is invalid. Any explicit
        scenario opponent, map, roster, seed or rule override is rejected, even
        when its value equals the fixed setting. Content validation errors propagate.

    Notes
    -----
    Uses only installed task content. It never reads mutable authoring drafts or
    compacts dead agents. A System remains the caller's exact object.
    """
    fixed = not isinstance(scenario_id, Omitted)
    if fixed:
        if not isinstance(team_b, Omitted):
            raise ValueError(SCENARIO_OPPONENT_ERROR)
        for name, value in (
            ("map_id", map_id),
            ("team_a_roster", team_a_roster),
            ("team_b_roster", team_b_roster),
            ("seed", seed),
            ("max_steps", max_steps),
            ("score_threshold", score_threshold),
            ("red_zone_depth", red_zone_depth),
        ):
            if not isinstance(value, Omitted):
                raise ValueError(
                    f"Scenarios are fixed. Remove {name}. "
                    "Only Team A controllers can be chosen."
                )
        if type(scenario_id) is not int or not 1 <= scenario_id <= 8:
            raise ValueError("scenario_id must be an integer from 1 through 8.")
        info = list_tdm_scenarios()[scenario_id - 1]
        first = _controllers(team_a, size=info.team_sizes[0], name="team_a")
        content = scenario_content(info)
        loaded = _load_tdm_scenario(info, content)
        schedule = build_tdm_qualification_seed_schedule()
        specification = _prepared_scenario_specification(loaded, content, schedule)
        scenario = DebuggerScenario(
            name=f"tdm_scenario_{scenario_id}",
            title=info.name,
            description=info.description,
            mode="interactive",
            build_scenario=lambda: (loaded.config, loaded.initial_state),
            frames=(),
            default_controlled_slot=0,
            scenario_specification=specification,
        )
        return BattleSetup(
            scenario,
            cast(int, schedule.realized_seed_protocols[0].root_seed),
            first,
            "scenario_5" if scenario_id in TDM_BETA_SCENARIO_IDS else "reactive_tdm",
        )
    if isinstance(map_id, Omitted):
        raise ValueError("Supply exactly one map_id or scenario_id.")
    actual_seed = 0 if isinstance(seed, Omitted) else seed
    if type(actual_seed) is not int or not 0 <= actual_seed < 2**32:
        raise ValueError("seed must be an integer from 0 through 2**32 - 1.")
    defaults = canonical_tournament_rosters()
    roster_a = defaults[0] if isinstance(team_a_roster, Omitted) else team_a_roster
    roster_b = defaults[1] if isinstance(team_b_roster, Omitted) else team_b_roster
    config = make_standard_team_deathmatch_config(
        map_id=map_id,
        team_a_roster=roster_a,
        team_b_roster=roster_b,
        max_steps=300 if isinstance(max_steps, Omitted) else max_steps,
        score_threshold=20 if isinstance(score_threshold, Omitted) else score_threshold,
        red_zone_depth=DEFAULT_TDM_RED_ZONE_DEPTH
        if isinstance(red_zone_depth, Omitted)
        else red_zone_depth,
    )
    first = _controllers(team_a, size=len(roster_a), name="team_a")
    second = _controllers(
        "manual" if isinstance(team_b, Omitted) else team_b,
        size=len(roster_b),
        name="team_b",
    )
    state, _, _, _ = reset(config, jax.random.key(actual_seed))
    map_info = list_tdm_maps()[map_id]
    scenario = DebuggerScenario(
        name=f"tdm_map_{map_id}",
        title=map_info.name,
        description=f"Packaged Map {map_id}",
        mode="interactive",
        build_scenario=lambda: (config, state),
        frames=(),
        default_controlled_slot=0,
        packaged_map=map_info,
    )
    return BattleSetup(scenario, actual_seed, first, second)
