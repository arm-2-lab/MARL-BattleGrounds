"""Kept so code written for 1.0.x keeps importing these names.

Their home is ``marl_battlegrounds.modes.team_deathmatch.scenarios``; import
them from there. This module only re-exports the six names it held in 1.0.x, as
the same objects, and defines nothing.
"""

from marl_battlegrounds.modes.team_deathmatch.scenarios import (
    TDM_BETA_SCENARIO_IDS,
    TDM_SCENARIO_PUBLIC_AGENT_IDS,
    build_tdm_qualification_seed_schedule,
    build_tdm_scenario_evaluation_record,
    build_tdm_scenario_specification,
    tdm_scenario_pressure_identity,
)

__all__ = [
    "TDM_BETA_SCENARIO_IDS",
    "TDM_SCENARIO_PUBLIC_AGENT_IDS",
    "build_tdm_qualification_seed_schedule",
    "build_tdm_scenario_evaluation_record",
    "build_tdm_scenario_specification",
    "tdm_scenario_pressure_identity",
]
