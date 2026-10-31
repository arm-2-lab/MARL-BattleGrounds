"""Build Capture the Flag configs for tests from Team Deathmatch ones.

capture_the_flag_config turns a valid scalar Team Deathmatch config into a
Capture the Flag config on the same map and roster: its Team Deathmatch
threshold and Red Zone depth become 0, and the Capture the Flag settings take
the defaults in marl_battlegrounds._task_settings. Tests use it until
make("ctf") exists.
"""

import jax.numpy as jnp

from marl_battlegrounds._task_settings import (
    DEFAULT_CTF_DROPPED_FLAG_AUTO_RETURN_STEPS,
    DEFAULT_CTF_FLAG_CAPTURE_DISTANCE,
    DEFAULT_CTF_FLAG_HOME_POSITIONS,
    DEFAULT_CTF_FLAG_INTERACTION_DISTANCE,
    DEFAULT_CTF_SCORE_THRESHOLD,
)
from marl_battlegrounds.core.types import TASK_MODE_CTF, EnvConfig


def capture_the_flag_config(config: EnvConfig) -> EnvConfig:
    return config._replace(
        task_mode=TASK_MODE_CTF,
        team_deathmatch_score_threshold=0,
        team_deathmatch_red_zone_depth=0.0,
        capture_the_flag_score_threshold=DEFAULT_CTF_SCORE_THRESHOLD,
        capture_the_flag_flag_home_positions=jnp.asarray(
            DEFAULT_CTF_FLAG_HOME_POSITIONS, dtype=jnp.float32
        ),
        capture_the_flag_flag_interaction_distance=DEFAULT_CTF_FLAG_INTERACTION_DISTANCE,
        capture_the_flag_flag_capture_distance=DEFAULT_CTF_FLAG_CAPTURE_DISTANCE,
        capture_the_flag_dropped_flag_auto_return_steps=(
            DEFAULT_CTF_DROPPED_FLAG_AUTO_RETURN_STEPS
        ),
    )
