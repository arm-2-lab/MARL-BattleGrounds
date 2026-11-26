"""Checks simple world comm's two-team instance (world_comm.py).

Run in .venv-outside, outside the two-stage gate::

    JAX_PLATFORMS=cpu UV_PROJECT_ENVIRONMENT=.venv-outside uv run --no-sync \\
        pytest scripts/paper_experiments/outside_environments/tests

Contracts checked:

- The teams: Team A is the four chasers with the leader first (observations 34 wide,
  20 actions for the leader and 5 for each follower), and Team B the two evaders
  (observations 28 wide, 5 actions).
- The leader's 20 actions decode to 20 different pairs of a move and a word.
- The word the leader sends on a step is in each follower's observation returned by
  that same step, so the followers act on it on the next tick; before it is sent,
  that part of their observation is zero.
"""

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from jaxmarl.environments.mpe.simple_world_comm import SimpleWorldCommMPE
from scripts.paper_experiments.outside_environments.world_comm import make_world_comm


def test_teams_widths_and_action_counts() -> None:
    teams = make_world_comm()
    assert teams.team_a == (
        "leadadversary_0",
        "adversary_0",
        "adversary_1",
        "adversary_2",
    )
    assert teams.team_b == ("agent_0", "agent_1")
    assert (teams.width_a, teams.width_b) == (34, 28)
    assert teams.action_counts_a == (20, 5, 5, 5)
    assert teams.action_counts_b == (5, 5)


def test_leader_actions_are_twenty_distinct_moves_and_words() -> None:
    teams = make_world_comm()
    env = cast(SimpleWorldCommMPE, teams.env)
    pairs: set[tuple[tuple[float, ...], int]] = set()
    for action in range(20):
        actions = {name: jnp.int32(0) for name in (*teams.team_a, *teams.team_b)}
        actions[env.leader] = jnp.int32(action)
        move, word = env.set_actions(actions)
        pairs.add((tuple(np.asarray(move[0]).tolist()), int(jnp.argmax(word[0]))))
    assert len(pairs) == 20


def test_followers_read_the_leaders_word_on_the_next_tick() -> None:
    teams = make_world_comm()
    team_step, state = teams.reset(jax.random.PRNGKey(0))
    np.testing.assert_array_equal(team_step.obs_a[1:, -4:], 0.0)

    word = 2
    actions_a = jnp.array([word * 5 + 1, 0, 0, 0], jnp.int32)
    actions_b = jnp.zeros(2, jnp.int32)
    team_step, _ = teams.step(jax.random.PRNGKey(1), state, actions_a, actions_b)
    sent = np.eye(4)[word]
    for follower in range(1, 4):
        np.testing.assert_array_equal(team_step.obs_a[follower, -4:], sent)
