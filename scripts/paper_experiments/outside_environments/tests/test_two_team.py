"""Checks the two-team adapter (two_team.py) against the JaxMARL games it wraps.

Run in .venv-outside, outside the two-stage gate::

    JAX_PLATFORMS=cpu UV_PROJECT_ENVIRONMENT=.venv-outside uv run --no-sync \\
        pytest scripts/paper_experiments/outside_environments/tests

Contracts checked:

- Order, shapes and masks: every TeamStep row equals the wrapped game's own entry for
  that agent. Simple world comm covers zero-padded observations (a 34-wide leader on
  a team with a 28-wide evader), action masks padded with False (the leader's 20
  actions beside 5), all-True alive flags and the default global state. SMAX covers
  the game's own action masks, alive flags, world state and team reward.
- Auto-reset: across episode ends, the adapter's game states equal JaxMARL's own
  MultiAgentEnv.step under the same keys bit for bit, with both compiled as the
  learners run them. Observation rows are compared to within float32 rounding
  (relative 1e-6), because XLA compiles them inside a different program.
- Episode totals: on the step that ends an episode, result holds the summed rewards
  and the step count; the totals restart from zero on the next step.
- Outcome: in SMAX, +1, -1 and 0 from hand-built final states (Team B wiped out,
  Team A wiped out, both at once, and a timeout).
- others(): each Team A agent gets its teammates in order without itself, every
  enemy, and masks that follow the alive flags.
- Construction: teams that do not name every agent once, and continuous actions,
  are refused.
- Batching: reset and step run under jax.jit and jax.vmap over 32 games.
"""

import dataclasses
from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxmarl.environments.mpe.simple_world_comm import SimpleWorldCommMPE
from jaxmarl.environments.multi_agent_env import MultiAgentEnv, State
from jaxmarl.environments.smax.smax_env import SMAX
from jaxmarl.environments.smax.smax_env import State as SMAXState
from scripts.paper_experiments.outside_environments.two_team import (
    TeamStep,
    TwoTeamEnv,
    TwoTeamState,
    others,
)

type Obs = dict[str, jax.Array]
type GameStep = tuple[Obs, State, dict[str, jax.Array], dict[str, jax.Array], object]
type Reset = Callable[[jax.Array], tuple[TeamStep, TwoTeamState]]
type Step = Callable[
    [jax.Array, TwoTeamState, jax.Array, jax.Array], tuple[TeamStep, TwoTeamState]
]


def _game_reset(env: MultiAgentEnv, key: jax.Array) -> tuple[Obs, State]:
    return cast(tuple[Obs, State], env.reset(key))


def _game_step(
    step: Callable[..., object], key: jax.Array, state: State, actions: Obs
) -> GameStep:
    return cast(GameStep, step(key, state, actions))


def _agents(env: MultiAgentEnv) -> list[str]:
    return env.agents


def _joint(teams: TwoTeamEnv, actions_a: jax.Array, actions_b: jax.Array) -> Obs:
    return {
        **dict(zip(teams.team_a, actions_a, strict=True)),
        **dict(zip(teams.team_b, actions_b, strict=True)),
    }


def _smax_alive(state: State) -> jax.Array:
    return cast(SMAXState, state).unit_alive


def _world_comm(max_steps: int = 25) -> SimpleWorldCommMPE:
    return SimpleWorldCommMPE(action_type="Discrete", max_steps=max_steps)


def _chasers_vs_evaders(env: SimpleWorldCommMPE) -> TwoTeamEnv:
    return TwoTeamEnv(env, [env.leader, *env.adversaries], env.good_agents)


def _smax() -> tuple[SMAX, TwoTeamEnv]:
    env = SMAX(num_allies=3, num_enemies=3)
    teams = TwoTeamEnv(
        env,
        env.agents[:3],
        env.agents[3:],
        global_state=lambda obs, state: obs["world_state"],
        team_reward=lambda reward: reward[0],
        alive=_smax_alive,
        has_outcome=True,
    )
    return env, teams


def _random_actions(key: jax.Array, teams: TwoTeamEnv) -> tuple[jax.Array, jax.Array]:
    key_a, key_b = jax.random.split(key)
    counts_a = jnp.array(teams.action_counts_a)
    counts_b = jnp.array(teams.action_counts_b)
    draw_a = jax.random.uniform(key_a, counts_a.shape)
    draw_b = jax.random.uniform(key_b, counts_b.shape)
    return (draw_a * counts_a).astype(jnp.int32), (draw_b * counts_b).astype(jnp.int32)


def _assert_rows(
    team_step: TeamStep, obs: dict[str, jax.Array], teams: TwoTeamEnv
) -> None:
    for rows, names, width in (
        (team_step.obs_a, teams.team_a, teams.width_a),
        (team_step.obs_b, teams.team_b, teams.width_b),
    ):
        assert rows.shape == (len(names), width)
        for i, name in enumerate(names):
            own = obs[name].reshape(-1)
            np.testing.assert_allclose(
                rows[i, : own.shape[0]], own, rtol=1e-6, atol=1e-7
            )
            np.testing.assert_array_equal(rows[i, own.shape[0] :], 0.0)


def _assert_states_equal(left: State, right: State) -> None:
    for x, y in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(x, y)


def test_world_comm_rows_match_the_game_with_padding() -> None:
    env = _world_comm()
    teams = TwoTeamEnv(
        env,
        [env.leader, env.good_agents[0]],
        [*env.adversaries, env.good_agents[1]],
    )
    key = jax.random.PRNGKey(0)
    team_step, state = teams.reset(key)
    obs, _ = _game_reset(env, key)

    assert (teams.width_a, teams.width_b) == (34, 34)
    _assert_rows(team_step, obs, teams)
    assert team_step.avail_a.shape == (2, 20)
    np.testing.assert_array_equal(team_step.avail_a[0], True)
    np.testing.assert_array_equal(team_step.avail_a[1, :5], True)
    np.testing.assert_array_equal(team_step.avail_a[1, 5:], False)
    np.testing.assert_array_equal(team_step.avail_b, True)
    np.testing.assert_array_equal(team_step.alive_a, True)
    joined = jnp.concatenate([obs[name] for name in _agents(env)])
    np.testing.assert_array_equal(team_step.state_a, joined)

    actions_a, actions_b = _random_actions(jax.random.PRNGKey(1), teams)
    team_step, _ = teams.step(jax.random.PRNGKey(2), state, actions_a, actions_b)
    env_state = _game_reset(env, key)[1]
    joint = _joint(teams, actions_a, actions_b)
    rewards = _game_step(env.step, jax.random.PRNGKey(2), env_state, joint)[2]
    expected = jnp.stack([rewards[name] for name in teams.team_a])
    np.testing.assert_allclose(team_step.reward_a, expected)
    np.testing.assert_allclose(team_step.team_reward_a, expected.sum())


def test_auto_reset_and_episode_totals_match_jaxmarl_step() -> None:
    env = _world_comm(max_steps=3)
    teams = _chasers_vs_evaders(env)
    teams_step = cast(Step, jax.jit(teams.step))
    env_step = jax.jit(env.step)
    team_step, state = teams.reset(jax.random.PRNGKey(3))
    _, env_state = _game_reset(env, jax.random.PRNGKey(3))
    summed = np.zeros(len(teams.team_a))
    for t in range(7):
        actions_a, actions_b = _random_actions(jax.random.PRNGKey(100 + t), teams)
        key = jax.random.PRNGKey(200 + t)
        team_step, state = teams_step(key, state, actions_a, actions_b)
        joint = _joint(teams, actions_a, actions_b)
        obs, env_state, rewards, dones, _ = _game_step(env_step, key, env_state, joint)

        _assert_rows(team_step, obs, teams)
        _assert_states_equal(state.env_state, env_state)
        assert bool(team_step.done) == bool(dones["__all__"]) == (t % 3 == 2)
        summed += np.array([rewards[name] for name in teams.team_a])
        np.testing.assert_allclose(team_step.result.return_a, summed, rtol=1e-6)
        assert int(team_step.result.length) == t % 3 + 1
        if team_step.done:
            summed[:] = 0.0
            np.testing.assert_array_equal(state.return_a, 0.0)
            assert int(state.length) == 0


def test_smax_rows_masks_alive_and_state_match_the_game() -> None:
    env, teams = _smax()
    key = jax.random.PRNGKey(4)
    team_step, state = teams.reset(key)
    obs, env_state = _game_reset(env, key)
    masks = cast(Obs, env.get_avail_actions(env_state))

    _assert_rows(team_step, obs, teams)
    for rows, names in (
        (team_step.avail_a, teams.team_a),
        (team_step.avail_b, teams.team_b),
    ):
        for i, name in enumerate(names):
            np.testing.assert_array_equal(rows[i], masks[name].astype(bool))
    np.testing.assert_array_equal(team_step.alive_a, _smax_alive(env_state)[:3])
    np.testing.assert_array_equal(team_step.state_a, obs["world_state"])

    stop = jnp.full(3, 4, jnp.int32)
    team_step, _ = teams.step(jax.random.PRNGKey(5), state, stop, stop)
    joint = _joint(teams, stop, stop)
    rewards = _game_step(env.step, jax.random.PRNGKey(5), env_state, joint)[2]
    np.testing.assert_allclose(team_step.team_reward_a, rewards[teams.team_a[0]])


@pytest.mark.parametrize(
    ("dead_a", "dead_b", "last_step", "outcome"),
    [
        (False, True, False, 1),
        (True, False, False, -1),
        (True, True, False, 0),
        (False, False, True, 0),
    ],
)
def test_smax_outcome_from_the_final_state(
    dead_a: bool, dead_b: bool, last_step: bool, outcome: int
) -> None:
    env, teams = _smax()
    _, state = teams.reset(jax.random.PRNGKey(6))
    dead = jnp.array([dead_a] * 3 + [dead_b] * 3)
    start = cast(SMAXState, state.env_state)
    env_state = dataclasses.replace(
        start,
        unit_health=jnp.where(dead, 0.0, start.unit_health),
        unit_alive=~dead,
        step=jnp.int32(env.max_steps - 1 if last_step else 0),
    )
    stop = jnp.full(3, 4, jnp.int32)
    team_step, _ = teams.step(
        jax.random.PRNGKey(7), state._replace(env_state=env_state), stop, stop
    )
    assert bool(team_step.done)
    assert int(team_step.result.outcome) == outcome


def test_others_leaves_each_agent_out_and_masks_the_dead() -> None:
    rows_a = jnp.arange(6.0).reshape(3, 2)
    rows_b = jnp.arange(10.0, 14.0).reshape(2, 2)
    alive_a = jnp.array([True, False, True])
    alive_b = jnp.array([False, True])
    result = others(rows_a, rows_b, alive_a, alive_b)

    for i, rest in enumerate(([1, 2], [0, 2], [0, 1])):
        np.testing.assert_array_equal(result.teammates[i], rows_a[jnp.array(rest)])
        np.testing.assert_array_equal(result.teammate_mask[i], alive_a[jnp.array(rest)])
        np.testing.assert_array_equal(result.enemies[i], rows_b)
        np.testing.assert_array_equal(result.enemy_mask[i], alive_b)


def test_construction_refuses_bad_teams_and_continuous_actions() -> None:
    env = _world_comm()
    with pytest.raises(ValueError, match="exactly once"):
        TwoTeamEnv(env, [env.leader], env.good_agents)
    continuous = SimpleWorldCommMPE()
    with pytest.raises(ValueError, match="discrete"):
        _chasers_vs_evaders(continuous)


def test_reset_and_step_batch_over_32_games() -> None:
    teams = _chasers_vs_evaders(_world_comm())
    keys = jax.random.split(jax.random.PRNGKey(8), 32)
    reset = cast(Reset, jax.jit(jax.vmap(teams.reset)))
    step = cast(Step, jax.jit(jax.vmap(teams.step)))
    team_step, state = reset(keys)

    def draw(key: jax.Array) -> tuple[jax.Array, jax.Array]:
        return _random_actions(key, teams)

    actions_a, actions_b = jax.vmap(draw)(keys)
    team_step, state = step(keys, state, actions_a, actions_b)

    assert isinstance(state, TwoTeamState)
    assert team_step.obs_a.shape == (32, 4, 34)
    assert team_step.obs_b.shape == (32, 2, 28)
    assert team_step.avail_a.shape == (32, 4, 20)
    assert team_step.state_a.shape == (32, 4 * 34 + 2 * 28)
    assert team_step.done.shape == (32,)
    assert team_step.result.return_a.shape == (32, 4)
