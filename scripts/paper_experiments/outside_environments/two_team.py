"""The two-team adapter for the paper's outside games.

This module gives every outside game (SMAX, simple world comm and predator-prey) one
array interface, so each JaxMARL reference learner reads Team A and Team B the same
way in every game. Team A is the side that learns the method; Team B is its
opponents. Each game's own file builds one ``TwoTeamEnv`` from a JaxMARL
``MultiAgentEnv`` plus the few facts that differ per game (which agents form each
team, the critic's global state, the team reward, who is alive, and whether the game
has a win or a loss).

Who uses it: the vendored JaxMARL learners in ``hosts/``, the opponent panels and the
run scripts in this folder. It runs only in ``.venv-outside``, the environment that
holds JaxMARL, and is never imported by the ``marl_battlegrounds`` package::

    UV_PROJECT_ENVIRONMENT=.venv-outside uv sync --locked --extra cuda13 \\
        --extra training --extra dev --group outside-envs

Main entry points:

- ``TwoTeamEnv``: ``reset(key)`` and ``step(key, state, actions_a, actions_b)``, each
  returning a ``TeamStep`` and the next ``TwoTeamState``.
- ``TeamStep`` and ``EpisodeResult``: what one step gives a learner.
- ``others``: each Team A agent's view of everyone else, as agent-model training
  targets.
- ``Opponent``: the one form that drives Team B (a frozen panel member or a scripted
  enemy).

Boundary: everything here is pure JAX and works under ``jax.jit`` and ``jax.vmap``.
``TeamStep.state_a`` and ``others`` hold what a team's agents cannot see. Only
critics, mixers and training losses may read them; nothing that picks an action may.
"""

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jaxmarl.environments.multi_agent_env import MultiAgentEnv, State
from jaxmarl.environments.spaces import Box, Discrete

type Observations = dict[str, jax.Array]
"""One step's observations from a JaxMARL game, keyed by agent name (SMAX adds
``"world_state"``)."""


class EpisodeResult(NamedTuple):
    """The running totals of the current episode, read when it ends.

    Each field counts the episode up to and including the step that returned it.
    When ``TeamStep.done`` is true they are the finished episode's totals, which is
    what the learners log (JaxMARL's ``returned_episode_returns`` and
    ``returned_episode_lengths``). After ``reset`` every field is zero.
    """

    return_a: jax.Array
    """Each Team A agent's summed reward this episode, shape ``(agents_a,)``,
    float32."""

    return_b: jax.Array
    """Each Team B agent's summed reward this episode, shape ``(agents_b,)``,
    float32."""

    length: jax.Array
    """Steps taken this episode, a scalar int32. 1 on an episode's first step."""

    outcome: jax.Array
    """Team A's result, a scalar int32: +1 for a win, -1 for a loss, 0 for a draw.
    Set only on the step where ``TeamStep.done`` is true, and only in a game built
    with ``has_outcome=True`` (SMAX); it is always 0 in games scored by return."""


class TeamStep(NamedTuple):
    """What one ``reset`` or ``step`` gives a learner, Team A's agents first.

    Every per-agent array lists agents in the team's fixed order, the order given to
    ``TwoTeamEnv``. Under ``jax.vmap`` each field gains a leading games axis.
    When ``done`` is true the game has already reset itself: the observations, masks
    and global state describe the next episode's first step, while the rewards and
    ``result`` belong to the episode that just ended (JaxMARL's auto-reset).
    """

    obs_a: jax.Array
    """Team A's observations, shape ``(agents_a, width_a)``, float32. An agent with a
    shorter observation than its team's widest is padded with zeros at the end."""

    obs_b: jax.Array
    """Team B's observations, shape ``(agents_b, width_b)``, float32, padded the same
    way."""

    avail_a: jax.Array
    """Team A's available actions, shape ``(agents_a, actions_a)``, bool. True marks
    an action the agent may take now. ``actions_a`` is the team's largest action
    count; the padding past an agent's own count is always False (in simple world
    comm the leader has 20 actions and its teammates 5)."""

    avail_b: jax.Array
    """Team B's available actions, shape ``(agents_b, actions_b)``, bool, built the
    same way."""

    alive_a: jax.Array
    """Which Team A agents are alive, shape ``(agents_a,)``, bool. All True in games
    where no agent dies (the particle games)."""

    alive_b: jax.Array
    """Which Team B agents are alive, shape ``(agents_b,)``, bool."""

    state_a: jax.Array
    """Team A's global state for its critic or mixer, shape ``(state_width,)``,
    float32. It is built as JaxMARL's own learners build it for that game, so it can
    hold what no Team A agent sees. Never feed it to anything that picks an
    action."""

    reward_a: jax.Array
    """Each Team A agent's reward for this step, shape ``(agents_a,)``, float32.
    Zero after ``reset``."""

    reward_b: jax.Array
    """Each Team B agent's reward for this step, shape ``(agents_b,)``, float32."""

    team_reward_a: jax.Array
    """Team A's single reward for this step, a scalar float32, as the game's
    ``team_reward`` defines it (the sum of ``reward_a`` unless the game says
    otherwise). Mixers such as QMIX's train on it."""

    done: jax.Array
    """Whether this step ended an episode, a scalar bool. False after ``reset``."""

    result: EpisodeResult
    """The current episode's running totals; the finished episode's when ``done``."""


class TwoTeamState(NamedTuple):
    """What ``TwoTeamEnv.step`` needs from the previous call. Pass it back unchanged.

    The caller owns it; ``step`` never changes it in place and returns a new one.
    """

    env_state: State
    """The wrapped JaxMARL game's own state."""

    return_a: jax.Array
    """Team A's summed rewards in the current episode so far, ``(agents_a,)``."""

    return_b: jax.Array
    """Team B's summed rewards in the current episode so far, ``(agents_b,)``."""

    length: jax.Array
    """Steps taken in the current episode so far, a scalar int32."""


class Others(NamedTuple):
    """Each Team A agent's view of every other agent, from ``others``.

    The leading axis is the Team A agent whose targets these are. Teammates keep the
    team's fixed order with the agent itself left out; enemies keep Team B's order.
    """

    teammates: jax.Array
    """Shape ``(agents_a, agents_a - 1, ...)``: row ``i`` holds every Team A row but
    agent ``i``'s."""

    teammate_mask: jax.Array
    """Shape ``(agents_a, agents_a - 1)``, bool: True where that teammate is alive, so
    its row is a real target."""

    enemies: jax.Array
    """Shape ``(agents_a, agents_b, ...)``: Team B's rows, the same for every Team A
    agent."""

    enemy_mask: jax.Array
    """Shape ``(agents_a, agents_b)``, bool: True where that enemy is alive."""


class Opponent(NamedTuple):
    """The one form that drives Team B: a frozen panel member or a scripted enemy.

    ``act(params, carry, obs_b, avail_b, done, key)`` returns ``(actions_b, carry)``:

    - ``params``: the policy's weights, or ``None`` for a scripted enemy. Pass them
      as an argument, never baked into ``act``, so new weights reuse the compiled
      program.
    - ``carry``: the policy's memory between steps (a recurrent state, or a scripted
      enemy's own state); start from ``init_carry``.
    - ``obs_b`` and ``avail_b``: ``TeamStep.obs_b`` and ``TeamStep.avail_b``.
    - ``done``: ``TeamStep.done``. When true, ``obs_b`` starts a new episode, and
      ``act`` must clear its memory before using it.
    - ``key``: a fresh random key for this step.
    - ``actions_b``: one action per Team B agent, shape ``(agents_b,)``, int32, each
      one available in ``avail_b``.

    A frozen panel member is ``Opponent(act=<the host's saved policy step>,
    init_carry=<its zero memory>)``; each game's file builds its scripted enemy the
    same way.
    """

    act: Callable[..., tuple[jax.Array, Any]]
    """The policy's step, in the form above."""

    init_carry: Any
    """The memory ``act`` starts from at the first episode (``()`` if none)."""


def _team_indices(env: MultiAgentEnv, team: Sequence[str]) -> np.ndarray:
    """Return each team member's position in ``env.agents``, as int32."""
    return np.array([env.agents.index(name) for name in team], dtype=np.int32)


class TwoTeamEnv:
    """A JaxMARL game seen as two teams of arrays: Team A first, then Team B.

    ``reset`` and ``step`` follow JaxMARL's own: the same random keys reach the game
    in the same order, and a finished game resets itself inside ``step``. Only the
    form changes, from per-agent dictionaries to one array per team, plus the
    episode totals the learners log.

    Parameters
    ----------
    env
        The JaxMARL game, built with discrete actions. Every agent named in
        ``team_a`` and ``team_b`` must have a ``Discrete`` action space.
    team_a
        Team A's agent names, in the order every Team A array uses. Team A is the
        side that learns the method.
    team_b
        Team B's agent names, in its fixed order. Together with ``team_a`` it must
        name every agent in ``env.agents`` exactly once.
    global_state
        ``global_state(obs, env_state)`` returns Team A's global state, a flat
        float32 vector the same width every step. The default joins every agent's
        observation in ``env.agents`` order, as JaxMARL's learners do for the
        particle games; SMAX passes ``obs["world_state"]``.
    team_reward
        ``team_reward(reward_a)`` turns the ``(agents_a,)`` rewards into Team A's
        single reward. The default is their sum, as in JaxMARL's Q-learning
        rollouts; SMAX, where every ally gets the same team reward, takes the
        first.
    alive
        ``alive(env_state)`` returns a ``(num_agents,)`` bool array in
        ``env.agents`` order, True for each living agent. ``None`` (the default)
        means no agent ever dies.
    has_outcome
        True for a game won by wiping out the other team (SMAX). The step that ends
        an episode then sets ``EpisodeResult.outcome`` from the final step's alive
        masks: +1 when Team B is all dead and a Team A agent lives, -1 for the
        reverse, 0 otherwise (a timeout, or both teams dying together). False (the
        default) keeps ``outcome`` at 0, for games scored by return.

    Attributes
    ----------
    env
        The wrapped JaxMARL game.
    team_a, team_b
        The two teams' agent names, in their fixed orders.
    has_outcome
        Whether ``EpisodeResult.outcome`` carries a win or a loss.
    width_a, width_b
        Each team's observation width after padding.
    actions_a, actions_b
        Each team's action count after padding: its largest agent's count.
    action_counts_a, action_counts_b
        Each agent's own action count, in team order.

    Raises
    ------
    ValueError
        If the two teams do not name every agent exactly once, or an agent's action
        space is not ``Discrete``.

    Notes
    -----
    Nothing here is compiled at construction. Wrap the methods yourself, for
    example ``jax.jit(jax.vmap(env.step))`` over a batch of games; the game and
    both teams are fixed Python structure, so building a new ``TwoTeamEnv``
    compiles anew, while new keys, states and actions do not. Compiled, ``step``
    reproduces the game states of a compiled ``MultiAgentEnv.step`` bit for bit;
    observations can still differ in a float's last bit, because XLA compiles them
    inside a different program.

    Available actions come from the game's ``get_avail_actions`` when the game
    defines it (SMAX). A game that does not (the particle games) has every one of an
    agent's own actions available at every step.
    """

    def __init__(
        self,
        env: MultiAgentEnv,
        team_a: Sequence[str],
        team_b: Sequence[str],
        *,
        global_state: Callable[[Observations, State], jax.Array] | None = None,
        team_reward: Callable[[jax.Array], jax.Array] = jnp.sum,
        alive: Callable[[State], jax.Array] | None = None,
        has_outcome: bool = False,
    ) -> None:
        self.env = env
        self.team_a = tuple(team_a)
        self.team_b = tuple(team_b)
        if sorted(self.team_a + self.team_b) != sorted(env.agents):
            raise ValueError(
                f"team_a {self.team_a} and team_b {self.team_b} must name every agent "
                f"in {env.agents} exactly once."
            )
        self.has_outcome = has_outcome
        self._index_a = _team_indices(env, self.team_a)
        self._index_b = _team_indices(env, self.team_b)
        self._global_state = global_state or self._joined_observations
        self._team_reward = team_reward
        self._alive = alive
        self._has_masks = (
            type(env).get_avail_actions is not MultiAgentEnv.get_avail_actions
        )

        def action_count(name: str) -> int:
            space = env.action_space(name)
            if not isinstance(space, Discrete):
                raise ValueError(
                    f"Agent {name!r} has a {type(space).__name__} action space; build "
                    "the game with discrete actions."
                )
            return int(space.n)

        def obs_width(name: str) -> int:
            shape = cast(Box, env.observation_space(name)).shape
            return int(np.prod(shape))

        self.action_counts_a = tuple(action_count(name) for name in self.team_a)
        self.action_counts_b = tuple(action_count(name) for name in self.team_b)
        self.actions_a = max(self.action_counts_a)
        self.actions_b = max(self.action_counts_b)
        self.width_a = max(obs_width(name) for name in self.team_a)
        self.width_b = max(obs_width(name) for name in self.team_b)

    def reset(self, key: jax.Array) -> tuple[TeamStep, TwoTeamState]:
        """Start a new episode.

        Parameters
        ----------
        key
            A random key, passed unchanged to the game's ``reset``.

        Returns
        -------
        tuple of (TeamStep, TwoTeamState)
            The first step (zero rewards, ``done`` False, a zero ``result``) and the
            state to pass to ``step``.
        """
        obs, env_state = self._reset_game(key)
        zeros_a = jnp.zeros(len(self.team_a), jnp.float32)
        zeros_b = jnp.zeros(len(self.team_b), jnp.float32)
        state = TwoTeamState(env_state, zeros_a, zeros_b, jnp.int32(0))
        result = EpisodeResult(zeros_a, zeros_b, jnp.int32(0), jnp.int32(0))
        team_step = self._team_step(
            obs, env_state, zeros_a, zeros_b, jnp.bool_(False), result
        )
        return team_step, state

    def step(
        self,
        key: jax.Array,
        state: TwoTeamState,
        actions_a: jax.Array,
        actions_b: jax.Array,
    ) -> tuple[TeamStep, TwoTeamState]:
        """Play one step for both teams, resetting the game if the episode ends.

        Parameters
        ----------
        key
            A fresh random key for this step. It is split exactly as JaxMARL's
            ``MultiAgentEnv.step`` splits it, so the game sees the same keys.
        state
            The state the previous ``reset`` or ``step`` returned.
        actions_a
            One action per Team A agent, shape ``(agents_a,)``, int, in team order.
            Each must be available in the previous ``TeamStep.avail_a``; the game
            does not check this.
        actions_b
            One action per Team B agent, shape ``(agents_b,)``, int, in team order.

        Returns
        -------
        tuple of (TeamStep, TwoTeamState)
            This step, and the state to pass to the next ``step``. When
            ``TeamStep.done`` is true, the game has reset: see ``TeamStep``.
        """
        actions = {name: actions_a[i] for i, name in enumerate(self.team_a)}
        actions.update({name: actions_b[i] for i, name in enumerate(self.team_b)})

        # MultiAgentEnv.step, unrolled so the final state is still in hand for the
        # outcome before the reset replaces it.
        key, key_reset = jax.random.split(key)
        obs_end, state_end, rewards, dones, _ = self.env.step_env(
            key, state.env_state, actions
        )
        obs_reset, state_reset = self._reset_game(key_reset)
        done = dones["__all__"]

        def pick(after_reset: jax.Array, before_reset: jax.Array) -> jax.Array:
            return jax.lax.select(done, after_reset, before_reset)

        env_state: State = jax.tree.map(pick, state_reset, state_end)
        obs: Observations = jax.tree.map(pick, obs_reset, obs_end)

        reward_a = jnp.stack([rewards[name] for name in self.team_a]).astype(
            jnp.float32
        )
        reward_b = jnp.stack([rewards[name] for name in self.team_b]).astype(
            jnp.float32
        )
        return_a = state.return_a + reward_a
        return_b = state.return_b + reward_b
        length = state.length + 1
        result = EpisodeResult(
            return_a, return_b, length, self._outcome(state_end, done)
        )
        team_step = self._team_step(obs, env_state, reward_a, reward_b, done, result)
        next_state = TwoTeamState(
            env_state,
            jnp.where(done, 0.0, return_a),
            jnp.where(done, 0.0, return_b),
            jnp.where(done, 0, length),
        )
        return team_step, next_state

    def _reset_game(self, key: jax.Array) -> tuple[Observations, State]:
        """Call the game's ``reset``. The cast restores the types its jit hides."""
        return cast(tuple[Observations, State], self.env.reset(key))

    def _joined_observations(self, obs: Observations, env_state: State) -> jax.Array:
        """Join every agent's flattened observation in ``env.agents`` order."""
        del env_state
        return jnp.concatenate([obs[name].reshape(-1) for name in self.env.agents])

    def _stack_obs(
        self, obs: Observations, team: tuple[str, ...], width: int
    ) -> jax.Array:
        """Stack one team's observations into ``(agents, width)``, zero-padded."""
        rows = [obs[name].reshape(-1).astype(jnp.float32) for name in team]
        return jnp.stack([jnp.pad(row, (0, width - row.shape[0])) for row in rows])

    def _stack_avail(
        self,
        env_state: State,
        team: tuple[str, ...],
        counts: tuple[int, ...],
        actions: int,
    ) -> jax.Array:
        """Build one team's ``(agents, actions)`` mask, False past each own count."""
        if self._has_masks:
            masks = cast(Observations, self.env.get_avail_actions(env_state))
            rows = [masks[name].astype(jnp.bool_) for name in team]
        else:
            rows = [jnp.ones(count, jnp.bool_) for count in counts]
        return jnp.stack([jnp.pad(row, (0, actions - row.shape[0])) for row in rows])

    def _alive_all(self, env_state: State) -> jax.Array:
        """Return the ``(num_agents,)`` alive mask in ``env.agents`` order."""
        if self._alive is None:
            return jnp.ones(len(self.env.agents), jnp.bool_)
        return self._alive(env_state).astype(jnp.bool_)

    def _outcome(self, state_end: State, done: jax.Array) -> jax.Array:
        """Team A's +1, 0 or -1 from the final state, or 0 before the end."""
        if not self.has_outcome:
            return jnp.int32(0)
        alive = self._alive_all(state_end)
        a_lives = alive[self._index_a].any()
        b_lives = alive[self._index_b].any()
        win = a_lives & ~b_lives
        loss = b_lives & ~a_lives
        return jnp.where(done, win.astype(jnp.int32) - loss.astype(jnp.int32), 0)

    def _team_step(
        self,
        obs: Observations,
        env_state: State,
        reward_a: jax.Array,
        reward_b: jax.Array,
        done: jax.Array,
        result: EpisodeResult,
    ) -> TeamStep:
        """Turn the game's dictionaries into one ``TeamStep``."""
        alive = self._alive_all(env_state)
        return TeamStep(
            obs_a=self._stack_obs(obs, self.team_a, self.width_a),
            obs_b=self._stack_obs(obs, self.team_b, self.width_b),
            avail_a=self._stack_avail(
                env_state, self.team_a, self.action_counts_a, self.actions_a
            ),
            avail_b=self._stack_avail(
                env_state, self.team_b, self.action_counts_b, self.actions_b
            ),
            alive_a=alive[self._index_a],
            alive_b=alive[self._index_b],
            state_a=self._global_state(obs, env_state).astype(jnp.float32),
            reward_a=reward_a,
            reward_b=reward_b,
            team_reward_a=self._team_reward(reward_a).astype(jnp.float32),
            done=done,
            result=result,
        )


def others(
    rows_a: jax.Array, rows_b: jax.Array, alive_a: jax.Array, alive_b: jax.Array
) -> Others:
    """Give each Team A agent the rows of every other agent: its training targets.

    The agent-model methods train on these: what each teammate and each enemy saw
    or did this step. They describe what the agent itself cannot see, so only
    training losses may read them, never an actor.

    Parameters
    ----------
    rows_a
        One row per Team A agent, shape ``(agents_a, ...)``, in team order: for
        example ``TeamStep.obs_a``, or the one-hot actions Team A just took.
    rows_b
        The matching rows for Team B, shape ``(agents_b, ...)``, with the same
        trailing shape as ``rows_a`` when both are used together in one loss.
    alive_a
        ``TeamStep.alive_a`` for the step the rows describe, shape ``(agents_a,)``.
    alive_b
        ``TeamStep.alive_b`` for that step, shape ``(agents_b,)``.

    Returns
    -------
    Others
        Each Team A agent's teammates (itself left out, order kept) and enemies,
        with masks that are False for dead agents.

    Notes
    -----
    A pure function of its inputs. Under ``jax.vmap`` over games, every input gains
    the same leading axis and so does every output.
    """
    count = rows_a.shape[0]
    rest = np.array(
        [[j for j in range(count) if j != i] for i in range(count)], dtype=np.int32
    ).reshape(count, count - 1)
    return Others(
        teammates=rows_a[rest],
        teammate_mask=alive_a[rest],
        enemies=jnp.broadcast_to(rows_b[None], (count, *rows_b.shape)),
        enemy_mask=jnp.broadcast_to(alive_b[None], (count, alive_b.shape[0])),
    )
