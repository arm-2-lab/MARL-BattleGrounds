"""Simple world comm as a two-team game: four chasers against two evaders.

This module owns the paper's settings for JaxMARL's simple world comm and the split of
its agents into two teams. Team A is the four chasers: a leader and three followers.
They are the side with real teammates. They share the reward for every touch, and the
leader, the only one who sees into forests, sends a four-way word that its followers
read on the next tick. Team B is the two evaders, which share no reward and cannot
talk.

Settings: JaxMARL's defaults (1 obstacle, 2 foods, 2 forests and 25-step episodes),
except ``action_type="Discrete"``. JaxMARL builds this one particle game with
continuous actions by default, and the learners in this folder take discrete actions.
The leader then has 20 actions, each a move (``action % 5``) and a word
(``action // 5``); every other agent has 5 moves. The global state and the team reward
are JaxMARL's own for the particle games: every agent's observation joined, and the
sum of the chasers' rewards.

Who uses it: the learners and run scripts in this folder, in ``.venv-outside``.
"""

from jaxmarl.environments.mpe.simple_world_comm import SimpleWorldCommMPE
from scripts.paper_experiments.outside_environments.two_team import TwoTeamEnv


def make_world_comm() -> TwoTeamEnv:
    """Build simple world comm with the chasers as Team A and the evaders as Team B.

    Returns
    -------
    TwoTeamEnv
        Team A is ``leadadversary_0``, ``adversary_0``, ``adversary_1`` and
        ``adversary_2``, in that order: observations 34 wide, 20 actions for the
        leader and 5 for each follower, padded with False to 20. Team B is
        ``agent_0`` and ``agent_1``: observations 28 wide and 5 actions each. No
        agent dies, and the game has no win or loss; it is scored by return.

    Examples
    --------
    >>> import jax
    >>> env = make_world_comm()
    >>> team_step, state = env.reset(jax.random.PRNGKey(0))
    >>> team_step.obs_a.shape, team_step.avail_a.shape
    ((4, 34), (4, 20))
    """
    env = SimpleWorldCommMPE(action_type="Discrete")
    return TwoTeamEnv(env, [env.leader, *env.adversaries], env.good_agents)
