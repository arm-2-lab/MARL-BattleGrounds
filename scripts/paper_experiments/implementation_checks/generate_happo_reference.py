"""Record what HAPPO's authors' code computes on fixed inputs, for equivalence tests.

HAPPO's authors maintain HARL (PKU-MARL/HARL, MIT license). This script installs
nothing and copies none of HARL's code: it runs HARL's own functions at commit
099d29e729a0d5796e141aaac2e862bd25ae0db7 on seeded NumPy inputs and saves those
inputs with HARL's outputs. tests/baselines/test_baseline_happo_harl.py then runs
our HAPPO on the same inputs and must match. Three parts of HAPPO are recorded:

- the advantage scaling of HARL's per-agent ("FP") mode, from
  OnPolicyHARunner.train: one mean and spread over every agent's active rows;
- each agent's factor, the product of the earlier agents' new-to-old probability
  ratios, from the same call, in the random agent order HARL draws by default;
- the weighted clipped loss and its gradient with respect to the action
  log-probabilities, from HAPPO.update.

The networks are left out on purpose: OnPolicyHARunner.train runs on a stand-in
runner whose actors return the recorded log-probabilities and capture what they
receive, and HAPPO.update runs on a real HAPPO whose evaluate_actions returns
them. So the numbers depend only on HAPPO's algorithm, not on any network. An
inactive agent has one legal action, so its log-probability is the same before
and after the update, as in HARL's own environments.

Run it in the pinned processor-only environment::

    python3.12 -m venv /tmp/happo-env
    /tmp/happo-env/bin/pip install --no-deps -r \
        scripts/paper_experiments/implementation_checks/requirements.txt
    /tmp/happo-env/bin/python \
        scripts/paper_experiments/implementation_checks/generate_happo_reference.py

It first checks that the installed HARL is the pinned commit, then overwrites
tests/fixtures/implementation_checks/happo_harl.npz. The run is deterministic:
seeded NumPy and torch, one thread, deterministic algorithms.
"""

import json
import os
from importlib.metadata import distribution
from pathlib import Path
from types import SimpleNamespace

os.environ["WANDB_MODE"] = "disabled"

import numpy as np
import torch
from gymnasium import spaces
from harl.algorithms.actors.happo import HAPPO
from harl.runners.on_policy_ha_runner import OnPolicyHARunner

HARL_COMMIT = "099d29e729a0d5796e141aaac2e862bd25ae0db7"
OUTPUT = (
    Path(__file__).resolve().parents[3]
    / "tests/fixtures/implementation_checks/happo_harl.npz"
)
STEPS, GAMES, AGENTS = 6, 4, 5
# Different spreads per agent, so joint scaling can be told apart from per-agent.
SCALES = np.array([1.0, 2.0, 0.5, 3.0, 0.1], np.float32)


class _Actor:
    """Stand in for one HARL actor: return recorded log-probs, keep what arrives."""

    def __init__(self, old: np.ndarray, new: np.ndarray) -> None:
        """Hold the log-probs to return before (old) and after (new) the update."""
        self.calls = [old, new]
        self.advantages: np.ndarray | None = None

    def evaluate_actions(self, *args: object) -> tuple[torch.Tensor, None, None]:
        """Return the next recorded (rows, 1) log-probs, as HARL's actor would."""
        del args
        return torch.as_tensor(self.calls.pop(0)), None, None

    def train(
        self, buffer: object, advantages: np.ndarray, state_type: str
    ) -> dict[str, float]:
        """Keep the advantages HARL hands this agent; train nothing."""
        del buffer
        assert state_type == "FP"
        self.advantages = advantages.copy()
        return {}


class _Buffer:
    """Stand in for one HARL actor buffer: shapes HARL reads, and the factor it sets."""

    def __init__(self, active: np.ndarray, agent: int, order: list[int]) -> None:
        """Hold agent's (steps + 1, games, 1) active mask; order logs factor calls."""
        self.active_masks = active
        self.obs = np.zeros((STEPS + 1, GAMES, 1), np.float32)
        self.rnn_states = np.zeros((STEPS + 1, GAMES, 1, 1), np.float32)
        self.actions = np.zeros((STEPS, GAMES, 1), np.float32)
        self.masks = np.ones((STEPS + 1, GAMES, 1), np.float32)
        self.available_actions = None
        self.factor: np.ndarray | None = None
        self.agent = agent
        self.order = order

    def update_factor(self, factor: np.ndarray) -> None:
        """Keep the factor HARL hands this agent and log the agent's turn."""
        self.factor = factor.copy()
        self.order.append(self.agent)


def _runner_case(rng: np.random.Generator, active: np.ndarray) -> dict[str, np.ndarray]:
    """Run HARL's OnPolicyHARunner.train on a stand-in runner and keep its outputs.

    active is float32 (steps + 1, games, agents): 1 where the agent acts. Return
    the inputs (advantages (steps, games, agents), active (steps, games, agents),
    old and new log-probs (agents, steps * games), equal on inactive rows) and
    what HARL computed: the agent order it drew (agents,), each agent's scaled
    advantages (agents, steps, games) and its factor (agents, steps, games).
    """
    advantages = (
        rng.normal(0.3, 1.0, (STEPS, GAMES, AGENTS)).astype(np.float32) * SCALES
    )
    old = np.log(rng.uniform(0.05, 0.9, (AGENTS, STEPS * GAMES, 1))).astype(np.float32)
    new = (old + rng.normal(0.0, 0.1, old.shape)).astype(np.float32)
    acting = np.moveaxis(active[:-1], -1, 0).reshape(AGENTS, -1, 1) > 0
    new = np.where(acting, new, old)
    returns = np.zeros((STEPS + 1, GAMES, AGENTS, 1), np.float32)
    returns[:-1, ..., 0] = advantages
    order: list[int] = []
    actors = [_Actor(old[agent], new[agent]) for agent in range(AGENTS)]
    buffers = [
        _Buffer(active[..., agent, None], agent, order) for agent in range(AGENTS)
    ]
    runner = SimpleNamespace(
        algo_args={"train": {"episode_length": STEPS, "n_rollout_threads": GAMES}},
        value_normalizer=None,
        critic_buffer=SimpleNamespace(
            returns=returns, value_preds=np.zeros_like(returns)
        ),
        state_type="FP",
        actor_buffer=buffers,
        actor=actors,
        num_agents=AGENTS,
        fixed_order=False,
        action_aggregation="prod",
        critic=SimpleNamespace(train=lambda *args: {}),
    )
    OnPolicyHARunner.train(runner)
    return {
        "advantages": advantages,
        "active": active[:-1],
        "old_log_prob": old[..., 0],
        "new_log_prob": new[..., 0],
        "order": np.array(order),
        "scaled": np.stack([actor.advantages[..., 0] for actor in actors]),
        "factor": np.stack([buffer.factor[..., 0] for buffer in buffers]),
    }


def _loss_case(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Run HARL's HAPPO.update on recorded log-probs and keep its loss and gradient.

    Return the inputs (log_prob, old_log_prob, advantages, factor, active, each
    float32 (rows,), and clip, a float32 scalar) and HARL's results: policy_loss
    (a scalar, the weighted clipped surrogate's masked mean, negated) and
    gradient (rows,), the gradient of HAPPO.update's objective with respect to
    log_prob.
    """
    rows = 64
    args = {
        "lr": 5e-4,
        "opti_eps": 1e-5,
        "weight_decay": 0.0,
        "data_chunk_length": 10,
        "use_recurrent_policy": False,
        "use_naive_recurrent_policy": False,
        "use_policy_active_masks": True,
        "action_aggregation": "prod",
        "hidden_sizes": [8],
        "activation_func": "relu",
        "use_feature_normalization": False,
        "initialization_method": "orthogonal_",
        "gain": 0.01,
        "recurrent_n": 1,
        "clip_param": 0.2,
        "ppo_epoch": 1,
        "actor_num_mini_batch": 1,
        "entropy_coef": 0.01,
        "use_max_grad_norm": True,
        "max_grad_norm": 10.0,
    }
    happo = HAPPO(args, spaces.Box(-1.0, 1.0, (4,)), spaces.Discrete(6))
    old = np.log(rng.uniform(0.05, 0.9, rows)).astype(np.float32)
    log_prob = (old + rng.normal(0.0, 0.3, rows)).astype(np.float32)
    advantages = rng.normal(0.0, 1.0, rows).astype(np.float32)
    factor = rng.uniform(0.5, 1.5, rows).astype(np.float32)
    active = (rng.random(rows) < 0.8).astype(np.float32)
    leaf = torch.tensor(log_prob[:, None], requires_grad=True)
    entropy = torch.tensor(0.0)
    happo.evaluate_actions = lambda *args: (leaf, entropy, None)
    # Observations, memory, actions and resets only reach evaluate_actions.
    sample = (
        None,
        None,
        None,
        None,
        active[:, None],
        old[:, None],
        advantages[:, None],
        None,
        factor[:, None],
    )
    policy_loss, *_ = happo.update(sample)
    assert leaf.grad is not None
    return {
        "log_prob": log_prob,
        "old_log_prob": old,
        "advantages": advantages,
        "factor": factor,
        "active": active,
        "clip": np.float32(args["clip_param"]),
        "policy_loss": np.float32(policy_loss.item()),
        "gradient": leaf.grad.numpy()[:, 0].copy(),
    }


def main() -> None:
    """Check HARL's commit, run the three recorded cases and write them to OUTPUT."""
    origin = json.loads(distribution("harl").read_text("direct_url.json") or "{}")
    installed = origin.get("vcs_info", {}).get("commit_id")
    if installed != HARL_COMMIT:
        raise SystemExit(f"installed HARL is {installed}, expected {HARL_COMMIT}")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    rng = np.random.default_rng(20261124)
    everyone = np.ones((STEPS + 1, GAMES, AGENTS), np.float32)
    some = (rng.random(everyone.shape) < 0.75).astype(np.float32)
    some[:, 0, :] = 1.0  # every agent keeps active rows
    cases = {
        "all_active": _runner_case(rng, everyone),
        "some_inactive": _runner_case(rng, some),
        "loss": _loss_case(rng),
    }
    arrays = {
        f"{case}/{name}": value
        for case, fields in cases.items()
        for name, value in fields.items()
    }
    arrays["harl_commit"] = np.array(HARL_COMMIT)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT, **arrays)
    print(f"wrote {OUTPUT} ({len(arrays)} arrays) from HARL {HARL_COMMIT}")


if __name__ == "__main__":
    main()
