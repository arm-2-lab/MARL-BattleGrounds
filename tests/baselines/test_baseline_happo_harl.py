"""Check our HAPPO against the numbers its authors' code computes on the same inputs.

tests/fixtures/implementation_checks/happo_harl.npz holds seeded inputs and what HARL
(PKU-MARL/HARL, MIT license, commit 099d29e729a0d5796e141aaac2e862bd25ae0db7)
computed from them, written by
scripts/paper_experiments/implementation_checks/generate_happo_reference.py in a
pinned processor-only environment that it first checks is that commit; torch never
runs here. Contracts, each against HARL's output to float32 rounding across
libraries: the fixture names the pinned commit; _standardized with HARL's 1e-5 over
every agent's active rows gives HARL's per-agent ("FP") advantage scaling, with
every agent active and with some rows inactive, and keeps each agent's own spread;
chaining _happo_factor through the agents in the random order HARL drew, each
agent's active rows as its samples, gives each agent the factor HARL hands it in
both cases (an inactive agent has one legal action, so its ratio is 1 either way);
and _policy_loss, given HARL's log-probabilities, old log-probabilities, advantages,
factor and active rows, gives HARL's HAPPO policy loss and the same gradient with
respect to the log-probabilities.
"""

# pyright: reportPrivateUsage=false
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from marl_battlegrounds.baselines import ppo

REFERENCE = np.load(
    Path(__file__).parents[1] / "fixtures/implementation_checks/happo_harl.npz"
)
RUNNER_CASES = ["all_active", "some_inactive"]


def _case(name: str) -> dict[str, np.ndarray]:
    prefix = f"{name}/"
    return {
        key.removeprefix(prefix): REFERENCE[key]
        for key in REFERENCE.files
        if key.startswith(prefix)
    }


def test_reference_names_the_pinned_harl_commit() -> None:
    assert str(REFERENCE["harl_commit"]) == "099d29e729a0d5796e141aaac2e862bd25ae0db7"


@pytest.mark.parametrize("name", RUNNER_CASES)
def test_advantage_scaling_matches_harl(name: str) -> None:
    case = _case(name)
    active = case["active"].astype(bool)
    scaled = ppo._standardized(
        jnp.asarray(case["advantages"]), jnp.asarray(active), epsilon=1e-5
    )
    expected = np.moveaxis(case["scaled"], 0, -1)
    np.testing.assert_allclose(np.asarray(scaled), expected, rtol=1e-5, atol=1e-6)
    spreads = [expected[..., agent][active[..., agent]].std() for agent in range(5)]
    assert max(spreads) > 10 * min(spreads)


@pytest.mark.parametrize("name", RUNNER_CASES)
def test_each_agents_factor_matches_harl(name: str) -> None:
    case = _case(name)
    agents, steps, games = case["factor"].shape
    log_ratio = (case["new_log_prob"] - case["old_log_prob"]).reshape(
        agents, steps, games
    )
    active = case["active"].astype(bool)
    factor = jnp.ones((steps, games), jnp.float32)
    for agent in case["order"]:
        np.testing.assert_allclose(factor, case["factor"][agent], rtol=1e-6)
        factor = ppo._happo_factor(
            factor, jnp.asarray(log_ratio[agent]), jnp.asarray(active[..., agent])
        )
    assert np.ptp(case["factor"][case["order"][-1]]) > 0.1


def test_weighted_clipped_loss_and_gradient_match_harl() -> None:
    case = _case("loss")
    old = jnp.asarray(case["old_log_prob"])
    active = jnp.asarray(case["active"].astype(bool))

    def loss(log_prob: jax.Array) -> jax.Array:
        return ppo._policy_loss(
            log_prob - old,
            jnp.asarray(case["advantages"]),
            jnp.asarray(case["factor"]),
            active,
            float(case["clip"]),
        )

    value, gradient = jax.value_and_grad(loss)(jnp.asarray(case["log_prob"]))
    np.testing.assert_allclose(float(value), float(case["policy_loss"]), rtol=1e-5)
    np.testing.assert_allclose(gradient, case["gradient"], rtol=1e-4, atol=1e-7)
    assert np.count_nonzero(case["gradient"]) > 10
