"""Check our TarMAC attention step against XuanCe's numbers on the same inputs.

TarMAC's authors released no code. The fixture
tests/fixtures/implementation_checks/tarmac_xuance.npz holds seeded queries, keys
and values at the paper's sizes and the messages XuanCe
(agi-brain/xuance, MIT license, commit 2e3f206c71d303c0c6cf62a919efb038331c616a)
computed from them with its own TarMAC.attention_block, written by
scripts/paper_experiments/implementation_checks/generate_tarmac_reference.py in a
pinned processor-only environment that it first checks is that commit; torch never
runs here. Contracts: the fixture names the pinned commit; with every teammate
heard, _tarmac_attention gives XuanCe's message for each of five receivers in three
games over four ticks, each receiver's own message included, to float32 rounding
across libraries; and the inputs make attention matter (the messages differ from
equal weighting). XuanCe has no sender mask, so only the case where every sender is
heard has a XuanCe counterpart; masking is checked against the paper's equations in
test_baseline_tarmac_reference.py.
"""

# pyright: reportPrivateUsage=false
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from marl_battlegrounds.baselines import ppo

REFERENCE = np.load(
    Path(__file__).parents[1] / "fixtures/implementation_checks/tarmac_xuance.npz"
)


def _per_tick(name: str) -> jnp.ndarray:
    # XuanCe lays arrays out [game, agent, tick, feature]; ours is per tick.
    return jnp.asarray(np.moveaxis(REFERENCE[name], 1, 2))


def test_reference_names_the_pinned_xuance_commit() -> None:
    assert str(REFERENCE["xuance_commit"]) == "2e3f206c71d303c0c6cf62a919efb038331c616a"


def test_attention_matches_xuance_with_every_sender_heard() -> None:
    query, key, value = _per_tick("query"), _per_tick("key"), _per_tick("value")
    heard = jnp.ones((*query.shape[:-1], 5), bool)
    message = ppo._tarmac_attention(query, key, value, heard, mean=False)
    expected = np.moveaxis(REFERENCE["message"], 1, 2)
    np.testing.assert_allclose(message, expected, rtol=1e-5, atol=1e-6)
    equal = ppo._tarmac_attention(query, key, value, heard, mean=True)
    assert np.abs(np.asarray(equal) - expected).max() > 0.1
