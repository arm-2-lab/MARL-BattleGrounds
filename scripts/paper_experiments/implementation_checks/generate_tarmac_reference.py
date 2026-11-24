"""Record what XuanCe's TarMAC attention computes on fixed inputs, for our tests.

TarMAC's authors released no code, so its attention step is checked against a
widely used open-source implementation, XuanCe (agi-brain/xuance, MIT license).
This script installs nothing and copies none of XuanCe's code: it runs XuanCe's
own TarMAC.attention_block at commit 2e3f206c71d303c0c6cf62a919efb038331c616a on
seeded queries, keys and values at the paper's sizes (16 for queries and keys,
32 for values, five agents) and saves those inputs with XuanCe's messages.
tests/baselines/test_baseline_tarmac_xuance.py then runs our attention step on
the same inputs and must match. attention_block reads one attribute, its scale,
which XuanCe sets to the square root of the query size; the stand-in object
holds 4, the square root of the paper's 16.

Run it in the pinned processor-only environment::

    python3.12 -m venv /tmp/tarmac-env
    /tmp/tarmac-env/bin/pip install --no-deps -r \
        scripts/paper_experiments/implementation_checks/requirements.txt
    /tmp/tarmac-env/bin/python \
        scripts/paper_experiments/implementation_checks/generate_tarmac_reference.py

It first checks that the installed XuanCe is the pinned commit, then overwrites
tests/fixtures/implementation_checks/tarmac_xuance.npz. The run is
deterministic: seeded NumPy, one torch thread, deterministic algorithms.
"""

import json
from importlib.metadata import distribution
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from xuance.torch.communications.attention_comm import TarMAC

XUANCE_COMMIT = "2e3f206c71d303c0c6cf62a919efb038331c616a"
OUTPUT = (
    Path(__file__).resolve().parents[3]
    / "tests/fixtures/implementation_checks/tarmac_xuance.npz"
)
GAMES, AGENTS, STEPS, KEY_SIZE, VALUE_SIZE = 3, 5, 4, 16, 32


def main() -> None:
    """Check XuanCe's commit, run its attention step and write OUTPUT.

    Saves query and key, float32 (games, agents, steps, 16); value, float32
    (games, agents, steps, 32); message, XuanCe's float32 (games, agents,
    steps, 32) output, in XuanCe's [game, agent, step, feature] layout; and
    the commit.
    """
    origin = json.loads(distribution("xuance").read_text("direct_url.json") or "{}")
    installed = origin.get("vcs_info", {}).get("commit_id")
    if installed != XUANCE_COMMIT:
        raise SystemExit(f"installed XuanCe is {installed}, expected {XUANCE_COMMIT}")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    rng = np.random.default_rng(20261124)
    shape = (GAMES, AGENTS, STEPS)
    query = rng.normal(0.0, 1.0, (*shape, KEY_SIZE)).astype(np.float32)
    key = rng.normal(0.0, 1.0, (*shape, KEY_SIZE)).astype(np.float32)
    value = rng.normal(0.0, 1.0, (*shape, VALUE_SIZE)).astype(np.float32)
    stand_in = SimpleNamespace(scale=KEY_SIZE**0.5)
    message = TarMAC.attention_block(
        stand_in,
        torch.as_tensor(query),
        torch.as_tensor(key),
        torch.as_tensor(value),
    )
    arrays = {
        "query": query,
        "key": key,
        "value": value,
        "message": message.numpy().copy(),
        "xuance_commit": np.array(XUANCE_COMMIT),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT, **arrays)
    print(f"wrote {OUTPUT} ({len(arrays)} arrays) from XuanCe {XUANCE_COMMIT}")


if __name__ == "__main__":
    main()
