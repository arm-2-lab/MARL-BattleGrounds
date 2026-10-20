"""Check that Team Deathmatch still plays exactly its recorded golden games.

``tests/fixtures/tdm_goldens.json`` holds one SHA-256 per game for every
installed map from both spawn ends, recorded on the CPU at commit b4a5b39,
before any Capture the Flag code (``tests/helpers/tdm_goldens.py`` explains
the recording). This gate test replays 8 maps from both ends on both paths:
direct ``core.step`` under ``vmap`` and the ``make("tdm")`` handle. The maps
cover the curriculum, training, validation and test splits, the shortest
games and the one game that reaches the 300-step horizon (map 36). It
requires every leaf that existed at b4a5b39 to match bit for bit and every
leaf added since to hold its neutral value.

CPU float results can depend on the processor and the JAX version, so the
test skips, saying why, when the machine type (``platform.machine()``, such as
x86_64 or arm64) or the JAX version differs from the recording. Within x86_64
the results matched on an AVX-512 machine and with XLA limited to AVX2, which
every hosted runner has; an x86_64 processor without AVX2 is not detected and
would fail. The remedy is to re-record from a frozen ``git archive`` of
b4a5b39, never from the current tree. The full 52-map sweep at every core
checkpoint runs the helper's ``check`` command directly and never skips.
"""

import json
import platform
from pathlib import Path

import jax
import pytest
from tests.helpers.tdm_goldens import check

GATE_MAPS = (0, 17, 29, 36, 43, 45, 47, 51)
FIXTURE = Path(__file__).parents[1] / "fixtures" / "tdm_goldens.json"


def test_team_deathmatch_plays_its_golden_games() -> None:
    recorded = json.loads(FIXTURE.read_text())
    here = (platform.machine(), jax.__version__)
    if here != (recorded["machine"], recorded["jax"]):
        pytest.skip(
            f"goldens were recorded on {recorded['machine']} with JAX "
            f"{recorded['jax']}; this run is {here[0]} with JAX {here[1]}"
        )
    assert check(FIXTURE, GATE_MAPS) == []
