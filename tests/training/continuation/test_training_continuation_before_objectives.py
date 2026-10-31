"""Check that a Team Deathmatch learner saved before the objective fields restores.

The Capture the Flag and King of the Hills settings and state joined
``EnvConfig`` and ``EnvState`` after Season 0, so older learner checkpoints
lack their leaves. The tripwire test builds this code's learner for the config
pinned in ``tests/fixtures/training/tdm_checkpoint_layout_before_objectives.json``
and checks that it differs from the pinned layout only by rows of the fields
``neutral_objective_fields`` names, and that the learner holds every one of
them; any other new learner leaf fails it. The restore test rewrites a saved
learner the way older code saved it (those rows removed, the state payload
renumbered) and checks that restore, from a shape-only template, returns the
same learner with those fields at their neutral values. A layout that differs
in any other row is refused before any array is read or file changed. CPU
only; the real Season 0 learner restore on the GPU is separate evidence.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import shutil
from collections.abc import Collection
from pathlib import Path
from typing import Any, cast

import jax
import orbax.checkpoint as ocp  # pyright: ignore[reportMissingTypeStubs]
import pytest
from tests.training.test_training_checkpoints import (
    _equal,  # pyright: ignore[reportPrivateUsage]
    _expected,  # pyright: ignore[reportPrivateUsage]
    _files,  # pyright: ignore[reportPrivateUsage]
    _metadata,  # pyright: ignore[reportPrivateUsage]
)

import marl_battlegrounds.training._content as _content
import marl_battlegrounds.training.checkpoints as checkpoints
from marl_battlegrounds.core.config import neutral_objective_fields
from marl_battlegrounds.evaluation.models import (
    canonical_digest_sha256,
    canonical_json_bytes,
)
from marl_battlegrounds.training import make_training_schedule, prepare_training_content
from marl_battlegrounds.training._content import TrainingContentBinding
from marl_battlegrounds.training.checkpoints import (
    read_checkpoint_details,
    restore_checkpoint,
    save_checkpoint,
)
from marl_battlegrounds.training.collection import TrainingCollection
from marl_battlegrounds.training.config import TrainConfig, config_from_dict
from marl_battlegrounds.training.learner import LearnerState, init_learner
from marl_battlegrounds.training.runner import (
    _history_setup,  # pyright: ignore[reportPrivateUsage]
)

PINNED = json.loads(
    (
        Path(__file__).parents[2]
        / "fixtures"
        / "training"
        / "tdm_checkpoint_layout_before_objectives.json"
    ).read_text()
)


@pytest.fixture(scope="module")
def pinned_learner() -> tuple[TrainConfig, TrainingCollection, LearnerState]:
    # Built as the runner builds it, so the history rows match the pinned run.
    config = config_from_dict(PINNED["config"])
    prepared = prepare_training_content()
    schedule, options = _history_setup(
        config,
        make_training_schedule(
            total_env_steps=config.total_env_steps, num_envs=config.num_envs
        ),
        prepared,
        check_spacing=False,
    )
    collection, state = init_learner(
        schedule=schedule,
        seed=config.seed,
        ppo=config.ppo,
        method=config.method,
        prepared=prepared,
        metrics=config.metrics,
        keep_past=options["keep_past"],
        history_capture_capacity=options["history_capture_capacity"],
        minimum_capture_rounds=options["minimum_capture_rounds"],
        capture_interval_rounds=options["capture_interval_rounds"],
    )
    return config, collection, state


SEASON_0_RULES = next(iter(_content._TDM_EQUIVALENT_SIMULATOR_RULES))  # pyright: ignore[reportPrivateUsage]


def _saved_before(path: Path, fields: Collection[str], rules: str) -> Path:
    # Rewrite a saved learner as older code saved it: drop the named fields'
    # rows, renumber the state payload, record the given simulator rules and
    # re-sign both records.
    details = read_checkpoint_details(path)
    kept = [
        checkpoints._row_field(row) not in fields  # pyright: ignore[reportPrivateUsage]
        for row in details["layout"]
    ]
    numbers = list(itertools.accumulate(kept, initial=0))
    with cast(Any, ocp.Checkpointer(ocp.StandardCheckpointHandler())) as reader:
        payload = cast(dict[str, Any], reader.restore(path / "state"))
    shutil.rmtree(path / "state")
    checkpoints._save_arrays(  # pyright: ignore[reportPrivateUsage]
        path / "state",
        {
            f"leaf_{numbers[int(key[5:])]:05d}": leaf
            for key, leaf in payload.items()
            if kept[int(key[5:])]
        },
    )
    details["layout"] = [
        row for row, keep in zip(details["layout"], kept, strict=True) if keep
    ]
    binding = TrainingContentBinding.model_validate_json(
        canonical_json_bytes(details["collection"]["content_binding"])
    )
    binding = binding.model_copy(
        update={
            "shared_definitions": tuple(
                row.model_copy(update={"canonical_digest": rules})
                if row.identifier == "simulator-rules"
                else row
                for row in binding.shared_definitions
            )
        }
    )
    details["collection"]["content_binding"] = binding.model_copy(
        update={
            "canonical_digest": canonical_digest_sha256(
                binding, exclude={"canonical_digest"}
            )
        }
    ).model_dump(mode="json")
    del details["checkpoint_id"]
    details["files"] = {
        name: digest
        for name, digest in checkpoints._inventory(path).items()  # pyright: ignore[reportPrivateUsage]
        if name != "checkpoint_details.json"
    }
    details["checkpoint_id"] = hashlib.sha256(
        checkpoints._json_bytes(details)  # pyright: ignore[reportPrivateUsage]
    ).hexdigest()
    (path / "checkpoint_details.json").write_text(json.dumps(details))
    return path


def test_this_learner_differs_from_the_pinned_layout_only_by_objective_rows(
    pinned_learner: tuple[TrainConfig, TrainingCollection, LearnerState],
) -> None:
    config, _, state = pinned_learner
    layout = checkpoints._layout(state)  # pyright: ignore[reportPrivateUsage]
    numbers = checkpoints._saved_row_numbers(PINNED["layout"], layout)  # pyright: ignore[reportPrivateUsage]
    filled = {
        checkpoints._row_field(row)  # pyright: ignore[reportPrivateUsage]
        for row, number in zip(layout, numbers, strict=True)
        if number is None
    }
    assert filled == set(neutral_objective_fields(host=True))
    assert (
        checkpoints._layout(checkpoints._actor_payload(state, config.method))  # pyright: ignore[reportPrivateUsage]
        == PINNED["actor_layout"]
    )


@pytest.mark.parametrize(
    ("extra_field", "rules", "refusal"),
    [
        ((), SEASON_0_RULES, None),
        (("step_count",), SEASON_0_RULES, "paths, shapes or dtypes differ"),
        ((), "0" * 64, "content or collection settings differ"),
    ],
    ids=["season-0", "other-layout", "other-rules"],
)
def test_a_season_0_learner_restores_with_neutral_objective_fields(
    pinned_learner: tuple[TrainConfig, TrainingCollection, LearnerState],
    tmp_path: Path,
    extra_field: tuple[str, ...],
    rules: str,
    refusal: str | None,
) -> None:
    config, collection, state = pinned_learner
    metadata = {**_metadata(), "config": PINNED["config"]}
    old = _saved_before(
        save_checkpoint(
            tmp_path,
            collection,
            state,
            metadata=metadata,
            ppo=config.ppo,
            method=config.method,
        ),
        {*neutral_objective_fields(host=True), *extra_field},
        rules,
    )
    before = _files(tmp_path)

    def restore() -> LearnerState:
        return cast(
            LearnerState,
            restore_checkpoint(
                old,
                collection,
                jax.eval_shape(lambda: state),
                expected_metadata=_expected(metadata),
                ppo=config.ppo,
                method=config.method,
            ).state,
        )

    if refusal is None:
        _equal(state, restore())
    else:
        with pytest.raises(ValueError, match=refusal):
            restore()
    assert _files(tmp_path) == before
