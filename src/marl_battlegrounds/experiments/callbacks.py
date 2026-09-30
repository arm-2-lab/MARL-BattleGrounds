"""Optional callback factories for ordinary training and study launchers.

PFSP is the competitive example's pure matchmaking rule. wandb_logging opens
one optional SDK run per local training attempt and closes it on exit. Neither
factory changes the default fixed opponent recipe. Importing this module starts
no numerical backend, SDK, client or experiment.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, cast


def pfsp(stats: dict[str, Any], env_steps: int) -> dict[str, object]:
    """Give harder named opponents more future games, keeping 20% current self.

    stats is the public matchmaking callback record. Each named member has
    cumulative games/wins/draws/losses. A draw counts as half a win; an unseen
    member starts at an explicitly chosen 50% prior, not an observed result.
    Named weight is 0.01 + (1 - win fraction)**2, normalized into the remaining
    80%. With no named members, return all self-play. env_steps is the completed
    transition count and is unused by this simple rule. Return only the next
    opponent rule; running games and the member pool stay unchanged.
    """
    del env_steps
    weights: dict[str, float] = {}
    for member in stats["members"]:
        if member["kind"] != "named":
            continue
        result = member["cumulative"]
        games = result["games"]
        win_fraction = (
            (result["wins"] + 0.5 * result["draws"]) / games if games else 0.5
        )
        weights[member["name"]] = 0.01 + (1.0 - win_fraction) ** 2
    total = sum(weights.values())
    return {
        "opponents": {
            "self": 0.2,
            **{name: 0.8 * value / total for name, value in weights.items()},
        }
        if total
        else {"self": 1.0}
    }


def make_pfsp() -> Callable[[dict[str, Any], int], dict[str, object]]:
    """Return the stateless example PFSP callback for future-game sampling.

    No arguments, resources or hidden state. The callback consumes only the
    trainer's supplied statistics and cannot add or replace population members.
    This is an optional researcher choice, never part of the fixed paper recipe.
    """
    return pfsp


@contextmanager
def wandb_logging(
    *,
    project: str,
    directory: str,
    entity: str | None = None,
    mode: Literal["online", "offline"] = "online",
) -> Generator[Callable[[dict[str, Any]], None]]:
    """Yield a lazy logger and close its SDK run when the training attempt ends.

    Parameters
    ----------
    project : str
        Nonempty W&B project name. Credentials come from the SDK environment or
        login store; never put them in YAML or factory kwargs.
    directory : str
        Explicit local SDK output folder. Use an absolute path or resolve it in
        YAML against the launch folder. Created only on the first saved update.
    entity : str or None, default=None
        Optional W&B account/team; None uses the SDK's configured account.
    mode : {"online", "offline"}, default="online"
        Whether W&B sends records remotely or saves them locally. To disable
        logging entirely, omit this factory or set runtime.on_update=null.

    Yields
    ------
    callable
        The trainer's on_update callback. Its first saved row provides run_id
        and attempt_id. Those set the SDK group and unique local-attempt ID.
        Resume creates a new attempt; repeated env_steps are kept separately.

    Raises
    ------
    ValueError
        Names, mode or output path are invalid, or a callback receives rows from
        different attempts. Settings fail on scope entry, before training starts.
    ImportError
        The first update needs the optional wandb extra but it is not installed.
        Install marl-battlegrounds[wandb]. SDK/network errors propagate too.

    Notes
    -----
    Importing and entering this factory starts no SDK or network connection.
    A failure before the first update creates no SDK run. Once opened, finish
    runs on normal return, callback failure or any other error leaving the scope.
    The local trainer row is saved before logging. SDK state is not a training
    checkpoint. Remote logging may add host wait time and must be measured for
    the user's provider; the default disabled path pays none of that cost.
    """
    if not isinstance(cast(object, project), str) or not project.strip():
        raise ValueError("W&B project must be nonempty")
    if not isinstance(cast(object, directory), str) or not directory.strip():
        raise ValueError("W&B directory must be an explicit nonempty path")
    if entity is not None and (
        not isinstance(cast(object, entity), str) or not entity.strip()
    ):
        raise ValueError("W&B entity must be nonempty or None")
    if mode not in ("online", "offline"):
        raise ValueError("W&B mode must be online or offline")
    from marl_battlegrounds.evaluation.revision import (
        _check_output_path,  # pyright: ignore[reportPrivateUsage]
    )

    destination = Path(directory).expanduser().resolve()
    _check_output_path(destination)
    sdk_run: Any = None
    callback: Callable[[dict[str, Any]], None] | None = None
    attempt: tuple[str, str] | None = None

    def log_row(row: dict[str, Any]) -> None:
        """Open at the first saved row, then log only this attempt's records."""
        nonlocal sdk_run, callback, attempt
        current = (row["run_id"], row["attempt_id"])
        if attempt is not None and current != attempt:
            raise ValueError("One logging scope must receive only one training attempt")
        if callback is None:
            try:
                wandb = import_module("wandb")
            except ImportError as error:
                raise ImportError(
                    "Install marl-battlegrounds[wandb] for W&B logging"
                ) from error
            from marl_battlegrounds.training import make_wandb_callback

            destination.mkdir(parents=True, exist_ok=True)
            attempt = current
            sdk_run = wandb.init(
                project=project,
                entity=entity,
                mode=mode,
                dir=str(destination),
                group=current[0],
                id=f"{current[0]}-{current[1]}",
                resume="never",
                reinit="create_new",
                job_type="training",
                config={"training_run_id": current[0], "attempt_id": current[1]},
                save_code=False,
            )
            callback = make_wandb_callback(sdk_run)
        callback(row)

    try:
        yield log_row
    finally:
        if sdk_run is not None:
            sdk_run.finish()
