"""Small optional adapters for the trainer's existing host callbacks.

make_wandb_callback logs saved update rows to a caller-owned W&B run. This
module imports no SDK or numerical library. Callback factories keep their
identity receipts here so ordinary trainer recovery can check them before writes.
"""

from __future__ import annotations

# Shared receipt reader is private to the training/launcher owners.
# pyright: reportUnusedFunction=false
import json
from collections.abc import Callable, Mapping
from typing import Any, Protocol, cast


class _LoggingRun(Protocol):
    """The two W&B Run methods used by the adapter; lifecycle stays with its owner."""

    def define_metric(self, name: str, *, step_metric: str | None = None) -> object:
        """Declare a logged field and, optionally, its plotted horizontal axis."""
        ...

    def log(self, data: Mapping[str, Any]) -> object:
        """Append one finite saved trainer row; W&B owns the remote write."""
        ...


def make_wandb_callback(run: _LoggingRun) -> Callable[[dict[str, Any]], None]:
    """Log saved training rows to a caller-owned W&B run.

    Parameters
    ----------
    run : wandb.Run
        An initialized run, or a compatible object with define_metric and log.
        The caller owns init, finish, credentials and optional SDK installation.

    Returns
    -------
    callable
        Pass to train(on_update=...) or extend_training(on_update=...). Each call
        logs the supplied saved row with env_steps as the explicit plot axis.
        The row includes durable run_id and local attempt_id. Recovery may replay
        steps; create a separate SDK run per attempt when that distinction matters.

    Raises
    ------
    TypeError
        run lacks the required callable methods. SDK errors propagate unchanged;
        the trainer already saved the local row before invoking this callback.

    Notes
    -----
    Calls define_metric twice during construction. Each update copies the row
    once for the SDK and calls log once. Does not initialize, finish, import W&B,
    transfer arrays or run once per environment step. None as the trainer's
    on_update skips this adapter entirely. It never changes training settings.
    """
    if not callable(getattr(run, "define_metric", None)) or not callable(
        getattr(run, "log", None)
    ):
        raise TypeError("W&B run must provide callable define_metric and log methods")
    run.define_metric("env_steps")
    run.define_metric("*", step_metric="env_steps")

    def log_row(row: dict[str, Any]) -> None:
        """Send one saved finite update row, retaining its run and attempt IDs."""
        run.log(dict(row))

    return log_row


def _callback_identities(**callbacks: Callable[..., Any] | None) -> dict[str, Any]:
    """Copy finite receipts attached by the shared trusted-factory adapter.

    Ordinary live Python callbacks have no receipt and keep their existing
    contract. A receipt records code/kwargs, not restorable opaque state. The
    trainer saves this mapping and compares it before ordinary recovery writes.
    """
    return cast(
        dict[str, Any],
        json.loads(
            json.dumps(
                {
                    name: identity
                    for name, callback in callbacks.items()
                    if (identity := getattr(callback, "_marl_callback_identity", None))
                    is not None
                },
                allow_nan=False,
            )
        ),
    )
