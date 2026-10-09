"""Validate and serialize training settings without starting numerical libraries.

TrainConfig and the JSON readers share the baseline settings and host checks.
The runner and original helper modules retain exact aliases for compatibility.
No config construction opens artifacts, calls a policy factory, starts JAX or
uses the network; released model names are checked for spelling only.
Array-valued shaping checks stay with training.shaping; ordinary configurations
use real host numbers. Numerical callers share the same scalar limits.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

from marl_battlegrounds._method_loading import validate_saved_method_reference
from marl_battlegrounds._task_settings import (
    DEFAULT_TDM_RED_ZONE_DEPTH,
    AgentClassName,
    _validate_roster,  # pyright: ignore[reportPrivateUsage]
)
from marl_battlegrounds.baselines.config import (
    DEFAULT_PQN_CONFIG,
    DEFAULT_QMIX_CONFIG,
    PPOConfig,
    PQNConfig,
    QMIXConfig,
    pqn_planned_learning_blocks,
    saved_settings,
    validate_ppo_batch_size,
    validate_pqn_batch_size,
    validate_qmix_batch_size,
)
from marl_battlegrounds.baselines.methods import (
    TrainingMethod,
    is_ppo_method,
    method_settings_field,
    validate_training_method,
)
from marl_battlegrounds.evaluation.models import (
    _validate_recorded_red_zone_depth,  # pyright: ignore[reportPrivateUsage]
)
from marl_battlegrounds.released_models import (
    _typed_name,  # pyright: ignore[reportPrivateUsage]
)

_FLOAT32_MAX = float.fromhex("0x1.fffffep+127")


def _validate_model_reference(reference: str) -> None:
    """Check one model field's spelling without files or the network.

    A typed released name such as ``"MARL-BattleGrounds/tdm-season-0-qmix@100M"``
    passes when its ``@`` suffix is a step form; the trainer later turns it
    into its exact ``hf://`` address and refuses it there if it is really a
    relative local path. Every other value must pass
    validate_saved_method_reference, so other relative paths stay refused here,
    as before. Raises TypeError or ValueError otherwise.
    """
    if not isinstance(cast(object, reference), str) or _typed_name(reference) is None:
        validate_saved_method_reference(reference)


# The five held-out validation maps, in ID order. This module owns the list so
# TrainConfig can check validation_maps without importing the validator;
# training.validation re-exports it as VALIDATION_MAPS.
VALIDATION_MAPS = (42, 43, 44, 45, 46)


def _validate_shaping_scalars(
    *, discount: object, coefficient: object = 0.01, mode: str = "potential"
) -> None:
    """Check host score-shaping settings, leaving device handling to shaping.

    discount is finite and in [0, 1]; coefficient is finite and nonnegative.
    mode accepts potential or score_delta. Reject Boolean and nonreal scalars
    with TypeError, and bad ranges or modes with ValueError. NumPy conversion
    is lazy for legacy scalar containers; ordinary Python values import no
    numerical package. Inputs are unchanged and no files are read.
    """
    _validate_shaping_mode(mode)
    for name, value in (("discount", discount), ("coefficient", coefficient)):
        if isinstance(value, bool):
            raise TypeError(f"{name} must be a real scalar, excluding Boolean values")
        if type(value) not in (int, float) or (
            type(value) is int and not -(2**63) <= value < 2**64
        ):
            import numpy as np

            scalar = np.asarray(value)
            if scalar.shape != ():
                raise ValueError(f"{name} must be a scalar")
            if scalar.dtype.kind not in "iuf":
                raise TypeError(
                    f"{name} must be a real scalar, excluding Boolean values"
                )
            value = scalar.item()
        _validate_shaping_number(cast(float, value), name=name)


def _validate_shaping_mode(mode: str) -> None:
    """Reject unknown score-shaping modes before checking scalar inputs."""
    if not isinstance(cast(object, mode), str) or mode not in (
        "potential",
        "score_delta",
    ):
        raise ValueError("shaping mode must be potential or score_delta")


def _validate_shaping_number(value: float, *, name: str) -> None:
    """Check one real shaping scalar's range without narrowing its precision.

    name is discount or coefficient; only discount has an upper limit of one.
    The caller has checked that value is real and not Boolean. NaN, infinity,
    negative values and discounts above one raise ValueError. No device reads.
    """
    if not 0 <= value < math.inf or (name == "discount" and value > 1):
        limit = "in [0, 1]" if name == "discount" else "nonnegative"
        raise ValueError(f"{name} must be finite and {limit}")


@dataclass(frozen=True)
class TrainConfig:
    """Declare one experiment without opening files or starting a backend.

    Parameters
    ----------
    seed : int, default=42
        Root seed for model and independent collection/shuffle/sampling streams.
    method : {"mappo", "ippo", "ff_mappo", "ff_ippo", "qmix", "pqn_vdn"}, \
default="mappo"
        Learner saved with the run. MAPPO uses physical critic inputs; IPPO
        uses each actor's permitted inputs. The ff_ methods use two 128-wide
        layers without recurrent memory. "qmix" is recurrent QMIX with compact
        replay, configured by qmix instead of ppo. "pqn_vdn" is recurrent
        PQN-VDN (no replay or target network), configured by pqn.
    num_envs : int, default=32
        Fixed even environment batch. Recurrent PPO methods require
        divisibility by PPO groups times minibatches. Feedforward methods
        require divisibility by groups, and rollout_length*num_envs by groups
        times minibatches. QMIX has no divisibility rule, but
        num_envs*qmix.rollout_length*5 must fit a signed 32-bit integer.
        PQN-VDN needs num_envs divisible by pqn.num_minibatches and the
        int32 guards of ``validate_pqn_batch_size``.
    total_env_steps : int, default=10000000
        Exact real environment transitions, divisible by num_envs. Resets and
        padding do not count. Per-lane rounds must fit a signed 32-bit integer.
        Whole-run totals may exceed it. QMIX needs at least min_buffer_size
        rounds per game, and rounds times qmix.epochs must fit int32.
        PQN-VDN needs at least memory_window + rollout_length + 1 rounds per
        game (W initial random rounds, counted in the budget, then at least
        one learning round) and planned optimizer steps within int32.
    curriculum, shaping : bool, default=False
        Enable the existing 17-stage schedule and chosen team reward adjustment.
        A custom list has 1..17 stages with positive share, maps, and either
        team_size or rosters={"system": [...], "opponent": [...]}. Shares sum
        to one. Optional score_threshold means the winning score. New stages
        take effect at new games only. A child may replace its future stages
        through changes.curriculum; the whole lineage can hold 17 distinct
        stage IDs so old experience keeps its original meaning.
    score_threshold_curriculum : bool, default=False
        Use K1..10, K12, K15 and K20 at future resets, with 10%, nine shares
        of 1/30, 5%, 5% and 50% of requested experience. Keeps canonical 5v5
        and all training maps. Cannot be combined with curriculum=True.
        Evaluation stays K20/H300. Actual played shares can lag stage changes.
    red_zone_depth : float, default=DEFAULT_TDM_RED_ZONE_DEPTH (5.0)
        Team Deathmatch Red Zone depth in map units for every training game
        and every validation game of this run. When an agent dies inside its
        own team's Red Zone (the strip this deep at its own spawn side), the
        enemy team gets 2 points instead of 1; it is still one kill and one
        death. 0.0 keeps one point per death. It must be a Python float that
        passes Core's scalar rules: finite, not negative (-0.0 is refused) and,
        when positive, a normal float32 value; training setup also checks it
        against every map width before any file is written. It is fixed for a
        run: a resume that declares another depth is refused. A JSON config
        may write it as an integer such as 6, which reads as 6.0; a saved run
        config without it was saved before the rule and reads as 0.0.
    shaping_coefficient : float, default=0.01
        Finite nonnegative shaping weight. Task reward remains separately logged.
    shaping_mode : {"potential", "score_delta"}, default="potential"
        Potential preserves the discounted task objective. Score_delta adds
        reward for the team's new points minus the enemy's new points without
        terminal cancellation; it deliberately changes the training objective.
        Points follow the task's scoring, so with a positive red_zone_depth a
        Red Zone death moves it by 2 instead of 1. Used only when shaping is
        True. Evaluation always uses native task rewards and win rules.
    opponents : mapping[str, float] or sequence[str] or None, default=None
        Named weights or a repeating game-start order. Names are "self", "past",
        method references, or aliases supplied by train's opponents argument.
        A name may be a released model name such as
        ``"MARL-BattleGrounds/tdm-season-0-qmix@100M"``; the name stays as typed
        and its saved binding records the exact ``hf://`` address.
        Weights are normalized. New games choose in ascending lane order;
        unfinished games keep their member. None preserves the default self/past
        recipe and the legacy pin settings. Explicit populations cannot also
        declare a legacy pin. Live bindings stay outside this JSON config.
    keep_past : int, default=20
        Number of most recent frozen actors eligible for new games. The bank
        holds one extra copy so unfinished games never lose their weights.
        Zero disables rotating history. An empty past share uses current weights.
        Every captured actor is exported; dropping it from draws does not delete it.
    past_capture_interval : int or None, default=None
        Recurring spacing in real environment transitions, independent of the
        run budget. Round up to a whole environment round, then capture at the
        first completed learner update reaching it;
        the next interval starts at that actual capture. Continuation keeps the
        saved window and cadence. Do not combine with history_capture_env_steps.
    history_capture_env_steps : tuple[int, ...] or None, default=None
        Explicit increasing capture targets, at most 20, in real transitions.
        Omit both capture settings to keep the 20 targets at 5% through 100%.
        Actual publication boundaries must be at least one maximum game length
        apart. An incompatible short run fails before files are written; use
        keep_past=0 or a sparse schedule for a short wiring check.
    learner_slots : tuple of int or None, default=None
        Physical team slots owned by the learner; None means all five. Slots
        never move when teammates are assigned. Every training stage needs at
        least one active learner slot. Frozen partners fill the other active slots.
    partners : mapping or sequence or None, default=None
        Frozen partner names with relative weights, or a repeating game-start
        order, using the same choice rule and released-name handling as
        opponents. Bind live members in
        train(partners=...). The learner and its self/past mirrors retain these
        partners on their other slots. Fixed partners receive no learner update.
    validation_partners : mapping of str to str or None, default=None
        Additional named method references for deployed-team validation;
        released model names are saved as their exact ``hf://`` addresses. Live
        members may be supplied through train(validation_partners=...). Training
        partners already receive their own familiar rows by default.
    validation_partner_labels : mapping of str to str or None, default=None
        Declared familiar, held_out or unknown labels for additional validation
        partners; missing labels mean unknown. Labels do not prove nonexposure.
    reward : str or None, default=None
        Optional pure ``module:function`` returning float32 (10,) training
        adjustments from before state, transition facts, after state and the
        completed-round count before that step. Multiply that count by num_envs
        to obtain environment transitions, including parent training. Native
        scores and evaluation rewards remain unchanged. A progress fade is one
        reward definition. Explicitly changing the definition on continuation
        clears and refills stored Q experience; resume rejects undeclared drift.
    pinned_opponent_share : float, default=0.0
        Legacy permanent-opponent share in [0, 0.8]. Without a named opponent,
        freeze the first completed update separately from the rolling bank.
        That pin remains available with keep_past=0. Before it exists its share
        uses current weights. The remaining recipe draws 20% past and current
        weights otherwise. New population settings do not inherit this share cap.
    pinned_opponent : str or None, default=None
        None keeps the pinned share playing that first-update actor. Otherwise
        a method reference that plays the pinned share instead: a built-in name
        ("random", "tdm-alpha", "tdm-beta", "tdm-gamma"), an absolute path to
        an actor export, complete learner checkpoint directory or portable
        model file (relative paths fail), a released model name such as
        ``"MARL-BattleGrounds/tdm-season-0-qmix@100M"`` or its exact ``hf://``
        address, or a
        ``module:function`` factory, resolved by
        ``load_method`` when the run is set up and again on resume. Construction
        checks a released name's spelling only; train turns it into its exact
        ``hf://`` address before any run file is written. Requires a
        positive pinned_opponent_share. JAX methods and host methods (for
        example LLM agents) are both accepted. Training does not enter a pinned
        System's resource_scope: supply an already open caller-owned LLM client,
        or keep the System's scope open around training yourself. Automatic
        managed-client setup is supported by evaluation, not pinned training.
        A host method with memory cannot be resumed while one of its games is
        unfinished. The pinned opponent
        is a training opponent: results against it, and its evidence of
        protected-controller exposure, are recorded with the run.
    initial_actor : str or None, default=None
        Compatible saved actor weights for a new run: an actor export or
        complete learner checkpoint folder, a portable model file, a released
        model name such as ``"MARL-BattleGrounds/tdm-season-0-qmix@final"`` or
        its exact ``hf://`` address. A relative path uses the current directory.
        The critic, optimizer,
        memory and counters start fresh. PQN also imports running statistics;
        QMIX starts its target Q from the same weights and uses fresh mixers.
        Resume restores the full saved learner and never reloads this path.
        The source identity and known protected-controller exposure are kept;
        a released name is saved as its exact ``hf://`` address.
    ppo : PPOConfig, default=PPOConfig()
        Immutable donor network update settings, including rollout length,
        input scale and spawn frame. A shared random initialization result can
        be reused only by a run with the same scale and frame. QMIX and
        PQN-VDN runs must leave it at the default; it is not saved for them.
    qmix : QMIXConfig or None, default=None
        QMIX settings (replay sizes, sampling, target rule, exploration, input
        scale and spawn frame). None means DEFAULT_QMIX_CONFIG for a QMIX run
        and is required for other runs. It is saved only for QMIX.
    pqn : PQNConfig or None, default=None
        PQN-VDN settings (T, H, epochs, minibatches, rates, schedules, input
        scale and spawn frame). None means DEFAULT_PQN_CONFIG for a PQN-VDN
        run and is required for other runs. It is saved only for PQN-VDN.
    metrics : {"priority", "none"}, default="priority"
        Existing episode metric level. No replay recording is implied.
    retain_all_actors : bool, default=False
        Export each saved learner boundary's actor before old recovery payloads
        can be pruned. Keeps initial, final and intermediate actors independently
        of selection. Save cadence controls archive size. False keeps the
        existing candidate/final export behavior. No extra actor export runs
        when disabled. A child may enable this for its own future boundaries.
    recording : bool, default=False
        Save episode tables through a separate training RunWriter. False skips
        recording drains; learner checkpoints and update summaries still save.
    validation_panel : str or None, default=None
        Existing frozen panel.json path. Use this or validation_opponents.
        Demonstration needs a panel by train setup; development may omit it.
    validation_opponents : tuple[str, ...] or None, default=None
        Built-in names, absolute actor export/checkpoint or model file paths,
        released model names or ``hf://`` addresses, or module:function
        factories. Released names are saved as their exact addresses.
        Freeze these once before training. Live methods instead belong in
        train's validation_opponents keyword. Conflicting declarations fail.
    slot_diagnostic_actor : str or None, default=None
        Explicit absolute actor export or model file path, released model
        name or ``hf://`` address (matching the run's method)
        for the optional slot comparison with a
        new System panel. Historical panels keep their final actor by default.
        Enabled diagnostics verify this artifact before learner setup.
    validation_fractions : tuple[float, ...], default=(0.1, ..., 1.0)
        Increasing experience fractions, rounded up to the next completed
        collection block. For PPO and QMIX blocks end at multiples of
        rollout_length rounds; for PQN-VDN they end at multiples of T rounds
        up to its W = memory_window + rollout_length initial rounds, then at
        W + j * T, and at the final round (for example 4,096, 4,224, 8,320
        transitions at 32 games with the defaults).
        Initialization is diagnostic only. The final fraction must be one.
    routine_seed_pairs, confirmation_seed_pairs : int, default=10, 50
        Independent seed pairs per validation map and opponent.
    validation_maps : tuple[int, ...] or None, default=None
        The held-out validation maps that this run's validation games use.
        None means all five maps, 42 through 46 (``validation.VALIDATION_MAPS``).
        A given value must be a nonempty list or tuple of distinct Python
        integers from those five maps, for example ``(42, 43, 44, 46)``; it is
        stored sorted, so ``[46, 42]`` becomes ``(42, 46)``. Routine checks,
        confirmation games, optional Random diagnostics and checkpoint
        selection all use exactly these maps, and every saved validation
        record names them. A run that sets it saves it in its config; None is
        left out of the saved config, so runs without it keep their old saved
        bytes. A resume or ``extend_training`` child keeps the saved maps; a
        resume config that names other maps is refused. Custom maps need a
        panel made with ``opponents=`` (or ``validation_opponents``); a
        historical two-actor panel refuses them.
    checkpoint_interval_updates : int or None, default=None
        Recovery-save interval in completed updates: PPO updates for PPO,
        optimizer steps for QMIX and PQN-VDN. A save happens whenever
        completed_updates crosses a multiple of it (a QMIX block adds epochs
        steps, so 24 to 28 crosses 25; a PQN-VDN block adds epochs *
        num_minibatches steps, and initial chunks add none). None means 25
        for PPO and 1600 for QMIX and PQN-VDN (at the defaults: 102,400
        transitions at 32 games for QMIX, 25 learning blocks for PQN-VDN);
        QMIX checkpoints carry the replay (about 1 GB at 32 games);
        the resolved number is what is saved. Initialization, capture points
        and the final boundary always save. The number is resolved when the
        config is built, so ``dataclasses.replace(ppo_config, method="qmix")``
        or ``method="pqn_vdn"`` keeps PPO's 25; pass
        ``checkpoint_interval_updates=None`` in that replace call to get the
        1600 default.
    checkpoint_env_steps : tuple[int, ...], default=()
        Additional exact actor capture points, each exactly at a collection
        boundary in transitions (the same boundaries as validation_fractions:
        for PQN-VDN these are offset by its W initial rounds, so 8,320 is a
        boundary at 32 games with the defaults and 8,192 is refused).
    random_diagnostic_seed_pairs : int or None, default=None
        Enable fixed Random diagnostics at initialization, checkpoint_env_steps
        and the final budget. The positive count is paired seeds per
        validation map (the five default maps, or validation_maps); four pairs
        on five maps give 40 games. No panel is required and no
        actor is selected from these results. None skips this optional work.
    random_initialization_result : str or None, default=None
        Original runner result copied to a JSON file for initialization reuse.
        Requires Random diagnostics. Its actor inference, task and complete evaluation
        evidence must match before output or recovery changes. Original evidence
        paths must remain available; later captures still run their own games.
    slot_diagnostic : bool, default=False
        Run the declared diagnostic after final training. Requires a frozen
        panel, the canonical full-learner roster and validation_maps=None.
    purpose : {"development", "demonstration"}, default="development"
        Demonstration requires frozen validation and final selection.
    verbose : bool, default=True
        Print existing host summaries at most every ten seconds and phase changes.
        Adds no numerical calls or device transfers. Full-run use is qualified by
        the separate measured reporting-cost check. False skips display and extra
        ETA calculation while retaining required counters, timings and status.

    Raises
    ------
    ValueError, TypeError
        A count, option or numerical setting violates this contract. Booleans
        are not accepted as integer counts. No files are changed on rejection.
    """

    seed: int = 42
    method: TrainingMethod = "mappo"
    num_envs: int = 32
    total_env_steps: int = 10_000_000
    curriculum: bool | Sequence[Mapping[str, object]] = False
    score_threshold_curriculum: bool = False
    red_zone_depth: float = DEFAULT_TDM_RED_ZONE_DEPTH
    shaping: bool = False
    reward: str | None = None
    shaping_coefficient: float = 0.01
    shaping_mode: Literal["potential", "score_delta"] = "potential"
    opponents: Mapping[str, float] | Sequence[str] | None = None
    learner_slots: tuple[int, ...] | None = None
    partners: Mapping[str, float] | Sequence[str] | None = None
    validation_partners: Mapping[str, str] | None = None
    validation_partner_labels: Mapping[str, str] | None = None
    keep_past: int = 20
    past_capture_interval: int | None = None
    history_capture_env_steps: tuple[int, ...] | None = None
    pinned_opponent_share: float = 0.0
    pinned_opponent: str | None = None
    initial_actor: str | None = None
    ppo: PPOConfig = field(default_factory=PPOConfig)
    qmix: QMIXConfig | None = None
    pqn: PQNConfig | None = None
    metrics: Literal["priority", "none"] = "priority"
    recording: bool = False
    retain_all_actors: bool = False
    validation_panel: str | None = None
    validation_opponents: tuple[str, ...] | None = None
    slot_diagnostic_actor: str | None = None
    validation_fractions: tuple[float, ...] = tuple(i / 10 for i in range(1, 11))
    routine_seed_pairs: int = 10
    confirmation_seed_pairs: int = 50
    validation_maps: tuple[int, ...] | None = None
    checkpoint_interval_updates: int | None = None
    checkpoint_env_steps: tuple[int, ...] = ()
    random_diagnostic_seed_pairs: int | None = None
    random_initialization_result: str | None = None
    slot_diagnostic: bool = False
    purpose: Literal["development", "demonstration"] = "development"
    verbose: bool = True

    def __post_init__(self) -> None:
        """Reject invalid host settings without allocating numerical arrays.

        Resolves qmix=None to DEFAULT_QMIX_CONFIG for QMIX runs, pqn=None to
        DEFAULT_PQN_CONFIG for PQN-VDN runs and checkpoint_interval_updates=None
        to 25 (PPO) or 1600 (QMIX and PQN-VDN). red_zone_depth is checked by
        the shared host copy of Core's scalar rules
        (``_validate_red_zone_depth``); map widths are checked later,
        when train prepares the content.
        """
        method = validate_training_method(self.method)
        if not isinstance(cast(object, self.ppo), PPOConfig):
            raise TypeError("ppo must be PPOConfig")
        if method == "qmix":
            if self.ppo != PPOConfig():
                raise ValueError("A QMIX run takes no PPO settings")
            if self.qmix is None:
                object.__setattr__(self, "qmix", DEFAULT_QMIX_CONFIG)
            if not isinstance(cast(object, self.qmix), QMIXConfig):
                raise TypeError("qmix must be QMIXConfig or None")
        elif self.qmix is not None:
            raise ValueError(
                "A PQN-VDN run takes no QMIX settings"
                if method == "pqn_vdn"
                else "A PPO run takes no QMIX settings"
            )
        if method == "pqn_vdn":
            if self.ppo != PPOConfig():
                raise ValueError("A PQN-VDN run takes no PPO settings")
            if self.pqn is None:
                object.__setattr__(self, "pqn", DEFAULT_PQN_CONFIG)
            if not isinstance(cast(object, self.pqn), PQNConfig):
                raise TypeError("pqn must be PQNConfig or None")
        elif self.pqn is not None:
            raise ValueError(
                "A QMIX run takes no PQN settings"
                if method == "qmix"
                else "A PPO run takes no PQN settings"
            )
        if self.checkpoint_interval_updates is None:
            object.__setattr__(
                self,
                "checkpoint_interval_updates",
                25 if is_ppo_method(method) else 1600,
            )
        if type(self.seed) is not int:
            raise TypeError("seed must be a Python integer, not bool")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be in [0, 2**32)")
        for name in (
            "num_envs",
            "total_env_steps",
            "routine_seed_pairs",
            "confirmation_seed_pairs",
            "checkpoint_interval_updates",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive Python integer")
        for name in (
            "score_threshold_curriculum",
            "shaping",
            "recording",
            "retain_all_actors",
            "slot_diagnostic",
            "verbose",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        if type(self.curriculum) is not bool:
            _custom_stages(self.curriculum)
            # Copy researcher-owned lists before the frozen config retains them.
            object.__setattr__(
                self,
                "curriculum",
                json.loads(json.dumps(self.curriculum, allow_nan=False)),
            )
        if self.curriculum and self.score_threshold_curriculum:
            raise ValueError("Choose team/map curriculum or score-threshold curriculum")

        _validate_red_zone_depth(self.red_zone_depth)
        if self.num_envs % 2:
            raise ValueError("num_envs must be even for paired spawn ends")
        if self.total_env_steps % self.num_envs:
            raise ValueError("total_env_steps must be divisible by num_envs")
        rounds = self.total_env_steps // self.num_envs
        if rounds > 2**31 - 1:
            raise ValueError("per-lane rounds must fit int32")
        if self.qmix is not None:
            validate_qmix_batch_size(self.num_envs, self.qmix)
            if rounds < self.qmix.min_buffer_size:
                raise ValueError(
                    "A QMIX budget needs at least min_buffer_size rounds per game"
                )
            if rounds * self.qmix.epochs > 2**31 - 1:
                raise ValueError("QMIX optimizer counts for this budget exceed int32")
            discount, rollout_length = self.qmix.gamma, self.qmix.rollout_length
        elif self.pqn is not None:
            validate_pqn_batch_size(self.num_envs, self.pqn)
            pqn_planned_learning_blocks(rounds, self.pqn)
            discount, rollout_length = self.pqn.gamma, self.pqn.rollout_length
        else:
            validate_ppo_batch_size(self.num_envs, self.ppo, method=method)
            discount, rollout_length = self.ppo.gamma, self.ppo.rollout_length
        if self.metrics not in ("priority", "none"):
            raise ValueError("training metrics must be priority or none")
        if (
            isinstance(self.shaping_coefficient, bool)
            or not math.isfinite(self.shaping_coefficient)
            or self.shaping_coefficient < 0
        ):
            raise ValueError("shaping_coefficient must be finite and nonnegative")
        _validate_shaping_scalars(
            discount=discount,
            coefficient=self.shaping_coefficient,
            mode=self.shaping_mode,
        )
        if self.reward is not None and (
            not isinstance(cast(object, self.reward), str)
            or self.reward.count(":") != 1
            or not all(self.reward.split(":"))
        ):
            raise ValueError("reward must be a module:function reference or None")
        if self.initial_actor is not None and (
            not isinstance(cast(object, self.initial_actor), str)
            or not self.initial_actor.strip()
        ):
            raise ValueError(
                "initial_actor must be a nonempty actor path, model file, "
                "released model name or hf:// address, or None"
            )
        if self.learner_slots is not None:
            slots = tuple(self.learner_slots)
            if (
                not slots
                or len(set(slots)) != len(slots)
                or any(type(slot) is not int or not 0 <= slot < 5 for slot in slots)
            ):
                raise ValueError("learner_slots needs distinct slots from 0 through 4")
            object.__setattr__(self, "learner_slots", slots)
        if self.partners is not None:
            names = opponent_selection_names(self.partners)
            if {"self", "past"} & set(names):
                raise ValueError("Partner names cannot be self or past")
            object.__setattr__(
                self,
                "partners",
                dict(self.partners)
                if isinstance(self.partners, Mapping)
                else tuple(self.partners),
            )
        for field_name in ("validation_partners", "validation_partner_labels"):
            value = getattr(self, field_name)
            if value is not None:
                if not isinstance(value, Mapping):
                    raise ValueError(f"{field_name} needs a mapping")
                value = cast(Mapping[object, object], value)
                if any(
                    not isinstance(name, str)
                    or not name.strip()
                    or not isinstance(reference, str)
                    or not reference.strip()
                    for name, reference in value.items()
                ):
                    raise ValueError(
                        f"{field_name} needs nonempty names and text values"
                    )
                object.__setattr__(self, field_name, dict(value))
        if self.validation_partner_labels is not None and any(
            label not in {"familiar", "held_out", "unknown"}
            for label in self.validation_partner_labels.values()
        ):
            raise ValueError("Partner labels must be familiar, held_out or unknown")
        if self.opponents is not None:
            opponent_selection_names(self.opponents)
            if self.pinned_opponent is not None or self.pinned_opponent_share != 0:
                raise ValueError(
                    "Declare opponents or legacy pinned settings, not both"
                )
            object.__setattr__(
                self,
                "opponents",
                dict(self.opponents)
                if isinstance(self.opponents, Mapping)
                else tuple(self.opponents),
            )
        if type(self.keep_past) is not int or not 0 <= self.keep_past < 2**31 - 1:
            raise ValueError("keep_past must be a nonnegative integer below 2**31 - 1")
        if self.past_capture_interval is not None:
            if (
                type(self.past_capture_interval) is not int
                or self.past_capture_interval <= 0
                or -(-self.past_capture_interval // self.num_envs) >= 2**31
            ):
                raise ValueError(
                    "past_capture_interval must be a positive transition count "
                    "that fits the training round clock"
                )
            if self.history_capture_env_steps is not None:
                raise ValueError(
                    "Declare past_capture_interval or history_capture_env_steps, "
                    "not both"
                )
        if self.history_capture_env_steps is not None:
            points = self.history_capture_env_steps
            if (
                not isinstance(cast(object, points), tuple)
                or len(points) > 20
                or any(
                    type(point) is not int
                    or not 0 < point <= self.total_env_steps
                    or point % self.num_envs
                    for point in points
                )
                or tuple(sorted(set(points))) != points
            ):
                raise ValueError(
                    "history_capture_env_steps needs at most 20 increasing "
                    "whole-round targets within the budget"
                )
        _pinned_share(self.pinned_opponent_share)
        if self.pinned_opponent is not None:
            if (
                not isinstance(cast(object, self.pinned_opponent), str)
                or not self.pinned_opponent.strip()
            ):
                raise ValueError("pinned_opponent must be None or a nonempty string")
            if not self.pinned_opponent_share > 0:
                raise ValueError(
                    "pinned_opponent needs a positive pinned_opponent_share"
                )
            _validate_model_reference(self.pinned_opponent)
        if self.purpose not in ("development", "demonstration"):
            raise ValueError("purpose must be development or demonstration")
        if self.validation_panel is not None and (
            not isinstance(cast(object, self.validation_panel), str)
            or not self.validation_panel
        ):
            raise ValueError("validation_panel must be a nonempty path string or None")
        if self.random_diagnostic_seed_pairs is not None and (
            type(self.random_diagnostic_seed_pairs) is not int
            or self.random_diagnostic_seed_pairs <= 0
        ):
            raise ValueError(
                "random_diagnostic_seed_pairs must be a positive integer or None"
            )
        if self.random_initialization_result is not None and (
            not isinstance(cast(object, self.random_initialization_result), str)
            or not self.random_initialization_result
            or self.random_diagnostic_seed_pairs is None
        ):
            raise ValueError(
                "random_initialization_result needs a path and enabled "
                "Random diagnostics"
            )
        if self.validation_opponents is not None:
            if (
                not isinstance(cast(object, self.validation_opponents), tuple)
                or not self.validation_opponents
            ):
                raise ValueError(
                    "validation_opponents must be a nonempty tuple of references"
                )
            for reference in self.validation_opponents:
                _validate_model_reference(reference)
            if self.validation_panel is not None:
                raise ValueError(
                    "Declare validation_panel or validation_opponents, not both"
                )
        if self.slot_diagnostic_actor is not None and (
            not isinstance(cast(object, self.slot_diagnostic_actor), str)
            or not (
                Path(self.slot_diagnostic_actor).is_absolute()
                or self.slot_diagnostic_actor.startswith("hf://")
                or _typed_name(self.slot_diagnostic_actor) is not None
            )
        ):
            raise ValueError(
                "slot_diagnostic_actor must be an absolute actor or model file "
                "path, a released model name or an hf:// address"
            )
        if self.slot_diagnostic_actor is not None and (
            self.slot_diagnostic_actor.startswith("hf://")
        ):
            validate_saved_method_reference(self.slot_diagnostic_actor)
        if self.slot_diagnostic and not (
            self.validation_panel
            or self.validation_opponents
            or self.slot_diagnostic_actor
        ):
            raise ValueError("slot_diagnostic requires a frozen validation panel")
        fractions = self.validation_fractions
        if (
            not isinstance(cast(object, fractions), tuple)
            or not fractions
            or fractions[-1] != 1
        ):
            raise ValueError(
                "validation_fractions must be a nonempty tuple ending at 1"
            )
        previous = 0.0
        for fraction in fractions:
            if (
                isinstance(fraction, bool)
                or not math.isfinite(fraction)
                or not previous < fraction <= 1
            ):
                raise ValueError("validation_fractions must increase within (0, 1]")
            previous = fraction
        if self.validation_maps is not None:
            object.__setattr__(
                self, "validation_maps", _checked_validation_maps(self.validation_maps)
            )
        if not isinstance(cast(object, self.checkpoint_env_steps), tuple):
            raise TypeError("checkpoint_env_steps must be a tuple")

        initial = self.pqn.initial_rounds if self.pqn is not None else 0
        previous_step = 0
        for step in self.checkpoint_env_steps:
            if (
                type(step) is not int
                or not previous_step < step <= self.total_env_steps
                or _collection_boundary(
                    step,
                    total_env_steps=self.total_env_steps,
                    num_envs=self.num_envs,
                    rollout_length=rollout_length,
                    initial_rounds=initial,
                )[0]
                != step
            ):
                raise ValueError(
                    "checkpoint_env_steps must increase at complete update boundaries"
                )
            previous_step = step


def _checked_validation_maps(value: object) -> tuple[int, ...]:
    """Check one validation map choice and return it as a sorted tuple.

    Parameters
    ----------
    value : object
        The declared validation_maps: a list or tuple of map IDs. A string,
        mapping, set or other value is refused.

    Returns
    -------
    tuple[int, ...]
        The same distinct map IDs in increasing order, for example
        ``(42, 43, 44, 46)``.

    Raises
    ------
    ValueError
        value is not a list or tuple; it is empty; an entry is not a Python
        integer (True and False are refused); an entry is not one of the five
        held-out maps in VALIDATION_MAPS; or an ID repeats.

    Notes
    -----
    Host-only; reads no files and does not change value.
    """
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            "validation_maps must be a list or tuple of validation map IDs or None"
        )
    maps = cast(Sequence[object], value)
    if not maps:
        raise ValueError("validation_maps must name at least one map")
    if any(type(item) is not int or item not in VALIDATION_MAPS for item in maps):
        raise ValueError(
            "validation_maps may only name the held-out validation maps "
            f"{', '.join(map(str, VALIDATION_MAPS))}"
        )
    if len(set(maps)) != len(maps):
        raise ValueError("validation_maps must not repeat a map")
    return tuple(sorted(cast(Sequence[int], maps)))


def config_to_dict(config: TrainConfig) -> dict[str, Any]:
    """Return version-1 JSON-ready settings; preserve every resolved option.

    Parameters
    ----------
    config : TrainConfig
        A validated training config. Its defaults are already resolved, for
        example qmix=None has become DEFAULT_QMIX_CONFIG for a QMIX run.

    Returns
    -------
    dict[str, Any]
        A new dictionary: "schema_version": 1 first, then every TrainConfig
        field in field order after a JSON round trip, so tuples become lists
        and settings objects become dictionaries. Only the run's own settings
        block is kept (``methods.method_settings_field``): a PPO config omits
        qmix and pqn, so historical PPO bytes are unchanged; a QMIX config
        keeps only its resolved qmix settings and a PQN-VDN config only its
        resolved pqn settings, in the form ``baselines.config.saved_settings``
        gives (QMIX/PQN reward_scale 1.0 and PQN normalization "batch_norm"
        are left out, so runs that do not use them keep their historical
        bytes). red_zone_depth is always written as a JSON
        number such as 5.0 (map units; 0.0 means the Red Zone rule is off), so
        every new saved config states its scoring rule. The schema version
        stays 1 (a config saved before the rule lacks the key).
        validation_maps is written as a sorted list only when it is set;
        None (all five validation maps) is left out, so default runs keep
        the exact bytes they had before the setting existed.

    Notes
    -----
    Host-only; config is not changed. train saves this dictionary in the run's
    metadata, which checkpoints carry as metadata.config, and compares two of
    them to check a resume config. config_from_dict reads it back.
    """
    data = json.loads(json.dumps(asdict(config)))
    if data["validation_maps"] is None:
        # Older runs never saved this key; leaving the default out keeps
        # their saved configs and every comparison with them unchanged.
        del data["validation_maps"]
    kept = method_settings_field(config.method)
    for name in ("ppo", "qmix", "pqn"):
        if name != kept:
            data.pop(name)
    settings = getattr(config, kept)
    if settings is not None:
        data[kept] = saved_settings(settings)
    return {"schema_version": 1, **data}


def config_from_dict(value: dict[str, Any]) -> TrainConfig:
    """Read version-1 settings and reject unknown fields before opening a run.

    Parameters
    ----------
    value : dict[str, Any]
        A JSON object with optional schema_version=1. Omitted settings use
        TrainConfig defaults; ppo, qmix and pqn are settings objects. A config
        may contain only its own method's block: a QMIX config no ppo or pqn,
        a PQN-VDN config no ppo or qmix, a PPO config no qmix or pqn (method
        defaults to "mappo"). red_zone_depth is the Red Zone depth in map
        units; 0.0 turns the rule off. An omitted validation_maps means None,
        all five validation maps, which is also how every config saved before
        the setting existed reads.

    Returns
    -------
    TrainConfig
        The validated config. Lists for validation fractions, extra save points
        (checkpoint_env_steps), validation_opponents and validation_maps
        become tuples. A JSON
        integer red_zone_depth (not true or false) becomes the equal float, so
        6 reads as 6.0; an omitted depth takes the TrainConfig default
        (DEFAULT_TDM_RED_ZONE_DEPTH, 5.0).

    Raises
    ------
    ValueError
        schema_version is not the integer 1; a field is unknown; a settings
        block belongs to another method; an integer red_zone_depth is too
        large to become a float ("red_zone_depth must be finite"); or
        TrainConfig or a settings object rejects a value.
    TypeError
        ppo, qmix or pqn is not a JSON object; one of the list settings
        (validation_fractions, checkpoint_env_steps, history_capture_env_steps
        or validation_opponents) is not a JSON array; a settings object gets an
        unknown key; or TrainConfig or a settings object rejects a value's type.

    Notes
    -----
    Host-only; the input dictionary is not changed. An omitted depth means
    5.0 here, so a config saved before the Red Zone rule must first pass
    through checkpoints.saved_training_config, which fills the missing depth
    with 0.0 (its original one-point scoring).
    """
    data = dict(value)
    version = data.pop("schema_version", 1)
    if type(version) is not int or version != 1:
        raise ValueError("Unsupported training config schema_version")
    unknown = set(data) - {item.name for item in fields(TrainConfig)}
    if unknown:
        raise ValueError(f"Unknown training config fields: {sorted(unknown)}")
    method = data.get("method", "mappo")
    label = {"qmix": "QMIX", "pqn_vdn": "PQN-VDN"}.get(str(method), "PPO")
    kept = method_settings_field(method)
    for name in ("ppo", "qmix", "pqn"):
        if name != kept and name in data:
            raise ValueError(f"A {label} training config takes no {name} settings")
    if "ppo" in data:
        if not isinstance(data["ppo"], dict):
            raise TypeError("ppo must be a JSON object")
        data["ppo"] = PPOConfig(**cast(dict[str, Any], data["ppo"]))
    if "qmix" in data and data["qmix"] is not None:
        if not isinstance(data["qmix"], dict):
            raise TypeError("qmix must be a JSON object")
        data["qmix"] = QMIXConfig(**cast(dict[str, Any], data["qmix"]))
    if "pqn" in data and data["pqn"] is not None:
        if not isinstance(data["pqn"], dict):
            raise TypeError("pqn must be a JSON object")
        data["pqn"] = PQNConfig(**cast(dict[str, Any], data["pqn"]))
    for name in (
        "validation_fractions",
        "checkpoint_env_steps",
        "history_capture_env_steps",
        "validation_opponents",
    ):
        if name in data and data[name] is not None:
            if not isinstance(data[name], (list, tuple)):
                raise TypeError(f"{name} must be a JSON array")
            data[name] = tuple(data[name])
    # JSON has one number type: a hand-written 6 means the depth 6.0.
    if type(data.get("red_zone_depth")) is int:
        try:
            data["red_zone_depth"] = float(data["red_zone_depth"])
        except OverflowError as error:
            raise ValueError("red_zone_depth must be finite") from error
    return TrainConfig(**data)


def read_config(path: str | Path) -> TrainConfig:
    """Load a UTF-8 JSON training config through the shared strict validator.

    path must name a JSON object. Read/parse errors propagate; invalid settings
    raise ValueError or TypeError before training files or a backend are opened.
    """
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Training config must be a JSON object")
    return config_from_dict(cast(dict[str, Any], value))


_STAGES = 17


_MAPS = 42


def _custom_stages(value: object) -> tuple[Mapping[str, object], ...]:
    """Validate a stage list and copy its JSON values into immutable mappings.

    Stages use positive finite shares summing exactly to one as decimal
    fractions. Maps are training IDs 0..41. Roster names use the task owner's
    validator; thresholds receive their depth-specific upper check when the
    source bank is prepared. This helper reads no assets and changes no inputs.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("curriculum must be bool or a sequence of stage mappings")
    stages = cast(Sequence[object], value)
    if not 1 <= len(stages) <= _STAGES:
        raise ValueError("A custom curriculum needs 1..17 stages")
    total = Fraction(0)
    for index, raw in enumerate(stages):
        if not isinstance(raw, Mapping):
            raise TypeError(f"Curriculum stage {index} must be a mapping")
        stage = cast(Mapping[str, Any], raw)
        keys = set(stage)
        if (
            not {"share", "maps"} <= keys
            or keys - {"share", "maps", "team_size", "rosters", "score_threshold"}
            or ("team_size" in keys) == ("rosters" in keys)
        ):
            raise ValueError(
                f"Curriculum stage {index} needs share, maps and exactly one of "
                "team_size or rosters"
            )
        share = stage["share"]
        if type(share) not in (int, float) or not 0 < share <= 1:
            raise ValueError("Stage shares must be positive finite numbers")
        total += Fraction(str(share))
        raw_maps = stage["maps"]
        if not isinstance(raw_maps, (list, tuple)):
            raise ValueError("Stage maps must be a list of training IDs from 0..41")
        maps = cast(Sequence[object], raw_maps)
        if (
            not maps
            or any(type(item) is not int or not 0 <= item < _MAPS for item in maps)
            or len(set(maps)) != len(maps)
        ):
            raise ValueError("Stage maps must be distinct training IDs from 0..41")
        if "team_size" in stage:
            size = stage["team_size"]
            if type(size) is not int or not 1 <= size <= 5:
                raise ValueError("Stage team_size must be a Python integer in 1..5")
        else:
            rosters = cast(Mapping[str, object], stage["rosters"])
            if not isinstance(cast(object, rosters), Mapping) or set(rosters) != {
                "system",
                "opponent",
            }:
                raise ValueError("Stage rosters must name system and opponent")
            for team in ("system", "opponent"):
                _validate_roster(
                    cast(Sequence[AgentClassName], rosters[team]), name=team
                )
        threshold = stage.get("score_threshold", 20)
        if type(threshold) is not int or not 1 <= threshold <= 2**24 - 4:
            raise ValueError(
                "Stage score_threshold is outside the supported score range"
            )
    if total != 1:
        raise ValueError("Curriculum stage shares must sum to one")
    frozen = _freeze_continuation({"stages": stages})
    return cast(tuple[Mapping[str, object], ...], frozen["stages"])


def _freeze_continuation(value: Mapping[str, object]) -> Mapping[str, object]:
    """Copy a finite JSON-like declaration into read-only nested records.

    Mapping keys must be strings; leaves must be None, strings, booleans, finite
    numbers or sequences of these. Raises ValueError for other values. No input
    is retained as mutable storage. Use _continuation_details to encode a schedule.
    """

    def freeze(item: object) -> object:
        """Freeze one mapping or sequence, checking its scalar leaves."""
        if isinstance(item, Mapping):
            mapping = cast(Mapping[object, object], item)
            if any(type(key) is not str for key in mapping):
                raise ValueError("Continuation keys must be strings")
            return MappingProxyType(
                {cast(str, key): freeze(val) for key, val in mapping.items()}
            )
        if isinstance(item, (tuple, list)):
            return tuple(
                freeze(val) for val in cast(tuple[object, ...] | list[object], item)
            )
        if item is None or type(item) in (str, bool, int):
            return item
        if type(item) is float and math.isfinite(item):
            return item
        raise ValueError("Continuation must contain finite JSON values")

    if not isinstance(cast(object, value), Mapping):
        raise ValueError("Continuation declaration must be a mapping")
    return cast(Mapping[str, object], freeze(value))


def opponent_selection_names(selection: object) -> tuple[str, ...]:
    """Validate training shares or a repeating order and return distinct names.

    None keeps the existing self/history recipe. A mapping gives finite,
    nonnegative relative shares with a positive total. A nonempty sequence
    gives an exact repeating game-start order; repeated names are allowed.
    Names must be nonempty strings. This does not load or resolve any member.
    """
    if selection is None:
        return ()
    if isinstance(selection, Mapping):
        if not selection:
            raise ValueError("Opponent shares must not be empty")
        values = list(cast(Mapping[Any, Any], selection).values())
        if any(
            isinstance(v, bool) or not isinstance(cast(object, v), (int, float))
            for v in values
        ):
            raise TypeError("Opponent shares must be real numbers")
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("Opponent shares must be finite and nonnegative")
        if not math.isfinite(sum(values)) or sum(values) <= 0:
            raise ValueError("Opponent shares must have a finite positive total")
        names = tuple(cast(Mapping[Any, Any], selection))
    elif isinstance(selection, Sequence) and not isinstance(selection, (str, bytes)):
        names = tuple(cast(Sequence[Any], selection))
        if not names:
            raise ValueError("Opponent order must not be empty")
    else:
        raise TypeError("Opponent selection must be a share mapping or a name sequence")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Opponent names must be nonempty strings")
    return tuple(dict.fromkeys(names))


def _pinned_share(value: object) -> None:
    """Require a static Python pinned-opponent share within [0, 0.8].

    Parameters
    ----------
    value : object
        The share to check. Zero reserves no probability for a permanent pin.

    Raises
    ------
    TypeError
        value is bool, a traced array or another non-number.
    ValueError
        value is not finite or lies outside [0, 0.8].
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            "The pinned opponent share must be a Python number, not bool, string or "
            "traced"
        )
    if not math.isfinite(value) or not 0 <= value <= 0.8:
        raise ValueError("The pinned opponent share must be finite within [0, 0.8]")


def validate_red_zone_depth(depth: object) -> float:
    """Check one Red Zone depth with Core's scalar rules, without a map width.

    Parameters
    ----------
    depth : object
        Red Zone depth in map units, declared for new content or read from a
        version 3 binding. 0.0 keeps one point per kill.

    Returns
    -------
    float
        The same depth, unchanged.

    Raises
    ------
    TypeError
        depth is not exactly a Python float. bool, int and NumPy scalars are
        refused, as Core refuses them.
    ValueError
        depth is not finite; has a negative sign (-0.0 included, so "off" has
        one spelling); or, when positive, its float32 value is not finite or
        is below the smallest normal float32 value.

    Notes
    -----
    Host-only; reads no files. The sign and float32 rules come from the
    recorded-config owner in evaluation.models. A binding records no map
    widths, so the width limit is given as the largest float32 value and never
    binds here; Core checks each map's width when preparation builds its
    configuration.
    """
    if type(depth) is not float:
        raise TypeError(
            f"red_zone_depth must be a Python float, not {type(depth).__name__}"
        )
    if not math.isfinite(depth):
        raise ValueError(f"red_zone_depth must be finite, not {depth}")
    _validate_recorded_red_zone_depth(depth, task_mode=1, map_width=_FLOAT32_MAX)
    return depth


# Keep existing internal imports on the same scalar rule.
_validate_red_zone_depth = validate_red_zone_depth


def _collection_boundary(
    requested_steps: int,
    *,
    total_env_steps: int,
    num_envs: int,
    rollout_length: int,
    initial_rounds: int = 0,
) -> tuple[int, int]:
    """Return the first reachable collection boundary at or above a request.

    Parameters
    ----------
    requested_steps : int
        Nonnegative requested experience in environment transitions.
    total_env_steps : int
        Positive declared budget, divisible by num_envs; its round count is
        R = total_env_steps / num_envs.
    num_envs : int
        Positive number of game lanes (B).
    rollout_length : int
        Positive rounds in a normal collection block (T).
    initial_rounds : int, default=0
        Nonnegative rounds collected before normal blocks start (W). Zero is the
        PPO and QMIX rule. PQN passes W = memory_window + rollout_length; W must
        then be less than R.

    Returns
    -------
    tuple[int, int]
        ``(steps, ordinal)``. In rounds, reachable boundaries are: zero; before
        W, multiples of T capped at W; after W, ``W + j * T`` capped at R.
        steps is the first reachable boundary at or above the request, capped at
        the total, times B. ordinal is the number of collection blocks that
        reach it (``ceil(W / T) + ceil((r - W) / T)`` after W, ``ceil(r / T)``
        before). A request of zero gives ``(0, 0)``.

    Raises
    ------
    ValueError
        A count is not a plain integer in range, the total is not divisible by
        num_envs, or a positive initial_rounds is not below R.

    Notes
    -----
    Integer ceil division only, so results are exact for any Python integer.
    With ``initial_rounds=0`` this equals the earlier rule
    ``min(total, ceil(requested / (B * T)) * B * T)``. One helper serves
    explicit save checks, fraction resolution, learner validation and host
    recovery; no list of every block is built. Host-only; no I/O.
    """
    requested = _integer(requested_steps, "requested_steps", minimum=0)
    total = _integer(total_env_steps, "total_env_steps", minimum=1)
    batch = _integer(num_envs, "num_envs", minimum=1)
    length = _integer(rollout_length, "rollout_length", minimum=1)
    initial = _integer(initial_rounds, "initial_rounds", minimum=0)
    if total % batch:
        raise ValueError("total_env_steps must be divisible by num_envs")
    budget = total // batch
    if initial and initial >= budget:
        raise ValueError("initial_rounds must be below the total rounds")
    wanted = -(-requested // batch)
    if wanted <= 0:
        return 0, 0
    if initial and wanted <= initial:
        rounds = min(initial, -(-wanted // length) * length)
        return rounds * batch, -(-rounds // length)
    rounds = min(budget, initial + -(-(wanted - initial) // length) * length)
    ordinal = -(-initial // length) + -(-(rounds - initial) // length)
    return rounds * batch, ordinal


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    """Require a plain integer at least minimum; Boolean values raise ValueError."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


TRAINING_KEY_SCHEMA_VERSION = 1
