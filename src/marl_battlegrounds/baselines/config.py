# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from Mava for MARL-BGs.
# See docs/method_sources.md in the source repository for source identities
# and deliberate changes.
#
# Value normalization follows on-policy under the following MIT terms.
# MIT License
#
# Copyright (c) 2021 MAPPO contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Copyright 2023 The JaxMARL Authors. All rights reserved.
"""Define baseline settings without importing numerical or training libraries.

PPOConfig, QMIXConfig and PQNConfig keep the baseline defaults and validation;
saved_settings gives the one JSON form that runs save for each of them, and
with_later_settings reads older saved blocks that predate a setting.
The numerical modules re-export these exact objects for existing callers.
Batch and budget checks run on the host and allocate no model or device arrays.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Literal, cast

type PPOMethod = Literal["mappo", "ippo", "ff_mappo", "ff_ippo", "happo"]

_METHODS = {
    "mappo": (True, False, "Recurrent MAPPO"),
    "ippo": (True, True, "Recurrent IPPO"),
    "ff_mappo": (False, False, "Feedforward MAPPO"),
    "ff_ippo": (False, True, "Feedforward IPPO"),
    "happo": (True, False, "Recurrent HAPPO"),
}

_PARAMETER_SHARING = ("all", "class", "none")

PQN_NORMALIZATIONS = ("batch_norm", "layer_norm")
"""PQN hidden-layer normalization choices, matching the donor's NORM_TYPE names."""

TEAM_SLOTS = 5
"""Team A actor slots mixed into one team value; smaller rosters pad inactive."""

_INT32_MAX = 2**31 - 1

SPAWN_FRAMES = ("world", "left")


def validate_ppo_method(method: str) -> PPOMethod:
    """Return a supported static PPO method name or raise ValueError.

    Accept mappo, ippo, ff_mappo, ff_ippo and happo. HAPPO (Kuba et al., ICLR
    2022) uses recurrent MAPPO's networks and training-only critic view and
    differs only in its update: one actor group at a time, in a random order.
    This host-only check allocates no
    arrays and owns the method choices shared by models, training and storage.
    """
    if not isinstance(cast(object, method), str) or method not in _METHODS:
        raise ValueError("PPO method must be mappo, ippo, ff_mappo, ff_ippo or happo")
    return cast(PPOMethod, method)


def is_recurrent_method(method: str) -> bool:
    """Return whether a checked PPO method uses recurrent actor/critic memory.

    method is a static name accepted by validate_ppo_method. Invalid names raise
    ValueError. No network, arrays or mutable state are created.
    """
    return _METHODS[validate_ppo_method(method)][0]


def _input_scale(value: float) -> float:
    """Check a fixed positive finite real input scale and return a Python float.

    This host check accepts real scalar numbers, excluding Booleans. Arrays,
    traced values, strings and nonpositive/nonfinite values raise ValueError.
    It performs no device work and stores no normalization statistics.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("input_scale must be a finite positive real number.")
    return float(value)


def _spawn_frame(value: object) -> str:
    """Check a static spawn frame name and return it as a Python string.

    This host check accepts exactly the strings "world" and "left". Any other
    value, including the removed "right", Booleans, None and traced arrays,
    raises ValueError. It performs no device work.
    """
    if not isinstance(value, str) or value not in SPAWN_FRAMES:
        raise ValueError('spawn_frame must be "world" or "left".')
    return value


def _parameter_sharing(value: str) -> str:
    """Check the fixed actor weight rule: all actors, own class or physical slot."""
    if value not in _PARAMETER_SHARING:
        raise ValueError("parameter_sharing must be all, class or none")
    return value


def validate_ppo_batch_size(
    games: int, config: PPOConfig, *, method: str = "mappo"
) -> None:
    """Check one static PPO game's batch against its grouping and minibatches.

    games is a positive Python count; config supplies positive rollout length,
    groups and minibatches. Recurrent game count must be divisible by
    groups*minibatches. Feedforward game count must be divisible by groups,
    and rollout_length*games must be divisible by groups*minibatches.
    Invalid method or divisibility raises ValueError. This host/static helper
    performs no device work; training separately requires an even game count.
    """
    recurrent = is_recurrent_method(method)
    if type(games) is not int or games < 1 or games % config.groups:
        raise ValueError("PPO needs positive B divisible by groups")
    rows = games if recurrent else config.rollout_length * games
    if rows % (config.groups * config.minibatches):
        kind = "B" if recurrent else "T*B"
        raise ValueError(f"PPO needs {kind} divisible by groups*minibatches")


def _count(value: object, name: str, *, minimum: int = 1) -> int:
    """Check one plain integer count within [minimum, int32 max] and return it.

    Booleans, floats, arrays and out-of-range values raise ValueError. This
    host-only check performs no device work.
    """
    if type(value) is not int or not minimum <= value <= _INT32_MAX:
        raise ValueError(f"{name} must be a plain integer in [{minimum}, 2**31-1].")
    return value


def _unit_interval(value: object, name: str) -> float:
    """Check one finite real number in [0, 1], excluding Booleans; return float."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number in [0, 1].")
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{name} must be a finite real number in [0, 1].")
    return number


def _positive(value: object, name: str) -> float:
    """Check one finite positive real number, excluding Booleans; return float."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive real number.")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive real number.")
    return number


def _pqn_normalization(value: object) -> str:
    """Check one PQN hidden-layer normalization name and return it.

    value must be the string "batch_norm" or "layer_norm"; anything else,
    including other types, raises ValueError. Host-only, no device work.
    """
    if not isinstance(value, str) or value not in PQN_NORMALIZATIONS:
        raise ValueError("normalization must be batch_norm or layer_norm")
    return value


def validate_qmix_batch_size(num_envs: int, config: QMIXConfig) -> None:
    """Check that QMIX's int32 counts for num_envs games cannot overflow.

    Parameters
    ----------
    num_envs : int
        Plain positive number of game lanes (B).
    config : QMIXConfig
        Settings whose rollout_length bounds one block and whose eps_decay
        sets the exploration clock.

    Raises
    ------
    ValueError
        ``num_envs * rollout_length * 5`` exceeds int32 (the shared update
        summary reduces one block's actor rows into int32 counters), or
        ``eps_decay + num_envs - 1`` exceeds int32 (the largest value of the
        saturated exploration clock in ``epsilon_at``).

    Notes
    -----
    Host-only; it creates no arrays. ``TrainConfig`` and
    ``init_qmix_learner`` both call it before any file or array work.
    Whole-run totals are Python integers and are not limited by this check.
    """
    _count(num_envs, "num_envs")
    if num_envs * config.rollout_length * TEAM_SLOTS > _INT32_MAX:
        raise ValueError("One QMIX block's actor counts would overflow int32")
    if config.eps_decay + num_envs - 1 > _INT32_MAX:
        raise ValueError("The QMIX exploration clock would overflow int32")


def _fraction(value: object, name: str) -> float:
    """Check one finite real number in (0, 1], excluding Booleans; return float."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number in (0, 1].")
    number = float(value)
    if not math.isfinite(number) or not 0 < number <= 1:
        raise ValueError(f"{name} must be a finite real number in (0, 1].")
    return number


def validate_pqn_batch_size(num_envs: int, pqn: PQNConfig) -> None:
    """Check that num_envs fits PQN's minibatches and int32 block counters.

    Parameters
    ----------
    num_envs : int
        Plain positive number of game lanes (B).
    pqn : PQNConfig
        Settings whose rollout_length, memory_window, epochs and
        num_minibatches bound one block.

    Raises
    ------
    ValueError
        num_envs is not a plain positive int32 value; num_minibatches does not
        divide it; ``B * T * 5`` exceeds int32 (the shared update summary
        reduces one block's actor rows into int32); or
        ``E * B * (H + T - 1) * 5`` exceeds int32 (one learning block's used
        agent utilities, the largest device reduction).

    Notes
    -----
    Host-only; it creates no arrays. ``TrainConfig`` and ``init_pqn_learner``
    both call it before any file or array work. Whole-run totals are Python
    integers and are not limited by this check.
    """
    _count(num_envs, "num_envs")
    if num_envs % pqn.num_minibatches:
        raise ValueError("PQN needs num_minibatches to divide num_envs")
    if num_envs * pqn.rollout_length * TEAM_SLOTS > _INT32_MAX:
        raise ValueError("One PQN block's actor counts would overflow int32")
    pairs = pqn.memory_window + pqn.rollout_length - 1
    if pqn.epochs * num_envs * pairs * TEAM_SLOTS > _INT32_MAX:
        raise ValueError("One PQN block's used utilities would overflow int32")


def pqn_planned_learning_blocks(total_rounds: int, pqn: PQNConfig) -> int:
    """Return the number of learning blocks a declared run will perform.

    Parameters
    ----------
    total_rounds : int
        Plain positive declared budget in rounds (R = total_env_steps /
        num_envs), at most int32.
    pqn : PQNConfig
        Settings giving T, H, E and M.

    Returns
    -------
    int
        N = ceil((R - W) / T), where W = H + T initial random rounds. Each
        learning block performs E * M optimizer steps, so the learning-rate
        schedule spans N * E * M steps.

    Raises
    ------
    ValueError
        R is not a plain int in [1, 2**31 - 1]; R < W + 1 (there would be no
        learning and no positive schedule horizon); N * E * M or
        ceil(W / T) + N exceeds int32 (optimizer and block counters).

    Notes
    -----
    Host-only and allocation-free; the runner's config check, learner setup,
    validation and restore all call it. A saved boundary may still lie at or
    before W inside a longer valid run.
    """
    rounds = _count(total_rounds, "total_rounds")
    initial = pqn.initial_rounds
    if rounds < initial + 1:
        raise ValueError(
            "A PQN run needs total rounds of at least memory_window + "
            f"rollout_length + 1 ({initial + 1}); got {rounds}"
        )
    planned = -(-(rounds - initial) // pqn.rollout_length)
    if planned * pqn.epochs * pqn.num_minibatches > _INT32_MAX:
        raise ValueError("PQN's planned optimizer steps would overflow int32")
    if -(-initial // pqn.rollout_length) + planned > _INT32_MAX:
        raise ValueError("PQN's block counter would overflow int32")
    return planned


@dataclass(frozen=True)
class PPOConfig:
    """Hold static donor update settings, with separate actor/critic optimizers.

    Parameters
    ----------
    actor_lr, critic_lr : float, default=0.00025
        Positive constant learning rates for the separate Adam optimizers.
    rollout_length : int, default=128
        Decisions collected per game before an update. Recurrent methods keep
        this complete sequence; shorter recurrent chunks are unsupported.
    epochs : int, default=4
        How many times each update uses the supplied rollout.
    minibatches : int, default=2
        Number of subsets in each group and epoch: whole game sequences for
        recurrent methods, or time/game rows for feedforward methods.
    groups : int, default=2
        Number of groups whose gradients are averaged for one learner. These
        are not independent runs. Game count must be divisible by groups.
        Recurrent game count, or feedforward rollout_length times game count,
        must also be divisible by groups*minibatches.
    gamma : float, default=0.99
        Reward discount per transition, from zero through one.
    gae_lambda : float, default=0.95
        Advantage mixing weight, from zero through one.
    clip_epsilon : float, default=0.2
        Positive limit for policy-ratio changes and changes in predicted value.
    entropy_coefficient : float, default=0.01
        Nonnegative weight of policy entropy in the actor loss.
    value_coefficient : float, default=0.5
        Nonnegative weight of the donor value loss. That loss already includes
        its separate factor of 0.5 for squared errors.
    max_grad_norm : float, default=0.5
        Positive gradient-length limit, applied after group averaging and before
        each network's Adam update.
    adam_epsilon : float, default=0.00001
        Positive offset in Adam's denominator.
    input_scale : float, default=1.0
        Positive finite multiplier applied to every actor and critic feature
        before the first Dense layer. One preserves the donor's raw inputs.
        Use the same setting for collection, learning and loaded inference.
        No feature is removed and no running statistics are collected.
    parameter_sharing : {"all", "class", "none"}, default="all"
        Share actor weights across everyone, within each class, or not between
        physical slots. Grouped modes have five parameter groups. Actor memory
        remains separate per physical slot; the critic keeps its existing rule.
    value_normalization : bool, default=True
        Train the critic on normalized reward targets using one learner-owned
        moving average. GAE still uses reward units; actor inputs and exports
        do not use these statistics. False preserves the historical raw loss.
        Saved configurations from before this field existed mean False.
    spawn_frame : str, default="left"
        Which spawn bank the actor always seems to start from. "left", the
        default, reflects the permitted view of any actor whose own team
        starts on the right bank about the map's vertical centerline, so every
        game looks like a left start, and maps the chosen move back before the
        game receives it. "world" keeps raw coordinates; use it for runs made
        before the default
        changed on 22 September 2026, and in old config files, which name no
        frame and would otherwise train in "left". Saved checkpoints, exports
        and run records without a frame still mean "world". On
        the built-in Team Deathmatch maps, which are their own mirror images,
        a reflected right start shows the own pads near x = 0.5; a custom
        layout is reflected the same way with no such guarantee, and
        coordinates keep float32 rounding. Use the same setting for
        collection, learning and loaded inference; it is saved with exported
        actors and is part of their inference identity.

    Raises
    ------
    ValueError
        A count is not a positive Python int, or a numerical setting is nonfinite
        or outside its stated range. Boolean counts and non-real/Boolean input
        scales are rejected, as is a spawn frame other than "world" or "left".

    Notes
    -----
    Values are immutable host setup settings. Capture one fixed instance or mark
    it static in jit. New settings may require a new compiled program. This
    record changes neither network width nor the environment's rules.
    """

    actor_lr: float = 0.00025
    critic_lr: float = 0.00025
    rollout_length: int = 128
    epochs: int = 4
    minibatches: int = 2
    groups: int = 2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    adam_epsilon: float = 0.00001
    input_scale: float = 1.0
    spawn_frame: str = "left"
    value_normalization: bool = True
    parameter_sharing: str = "all"

    def __post_init__(self) -> None:
        """Reject invalid static update counts and numerical settings on the host."""
        _input_scale(self.input_scale)
        _spawn_frame(self.spawn_frame)
        _parameter_sharing(self.parameter_sharing)
        if type(self.value_normalization) is not bool:
            raise ValueError("value_normalization must be a Python bool")
        for name in ("rollout_length", "epochs", "minibatches", "groups"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in (
            "actor_lr",
            "critic_lr",
            "clip_epsilon",
            "max_grad_norm",
            "adam_epsilon",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("gamma", "gae_lambda"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1].")
        for name in ("entropy_coefficient", "value_coefficient"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")


@dataclass(frozen=True)
class QMIXConfig:
    """Hold the static recurrent QMIX settings, using the donor's names.

    Parameters
    ----------
    rollout_length : int, default=8
        Most real rounds collected in one block before replay insertion and
        learning (T). One round is one decision in every game lane. Blocks may
        be shorter at the end of a budget or in low-level use.
    buffer_size : int, default=1000
        Replay capacity in time rows per game lane (C). The total number of rows
        is C times the number of lanes. It must be at least min_buffer_size and
        at least rollout_length.
    min_buffer_size : int, default=32
        Rows per lane that must be stored before any sample or optimizer step.
        It must be at least sample_sequence_length.
    sample_sequence_length : int, default=20
        Consecutive rows per sampled sequence (S). S rows give S-1 adjacent TD
        pairs. Must be at least 2.
    sample_batch_size : int, default=128
        Sequences drawn for each optimizer step (M), with replacement.
    epochs : int, default=4
        Fresh samples and optimizer steps per ready block.
    q_lr : float, default=0.00003
        Adam learning rate shared by the online Q-network and mixer. Adam keeps
        its defaults (b1 0.9, b2 0.999, eps 1e-8). There is no gradient clipping:
        the donor configuration names ``max_grad_norm`` but its optimizer never
        uses it, so it is not copied.
    hard_update : bool, default=True
        Python bool choosing the target-network rule. True copies the online
        networks when the optimizer count before a step is a multiple of
        update_period (counts 0, 200, 400 and so on). False blends the online
        networks into the targets with weight tau after every step.
    update_period : int, default=200
        Optimizer steps between hard target copies. Kept and checked even when
        hard_update is False.
    tau : float, default=0.01
        Soft-blend weight in [0, 1]. Kept and checked even when hard_update is
        True.
    gamma : float, default=0.99
        Discount per transition, in [0, 1].
    eps_min : float, default=0.05
        Exploration rate reached at the end of the decay, in [0, 1].
    eps_decay : int, default=100000
        Real environment transitions (all game lanes counted) over which the
        exploration rate falls linearly from 1 to eps_min. It is a number of
        transitions, not a rate.
    input_scale : float, default=1.0
        Positive finite multiplier applied to every actor feature before the
        first Dense layer. The default 1.0 keeps the donor's calculation and
        old models. The donor's inputs are about unit size, but ours reach
        about 300: at 1.0 the first layer starts with activations about 14
        times the intended size, which pushes the GRU's gates to their limits.
        0.01, the value the PPO baselines use, starts them near 0.14. There is
        no input normalization to absorb the scale. The mixer's physical state
        is never scaled; the mixer applies its own LayerNorm to it.
    spawn_frame : str, default="left"
        Network frame for actor features, masks and actions: "left" reflects
        every game to look like a start from the left bank, "world" keeps raw
        coordinates. Replay rows and physical state stay in the world frame.

    parameter_sharing : {"all", "class", "none"}, default="all"
        Share actor weights across everyone, within each class, or not between
        physical slots. Grouped modes have five parameter groups. Memory stays
        separate per physical actor. Class changes at resets select new weights.
    reward_scale : float, default=1.0
        Positive finite multiplier the learner applies to every team reward
        (task, custom and shaping parts together) before its TD targets. For
        example 10.0 makes a kill worth ten times as much to the Q-values.
        Only the learner's own targets change: logged rewards, game results
        and validation stay in true game units, while mean Q and target
        values in the training log are in scaled units. A constant positive
        scale does not change which actions are best. The default 1.0 leaves
        the update unchanged, and a saved run without this field used 1.0.

    Raises
    ------
    ValueError
        A count is not a plain integer in range, a float is nonfinite, Boolean
        or outside its range, hard_update is not a bool, the spawn frame is
        unknown, the buffer ordering ``buffer_size >= min_buffer_size >=
        sample_sequence_length >= 2`` or ``rollout_length <= buffer_size``
        fails, or a per-step or per-block sample count would overflow int32.

    Notes
    -----
    The network widths (Q 256, recurrent 256, mixer embedding 32, hypernetwork
    64) and the replay period of 1 are fixed contracts, not settings. The
    record is immutable host setup; capture one instance in compiled code.
    """

    rollout_length: int = 8
    buffer_size: int = 1000
    min_buffer_size: int = 32
    sample_sequence_length: int = 20
    sample_batch_size: int = 128
    epochs: int = 4
    q_lr: float = 0.00003
    hard_update: bool = True
    update_period: int = 200
    tau: float = 0.01
    gamma: float = 0.99
    eps_min: float = 0.05
    eps_decay: int = 100000
    input_scale: float = 1.0
    spawn_frame: str = "left"
    parameter_sharing: str = "all"
    reward_scale: float = 1.0

    def __post_init__(self) -> None:
        """Reject invalid settings on the host before any array is created."""
        for name in (
            "rollout_length",
            "buffer_size",
            "min_buffer_size",
            "sample_batch_size",
            "epochs",
            "update_period",
            "eps_decay",
        ):
            _count(getattr(self, name), name)
        _count(self.sample_sequence_length, "sample_sequence_length", minimum=2)
        if not (
            self.buffer_size >= self.min_buffer_size >= self.sample_sequence_length
        ):
            raise ValueError(
                "QMIX needs buffer_size >= min_buffer_size >= sample_sequence_length"
            )
        if self.rollout_length > self.buffer_size:
            raise ValueError("QMIX needs rollout_length <= buffer_size")
        _positive(self.q_lr, "q_lr")
        _positive(self.reward_scale, "reward_scale")
        _input_scale(self.input_scale)
        _spawn_frame(self.spawn_frame)
        _parameter_sharing(self.parameter_sharing)
        for name in ("tau", "gamma", "eps_min"):
            _unit_interval(getattr(self, name), name)
        if type(self.hard_update) is not bool:
            raise ValueError("hard_update must be a Python bool")
        per_step = self.sample_batch_size * (self.sample_sequence_length - 1)
        if per_step * TEAM_SLOTS > _INT32_MAX:
            raise ValueError("One QMIX step's utility count would overflow int32")
        if self.epochs * per_step * TEAM_SLOTS > _INT32_MAX:
            raise ValueError("One QMIX block's sample counts would overflow int32")


@dataclass(frozen=True)
class PQNConfig:
    """Hold the static recurrent PQN-VDN settings.

    Parameters
    ----------
    rollout_length : int, default=128
        Real rounds in one normal collection block (T). One round is one
        decision in every game lane. Only the block that reaches the declared
        total budget may be shorter.
    memory_window : int, default=4
        Retained real rows per game lane (H), 1 <= H <= rollout_length. Each
        learning window is these H older rows followed by the block's new rows,
        and the H older rows take part in the loss (the donor's recent-data
        rule). Initial random collection runs H + T rounds before learning.
    epochs : int, default=4
        Passes over each learning window (E). Each pass reshuffles the games.
    num_minibatches : int, default=16
        Equal game groups per pass (M). It must divide num_envs; each optimizer
        step sees num_envs / M whole game sequences with all five teammates.
    q_lr : float, default=0.00025
        Starting RAdam learning rate. Positive finite real, not bool.
    lr_linear_decay : bool, default=True
        Python bool. True decays the rate linearly to 1e-10 over all planned
        optimizer steps of the declared run; False keeps q_lr constant.
    max_grad_norm : float, default=1.0
        Global gradient-norm limit applied before RAdam. Positive finite.
    gamma : float, default=0.99
        Discount per transition, in [0, 1].
    td_lambda : float, default=0.85
        Lambda-return weight, in [0, 1]. Zero gives one-step targets.
    eps_start : float, default=1.0
        Exploration rate used for the first learning block, in [0, 1]. Initial
        random collection always uses 1.0, whatever this value is.
    eps_finish : float, default=0.01
        Final exploration rate, 0 <= eps_finish <= eps_start.
    eps_decay_fraction : float, default=0.1
        Share of the planned learning blocks over which exploration falls
        linearly from eps_start to eps_finish, in (0, 1]. A fractional block
        span is kept, as in the donor.
    input_scale : float, default=1.0
        Positive finite multiplier applied to every actor feature before the
        input BatchNorm. That BatchNorm divides each feature by its own
        standard deviation, so the scale almost cancels: it changes only how
        large BatchNorm's epsilon (0.00001) is next to the feature variance,
        and the untrained network's first outputs. It is kept for identity
        and compatibility with the other methods; it is not a useful knob to
        tune.
    spawn_frame : str, default="left"
        Network frame for actor features, masks and actions: "left" reflects
        every game to look like a start from the left bank, "world" keeps raw
        coordinates. Stored recent rows stay in the world frame.

    parameter_sharing : {"all", "class", "none"}, default="all"
        Share actor weights across everyone, within each class, or not between
        physical slots. Grouped modes have five parameter groups. Memory stays
        separate per physical actor. Class changes at resets select new weights.
    reward_scale : float, default=1.0
        Positive finite multiplier the learner applies to every team reward
        (task, custom and shaping parts together) before its lambda-return
        targets, like the donor's ``REW_SCALE``. For example 10.0 makes a kill
        worth ten times as much to the Q-values. Only the learner's own
        targets change: logged rewards, game results and validation stay in
        true game units, while mean Q values in the training log are in scaled
        units. A constant positive scale does not change which actions are
        best. The default 1.0 leaves the update unchanged, and a saved run
        without this field used 1.0.
    normalization : {"batch_norm", "layer_norm"}, default="batch_norm"
        Normalization after each of the two hidden Dense layers, like the
        donor's ``NORM_TYPE``. "batch_norm" is the donor's SMAX choice and
        keeps running statistics for those layers. "layer_norm" normalizes
        each actor's row on its own (no running statistics), which the PQN
        paper reports as the more stable choice. The input BatchNorm stays in
        both cases, as with the donor's ``NORM_INPUT``. The choice changes the
        saved weight layout, so an actor is always rebuilt with the
        normalization it was trained with. A saved run or actor without this
        field used "batch_norm".

    Raises
    ------
    ValueError
        A count is not a plain integer in range, a float is nonfinite, Boolean
        or outside its range, lr_linear_decay is not a bool, the spawn frame or
        normalization is unknown, memory_window exceeds rollout_length,
        eps_finish exceeds eps_start, or H + T exceeds int32.

    Notes
    -----
    The architecture (input BatchNorm; two Dense 512, normalization, ReLU
    layers; GRU 512; Dense 198), the BatchNorm constants (momentum 0.99,
    epsilon 1e-5), Flax's LayerNorm defaults (epsilon 1e-6, learned scale and
    bias), RAdam's own defaults and the 1e-10 schedule end are fixed
    contracts, not settings. There is no target network, replay or value
    normalization. The record is immutable host setup; capture one instance in
    compiled code.
    """

    rollout_length: int = 128
    memory_window: int = 4
    epochs: int = 4
    num_minibatches: int = 16
    q_lr: float = 0.00025
    lr_linear_decay: bool = True
    max_grad_norm: float = 1.0
    gamma: float = 0.99
    td_lambda: float = 0.85
    eps_start: float = 1.0
    eps_finish: float = 0.01
    eps_decay_fraction: float = 0.1
    input_scale: float = 1.0
    spawn_frame: str = "left"
    parameter_sharing: str = "all"
    reward_scale: float = 1.0
    normalization: str = "batch_norm"

    def __post_init__(self) -> None:
        """Reject invalid settings on the host before any array is created."""
        for name in ("rollout_length", "memory_window", "epochs", "num_minibatches"):
            _count(getattr(self, name), name)
        if self.memory_window > self.rollout_length:
            raise ValueError("PQN needs memory_window <= rollout_length")
        if self.memory_window + self.rollout_length > _INT32_MAX:
            raise ValueError("PQN's initial rounds would overflow int32")
        _positive(self.q_lr, "q_lr")
        _positive(self.max_grad_norm, "max_grad_norm")
        _positive(self.reward_scale, "reward_scale")
        _pqn_normalization(self.normalization)
        if type(self.lr_linear_decay) is not bool:
            raise ValueError("lr_linear_decay must be a Python bool")
        for name in ("gamma", "td_lambda", "eps_start", "eps_finish"):
            _unit_interval(getattr(self, name), name)
        if self.eps_finish > self.eps_start:
            raise ValueError("PQN needs eps_finish <= eps_start")
        _fraction(self.eps_decay_fraction, "eps_decay_fraction")
        _input_scale(self.input_scale)
        _spawn_frame(self.spawn_frame)
        _parameter_sharing(self.parameter_sharing)

    @property
    def initial_rounds(self) -> int:
        """Rounds of initial random collection before learning: W = H + T."""
        return self.memory_window + self.rollout_length


_LATER_SETTING_DEFAULTS: dict[type, dict[str, object]] = {
    QMIXConfig: {"reward_scale": 1.0},
    PQNConfig: {"reward_scale": 1.0, "normalization": "batch_norm"},
}
"""Settings added after runs were first saved, with the value those runs used."""


def saved_settings(settings: PPOConfig | QMIXConfig | PQNConfig) -> dict[str, object]:
    """Return one settings record as the JSON object that runs save.

    Parameters
    ----------
    settings : PPOConfig, QMIXConfig or PQNConfig
        A validated settings record.

    Returns
    -------
    dict[str, object]
        A new JSON-ready dictionary of every field in field order (after a
        JSON round trip). Settings added after runs were first saved
        (QMIX and PQN ``reward_scale``; PQN ``normalization``) are left out
        while they hold the value older runs used (1.0 and "batch_norm"). So
        a run that does not use them saves the same bytes as before they
        existed, and an older saved block compares equal to the record it
        reads back as.

    Notes
    -----
    Host-only; settings is not changed. ``training.config.config_to_dict``
    uses this rule. Reading a saved block back through the settings class
    fills a missing field with its default, which is the same value; the
    checkpoint settings checks fill it with ``with_later_settings``.
    """
    data = cast(dict[str, object], json.loads(json.dumps(asdict(settings))))
    for name, value in _LATER_SETTING_DEFAULTS.get(type(settings), {}).items():
        if data[name] == value:
            del data[name]
    return data


def with_later_settings(
    record: type[PPOConfig | QMIXConfig | PQNConfig], block: Mapping[str, object]
) -> dict[str, object]:
    """Return a saved settings block with later-added settings filled in.

    Parameters
    ----------
    record : type
        PPOConfig, QMIXConfig or PQNConfig, the class the block belongs to.
    block : Mapping[str, object]
        One saved JSON settings object, such as a run config's "pqn" block.

    Returns
    -------
    dict[str, object]
        A new dictionary: the block with each setting added after runs were
        first saved (QMIX and PQN ``reward_scale``; PQN ``normalization``) set
        to the value older runs used, when the block omits it. A block that
        states a value keeps it. So a block written before the setting
        existed, one written by ``saved_settings`` and one stating the old
        default explicitly all compare equal to the full JSON form of the same
        settings record.

    Notes
    -----
    Host-only; block is not changed and nothing is validated here.
    """
    return {**_LATER_SETTING_DEFAULTS.get(record, {}), **block}


DEFAULT_PPO_CONFIG = PPOConfig()

DEFAULT_QMIX_CONFIG = QMIXConfig()
"""The donor's recurrent QMIX settings with the BG left spawn frame."""

DEFAULT_PQN_CONFIG = PQNConfig()
"""The donor's SMAX PQN-VDN settings with reward scale 1.0 (not its 10), left frame."""

# Host-readable format versions shared by model code and saved descriptions.
ACTION_SCHEMA_VERSION = 1

ACTOR_INPUT_SCHEMA_VERSION = 2
TRAINING_STATE_SCHEMA_VERSION = 2

QMIX_TIE_RULE = "first_legal_maximum"
"""Greedy rule: the lowest legal action index among equal best scores wins."""
QMIX_KEY_SCHEMA_VERSION = 1
"""Version of the learner's model-initialization and sampling key tags."""
QMIX_REPLAY_SCHEMA_VERSION = 1
"""Version of the compact QMIX replay row layout."""

PQN_TIE_RULE = QMIX_TIE_RULE
"""Greedy rule: the lowest legal action index among equal best values wins."""
PQN_KEY_SCHEMA_VERSION = 1
"""Version of the learner's model-initialization and shuffle key tags."""
PQN_RECENT_WINDOW_SCHEMA_VERSION = 1
"""Version of the compact retained recent-row layout saved with the learner."""
PQN_NORMALIZATION_SCHEMA_VERSION = 1
"""Version of the BatchNorm statistics layout and its update rule."""
