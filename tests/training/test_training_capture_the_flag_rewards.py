"""Check the Capture the Flag reward recipes against independent arithmetic.

Transitions are built by hand on map 19 (walls on the map's center line): a
real reset, then each flag's carrier, countdown and position and the scores,
plus the Capture the Flag facts. The oracle uses plain Python and NumPy with the
version 1 values, and walks from a route field built directly toward the
config's own flag homes, not the recipes' bank. It checks every part of both
recipes in both forms: the objective potential (0.10 per capture of lead; a
flag is worth 0 at home, 0.02 dropped and 0.05 carried), the carrier's own
0.02, the zero-sum capture and return credits, and the route potential (0.03
times each carrier's progress home). Scenes cover a pickup, carrying, a
capture, a winning end and a timeout end (the next potential is 0), a drop,
both teams' returns, the automatic return, a standoff and padding, at 5v5 and
3v3 and with the flag homes exchanged. The team form gives every configured
Team A slot the shared parts once plus the Team A total of the others.

The loop table holds: with no discount, a pickup carried home pays the capturer
0.10 + 0.02 and each teammate 0.10; a lost carry the enemy returns costs every
agent 0.02 / 5; a defense pays the returner 0.02. At discount 0.99 the capture
tick pays every Team A agent a gain, and a path back to its starting state pays
exactly what waiting there would (so no loop beats waiting). The bank's lookup
equals a direct route field on all 52 maps from both ends; another map falls
back to straight lines. Missing facts raise ValueError; jit and vmap equal eager
calls, with the bank first built inside a trace; the shared reward contract
passes the discount and refuses to run without it; and every recipe's identity
records both keyword contracts and is the declared version 1 identity.
Hand-built states prove the rewards' arithmetic, not learning.
"""

# Tests read the recipes' private bank on purpose.
# pyright: reportPrivateUsage=false
from collections.abc import Callable
from functools import partial
from typing import TypedDict, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from marl_battlegrounds import tasks
from marl_battlegrounds._task_settings import DEFAULT_CTF_FLAG_HOME_POSITIONS
from marl_battlegrounds.core import env as core
from marl_battlegrounds.core.types import (
    CaptureTheFlagTransitionFacts,
    EnvConfig,
    EnvState,
)
from marl_battlegrounds.environment import TrainingFacts
from marl_battlegrounds.policies.route_field import (
    FINE_CELL_SIZE,
    blocked_cells,
    route_distance_at,
    route_distances,
)
from marl_battlegrounds.training import reward_recipes
from marl_battlegrounds.training.reward_recipes import (
    ctf_objective_reward,
    ctf_objective_route_reward,
    ctf_objective_route_team_reward,
    ctf_objective_team_reward,
)
from marl_battlegrounds.training.shaping import (
    RewardOutput,
    resolve_reward,
    reward_adjustments,
)

K, C, D, B, E_C, E_R, R = 0.10, 0.05, 0.02, 0.02, 0.02, 0.02, 0.03
ROSTER = ("mage", "warrior", "priest", "hunter", "rogue")
RECIPES: dict[str, tuple[Callable[..., RewardOutput], Callable[..., RewardOutput]]] = {
    "objective": (
        ctf_objective_reward,
        ctf_objective_team_reward,
    ),
    "route": (
        ctf_objective_route_reward,
        ctf_objective_route_team_reward,
    ),
}
ALL_RECIPES = [recipe for pair in RECIPES.values() for recipe in pair]
SHARED = ("objective_potential", "route_potential")
type Point = tuple[float, float]


class Flags(TypedDict, total=False):
    scores: tuple[int, int]
    slots: tuple[int, int]
    countdowns: tuple[int, int]
    positions: tuple[Point | None, Point | None]


class Events(TypedDict, total=False):
    picked: tuple[int, ...]
    dropped: tuple[int, ...]
    returned: tuple[int, ...]
    captured: tuple[int, ...]
    automatic: tuple[int, ...]
    outcome: int
    has_transition: bool


type Step = tuple[Flags, Flags, Events]

# None in positions means the flag's home. Team A's carrier holds flag 1.
HOME: Flags = {}
CARRIED: Flags = {"slots": (-1, 2), "positions": (None, (15.0, 5.0))}
NEAR_HOME: Flags = {"slots": (-1, 2), "positions": (None, (1.0, 5.0))}
DROPPED: Flags = {"countdowns": (0, 30), "positions": (None, (5.0, 5.0))}
STANDOFF: Flags = {"slots": (6, 2), "positions": ((5.0, 5.0), (15.0, 5.0))}
SCENES: dict[str, Step] = {
    "pickup": (HOME, CARRIED, {"picked": (2,)}),
    "carry": (CARRIED, {"slots": (-1, 2), "positions": (None, (5.0, 5.0))}, {}),
    "capture": (NEAR_HOME | {"scores": (0, 1)}, {"scores": (1, 1)}, {"captured": (2,)}),
    "winning capture": (
        NEAR_HOME | {"scores": (2, 1)},
        {"scores": (3, 1)},
        {"captured": (2,), "outcome": 1},
    ),
    "drop": (
        {"slots": (-1, 2), "positions": (None, (5.0, 5.0))},
        DROPPED,
        {"dropped": (2,)},
    ),
    "enemy return": (DROPPED, HOME, {"returned": (7,)}),
    "our return": (
        {"countdowns": (12, 0), "positions": ((15.0, 5.0), None)},
        HOME,
        {"returned": (1,)},
    ),
    "enemy capture": (
        {"slots": (6, -1), "positions": ((19.0, 5.0), None)},
        {"scores": (0, 1)},
        {"captured": (6,)},
    ),
    "automatic return": (
        {"countdowns": (0, 1), "positions": (None, (5.0, 5.0))},
        HOME,
        {"automatic": (1,)},
    ),
    "standoff": (
        STANDOFF,
        {"slots": (6, 2), "positions": ((15.0, 5.0), (5.0, 5.0))},
        {},
    ),
    "timeout draw": (
        STANDOFF | {"scores": (1, 1)},
        STANDOFF | {"scores": (1, 1)},
        {"outcome": 3},
    ),
    "padding": (HOME, CARRIED, {"picked": (2,), "has_transition": False}),
}


def _config(*, map_id: int = 19, size: int = 5, exchanged: bool = False) -> EnvConfig:
    config = tasks.make_standard_capture_the_flag_config(
        map_id=map_id, team_a_roster=ROSTER[:size], team_b_roster=ROSTER[:size]
    )
    if not exchanged:
        return config
    homes = np.asarray(DEFAULT_CTF_FLAG_HOME_POSITIONS, np.float32)
    return config._replace(
        capture_the_flag_flag_home_positions=jnp.asarray(homes[::-1])
    )


def _state(config: EnvConfig, flags: Flags) -> EnvState:
    state = core.reset(config, jax.random.key(0))[0]
    homes = np.asarray(config.capture_the_flag_flag_home_positions)
    places = flags.get("positions", (None, None))
    positions = [
        homes[flag] if place is None else np.asarray(place, np.float32)
        for flag, place in enumerate(places)
    ]
    return state._replace(
        capture_the_flag_scores=jnp.asarray(flags.get("scores", (0, 0)), jnp.int32),
        capture_the_flag_flag_carrier_slots=jnp.asarray(
            flags.get("slots", (-1, -1)), jnp.int32
        ),
        capture_the_flag_dropped_flag_countdowns=jnp.asarray(
            flags.get("countdowns", (0, 0)), jnp.int32
        ),
        capture_the_flag_flag_positions=jnp.asarray(np.stack(positions)),
    )


def _facts(events: Events) -> TrainingFacts:
    def marked(count: int, chosen: tuple[int, ...]) -> Array:
        return jnp.asarray(np.isin(np.arange(count), chosen))

    record = CaptureTheFlagTransitionFacts(
        flag_was_picked_up_by_agent=marked(10, events.get("picked", ())),
        flag_was_dropped_by_agent=marked(10, events.get("dropped", ())),
        ally_flag_was_returned_by_agent=marked(10, events.get("returned", ())),
        enemy_flag_was_captured_by_agent=marked(10, events.get("captured", ())),
        flag_was_automatically_returned_by_team=marked(2, events.get("automatic", ())),
        outcome=jnp.int32(events.get("outcome", 0)),
    )
    none, zeros = jnp.zeros(10, jnp.bool_), jnp.zeros(10, jnp.float32)
    return TrainingFacts(
        jnp.bool_(events.get("has_transition", True)),
        jnp.full(10, -1, jnp.int32),
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        none,
        none,
        zeros,
        none,
        zeros,
        capture_the_flag_facts=record,
    )


def _scene(step: Step, config: EnvConfig) -> tuple[EnvState, TrainingFacts, EnvState]:
    before, after, events = step
    return _state(config, before), _facts(events), _state(config, after)


def _call(
    recipe: Callable[..., RewardOutput],
    scene: tuple[EnvState, TrainingFacts, EnvState],
    config: EnvConfig,
    discount: float = 0.99,
) -> RewardOutput:
    before, facts, after = scene
    return recipe(
        before,
        facts,
        after,
        jnp.int32(0),
        config=config,
        discount=jnp.float32(discount),
    )


_fields = cast(
    Callable[..., Array], jax.jit(route_distances, static_argnames="cell_size")
)
_read = cast(
    Callable[[Array, Array], Array],
    jax.jit(jax.vmap(partial(route_distance_at, cell_size=FINE_CELL_SIZE), (None, 0))),
)


def _direct_walks(config: EnvConfig) -> Callable[[int, np.ndarray], np.ndarray]:
    blocked = blocked_cells(
        config.obstacles, map_width=20.0, map_height=10.0, cell_size=FINE_CELL_SIZE
    )
    homes = config.capture_the_flag_flag_home_positions
    fields = _fields(blocked, homes, cell_size=FINE_CELL_SIZE)

    def walk(team: int, positions: np.ndarray) -> np.ndarray:
        points = jnp.asarray(np.reshape(positions, (-1, 2)), jnp.float32)
        return np.asarray(_read(fields[team], points))

    return walk


def _oracle(
    config: EnvConfig,
    scene: tuple[EnvState, TrainingFacts, EnvState],
    discount: float,
) -> dict[str, np.ndarray]:
    before, facts, after = scene
    record = facts.capture_the_flag_facts
    assert record is not None
    keep = 0.0 if int(record.outcome) else discount
    homes = np.asarray(config.capture_the_flag_flag_home_positions)
    walk = _direct_walks(config)

    def value(state: EnvState, flag: int) -> float:
        if int(state.capture_the_flag_flag_carrier_slots[flag]) >= 0:
            return C
        return D if int(state.capture_the_flag_dropped_flag_countdowns[flag]) else 0.0

    def objective(state: EnvState) -> float:
        ours, theirs = (int(score) for score in state.capture_the_flag_scores)
        return K * (ours - theirs) + value(state, 1) - value(state, 0)

    def carrier(state: EnvState) -> np.ndarray:
        shares = np.zeros(5)
        slot = int(state.capture_the_flag_flag_carrier_slots[1])
        if 0 <= slot < 5:
            shares[slot] = B
        return shares

    def route(state: EnvState) -> float:
        total = 0.0
        for team, sign in ((0, 1.0), (1, -1.0)):
            if int(state.capture_the_flag_flag_carrier_slots[1 - team]) >= 0:
                position = np.asarray(state.capture_the_flag_flag_positions[1 - team])
                ratio = walk(team, position)[0] / walk(team, homes[1 - team])[0]
                total += sign * min(max(1 - ratio, 0.0), 1.0)
        return R * total

    active = np.asarray(config.agent_profile.active_mask[:5])
    players = active.sum()
    captured = np.asarray(record.enemy_flag_was_captured_by_agent, np.float64)
    returned = np.asarray(record.ally_flag_was_returned_by_agent, np.float64)
    parts = {
        "objective_potential": np.full(5, keep * objective(after) - objective(before)),
        "carrier_potential": keep * carrier(after) - carrier(before),
        "capture_credit": E_C * captured[:5] - E_C * captured[5:].any() / players,
        "return_credit": E_R * returned[:5] - E_R * returned[5:].any() / players,
        "route_potential": np.full(5, keep * route(after) - route(before)),
    }
    recipients = active & bool(facts.has_transition)
    return {name: np.where(recipients, value, 0.0) for name, value in parts.items()}


@pytest.mark.parametrize("exchanged", [False, True])
@pytest.mark.parametrize("size", [5, 3])
@pytest.mark.parametrize("name", list(SCENES))
def test_every_part_matches_the_oracle_in_both_forms(
    name: str, size: int, exchanged: bool
) -> None:
    config = _config(size=size, exchanged=exchanged)
    scene = _scene(SCENES[name], config)
    expected = _oracle(config, scene, 0.99)
    recipients = np.asarray(config.agent_profile.active_mask[:5]) & bool(
        scene[1].has_transition
    )
    for route, (individual, team) in RECIPES.items():
        names = list(expected)[: 5 if route == "route" else 4]
        output = _call(individual, scene, config)
        assert list(output.components) == names
        for part in names:
            np.testing.assert_allclose(
                output.components[part][:5], expected[part], atol=1e-6
            )
            assert not np.asarray(output.components[part][5:]).any()
        np.testing.assert_allclose(
            output.adjustments, sum(output.components.values()), atol=1e-6
        )
        amount = sum(
            expected[part][0] if part in SHARED else expected[part].sum()
            for part in names
        )
        shared = _call(team, scene, config).adjustments
        np.testing.assert_allclose(
            shared[:5], np.where(recipients, amount, 0.0), atol=1e-6
        )
        assert not np.asarray(shared[5:]).any()


def _total(
    steps: list[Step], recipe: Callable[..., RewardOutput], discount: float
) -> np.ndarray:
    config = _config()
    total = np.zeros(10)
    for step, transition in enumerate(steps):
        output = _call(recipe, _scene(transition, config), config, discount)
        total += discount**step * np.asarray(output.adjustments)
    return total


@pytest.mark.parametrize("route", list(RECIPES))
def test_the_loop_table_holds(route: str) -> None:
    recipe = RECIPES[route][0]
    carry: list[Step] = [
        (HOME, CARRIED, {"picked": (2,)}),
        (CARRIED, NEAR_HOME, {}),
        (NEAR_HOME, {"scores": (1, 0)}, {"captured": (2,)}),
    ]
    total = _total(carry, recipe, 1.0)
    np.testing.assert_allclose(total[:5], [K, K, K + E_C, K, K], atol=1e-6)
    assert (_total(carry[2:], recipe, 0.99)[:5] > 0).all()
    lost: list[Step] = [
        (HOME, CARRIED, {"picked": (2,)}),
        (CARRIED, DROPPED, {"dropped": (2,)}),
        (DROPPED, HOME, {"returned": (7,)}),
    ]
    np.testing.assert_allclose(_total(lost, recipe, 1.0)[:5], -E_R / 5, atol=1e-6)
    theirs: Flags = {"slots": (6, -1), "positions": ((15.0, 5.0), None)}
    ours_dropped: Flags = {"countdowns": (30, 0), "positions": ((15.0, 5.0), None)}
    defense: list[Step] = [
        (HOME, theirs, {"picked": (6,)}),
        (theirs, ours_dropped, {"dropped": (6,)}),
        (ours_dropped, HOME, {"returned": (3,)}),
    ]
    np.testing.assert_allclose(
        _total(defense, recipe, 1.0)[:5], [0, 0, 0, E_R, 0], atol=1e-6
    )
    starts: list[Flags] = [{"scores": (1, 0)}, theirs | {"scores": (0, 1)}]
    for start in starts:
        elsewhere: Flags = STANDOFF | {"scores": start.get("scores", (0, 0))}
        loop: list[Step] = [(start, elsewhere, {}), (elsewhere, start, {})]
        wait: list[Step] = [(start, start, {}), (start, start, {})]
        np.testing.assert_allclose(
            _total(loop, recipe, 0.99), _total(wait, recipe, 0.99), atol=1e-6
        )


@pytest.mark.parametrize("exchanged", [False, True])
def test_the_bank_matches_a_direct_route_field_on_every_map(exchanged: bool) -> None:
    rng = np.random.default_rng(19)
    for info in tasks.list_ctf_maps():
        config = _config(map_id=info.map_id, exchanged=exchanged)
        walk = _direct_walks(config)
        homes = np.asarray(config.capture_the_flag_flag_home_positions)
        candidates = rng.uniform([0.5, 0.5], [19.5, 9.5], (40, 2)).astype(np.float32)
        open_cells = candidates[
            np.isfinite(walk(0, candidates)) & np.isfinite(walk(1, candidates))
        ][:2]
        ours, theirs = ((float(x), float(y)) for x, y in open_cells)
        carried: Flags = {"slots": (6, 2), "positions": (theirs, ours)}
        scene = (_state(config, HOME), _facts({}), _state(config, carried))
        output = _call(ctf_objective_route_reward, scene, config, 1.0)
        progress = [
            np.clip(
                1 - walk(team, open_cells[team])[0] / walk(team, homes[1 - team])[0],
                0,
                1,
            )
            for team in (0, 1)
        ]
        np.testing.assert_allclose(
            output.components["route_potential"][0],
            R * (progress[0] - progress[1]),
            atol=1e-6,
        )


def test_another_map_uses_straight_lines() -> None:
    config = _config()
    config = config._replace(obstacles=config.obstacles.at[0, 0].add(0.25))
    scene = (_state(config, HOME), _facts({}), _state(config, STANDOFF))
    output = _call(ctf_objective_route_reward, scene, config, 1.0)
    homes = np.asarray(config.capture_the_flag_flag_home_positions)
    between = np.linalg.norm(homes[0] - homes[1])
    progress = [
        1 - np.linalg.norm(np.asarray(position) - homes[team]) / between
        for team, position in ((0, (15.0, 5.0)), (1, (5.0, 5.0)))
    ]
    np.testing.assert_allclose(
        output.components["route_potential"][0],
        R * (progress[0] - progress[1]),
        atol=1e-6,
    )


@pytest.mark.parametrize("recipe", ALL_RECIPES)
def test_missing_capture_the_flag_facts_raise(
    recipe: Callable[..., RewardOutput],
) -> None:
    config = _config()
    before, facts, after = _scene(SCENES["pickup"], config)
    without = facts._replace(capture_the_flag_facts=None)
    with pytest.raises(ValueError, match="Capture the Flag facts"):
        _call(recipe, (before, without, after), config)


def test_jit_and_vmap_equal_eager_calls_with_the_bank_built_while_tracing() -> None:
    config = _config()
    scenes = [_scene(SCENES[name], config) for name in ("carry", "standoff")]
    eager = [_call(ctf_objective_route_team_reward, scene, config) for scene in scenes]
    reward_recipes._route_bank.cache_clear()

    def one(
        before: EnvState, facts: TrainingFacts, after: EnvState, config: EnvConfig
    ) -> Array:
        return ctf_objective_route_team_reward(
            before,
            facts,
            after,
            jnp.int32(0),
            config=config,
            discount=jnp.float32(0.99),
        ).adjustments

    stacked = jax.tree.map(
        lambda *leaves: jnp.stack(leaves), *[(*scene, config) for scene in scenes]
    )
    batched = cast(Array, jax.jit(jax.vmap(one))(*stacked))
    for row, output in zip(batched, eager, strict=True):
        np.testing.assert_allclose(row, output.adjustments, atol=1e-6)


@pytest.mark.parametrize("name", [recipe.__name__ for recipe in ALL_RECIPES])
def test_the_shared_reward_contract_passes_the_discount(name: str) -> None:
    reward, identity = resolve_reward(
        f"marl_battlegrounds.training.reward_recipes:{name}"
    )
    assert reward is not None and identity is not None
    assert identity == {
        "reference": f"marl_battlegrounds.training.reward_recipes:{name}",
        **reward_recipes._CTF_SAVED_IDENTITY_V1[name],
    }
    assert identity["call_contract"] == "producing_config_keyword_v1"
    assert identity["discount_contract"] == "learner_discount_keyword_v1"
    config = _config()
    before, facts, after = _scene(SCENES["carry"], config)
    adjustments = reward_adjustments(
        reward,
        before,
        facts,
        after,
        jnp.int32(0),
        config=config,
        discount=jnp.float32(0.9),
    )
    recipe = cast(Callable[..., RewardOutput], reward)
    np.testing.assert_allclose(
        adjustments, _call(recipe, (before, facts, after), config, 0.9).adjustments
    )
    with pytest.raises(ValueError, match="discount"):
        reward_adjustments(reward, before, facts, after, jnp.int32(0), config=config)
