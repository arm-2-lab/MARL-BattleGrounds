"""Check that evaluation saves Capture the Flag replays that carry every flag event.

Four hand-built starts on map 0 (three champions a team, no obstacles) run
through evaluate_episodes with save_replays. Team A stands still; Team B is the
built-in tdm-alpha bot, which kills a carrier left at 1 health beside it.
Contracts: each game saves a replay V5 that load_replay reads back with one
frame per tick, and each frame holds every agent's position and the flags'
positions, carriers, countdowns and scores. On the first tick the replays hold
a pickup from the flag's home and a timed return (game 1), a return by touch
and a capture that scores 1 to 0 (game 2), a carrier's drop where it died
(game 3), and a pickup from the ground (game 4). Runs on CPU in about 10 s.
"""

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import pytest

from marl_battlegrounds.core.env import reset
from marl_battlegrounds.core.types import ActionMask, EnvState
from marl_battlegrounds.evaluation._replays.replay_v5 import ReplayArtifactV5
from marl_battlegrounds.evaluation.evaluate import EpisodeSpec, evaluate_episodes
from marl_battlegrounds.evaluation.policy_execution import Policy
from marl_battlegrounds.evaluation.replay_io import load_replay
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.input import ActorInput
from marl_battlegrounds.tasks import (
    AgentClassName,
    make_standard_capture_the_flag_config,
)

_ROSTER: tuple[AgentClassName, ...] = ("mage", "warrior", "priest")
_CONFIG = make_standard_capture_the_flag_config(
    map_id=0, team_a_roster=_ROSTER, team_b_roster=_ROSTER, max_steps=3
)
_FAR = {
    0: (3.0, 1.0),
    1: (3.0, 3.0),
    2: (3.0, 9.0),
    5: (12.0, 1.0),
    6: (12.0, 3.0),
    7: (12.0, 9.0),
}
_Point = tuple[float, float]


def _still(
    variables: jax.Array,
    carry: jax.Array,
    actor: ActorInput,
    mask: ActionMask,
    key: jax.Array,
) -> tuple[ActorAction, jax.Array]:
    zero = jnp.zeros((), jnp.int32)
    return ActorAction(zero, zero, zero), carry


def _start(
    positions: Mapping[int, _Point],
    *,
    flags: Mapping[int, _Point],
    carriers: Mapping[int, int] | None = None,
    countdowns: Mapping[int, int] | None = None,
    health: Mapping[int, int] | None = None,
) -> EnvState:
    state = reset(_CONFIG, jax.random.key(0))[0]
    where = state.agent_positions
    for slot, point in {**_FAR, **positions}.items():
        where = where.at[slot].set(jnp.asarray(point, jnp.float32))
    flag_positions = state.capture_the_flag_flag_positions
    for flag, point in flags.items():
        flag_positions = flag_positions.at[flag].set(jnp.asarray(point, jnp.float32))
    carrier_slots = state.capture_the_flag_flag_carrier_slots
    for flag, slot in (carriers or {}).items():
        carrier_slots = carrier_slots.at[flag].set(slot)
    flag_countdowns = state.capture_the_flag_dropped_flag_countdowns
    for flag, steps in (countdowns or {}).items():
        flag_countdowns = flag_countdowns.at[flag].set(steps)
    current_health = state.current_health
    for slot, points in (health or {}).items():
        current_health = current_health.at[slot].set(points)
    return state._replace(
        agent_positions=where,
        capture_the_flag_flag_positions=flag_positions,
        capture_the_flag_flag_carrier_slots=carrier_slots,
        capture_the_flag_dropped_flag_countdowns=flag_countdowns,
        current_health=current_health,
        spawn_shield_durations=jnp.zeros_like(state.spawn_shield_durations),
    )


# Flag 0 is Team A's, flag 1 Team B's; Team A's home is (0.5, 5), Team B's
# (19.5, 5). Each game lists its first tick's flag events.
_GAMES = {
    # Our Mage stands at their home; our flag lies with 1 tick left.
    1: (
        _start({0: (19.0, 5.0)}, flags={0: (10.0, 9.0)}, countdowns={0: 1}),
        [("FlagPickedUpEventV1", "home"), ("FlagAutomaticallyReturnedEventV1", None)],
    ),
    # Our Warrior stands on our dropped flag; our Mage carries theirs home.
    2: (
        _start(
            {0: (1.5, 5.0), 1: (5.0, 5.0)},
            flags={0: (5.0, 5.0), 1: (1.5, 5.0)},
            carriers={1: 0},
            countdowns={0: 10},
        ),
        [("FlagReturnedEventV1", None), ("FlagCapturedEventV1", None)],
    ),
    # Our Mage carries theirs at 1 health beside their Mage and Warrior.
    3: (
        _start(
            {0: (10.0, 5.0), 5: (11.0, 5.0), 6: (10.0, 6.0)},
            flags={1: (10.0, 5.0)},
            carriers={1: 0},
            health={0: 1},
        ),
        [("FlagDroppedEventV1", None)],
    ),
    # Our Warrior stands on their dropped flag.
    4: (
        _start({1: (8.0, 5.0)}, flags={1: (8.0, 5.0)}, countdowns={1: 10}),
        [("FlagPickedUpEventV1", "ground")],
    ),
}


@pytest.fixture(scope="module")
def replays(tmp_path_factory: pytest.TempPathFactory) -> dict[int, ReplayArtifactV5]:
    out = tmp_path_factory.mktemp("ctf-replays")
    evaluate_episodes(
        Policy("still", _still, jnp.int32(0), jnp.int32(0)),
        "tdm-alpha",
        [
            EpisodeSpec(game, _CONFIG, initial_state=start)
            for game, (start, _) in _GAMES.items()
        ],
        num_envs=2,
        save_replays=len(_GAMES),
        output_dir=out,
    )
    loaded: dict[int, ReplayArtifactV5] = {}
    for path in out.rglob("*.marlbg-replay.json"):
        replay = load_replay(path).replay
        assert isinstance(replay, ReplayArtifactV5)
        # The episode ID ends with the scheduled game: "...:episode-3".
        game = int(replay.header.context.identity.episode_id.rsplit("-", 1)[1])
        loaded[game] = replay
    return loaded


def test_every_game_saves_a_replay_with_one_frame_per_tick(
    replays: dict[int, ReplayArtifactV5],
) -> None:
    assert sorted(replays) == sorted(_GAMES)
    for replay in replays.values():
        assert len(replay.frames) == len(replay.transitions) + 1 == 4
        snapshot = replay.frames[1].snapshot
        assert len(snapshot.agent_positions) == 10
        assert len(snapshot.capture_the_flag_flag_positions) == 2


@pytest.mark.parametrize("game", sorted(_GAMES))
def test_the_first_tick_holds_the_games_flag_events(
    replays: dict[int, ReplayArtifactV5], game: int
) -> None:
    first = replays[game].transitions[0]
    events = [
        (type(event).__name__, getattr(event, "picked_up_from", None))
        for event in first.events
        if type(event).__name__.startswith("Flag")
    ]
    assert events == _GAMES[game][1]


def test_a_capture_scores_and_a_drop_lies_where_the_carrier_died(
    replays: dict[int, ReplayArtifactV5],
) -> None:
    assert replays[2].frames[1].snapshot.capture_the_flag_scores == (1, 0)
    died = replays[3].frames[1].snapshot
    assert died.capture_the_flag_flag_carrier_slots[1] is None
    assert died.capture_the_flag_dropped_flag_countdowns[1] > 0
    assert died.capture_the_flag_flag_positions[1] == pytest.approx(
        died.agent_positions[0]
    )
