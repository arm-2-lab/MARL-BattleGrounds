"""Check two host facts: recent_falls (both servers) and next_end_spots.

The game: the Vision Moment "fall" start under NoSharedObs, where the Team A
Warrior Charges the Team B Mage, who falls on the first tick; then nine quiet
ticks, played live and saved as a replay. Contracts checked:

- ``recent_falls`` (the Fallen panel's earlier rows; viewer.champion_streaks
  is the one owner of kill credit):
  - Oracle View names the fall from the frame after it, with the Warrior as
    the killer, 1 point and the Charge as the Ultimate.
  - Agent POV keeps its own rules: a recipient that saw the fall but not the
    blow gets the row with no killer and no Ultimate; a recipient that did
    not see the fall gets no row; the victim's own view names the Warrior.
  - The window: the fall's own frame and frame zero list nothing (the plan
    draws the shown tick), the row stays for the 7 ticks after it and is
    gone on the 8th; a seek straight to a frame equals counting up to it.
  - Live play: the BattleClient and DevClient service sends each view's own
    rows from its session (ChampionTakedownsV1.recent_falls), equal to the
    Replay Viewer's on every checked frame for Oracle View, the Warrior's
    victim's view, a recipient that saw only the fall and one that saw
    nothing, so a view switch or a reload no longer empties the panel.
  - The session record keeps each view's ticks with a fall: a tick lists the
    7 ticks before it and never itself, a quiet tick keeps the rows, a later
    advance drops rows outside the window, and rows out of order or without
    a fall are refused.
  - A live frame refuses a recent_falls row on its own tick and a victim
    listed twice (the same check as replay frames).
- ``next_end_spots`` (Next Step Ghost):
  - Oracle View lists every champion alive now at its recorded position in
    the next frame.
  - Agent POV (recipients 0 and 5, every frame) lists exactly the champions
    alive in this POV scene that the next POV scene also holds, at their
    next positions, so a champion seen now but not next tick gets no spot
    (checked on synthetic scenes too).
  - The final frame lists nobody.
- TickTakedownsV1 refuses an Ultimate that is not a takedown and a home fall
  that is not a fall.
- The damaging-blow table of the server rule (a Basic deals damage unless it
  is the Priest's; an Ultimate for the Warrior, Hunter and Rogue) equals the
  table that tests/fallen-feed-history.test.js checks against the browser's
  activation tokens, so a drift on either side fails loudly.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import cast

import jax.numpy as jnp
import pytest
from pydantic import ValidationError
from tests.test_viewer_champion_streaks import _replay_of, _session

from marl_battlegrounds._battle_client.control import (
    make_neutral_joint_action,
    submit_joint_action,
)
from marl_battlegrounds._battle_client.model import DebuggerSession
from marl_battlegrounds._battle_client.service import DebuggerService
from marl_battlegrounds.core.types import Action
from marl_battlegrounds.evaluation.replay_io import LoadedReplay
from marl_battlegrounds.rendering.authorized_presentation import (
    AuthorizedBattlefieldSceneV1,
)
from marl_battlegrounds.rendering.vocabulary import (
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    WARRIOR_CLASS_ID,
)
from marl_battlegrounds.viewer.champion_streaks import (
    DAMAGING_ULTIMATE_CLASS_IDS,
    NO_TAKEDOWNS,
    ChampionTakedownsV1,
    TickTakedownsV1,
)
from marl_battlegrounds.viewer.presentation import _scene_next_end_spots_v1
from marl_battlegrounds.viewer.presentation_protocol import (
    LiveOracleAuthorizedPresentationFrameV1,
    ReplayNoSharedObsAuthorizedPresentationFrameV1,
    ReplayOracleAuthorizedPresentationFrameV1,
    ReplaySharedObsAuthorizedPresentationFrameV1,
)
from marl_battlegrounds.viewer.replay_protocol import (
    ReplayAbsoluteSeekCommandV1,
    ReplayCommandRequestV1,
)
from marl_battlegrounds.viewer.replay_service import ReplayViewerService

# (Basic deals damage, Ultimate deals damage) per class. Keep equal to
# DAMAGING_BY_CLASS in web/visual_debugger/tests/fallen-feed-history.test.js.
DAMAGING_BY_CLASS = {
    MAGE_CLASS_ID: (True, False),
    WARRIOR_CLASS_ID: (True, True),
    HUNTER_CLASS_ID: (True, True),
    ROGUE_CLASS_ID: (True, True),
    PRIEST_CLASS_ID: (False, False),
}


@pytest.fixture(scope="module")
def sessions() -> tuple[DebuggerSession, ...]:
    zeros = jnp.zeros((10,), dtype=jnp.int32)
    charge = Action(
        move=zeros,
        select_target=zeros.at[1].set(6),
        use_ultimate=zeros.at[1].set(1),
    )
    sessions = [_session("no_shared_obs")]
    for action in (charge, *[make_neutral_joint_action()] * 9):
        sessions.append(
            submit_joint_action(
                sessions[-1],
                action,
                submission_kind="interactive",
                report_actor_slots=(),
            )
        )
    return tuple(sessions)


@pytest.fixture(scope="module")
def loaded(sessions: tuple[DebuggerSession, ...]) -> LoadedReplay:
    return _replay_of(sessions)


type _ReplayFrame = (
    ReplayOracleAuthorizedPresentationFrameV1
    | ReplayNoSharedObsAuthorizedPresentationFrameV1
    | ReplaySharedObsAuthorizedPresentationFrameV1
)


def _viewer(
    loaded: LoadedReplay, recipient: int | None
) -> Callable[[int], _ReplayFrame]:
    service = ReplayViewerService(
        loaded,
        view_mode="researcher" if recipient is None else "pov",
        pov_global_slot=recipient,
    )

    def at(frame: int) -> _ReplayFrame:
        if frame != service.current_frame().cursor.frame_index:
            result = service.apply_command(
                ReplayCommandRequestV1(
                    client_id="recent-falls-test",
                    command_id=f"seek-{recipient}-{frame}-{service.revision}",
                    base_revision=service.revision,
                    command=ReplayAbsoluteSeekCommandV1(frame_index=frame),
                )
            )
            assert result.outcome == "response"
        return cast(_ReplayFrame, service.current_presentation().payload)

    return at


def _falls(
    payload: _ReplayFrame,
) -> list[tuple[int, str, tuple[str, ...], int | None, str | None]]:
    return [
        (
            row.tick,
            row.victim_public_agent_id,
            row.killer_public_agent_ids,
            row.points,
            row.ultimate_public_agent_id,
        )
        for row in payload.recent_falls or ()
    ]


def _scene(payload: _ReplayFrame) -> AuthorizedBattlefieldSceneV1:
    if isinstance(payload, ReplayOracleAuthorizedPresentationFrameV1):
        return payload.current_endpoint.scene
    return payload.current_endpoint.parts.scene


def test_recent_falls_follow_each_views_rules_and_the_window(
    loaded: LoadedReplay,
) -> None:
    frames = ReplayViewerService(loaded)._replay.frames
    first = frames[0].simulator_step_count
    fall = (first + 1, "5", ("1",), 1, "1")
    for recipient, expected in (
        (None, fall),
        (5, fall),
        (0, (first + 1, "5", (), 1, None)),
        (6, None),
    ):
        at = _viewer(loaded, recipient)
        for frame in (0, 1, 2, 8, 9, 10):
            rows = _falls(at(frame))
            inside = 2 <= frame <= 8 and expected is not None
            assert rows == ([expected] if inside else []), (recipient, frame)
    # Counting up to a frame and a seek straight to it give the same rows.
    sequential = ReplayViewerService(loaded)
    for frame in range(len(frames)):
        assert sequential._recent_falls(frame, recipient_global_slot=None) == (
            ReplayViewerService(loaded)._recent_falls(frame, recipient_global_slot=None)
        )


def test_live_recent_falls_equal_the_replay_viewers_in_every_view(
    sessions: tuple[DebuggerSession, ...], loaded: LoadedReplay
) -> None:
    for recipient in (None, 5, 0, 6):
        at = _viewer(loaded, recipient)
        for frame in (0, 1, 2, 8, 9, 10):
            service = DebuggerService(
                replace(
                    sessions[frame],
                    controlled_global_slot=1 if recipient is None else recipient,
                ),
                view_mode="researcher" if recipient is None else "pov",
                preset="analysis",
                include_stress=False,
            )
            try:
                live = service.current_presentation().payload
            finally:
                service.close()
            replayed = at(frame)
            assert getattr(live, "recent_falls", None) == replayed.recent_falls, (
                recipient,
                frame,
            )
    # Not empty-equals-empty: Oracle View lists the fall on the frame after it.
    assert len(_falls(_viewer(loaded, None)(2))) == 1


def test_live_frames_refuse_falls_on_their_own_tick_or_listed_twice(
    sessions: tuple[DebuggerSession, ...],
) -> None:
    service = DebuggerService(
        replace(sessions[2], controlled_global_slot=1),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )
    try:
        live = service.current_presentation().payload
    finally:
        service.close()
    assert isinstance(live, LiveOracleAuthorizedPresentationFrameV1)
    rows = live.recent_falls
    assert rows is not None and len(rows) == 1

    def rebuilt(recent: object) -> None:
        LiveOracleAuthorizedPresentationFrameV1.model_validate(
            {
                **{
                    name: getattr(live, name)
                    for name in LiveOracleAuthorizedPresentationFrameV1.model_fields
                },
                "recent_falls": recent,
            }
        )

    rebuilt(rows)
    shown_tick = live.source.source_simulator_step_count
    with pytest.raises(ValidationError, match="recent falls must lie"):
        rebuilt((rows[0].model_copy(update={"tick": shown_tick}),))
    with pytest.raises(ValidationError, match="recent falls must lie"):
        rebuilt((rows[0], rows[0]))


def test_session_record_keeps_each_views_window_of_falls(
    sessions: tuple[DebuggerSession, ...],
) -> None:
    context = sessions[0].evaluation_context
    fall = TickTakedownsV1(falls=frozenset({5}), takedowns=frozenset({(1, 5)}))
    seen_by_warrior = tuple(fall if slot == 1 else NO_TAKEDOWNS for slot in range(10))
    record = ChampionTakedownsV1().advance(fall, seen_by_warrior, tick=4)
    assert record.oracle_recent == ((4, fall),)
    assert record.agent_pov_recent[1] == ((4, fall),)
    assert record.agent_pov_recent[0] == ()
    assert record.quiet().oracle_recent == record.oracle_recent

    def listed(view: int | None, tick: int) -> list[int]:
        return [row.tick for row in record.recent_falls(context, view, tick)]

    # A tick lists the 7 ticks before it, never itself; a view lists only its own.
    assert listed(None, 4) == []
    assert listed(None, 5) == [4]
    assert listed(None, 11) == [4]
    assert listed(None, 12) == []
    assert listed(1, 5) == [4]
    assert listed(0, 5) == []
    # A later advance drops the rows no later frame can list.
    later = record.advance(NO_TAKEDOWNS, (NO_TAKEDOWNS,) * 10, tick=12)
    assert later.oracle_recent == ()
    assert record.advance(
        NO_TAKEDOWNS, (NO_TAKEDOWNS,) * 10, tick=11
    ).oracle_recent == ((4, fall),)
    with pytest.raises(ValueError, match="rising"):
        ChampionTakedownsV1(oracle_recent=((5, fall), (4, fall)))
    with pytest.raises(ValueError, match="rising"):
        ChampionTakedownsV1(oracle_recent=((4, NO_TAKEDOWNS),))
    with pytest.raises(ValueError, match="rising"):
        record.advance(fall, seen_by_warrior, tick=4)


def test_next_end_spots_name_next_positions_of_champions_seen_at_both_ends(
    loaded: LoadedReplay,
) -> None:
    last = len(ReplayViewerService(loaded)._replay.frames) - 1
    # Oracle View, a POV that sees only part of the field, and one that sees
    # the field shrink after its champion falls.
    for recipient in (None, 0, 5):
        at = _viewer(loaded, recipient)
        payloads = [at(frame) for frame in range(last + 1)]
        scenes = [_scene(payload) for payload in payloads]
        for frame in range(last + 1):
            spots = payloads[frame].next_end_spots
            assert spots is not None
            if frame == last:
                assert spots == ()
                continue
            alive = {
                agent.public_agent_id
                for agent in scenes[frame].agents
                if agent.life_state == "alive"
            }
            expected = [
                (agent.public_agent_id, agent.position)
                for agent in scenes[frame + 1].agents
                if agent.public_agent_id in alive
            ]
            got = [(spot.public_agent_id, spot.position) for spot in spots]
            if recipient is None:
                assert sorted(got) == sorted(expected), frame
                assert {name for name, _ in got} == alive
            else:
                assert got == expected, (recipient, frame)


@dataclass(frozen=True)
class _Body:
    public_agent_id: str
    life_state: str
    position: tuple[float, float]


@dataclass(frozen=True)
class _Scene:
    agents: tuple[_Body, ...]


def test_a_champion_seen_now_but_not_next_tick_gets_no_spot() -> None:
    now = _Scene(
        (
            _Body("0", "alive", (1.0, 1.0)),
            _Body("6", "alive", (5.0, 5.0)),
            _Body("7", "corpse", (6.0, 6.0)),
        )
    )
    after = _Scene((_Body("0", "alive", (1.5, 1.0)), _Body("7", "alive", (2.0, 2.0))))
    spots = _scene_next_end_spots_v1(
        cast(AuthorizedBattlefieldSceneV1, now),
        cast(AuthorizedBattlefieldSceneV1, after),
    )
    # "6" left sight (no spot); "7" was not alive now (no spot, even after it returns).
    assert [(spot.public_agent_id, spot.position) for spot in spots] == [
        ("0", (1.5, 1.0))
    ]


def test_tick_takedowns_refuse_an_ultimate_or_home_fall_without_its_fall() -> None:
    with pytest.raises(ValueError, match="Ultimate"):
        TickTakedownsV1(
            falls=frozenset({5}),
            takedowns=frozenset({(1, 5)}),
            ultimates=frozenset({(2, 5)}),
        )
    with pytest.raises(ValueError, match="home fall"):
        TickTakedownsV1(
            falls=frozenset({5}),
            takedowns=frozenset({(1, 5)}),
            home_falls=frozenset({4}),
        )


def test_the_server_damaging_rule_equals_the_shared_table() -> None:
    for class_id, (basic, ultimate) in DAMAGING_BY_CLASS.items():
        assert (class_id != PRIEST_CLASS_ID) is basic, class_id
        assert (class_id in DAMAGING_ULTIMATE_CLASS_IDS) is ultimate, class_id
