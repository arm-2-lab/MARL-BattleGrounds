"""Check the Replay Viewer's Capture the Flag games and match facts.

Contracts: a Capture the Flag replay V5, NoSharedObs or SharedObs, that holds
all six flag events and a grab and fall opens in the Replay Viewer in Oracle
View and in each Agent POV, and at every frame each view gets its own flags.
Oracle View shows both flags with their carriers and gets the six flag events
with their agents, in recorded order. Each Agent POV shows both flags as its
own objective rows give them (a carrier only when the actor is or sees it,
decision 155), gets one flag_changed line per flag whose state changed, and
never gets an Oracle flag event. Every view's match summary ends with Team
A's winning capture. build_match_summary_v1 summarizes a Capture the Flag
frame (context V5) as task mode 3, with the captures as scores, the captures
to win as the threshold, the result from the capture_the_flag_completed event
and the configured last tick as the episode limit. Without the start frame,
the score before the tick comes from the flag_captured event, credited to the
carrier's team (the captured flag's enemy), whether the events are the
transition's own or the visual batch's. Every view's summary carries the
recorded flag return ticks, a public rule setting. A fall scores no points in
Capture the Flag, so the Fallen panel's recent falls carry none. The static
Matplotlib view opens a Capture the Flag replay V5 and draws both bases (Team B
double-lined), the capture rings, the ground and carried pennants (each flying
from its home's side, even at the other end) and a dropped flag's ticks left.
The Rune Scrubber marks every capture at its tick, by the capturing team, in
Oracle View and in every Agent POV, and names the capturer only where the view
saw the carrier. The frames,
objective rows and flag facts are hand-set, as in
tests/rendering/test_capture_the_flag_presentation.py, until Core scores Capture
the Flag (his C1.3 and C1.4). Host only, on CPU.
"""

import json
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from tests.helpers.evaluation_fixtures import neutral_action
from tests.rendering.test_capture_the_flag_presentation import (
    _CONTEXT,  # pyright: ignore[reportPrivateUsage]
    _RUNTIME,  # pyright: ignore[reportPrivateUsage]
    _frame,  # pyright: ignore[reportPrivateUsage]
    _view,  # pyright: ignore[reportPrivateUsage]
    game,  # noqa: F401  # pyright: ignore[reportUnusedImport]
)

from marl_battlegrounds.core.env import reset, step
from marl_battlegrounds.evaluation._recording.capture import (
    _capture_evaluation_transition_unit,  # pyright: ignore[reportPrivateUsage]
    _capture_initial_evaluation_frame,  # pyright: ignore[reportPrivateUsage]
)
from marl_battlegrounds.evaluation._recording.recording_context import (
    build_recording_context,
)
from marl_battlegrounds.evaluation._replays.replay_v5 import (
    ReplayArtifactV5,
    build_replay_v5,
)
from marl_battlegrounds.evaluation.models import (
    CodeRevisionV2,
    EvaluationEpisodeContextV5,
    EvaluationFrameV4,
    EvaluationTransitionV2,
)
from marl_battlegrounds.evaluation.replay_io import LoadedReplay
from marl_battlegrounds.policies.shared_obs import (
    build_default_shared_obs_information_availability,
)
from marl_battlegrounds.rendering.authorized_presentation import (
    AgentPovVisualIncomingFlagChangedEventV1,
)
from marl_battlegrounds.rendering.evaluation_adapter import build_visual_event_batch_v2
from marl_battlegrounds.tasks import make_standard_capture_the_flag_config
from marl_battlegrounds.viewer.champion_streaks import (
    TickTakedownsV1,
    recent_falls_v1,
)
from marl_battlegrounds.viewer.match_summary import build_match_summary_v1
from marl_battlegrounds.viewer.presentation_protocol import (
    ReplayNoSharedObsAuthorizedPresentationFrameV1,
    ReplayOracleAuthorizedPresentationFrameV1,
    ReplaySharedObsAuthorizedPresentationFrameV1,
)
from marl_battlegrounds.viewer.replay_protocol import (
    ReplayAbsoluteSeekCommandV1,
    ReplayCommandRequestV1,
)
from marl_battlegrounds.viewer.replay_service import ReplayViewerService


def test_a_winning_capture_summarizes_as_capture_the_flag(
    game: tuple[EvaluationFrameV4, EvaluationTransitionV2, EvaluationFrameV4],  # noqa: F811
) -> None:
    start, transition, successor = game
    homes = _CONTEXT.resolved_env_config.capture_the_flag_flag_home_positions
    carrying = _frame(
        start,
        scores=(2, 0),
        flag_carrier_slots=(None, 1),
        flag_positions=(homes[0], tuple(start.snapshot.agent_positions[1])),
    )
    won = _frame(successor, scores=(3, 0))
    view = _view(
        carrying, transition, won, outcome=1, enemy_flag_was_captured_by_agent={1}
    )
    summary = build_match_summary_v1(
        _CONTEXT, won, view.transition.events, transition_start_frame=carrying
    )
    assert (
        summary.task_mode,
        summary.score_threshold,
        summary.scores,
        summary.previous_scores,
        summary.outcome,
        summary.episode_limit,
        summary.flag_return_steps,
    ) == (
        3,
        3,
        (3, 0),
        (2, 0),
        "team_a_win",
        50,
        _CONTEXT.resolved_env_config.capture_the_flag_dropped_flag_auto_return_steps,
    )
    for events in (view.transition.events, build_visual_event_batch_v2(view).events):
        assert build_match_summary_v1(_CONTEXT, won, events).previous_scores == (2, 0)


def test_a_capture_the_flag_fall_scores_no_points() -> None:
    # Team Deathmatch's points ride on its Red Zone rule; in Capture the Flag a
    # fall scores nothing, so the Fallen panel's rows carry no points.
    tick = TickTakedownsV1(
        falls=frozenset({1, 6}), takedowns=frozenset(), home_falls=frozenset({1})
    )
    assert [row.points for row in recent_falls_v1(_CONTEXT, [(4, tick)])] == [
        None,
        None,
    ]


def _with_c1c_rows(
    context: EvaluationEpisodeContextV5, frame: EvaluationFrameV4
) -> EvaluationFrameV4:
    # Fill each active actor's context columns 6-18 and flag rows 0-3 from the
    # snapshot, the way C1.C defines them and Core will from C1.4 on.
    config = context.resolved_env_config
    homes = config.capture_the_flag_flag_home_positions
    capture = config.capture_the_flag_flag_capture_distance
    touch = config.capture_the_flag_flag_interaction_distance
    threshold = float(config.capture_the_flag_score_threshold)
    snapshot = frame.snapshot
    scores = snapshot.capture_the_flag_scores
    data = frame.model_dump()
    observation = data["base_observation"]
    contexts = [list(row) for row in observation["context_features"]]
    objectives = [list(map(list, agent)) for agent in observation["objective_features"]]
    # Each team's observation rows follow its active global slots.
    team_slots = {
        team: [
            slot
            for slot, row in enumerate(context.roster)
            if row.configured_active and row.configured_team_id == team
        ]
        for team in (1, 2)
    }
    for slot, roster in enumerate(context.roster):
        if not roster.configured_active:
            continue
        team = roster.configured_team_id
        own, enemy = team - 1, 2 - team
        # Columns 6-18: is TDM, KOTH, CTF; active objective rows; TDM and KOTH
        # scores; CTF scores (own, enemy); TDM, KOTH and CTF thresholds.
        contexts[slot][6:19] = [
            *(0.0, 0.0, 1.0, 4.0, 0.0, 0.0, 0.0, 0.0),
            *(float(scores[own]), float(scores[enemy]), 0.0, 0.0, threshold),
        ]
        objectives[slot][0] = [1.0, *homes[own], capture, *[0.0] * 8]
        objectives[slot][1] = [1.0, *homes[enemy], capture, *[0.0] * 8]
        for objective_row, flag in ((2, own), (3, enemy)):
            carrier = snapshot.capture_the_flag_flag_carrier_slots[flag]
            countdown = snapshot.capture_the_flag_dropped_flag_countdowns[flag]
            # Decision 139: a home flag is its owner's, a carried flag its
            # carrier's team's, and a dropped flag nobody's.
            holder = None if countdown else flag + 1
            carrier_rows = [0.0] * 5
            if carrier is not None:
                holder = context.roster[carrier].configured_team_id
                unit_row = team_slots[holder].index(carrier)
                side = "ally" if holder == team else "enemy"
                seen = observation[f"{side}_visibility_mask"][slot][unit_row]
                # Decision 155: a row names only a carrier the actor is or sees.
                if carrier == slot or seen:
                    carrier_rows[unit_row] = 1.0
            objectives[slot][objective_row] = [
                1.0,
                *snapshot.capture_the_flag_flag_positions[flag],
                touch,
                float(holder == team),
                float(holder == 3 - team),
                float(countdown),
                *carrier_rows,
            ]
    observation["context_features"] = tuple(tuple(row) for row in contexts)
    observation["objective_features"] = tuple(
        tuple(tuple(row) for row in agent) for agent in objectives
    )
    return EvaluationFrameV4.model_validate(data)


# A short game: a dropped flag waits 2 ticks, and the first capture wins.
_STORY_CONFIG = make_standard_capture_the_flag_config(
    map_id=42,
    team_a_roster=("mage", "warrior"),
    team_b_roster=("rogue",),
    score_threshold=1,
    max_steps=50,
    dropped_flag_auto_return_steps=2,
)
_MODES = ("no_shared_obs", "shared_obs")


@pytest.fixture(scope="module")
def story() -> dict[str, tuple[EvaluationEpisodeContextV5, ReplayArtifactV5]]:
    return build_capture_the_flag_story()


def build_capture_the_flag_story() -> dict[
    str, tuple[EvaluationEpisodeContextV5, ReplayArtifactV5]
]:
    contexts: dict[str, EvaluationEpisodeContextV5] = {}
    for mode in _MODES:
        context, _ = build_recording_context(
            jax.device_get(_STORY_CONFIG),
            run_id="run-1",
            phase="evaluation",
            pass_id="1",
            episode={"episode_id": 7, "map_id": 42},
            policies={"team_a": "Alpha", "team_b": "Beta"},
            details={
                "code_revision": CodeRevisionV2(package_version="0.0.0").model_dump(
                    mode="json"
                ),
                "runtime_provenance": _RUNTIME.model_dump(mode="json"),
                "execution_information_mode": mode,
            },
        )
        assert type(context) is EvaluationEpisodeContextV5
        contexts[mode] = context
    roster = contexts["shared_obs"].roster
    # SharedObs lets each actor receive its active teammates' sensors.
    availability = {
        "no_shared_obs": None,
        "shared_obs": build_default_shared_obs_information_availability(
            jnp.asarray([row.configured_active for row in roster]),
            jnp.asarray([row.configured_team_id for row in roster]),
        ),
    }
    # Core plays six neutral ticks once, and each mode records them.
    state, observation, mask, _ = reset(_STORY_CONFIG, jax.random.PRNGKey(0))
    frames: dict[str, list[EvaluationFrameV4]] = {}
    recorded: dict[str, list[EvaluationTransitionV2]] = {mode: [] for mode in _MODES}
    for mode in _MODES:
        initial = _capture_initial_evaluation_frame(
            contexts[mode], state, observation, mask, availability[mode]
        )
        assert isinstance(initial, EvaluationFrameV4)
        frames[mode] = [initial]
    for index in range(6):
        state, observation, reward, done, mask, info = step(
            _STORY_CONFIG, state, mask, neutral_action(), jax.random.PRNGKey(index + 1)
        )
        for mode in _MODES:
            transition, frame = _capture_evaluation_transition_unit(
                contexts[mode],
                frames[mode][-1],
                state,
                observation,
                mask,
                info.transition_facts,
                reward,
                done,
                successor_shared_obs_information_availability_by_recipient_and_sensor_source=(
                    availability[mode]
                ),
            )
            assert isinstance(frame, EvaluationFrameV4)
            assert isinstance(transition, EvaluationTransitionV2)
            frames[mode].append(frame)
            recorded[mode].append(transition)
    return {
        mode: (
            contexts[mode],
            _with_story(contexts[mode], frames[mode], recorded[mode]),
        )
        for mode in _MODES
    }


def _with_story(
    context: EvaluationEpisodeContextV5,
    frames: list[EvaluationFrameV4],
    recorded: list[EvaluationTransitionV2],
) -> ReplayArtifactV5:
    homes = context.resolved_env_config.capture_the_flag_flag_home_positions

    def body(index: int, slot: int) -> tuple[float, ...]:
        return tuple(frames[index].snapshot.agent_positions[slot])

    # Per tick: the successor's flags, the outcome and the flag facts.
    # Flag 0 is Team A's and flag 1 Team B's; agents 0 and 1 are Team A's.
    story: list[tuple[dict[str, object], int, dict[str, set[int]]]] = [
        # Agent 5 grabs Team A's flag and falls with it on the same tick
        # (decision 169), while agent 0 picks up Team B's flag.
        (
            {
                "flag_carrier_slots": (None, 0),
                "flag_positions": ((1.5, 5.0), body(1, 0)),
                "dropped_flag_countdowns": (2, 0),
            },
            0,
            {"flag_was_picked_up_by_agent": {0, 5}, "flag_was_dropped_by_agent": {5}},
        ),
        # Agent 1 returns Team A's flag; agent 0 falls and drops Team B's.
        (
            {
                "flag_positions": (homes[0], body(2, 0)),
                "dropped_flag_countdowns": (0, 2),
            },
            0,
            {
                "ally_flag_was_returned_by_agent": {1},
                "flag_was_dropped_by_agent": {0},
            },
        ),
        # Team B's flag counts down, then goes home by itself.
        (
            {
                "flag_positions": (homes[0], body(2, 0)),
                "dropped_flag_countdowns": (0, 1),
            },
            0,
            {},
        ),
        ({}, 0, {"flag_was_automatically_returned_by_team": {1}}),
        # Agent 1 takes Team B's flag and brings it home for the win.
        (
            {"flag_carrier_slots": (None, 1), "flag_positions": (homes[0], body(5, 1))},
            0,
            {"flag_was_picked_up_by_agent": {1}},
        ),
        ({"scores": (1, 0)}, 1, {"enemy_flag_was_captured_by_agent": {1}}),
    ]
    story_frames = [_with_c1c_rows(context, frames[0])] + [
        _with_c1c_rows(context, _frame(frame, **flags))
        for frame, (flags, _, _) in zip(frames[1:], story, strict=True)
    ]
    transitions: list[EvaluationTransitionV2] = []
    for index, (_, outcome, facts) in enumerate(story):
        view = _view(
            story_frames[index],
            recorded[index],
            story_frames[index + 1],
            outcome,
            context,
            **facts,
        )
        assert isinstance(view.transition, EvaluationTransitionV2)
        transitions.append(view.transition)
    return build_replay_v5(
        context, story_frames, transitions, runtime_provenance=_RUNTIME
    )


# Both flags per frame, the same public facts in every view: (state, carrier
# global slot, countdown).
_FLAGS = [
    [("home", None, 0), ("home", None, 0)],
    [("dropped", None, 2), ("carried", 0, 0)],
    [("home", None, 0), ("dropped", None, 2)],
    [("home", None, 0), ("dropped", None, 1)],
    [("home", None, 0), ("home", None, 0)],
    [("home", None, 0), ("carried", 1, 0)],
    [("home", None, 0), ("home", None, 0)],
]
# Oracle View's flag events per frame: (event kind, agent global slot).
_ORACLE_EVENTS = [
    [],
    [("flag_picked_up", 0), ("flag_picked_up", 5), ("flag_dropped", 5)],
    [("flag_returned", 1), ("flag_dropped", 0)],
    [],
    [("flag_automatically_returned", None)],
    [("flag_picked_up", 1)],
    [("flag_captured", 1), ("capture_the_flag_completed", None)],
]
# An Agent POV's flag lines per frame: (flag team ID, state before, state
# after, agent global slot).
_POV_LINES = [
    [],
    [(1, "home", "dropped", None), (2, "home", "carried", 0)],
    [(1, "dropped", "home", None), (2, "carried", "dropped", 0)],
    [],
    [(2, "dropped", "home", None)],
    [(2, "home", "carried", 1)],
    [(2, "carried", "home", 1)],
]


def _anchor_key(event: object) -> str | None:
    anchor = getattr(event, "agent_anchor", None)
    return None if anchor is None else anchor.presentation_key


@pytest.mark.parametrize("execution_information_mode", _MODES)
def test_a_capture_the_flag_replay_shows_each_view_its_own_flags(
    story: dict[str, tuple[EvaluationEpisodeContextV5, ReplayArtifactV5]],
    execution_information_mode: str,
) -> None:
    context, replay = story[execution_information_mode]
    slot_by_public_id = {
        row.public_agent_id: slot for slot, row in enumerate(context.roster)
    }
    pov_frame = (
        ReplaySharedObsAuthorizedPresentationFrameV1
        if execution_information_mode == "shared_obs"
        else ReplayNoSharedObsAuthorizedPresentationFrameV1
    )
    # Oracle View; agent 0, which carries Team B's flag at frame 1 and sees its
    # teammate carry it at frame 5; and Team B's rogue, which never sees Team
    # A's carriers across the map.
    for recipient in (None, 0, 5):
        service = ReplayViewerService(
            LoadedReplay(replay),
            view_mode="researcher" if recipient is None else "pov",
            pov_global_slot=recipient,
        )
        sees_carriers = recipient != 5
        for index, flags in enumerate(_FLAGS):
            if index:
                result = service.apply_command(
                    ReplayCommandRequestV1(
                        client_id="capture-the-flag-test",
                        command_id=f"seek-{index}",
                        base_revision=service.revision,
                        command=ReplayAbsoluteSeekCommandV1(frame_index=index),
                    )
                )
                assert result.outcome == "response"
            payload = service.current_presentation().payload
            if recipient is None:
                assert isinstance(payload, ReplayOracleAuthorizedPresentationFrameV1)
                scene = payload.current_endpoint.scene
                summary = payload.latest_events
            else:
                assert isinstance(payload, pov_frame)
                scene = payload.current_endpoint.parts.scene
                summary = payload.visual_events
            slot_by_key: dict[str | None, int | None] = {None: None} | {
                agent.presentation_key: slot_by_public_id[agent.public_agent_id]
                for agent in scene.agents
            }
            assert scene.capture_the_flag is not None
            assert [
                (
                    flag.state,
                    slot_by_key[flag.carrier_presentation_key],
                    flag.countdown_steps,
                )
                for flag in scene.capture_the_flag.flags
            ] == [
                (state, carrier if sees_carriers else None, countdown)
                for state, carrier, countdown in flags
            ]
            # In a Capture the Flag game only the flag events and the
            # completion rank from 130.
            events = [
                event
                for event in (() if summary is None else summary.events)
                if event.phase_rank >= 130
            ]
            if recipient is None:
                assert [
                    (event.event_kind, slot_by_key[_anchor_key(event)])
                    for event in events
                ] == _ORACLE_EVENTS[index]
            else:
                lines = [
                    event
                    for event in events
                    if isinstance(event, AgentPovVisualIncomingFlagChangedEventV1)
                ]
                assert len(lines) == len(events)
                assert [
                    (
                        line.flag_team_id,
                        line.state_before,
                        line.state_after,
                        slot_by_key[_anchor_key(line)],
                    )
                    for line in lines
                ] == [
                    (team_id, before, after, agent if sees_carriers else None)
                    for team_id, before, after, agent in _POV_LINES[index]
                ]
            if index == len(_FLAGS) - 1:
                match = payload.match_summary
                assert match is not None
                assert (
                    match.task_mode,
                    match.score_threshold,
                    match.scores,
                    match.outcome,
                ) == (3, 1, (1, 0), "team_a_win")
            # The return ticks are a public rule setting: every view's match
            # summary carries the recorded value.
            assert payload.match_summary is not None
            assert payload.match_summary.flag_return_steps == (
                context.resolved_env_config.capture_the_flag_dropped_flag_auto_return_steps
            )


def test_the_static_view_opens_a_capture_the_flag_replay_and_draws_its_flags(
    story: dict[str, tuple[EvaluationEpisodeContextV5, ReplayArtifactV5]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    from marl_battlegrounds.rendering import render_scene_geometry
    from marl_battlegrounds.viewer import static

    _, replay = story["no_shared_obs"]
    drawn: list[tuple[str, float]] = []
    pennants: dict[str, list[list[float]]] = {}

    def paint(scene: object, **options: object) -> object:
        result = render_scene_geometry(scene, **options)  # pyright: ignore[reportArgumentType]
        axes: Any = result.axes
        for artist in (*axes.patches, *axes.texts):
            gid: object = artist.get_gid()
            if isinstance(gid, str) and gid.startswith("scene:v2:flag:"):
                drawn.append((gid, float(artist.get_zorder())))
                if gid.endswith(":pennant"):
                    pennants[gid] = artist.get_xy().tolist()
        pyplot.close(result.figure)  # pyright: ignore[reportArgumentType]
        return result

    monkeypatch.setattr(static, "render_scene_geometry", paint)
    monkeypatch.setattr(static, "_load_pyplot", lambda: pyplot)
    monkeypatch.setattr(pyplot, "show", lambda: None)
    # Frame 1: Team A's flag lies dropped and Team B's is carried by agent 0.
    assert (
        static.run_static_replay_artifact_renderer(
            replay=replay, frame_index=1, show_ranges=False
        )
        == 0
    )
    kinds = {gid.rsplit(":", 1)[1] for gid, _ in drawn}
    assert {"base", "capture-ring", "pole", "pennant", "ticks"} <= kinds
    # Team B is double-lined.
    assert sum(gid == "scene:v2:flag:2:base" for gid, _ in drawn) == 2
    assert sum(gid == "scene:v2:flag:1:base" for gid, _ in drawn) == 1
    # Team A's dropped flag lies under the bodies (z-order 24 and up); Team B's,
    # carried, rides over them and their selection rings (up to 32).
    order = dict(drawn)
    assert order["scene:v2:flag:1:pennant"] < 24 < 32 < order["scene:v2:flag:2:pennant"]
    # Each pennant flies from its home's side, as in the browser: Team B's,
    # carried at Team A's end, still flies left (its second point is the tip).
    team_a, team_b = (
        pennants["scene:v2:flag:1:pennant"],
        pennants["scene:v2:flag:2:pennant"],
    )
    assert team_a[1][0] > team_a[0][0]
    assert team_b[1][0] < team_b[0][0]


@pytest.mark.parametrize("execution_information_mode", _MODES)
def test_the_rune_scrubber_marks_every_capture_and_names_only_a_seen_capturer(
    story: dict[str, tuple[EvaluationEpisodeContextV5, ReplayArtifactV5]],
    execution_information_mode: str,
) -> None:
    context, replay = story[execution_information_mode]
    capturer = context.roster[1].public_agent_id
    # Oracle View; agent 0, the capturer's teammate; and Team B's rogue, which
    # never sees Team A's carriers but sees the score step up.
    for recipient, seen in ((None, True), (0, True), (5, False)):
        service = ReplayViewerService(
            LoadedReplay(replay),
            view_mode="researcher" if recipient is None else "pov",
            pov_global_slot=recipient,
        )
        ticks: list[dict[str, Any]] = json.loads(service.timeline_marks())["ticks"]
        captures: dict[int, list[tuple[int, str | None]]] = {
            tick["frame_index"]: [
                (mark["team_id"], (mark["capturer"] or {}).get("public_agent_id"))
                for mark in tick["captures"]
            ]
            for tick in ticks
            if tick["captures"]
        }
        assert captures == {6: [(1, capturer if seen else None)]}, recipient
