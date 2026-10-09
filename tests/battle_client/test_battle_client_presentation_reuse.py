"""Check that live presentations are built once, reused exactly, and validated once.

Agent POV used to be several times slower than Oracle View because each request
redid the same work: every one of the ten source rows re-validated the whole
context/frame join, and one switch built the same presentation two or three
times (at commit, for the page's request and for the live snapshot). This file
checks:

- one Agent POV switch builds its presentation once; the page's request and the
  live snapshot reuse it, as the same object;
- every reused presentation equals a fresh build, in Agent POV and Oracle View,
  across focus changes, view switches and a Lock In, so reuse never changes what
  either view shows;
- the context/frame join is validated once for all ten source rows, a new frame
  object is validated again, and a failing pair fails on every call.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import pytest
from tests.battle_client.test_battle_client_view_recipient import (
    _hot_seat_session,
    _scenario_session,
    _send,
    _service,
)

from marl_battlegrounds._battle_client import service as service_module
from marl_battlegrounds._battle_client.protocol import (
    KeyboardCommandV1,
    RosterSelectionCommandV1,
    SetViewCommandV1,
)
from marl_battlegrounds._battle_client.service import DebuggerService
from marl_battlegrounds.evaluation.models import (
    EvaluationEpisodeContextV4,
    EvaluationFrameV3,
)
from marl_battlegrounds.rendering import evaluation_adapter
from marl_battlegrounds.rendering.evaluation_adapter import (
    build_shared_obs_authority_source_material_projection_v1,
)
from marl_battlegrounds.viewer.presentation_protocol import (
    AuthorizedPresentationFrameV1,
)


def _fresh(service: DebuggerService) -> AuthorizedPresentationFrameV1:
    with service._lock:
        service._presentations.clear()
        return service.current_presentation().payload  # pyright: ignore[reportReturnType]


def test_an_agent_pov_switch_builds_its_presentation_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(_scenario_session())
    real = service_module.build_live_shared_obs_authorized_presentation_v1
    builds: list[int] = []

    def counted(*args: object, **kwargs: object) -> object:
        builds.append(1)
        return real(*args, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(
        service_module, "build_live_shared_obs_authorized_presentation_v1", counted
    )
    try:
        _send(service, RosterSelectionCommandV1(role="control", global_slot=2), "p2")
        first = service.current_presentation()
        snapshot = service.current_snapshot()
        assert service.current_presentation() is first
        assert snapshot.presentation is first.payload
        assert builds == [1]
    finally:
        service.close()


def test_reused_presentations_always_equal_a_fresh_build() -> None:
    service = _service(_hot_seat_session())
    steps: list[tuple[str, object]] = [
        ("start", None),
        ("pick-3", RosterSelectionCommandV1(role="control", global_slot=3)),
        (
            "tab",
            KeyboardCommandV1(
                key="Tab",
                shift_key=False,
                ctrl_key=False,
                alt_key=False,
                meta_key=False,
                repeat=False,
            ),
        ),
        ("to-oracle", SetViewCommandV1(view_mode="researcher")),
        ("pick-7", RosterSelectionCommandV1(role="control", global_slot=7)),
        ("to-pov", SetViewCommandV1(view_mode="pov")),
        (
            "lock-in",
            KeyboardCommandV1(
                key=" ",
                shift_key=False,
                ctrl_key=False,
                alt_key=False,
                meta_key=False,
                repeat=False,
            ),
        ),
        ("pick-6", RosterSelectionCommandV1(role="control", global_slot=6)),
    ]
    try:
        for name, command in steps:
            if command is not None:
                _send(service, command, name)
            reused = service.current_presentation().payload
            fresh = _fresh(service)
            assert reused is not fresh, name
            assert reused == fresh, name
            assert reused.model_dump_json() == fresh.model_dump_json(), name
        assert service.session.current_evaluation_frame.frame_index == 1
    finally:
        service.close()


def test_the_frame_join_is_validated_once_for_all_ten_source_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _hot_seat_session()
    context = session.evaluation_context
    frame = session.current_evaluation_frame
    assert type(context) is EvaluationEpisodeContextV4
    assert type(frame) is EvaluationFrameV3
    real = evaluation_adapter.validate_context_joined_evaluation_frame_v1
    checked: list[object] = []

    def counted(
        joined_context: EvaluationEpisodeContextV4, joined_frame: EvaluationFrameV3
    ) -> None:
        checked.append(joined_frame)
        real(joined_context, joined_frame)

    monkeypatch.setattr(
        evaluation_adapter, "validate_context_joined_evaluation_frame_v1", counted
    )
    rows = [
        build_shared_obs_authority_source_material_projection_v1(
            context, frame, selected_global_slot=slot
        )
        for slot in range(10)
    ]
    assert checked == [frame]
    assert len(rows) == 10
    equal_copy = frame.model_copy()
    build_shared_obs_authority_source_material_projection_v1(
        context, equal_copy, selected_global_slot=0
    )
    assert checked == [frame, equal_copy]

    def failing(
        joined_context: EvaluationEpisodeContextV4, joined_frame: EvaluationFrameV3
    ) -> None:
        checked.append(joined_frame)
        raise ValueError("bad join")

    monkeypatch.setattr(
        evaluation_adapter, "validate_context_joined_evaluation_frame_v1", failing
    )
    other = frame.model_copy()
    for _ in range(2):
        with pytest.raises(ValueError, match="bad join"):
            build_shared_obs_authority_source_material_projection_v1(
                context, other, selected_global_slot=0
            )
    assert checked == [frame, equal_copy, other, other]
