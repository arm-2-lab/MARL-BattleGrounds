"""Check that replays keep every controller's name, per team and per champion.

Owner decision, 2 October 2026. The contracts checked here:

- ``team_controller_names`` names each of a team's five slots: a Policy, a
  shared or independent Policy adapter, a ``marl_bgs.team`` (including class
  selectors and nested teams) and a ``marl_bgs.pool``.
- ``marl_bgs.evaluate`` records each team's name (``marl_bgs.system_name.*``)
  and one name per used champion slot (``marl_bgs.controller_name.slot_N``),
  as JSON strings, both for in-memory replays and for replays saved in a run
  folder. A team of several named methods names each member's champions.
- Those names live only in the replay context: evaluate's run details (pass
  identity, systems), result tables and replay file names are the same as
  when no slot names are recorded.
- A replay saved from a live BattleClient session records the session's slot
  labels (Manual as "Manual", Unicode kept, a ``team(...)`` entry named by its
  member) and the team names joined with " / "; built-in controller kinds use
  their public names ("tdm-beta").
- A DevClient session switched to an offered System by alias
  (``system:counter``, the service's whole-team route, with no labels) labels
  and records each champion by the loaded System's own names, never the
  alias: a shared Policy names all five champions, a ``team(...)`` names each
  member's champions. These session labels are also the live labels on
  screen. A plain restart keeps them.
- The Replay Viewer's match summary and presentation show recorded names
  exactly as recorded ("tdm-beta" stays "tdm-beta", "researcher_policy" stays
  "researcher_policy"); file-style cleanup applies only to old policy IDs.
- Old replays without slot names open with ``slot_controller_names`` None and
  their older labels, and ``load_replay`` and the replay file-name builder
  accept records with and without the new keys. Without slot names, a fixed
  scenario's Team B title still wins over a recorded team name.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from tests.helpers.evaluation_fixtures import captured_evaluation_trajectory
from tests.helpers.visual_debugger_fixtures import debugger_test_launch_specification

import marl_battlegrounds as marl_bgs
from marl_battlegrounds._battle_client import control
from marl_battlegrounds._battle_client.evaluation_bridge import (
    DebuggerActionSourceKindV1,
    build_debugger_evaluation_launch_specification_v1,
)
from marl_battlegrounds._battle_client.methods import resolve_controllers
from marl_battlegrounds._battle_client.recording import (
    build_debugger_recording_specification_v1,
    recording_action_source,
    recording_policy_execution_included,
)
from marl_battlegrounds._battle_client.replay_recorder import DebuggerReplayRecorder
from marl_battlegrounds._battle_client.setup import prepare_battle_setup
from marl_battlegrounds.evaluation._replays.controller_names import (
    recorded_slot_controller_names,
    recorded_team_controller_name,
    slot_controller_name_keys,
)
from marl_battlegrounds.evaluation.models import (
    AggregationKeyV1,
    AssignedPolicySlotV1,
    EvaluationEpisodeContext,
)
from marl_battlegrounds.evaluation.policy_execution import (
    Policy,
    System,
    independent_policies,
)
from marl_battlegrounds.evaluation.recording_identity import team_controller_names
from marl_battlegrounds.evaluation.replay_io import (
    generated_replay_filename,
    load_replay,
    preflight_replay_destination,
)
from marl_battlegrounds.evaluation.run_writer import RunWriter
from marl_battlegrounds.evaluation.runtime_provenance import capture_runtime_provenance
from marl_battlegrounds.viewer.match_summary import (
    MatchSummaryV1,
    build_match_summary_v1,
)
from marl_battlegrounds.viewer.presentation_protocol import (
    ReplayOracleAuthorizedPresentationFrameV1,
)
from marl_battlegrounds.viewer.replay_service import ReplayViewerService

_FIXTURES = Path(__file__).parents[2] / "fixtures" / "historical_scenario_v4"
_LONG = "Recurrent MAPPO · Pool Seed 7 Best"


def _named(reference: str, name: str) -> Policy:
    method = marl_bgs.load_method(reference)
    assert isinstance(method, Policy)
    return replace(method, name=name)


def _names(context: EvaluationEpisodeContext) -> dict[str, str]:
    return {
        row.name: row.value
        for row in context.aggregation_keys
        if row.name.startswith(("marl_bgs.controller_name.", "marl_bgs.system_name."))
    }


def _without_slot_names(
    context: EvaluationEpisodeContext,
) -> EvaluationEpisodeContext:
    return context.model_copy(
        update={
            "aggregation_keys": tuple(
                row
                for row in context.aggregation_keys
                if not row.name.startswith("marl_bgs.controller_name.")
            )
        }
    )


def _replay_summary(path: str | Path) -> MatchSummaryV1:
    loaded = load_replay(path)
    payload = ReplayViewerService(loaded, view_mode="researcher").current_presentation()
    assert isinstance(payload.payload, ReplayOracleAuthorizedPresentationFrameV1)
    assert payload.payload.match_summary is not None
    return payload.payload.match_summary


def test_team_controller_names_follow_each_member() -> None:
    alpha = _named("tdm-alpha", "Alpha Lead")
    beta = _named("tdm-beta", "tdm-beta")
    rand = _named("random", "researcher_policy")
    classes = [1, 2, 3, 4, 5]
    assert team_controller_names(beta, classes) == ("tdm-beta",) * 5
    mixed = marl_bgs.team(alpha, beta, slots=[[0, 1, 2], [3, 4]])
    assert team_controller_names(mixed, classes) == (
        *("Alpha Lead",) * 3,
        *("tdm-beta",) * 2,
    )
    by_class = marl_bgs.team(alpha, beta, slots=[["Mage", "Priest"], [1, 2, 3]])
    assert team_controller_names(by_class, [5, 2, 3, 4, 1]) == (
        "Alpha Lead",
        "tdm-beta",
        "tdm-beta",
        "tdm-beta",
        "Alpha Lead",
    )
    nested = marl_bgs.team(mixed, rand, slots=[[0, 1, 2, 3], [4]])
    assert team_controller_names(nested, classes) == (
        *("Alpha Lead",) * 3,
        "tdm-beta",
        "researcher_policy",
    )
    pooled = marl_bgs.pool({alpha: 1.0, beta: 1.0})
    assert team_controller_names(pooled, classes) == (
        ("Pool: Alpha Lead / tdm-beta",) * 5
    )
    independent = independent_policies([alpha, beta, rand])
    assert team_controller_names(independent, [1, 2, 3, 0, 0]) == (
        "Alpha Lead",
        "tdm-beta",
        "researcher_policy",
        "Alpha Lead / tdm-beta / researcher_policy",
        "Alpha Lead / tdm-beta / researcher_policy",
    )
    with pytest.raises(ValueError, match="five class IDs"):
        team_controller_names(beta, [1, 2])
    with pytest.raises(TypeError):
        team_controller_names(cast(System, "tdm-beta"), classes)


# Renamed Random policies compile fast on CPU; only their names matter here.
_SECOND = "Planner-v2_beta"


@pytest.fixture(scope="module")
def mixed_team() -> System:
    return marl_bgs.team(
        _named("random", _LONG),
        _named("random", _SECOND),
        slots=[["Mage", "Priest"], [1, 2, 3]],
    )


_EXPECTED_SLOTS = (
    _LONG,
    _SECOND,
    _SECOND,
    _SECOND,
    _LONG,
    *("random",) * 5,
)
_EVALUATE: dict[str, Any] = {
    "num_episodes": 2,
    "maps": [47],
    "num_envs": 2,
    "save_replays": 2,
    "max_steps": 2,
    "metrics": "none",
}


def test_evaluate_records_team_and_slot_names_in_memory_and_on_disk(
    mixed_team: System, tmp_path: Path
) -> None:
    in_memory = marl_bgs.evaluate(mixed_team, "random", **_EVALUATE)
    saved = marl_bgs.evaluate(mixed_team, "random", output_dir=tmp_path, **_EVALUATE)
    replays = [*in_memory.replays] + [
        load_replay(path).replay for path in saved.replay_paths
    ]
    assert len(replays) == 4
    for replay in replays:
        context = replay.header.context
        names = _names(context)
        assert names["marl_bgs.system_name.team_a"] == json.dumps(
            f"Team: {_LONG} / {_SECOND}"
        )
        assert names["marl_bgs.system_name.team_b"] == '"random"'
        # Non-ASCII names are escaped, so the value stays printable ASCII.
        assert names["marl_bgs.controller_name.slot_0"] == (
            '"Recurrent MAPPO \\u00b7 Pool Seed 7 Best"'
        )
        assert recorded_slot_controller_names(context) == _EXPECTED_SLOTS
        assert recorded_team_controller_name(context, "team_b") == "random"
        summary = build_match_summary_v1(context, replay.frames[0])
        assert summary.slot_controller_names == _EXPECTED_SLOTS
        assert summary.teams[0].display_name == f"{_LONG} / {_SECOND}"
        assert summary.teams[1].display_name == "random"
    viewer = _replay_summary(saved.replay_paths[0])
    assert viewer.slot_controller_names == _EXPECTED_SLOTS
    assert [team.display_name for team in viewer.teams] == [
        f"{_LONG} / {_SECOND}",
        "random",
    ]


def test_slot_names_change_only_the_replay_context(
    mixed_team: System, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    named = marl_bgs.evaluate(
        mixed_team, "random", output_dir=tmp_path / "named", **_EVALUATE
    )

    def skip(*_: object) -> None:
        return None

    with monkeypatch.context() as patch:
        # The same run without the live methods records team names only.
        patch.setattr(RunWriter, "_name_replay_controllers", skip)
        plain = marl_bgs.evaluate(
            mixed_team, "random", output_dir=tmp_path / "plain", **_EVALUATE
        )
    folders = [named.run_dir, plain.run_dir]
    assert folders[0] is not None and folders[1] is not None
    runs = cast(list[Path], folders)
    details = [
        cast(dict[str, Any], json.loads((run / "run_details.json").read_text()))
        for run in runs
    ]
    for key in ("systems", "configurations", "details"):
        assert details[0][key] == details[1][key]
    passes = [next(iter(value["passes"].values())) for value in details]
    for key in ("details", "policies", "system_ids", "episodes"):
        assert passes[0][key] == passes[1][key]
    for table in sorted(path.name for path in runs[0].glob("*.csv")):
        rows = [
            (run / table).read_text().replace(str(detail["run_id"]), "RUN")
            for run, detail in zip(runs, details, strict=True)
        ]
        assert rows[0] == rows[1], table
    for first, second in zip(
        sorted(named.replay_paths), sorted(plain.replay_paths), strict=True
    ):
        assert (
            Path(first).name.rsplit("__", 1)[0]
            == (Path(second).name.rsplit("__", 1)[0])
        )
        new = load_replay(first).replay
        old = load_replay(second).replay
        assert recorded_slot_controller_names(old.header.context) is None
        assert recorded_slot_controller_names(new.header.context) == _EXPECTED_SLOTS
        stripped = _without_slot_names(new.header.context).aggregation_keys
        assert [
            (row.name, row.value) for row in stripped if row.name != "marl_bgs.run_id"
        ] == [
            (row.name, row.value)
            for row in old.header.context.aggregation_keys
            if row.name != "marl_bgs.run_id"
        ]
        digest = "0" * 64
        assert generated_replay_filename(
            new.header.context, digest, episode_id=1
        ) == generated_replay_filename(
            _without_slot_names(new.header.context), digest, episode_id=1
        )
        assert len(new.transitions) == len(old.transitions)


def test_battle_client_saved_replay_keeps_live_slot_names(tmp_path: Path) -> None:
    long_method = _named("random", _LONG)
    owned_by_slot = marl_bgs.team(
        _named("random", "QMIX Mixer Ablation"),
        _named("random", _SECOND),
        slots=[[0, 1, 2], [3, 4]],
    )
    planner = _named("random", "My LLM Planner (Qwen3.5-4B)")
    setup = replace(
        prepare_battle_setup(
            map_id=0,
            max_steps=6,
            team_b_roster=("priest", "warrior", "hunter", "rogue", "mage"),
        ),
        team_a=("manual", long_method, long_method, owned_by_slot, "random"),
        team_b=marl_bgs.team(
            planner, "random", slots=[["Mage", "Warrior"], ["Priest", 2, 3]]
        ),
    )
    resolved = resolve_controllers(setup)
    # Team B's roster is Priest, Warrior, Hunter, Rogue, Mage: class selectors
    # follow the class, and the team(...) entry at slot 3 names its member.
    expected = (
        "Manual",
        _LONG,
        _LONG,
        _SECOND,
        "random",
        "random",
        "My LLM Planner (Qwen3.5-4B)",
        "random",
        "random",
        "My LLM Planner (Qwen3.5-4B)",
    )
    assert resolved.slot_labels == expected
    fixture = debugger_test_launch_specification(setup.seed)
    session = control.create_session(
        setup.scenario,
        seed=setup.seed,
        evaluation_launch_specification=build_debugger_evaluation_launch_specification_v1(
            root_seed=setup.seed,
            code_revision=fixture.code_revision,
            capture_profile="evaluation_metric_complete",
        ),
        controlled_global_slot=None,
        show_ranges=False,
        verbose_logging=False,
        team_a_controller=resolved.controllers[0],
        team_b_controller=resolved.controllers[1],
        execution_information_mode="shared_obs",
        systems=resolved.systems,
        borrow_systems=True,
        manual_slots=resolved.manual_slots,
        slot_labels=resolved.slot_labels,
    )
    destination = tmp_path / "battle.marlbg-replay.json"
    recorder = DebuggerReplayRecorder(
        specification=build_debugger_recording_specification_v1(
            action_source_kind=cast(
                DebuggerActionSourceKindV1,
                recording_action_source(session.evaluation_context),
            ),
            runtime_provenance=capture_runtime_provenance(
                "0.1.0",
                policy_execution_included=recording_policy_execution_included(
                    session.evaluation_context
                ),
            ),
        ),
        destination=preflight_replay_destination(destination),
        context=session.evaluation_context,
        initial_frame=session.current_evaluation_frame,
    )
    session = control.submit_interactive(session)
    assert session.incoming_evaluation_view is not None
    recorder.append(
        session.incoming_evaluation_view.transition, session.current_evaluation_frame
    )
    assert recorder.finalize_and_save("finish_and_review") == "saved"
    context = load_replay(destination).replay.header.context
    assert context == session.evaluation_context
    assert recorded_slot_controller_names(context) == expected
    assert recorded_team_controller_name(context, "team_a") == (
        f"Manual / {_LONG} / {_SECOND} / random"
    )
    assert recorded_team_controller_name(context, "team_b") == (
        "random / My LLM Planner (Qwen3.5-4B)"
    )
    viewer = _replay_summary(destination)
    assert viewer.slot_controller_names == expected
    assert [team.display_name for team in viewer.teams] == [
        f"Manual / {_LONG} / {_SECOND} / random",
        "random / My LLM Planner (Qwen3.5-4B)",
    ]


def test_derived_built_in_labels_record_public_names() -> None:
    session = control.create_session(
        prepare_battle_setup(map_id=0, max_steps=6).scenario,
        seed=0,
        evaluation_launch_specification=debugger_test_launch_specification(0),
        controlled_global_slot=None,
        show_ranges=False,
        verbose_logging=False,
        team_a_controller="manual",
        team_b_controller="scenario_5",
        execution_information_mode="shared_obs",
    )
    assert session.slot_labels == ("Manual",) * 5 + ("tdm-beta",) * 5
    assert recorded_slot_controller_names(session.evaluation_context) == (
        session.slot_labels
    )
    assert _names(session.evaluation_context)["marl_bgs.system_name.team_b"] == (
        '"tdm-beta"'
    )


def test_offered_system_alias_records_the_system_names() -> None:
    session = control.create_session(
        prepare_battle_setup(map_id=0, max_steps=6).scenario,
        seed=0,
        evaluation_launch_specification=debugger_test_launch_specification(0),
        controlled_global_slot=None,
        show_ranges=False,
        verbose_logging=False,
        team_a_controller="manual",
        team_b_controller="scenario_5",
        execution_information_mode="shared_obs",
    )
    # The review's repro: the same call the DevClient service makes for an
    # offered System, with no slot labels. It used to record "system:counter".
    shared = control.set_combat_configuration(
        session,
        team_a_controller="system:counter",
        team_b_controller="scenario_5",
        execution_information_mode="shared_obs",
        systems=(marl_bgs.shared_policy(_named("random", "My Counter System")), None),
    )
    alone = ("My Counter System",) * 5 + ("tdm-beta",) * 5
    assert shared.slot_labels == alone
    assert recorded_slot_controller_names(shared.evaluation_context) == alone
    assert recorded_team_controller_name(shared.evaluation_context, "team_a") == (
        "My Counter System"
    )
    counter = marl_bgs.team(
        _named("random", "My Counter System"),
        _named("random", _SECOND),
        slots=[[0, 1, 2], [3, 4]],
    )
    switched = control.set_combat_configuration(
        shared,
        team_a_controller="system:counter",
        team_b_controller="scenario_5",
        execution_information_mode="shared_obs",
        systems=(counter, None),
    )
    expected = (
        *("My Counter System",) * 3,
        *(_SECOND,) * 2,
        *("tdm-beta",) * 5,
    )
    assert switched.slot_labels == expected
    context = switched.evaluation_context
    assert recorded_slot_controller_names(context) == expected
    assert recorded_team_controller_name(context, "team_a") == (
        f"My Counter System / {_SECOND}"
    )
    restarted = control.reset_session(switched)
    assert restarted.slot_labels == expected
    assert recorded_slot_controller_names(restarted.evaluation_context) == expected


@pytest.mark.parametrize("name", ["tdm-beta", "researcher_policy", "Équipe One"])
def test_replay_viewer_shows_recorded_names_exactly(name: str) -> None:
    trajectory = captured_evaluation_trajectory(transition_count=0)
    base = trajectory.context.model_copy(
        update={
            "policy_assignments": tuple(
                row.model_copy(update={"policy_id": "a" * 64, "policy_kind": "system"})
                if isinstance(row, AssignedPolicySlotV1)
                else row
                for row in trajectory.context.policy_assignments
            )
        }
    )
    frame = trajectory.frames[0]
    # No recorded name: a bare hash keeps the old plain label.
    assert build_match_summary_v1(base, frame).teams[0].display_name == (
        "Recorded System"
    )
    team_only = base.model_copy(
        update={
            "aggregation_keys": tuple(
                sorted(
                    (
                        *base.aggregation_keys,
                        AggregationKeyV1(
                            name="marl_bgs.system_name.team_a",
                            value=json.dumps(name, ensure_ascii=True),
                        ),
                    ),
                    key=lambda row: row.name,
                )
            )
        }
    )
    summary = build_match_summary_v1(team_only, frame)
    assert summary.teams[0].display_name == name
    assert summary.slot_controller_names is None
    active = [row.configured_active for row in base.roster]
    slots = tuple(
        (name if slot % 2 == 0 else "Manual") if active[slot] else None
        for slot in range(10)
    )
    per_slot = team_only.model_copy(
        update={
            "aggregation_keys": tuple(
                sorted(
                    (*team_only.aggregation_keys, *slot_controller_name_keys(slots)),
                    key=lambda row: row.name,
                )
            )
        }
    )
    summary = build_match_summary_v1(per_slot, frame)
    assert summary.slot_controller_names == slots
    assert summary.teams[0].display_name == " / ".join(
        dict.fromkeys(value for value in slots[:5] if value is not None)
    )


@pytest.mark.parametrize(
    ("fixture", "labels"),
    [
        ("scenario_1", ["Tdm alpha", "TDM Scenario 1 ALPHA Controller"]),
        ("no_shared_obs", ["Policy 0 / Policy 1 / Policy 2", "Policy 5 / Policy 6"]),
    ],
)
def test_old_replays_open_with_their_older_labels(
    fixture: str, labels: list[str]
) -> None:
    path = _FIXTURES / f"{fixture}.marlbg-replay.json"
    loaded = load_replay(path)
    context = loaded.replay.header.context
    assert recorded_slot_controller_names(context) is None
    assert recorded_team_controller_name(context, "team_a") is None
    summary = _replay_summary(path)
    assert summary.slot_controller_names is None
    assert [team.display_name for team in summary.teams] == labels
    assert generated_replay_filename(
        context, loaded.replay.canonical_digest_sha256, episode_id=1
    ).endswith(f"__{loaded.replay.canonical_digest_sha256}.marlbg-replay.json")


def test_fixed_scenario_title_wins_over_a_recorded_team_name() -> None:
    loaded = load_replay(_FIXTURES / "scenario_1.marlbg-replay.json").replay
    context = loaded.header.context
    named = context.model_copy(
        update={
            "aggregation_keys": tuple(
                sorted(
                    (
                        *context.aggregation_keys,
                        AggregationKeyV1(
                            name="marl_bgs.system_name.team_b",
                            value=json.dumps("my-opponent_v2"),
                        ),
                    ),
                    key=lambda row: row.name,
                )
            )
        }
    )
    summary = build_match_summary_v1(named, loaded.frames[0])
    assert [team.display_name for team in summary.teams] == [
        "Tdm alpha",
        "TDM Scenario 1 ALPHA Controller",
    ]


def test_slot_name_keys_reject_bad_rows() -> None:
    with pytest.raises(ValueError, match="ten entries"):
        slot_controller_name_keys(["A"] * 9)
    with pytest.raises(ValueError, match="nonempty"):
        slot_controller_name_keys([" ", *[None] * 9])
    keys = slot_controller_name_keys(["tdm-beta", *[None] * 8, "Manual"])
    assert [(row.name, row.value) for row in keys] == [
        ("marl_bgs.controller_name.slot_0", '"tdm-beta"'),
        ("marl_bgs.controller_name.slot_9", '"Manual"'),
    ]
    assert np.all([row.value.isascii() for row in keys])
