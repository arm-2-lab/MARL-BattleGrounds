"""Evaluate a served LLM with standard, note or word-based prompts and replies.

Install the hydra extra and start a compatible provider separately::

    python examples/llm.py model=MODEL server_url=URL output_dir=runs/llm
    python examples/llm.py model=MODEL format=words 'llm_slots=[4]'

The lesson never downloads weights or starts a server. Every model call uses
only its actor's permitted input and history. Unowned slots use Random when a
slot list is given. The client closes even if evaluation fails. Model-call
records are distinct from game metrics. Use --help or --cfg job for settings;
import run and the prompt/parser functions for ordinary Python use.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import marl_battlegrounds as marl_bgs

if TYPE_CHECKING:
    from marl_battlegrounds.core.types import ActionMask
    from marl_battlegrounds.llm.system import History
    from marl_battlegrounds.policies.actor import ActorAction
    from marl_battlegrounds.policies.input import ActorInput


def _view_and_history(
    actor: ActorInput,
    masks: ActionMask,
    history: History,
    *,
    reply_instruction: str,
) -> str:
    """Compose permitted views with clear dates and one current reply instruction.

    actor/masks describe the current left-frame decision. history contains only
    this actor's earlier living turns. Historical menus never govern the current
    reply. No own scheduler, memory store or native action rules are needed.
    """

    from marl_battlegrounds import llm
    from marl_battlegrounds.llm.actions import COMBAT_NAMES, MOVE_NAMES

    parts = ["You control the actor named on the 'You are' line."]
    for entry in history:
        parts.extend(
            [
                f"Earlier turn {entry.timestep}; its menu is historical:",
                llm.format_actor_view(
                    entry.actor,
                    entry.masks,
                    include_static=False,
                    reply_instruction=None,
                ),
                f"Submitted: move={MOVE_NAMES[int(entry.action.move)]} "
                f"combat={COMBAT_NAMES[int(entry.action.select_target)][int(entry.action.use_ultimate)]}",
            ]
        )
    parts.extend(
        [
            "Current decision:",
            llm.format_actor_view(actor, masks, reply_instruction=reply_instruction),
        ]
    )
    return "\n".join(parts)


def prompt_with_note(actor: ActorInput, masks: ActionMask, history: History) -> str:
    """Change the presentation while retaining the default named JSON parser.

    Use this example with make_system's default left frame. This pure
    function may run again if context fitting removes an older history entry.
    """
    return _view_and_history(
        actor,
        masks,
        history,
        reply_instruction=(
            'Reply with exactly {"move":"<legal move>","combat":"<legal combat>"}.'
        ),
    )


def prompt_for_words(actor: ActorInput, masks: ActionMask, history: History) -> str:
    """Ask for two stable names using make_system's default left-frame inputs."""
    return _view_and_history(
        actor,
        masks,
        history,
        reply_instruction=(
            "Reply with two names on one line: <move> <combat>. "
            "Copy each name from the current legal lists. Use one space, with no "
            "move= or combat= labels, JSON, extra punctuation or explanation."
        ),
    )


def parse_words(
    reply: str, actor: ActorInput, masks: ActionMask, history: History
) -> ActorAction:
    """Read two names and reuse the shared name mapping and legal-pair check.

    actor, masks and history describe this exact fitted request. This parser
    needs only masks. Bad text raises ReplyFormatError. make_system checks the
    returned native ActorAction again and owns world-frame conversion.
    """

    from marl_battlegrounds import llm

    del actor, history
    words = reply.split()
    if len(words) != 2:
        raise llm.ReplyFormatError("Reply must contain one move and one combat name")
    return llm.parse_action_reply(
        json.dumps({"move": words[0], "combat": words[1]}), masks
    )


def _factory(*, custom: bool) -> marl_bgs.System:
    """Build a lazy CLI/BattleClient System from the example's environment settings.

    MARL_LLM_MODEL is the required served model name. MARL_LLM_URL defaults to
    http://127.0.0.1:8000/v1. MARL_LLM_REVISION optionally records pinned model
    content; omission means unknown. MARL_LLM_HISTORY defaults to 0 earlier turns.
    custom selects the two-word tutorial functions. No request or server starts.
    """

    from marl_battlegrounds import llm

    model = os.environ.get("MARL_LLM_MODEL")
    if not model:
        raise ValueError("Set MARL_LLM_MODEL to the model name served by your server")
    return llm.make_system(
        model,
        os.environ.get("MARL_LLM_URL", "http://127.0.0.1:8000/v1"),
        model_revision=os.environ.get("MARL_LLM_REVISION"),
        history_turns=int(os.environ.get("MARL_LLM_HISTORY", "0")),
        prompt_builder=prompt_for_words if custom else None,
        reply_parser=parse_words if custom else None,
        custom_version="tutorial-v1" if custom else None,
        custom_settings={"format": "words" if custom else "default"},
    )


def make_default_system() -> marl_bgs.System:
    """Return the default JSON System for a trusted module:function declaration.

    Set MARL_LLM_MODEL first. Optional MARL_LLM_URL, MARL_LLM_REVISION and
    MARL_LLM_HISTORY select the endpoint, declared content and earlier turns.
    This zero-argument factory starts no request. The runner owns its client.
    """
    return _factory(custom=False)


def make_custom_system() -> marl_bgs.System:
    """Return the two-word tutorial System with the same managed runner lifetime.

    Use the same environment settings as make_default_system. Only prompt and
    reply format change; history, action checks, recording and cleanup are shared.
    """
    return _factory(custom=True)


def run(
    *,
    model: str,
    server_url: str = "http://127.0.0.1:8000/v1",
    format: str = "default",
    history_turns: int = 0,
    records: Literal["none", "light", "full"] = "light",
    games: int = 2,
    num_envs: int = 2,
    map_id: int = 0,
    max_steps: int = 300,
    output_dir: str | Path | None = None,
    llm_slots: list[int] | None = None,
) -> None:
    """Evaluate a served language model using permitted actor inputs.

    Parameters
    ----------
    model : str
        Served model name sent to the existing compatible provider. This lesson does
        not download or launch a model.
    server_url : str, default='http://127.0.0.1:8000/v1'
        Base URL of the running compatible HTTP provider, including its /v1 prefix.
    format : str, default='default'
        default uses the standard structured reply; note adds a prompt note; words
        uses this module's word parser.
    history_turns : int, default=0
        Number of earlier per-actor turns included in each prompt. Zero uses the
        current permitted input only.
    records : Literal['none', 'light', 'full'], default='light'
        none, light or full model-call records. Full can include prompts and
        replies. These are separate from game metrics.
    games : int, default=2
        Total evaluation games against Random, including both spawn ends in the
        paired schedule.
    num_envs : int, default=2
        Number of parallel games. This changes batching, not the declared game or
        transition budget.
    map_id : int, default=0
        Installed development map ID for these example games.
    max_steps : int, default=300
        Game horizon in transitions. A short horizon is a software check, not
        evidence of learned skill.
    output_dir : str | Path | None, default=None
        Optional parent for a new saved evaluation run, including selected model-
        call records. None returns results in memory. The client closes on success
        and failure.
    llm_slots : list[int] | None, default=None
        Physical Team A slots from 0 through 4 assigned to the LLM. None assigns the
        whole team. Other slots use Random; ownership grants no extra information.

    Returns
    -------
    None
        Print results and saved locations. No scientific learning claim is made.

    Notes
    -----
    API errors propagate without an automatic retry. Python callers use current
    working-directory paths; Hydra preserves its original launch directory.
    """

    from marl_battlegrounds import llm

    output_dir = None if output_dir is None else Path(output_dir)
    if format not in ("default", "note", "words"):
        raise ValueError("format must be default, note or words")
    if llm_slots is not None and any(
        type(slot) is not int or not 0 <= slot < 5 for slot in llm_slots
    ):
        raise ValueError("llm_slots must contain physical slots 0 through 4")
    builder = (
        prompt_with_note
        if format == "note"
        else prompt_for_words
        if format == "words"
        else None
    )
    decoder = parse_words if format == "words" else None
    with llm.Client(server_url) as client:
        system = llm.make_system(
            model,
            client=client,
            history_turns=history_turns,
            records=records,
            prompt_builder=builder,
            reply_parser=decoder,
            custom_version="tutorial-v1" if builder else None,
            custom_settings={"format": format},
        )
        if llm_slots is not None:
            selected = sorted(set(llm_slots))
            remaining = [slot for slot in range(5) if slot not in selected]
            system = marl_bgs.team(system, "random", slots=[selected, remaining])
        result = marl_bgs.evaluate(
            system,
            "random",
            num_episodes=games,
            maps=[map_id],
            num_envs=num_envs,
            max_steps=max_steps,
            output_dir=output_dir,
        )
    episodes = result.table("episodes")
    print(
        {
            column: episodes[column]
            for column in (
                "episode_id",
                "outcome",
                "episode_length",
                "team_a_score",
                "team_b_score",
            )
        }
    )
    print("Model calls:", llm.call_summary(result))
    if result.paths:
        print("Game files:", result.paths)


def main() -> int:
    """Compose packaged YAML and call run; use --help or --cfg job to inspect."""
    from marl_battlegrounds.experiments.hydra import example_main

    return example_main("llm", run)


if __name__ == "__main__":
    raise SystemExit(main())
