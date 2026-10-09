"""Check that a custom request editor's request is what is counted, saved and sent.

Contracts: make_system's request_editor output, any mapping, is counted, saved
byte for byte in full call records and sent unchanged; vLLM counting keeps
chat_template_kwargs, passes reasoning_effort, its derived thinking switch and
documents to the template as vLLM's chat route does, and forwards an edited
request's tools;
fitting reserves the declared max_tokens after an editor renames or lowers it,
or a larger limit the editor sets; replies with visible reasoning text add to
reasoning_replies; editor exceptions, non-JSON results and edits that change the
model, stream or ask for other than one choice stop execution instead of becoming
fallback actions; and only Systems with an editor add it to their saved method
identity. Loopback fake models stand in for vLLM, so no weights or GPU are
needed.
"""

import json
import threading
from collections.abc import Callable, Mapping
from hashlib import sha256
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import jax
import pytest
from tests.llm_fixtures import FakeModel, send, server

import marl_battlegrounds as marl_bgs
from marl_battlegrounds import llm
from marl_battlegrounds.environment import Environment, EnvironmentState
from marl_battlegrounds.evaluation.policy_execution import SystemInput
from marl_battlegrounds.llm.actions import ReplyFormatError
from marl_battlegrounds.llm.client import Client, ContextLimitError
from marl_battlegrounds.llm.system import make_system
from marl_battlegrounds.policies.input import Observations
from marl_battlegrounds.tasks import make_standard_team_deathmatch_config

STAY = '{"move":"stay","combat":"no_combat"}'
TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": {"name": "noop", "parameters": {}}}
]
DOCUMENTS: list[dict[str, str]] = [{"title": "Rules", "text": "Win the fight."}]


@pytest.fixture(scope="module")
def game() -> tuple[Environment, Observations, EnvironmentState]:
    roster = ("warrior", "mage", "hunter", "rogue", "priest")
    config = make_standard_team_deathmatch_config(
        map_id=0, max_steps=30, team_a_roster=roster, team_b_roster=roster
    )
    env = marl_bgs.make("tdm", env_config=config, num_envs=2, metrics="none")
    observations, state = env.reset(jax.random.key(2028))
    return env, observations, state


def inputs(game: tuple[Environment, Observations, EnvironmentState]) -> SystemInput:
    env, observations, state = game
    return cast(SystemInput, jax.device_get(env.policy_inputs(observations, state)))


def prompt_tokens(payload: dict[str, Any], *, generation: bool) -> int:
    # Like vLLM: only generation reads reasoning_effort and documents, and
    # passes them to the template. This template prints the effort, documents
    # and tools, and a closed thinking block adds tokens to the prompt.
    template = dict(payload.get("chat_template_kwargs") or {})
    effort = payload.get("reasoning_effort") if generation else None
    if effort is not None:
        if "enable_thinking" not in template:
            template["enable_thinking"] = effort != "none"
        template["reasoning_effort"] = effort
    if generation and "documents" in payload:
        template["documents"] = payload["documents"]
    closed = template.get("enable_thinking", True) is False
    extra = len(str(template.get("reasoning_effort", "")))
    extra += len(json.dumps(template.get("documents", [])))
    extra += len(json.dumps(payload.get("tools", [])))
    return len(payload["messages"][0]["content"]) + (4 if closed else 0) + extra


class VllmModel:
    def __init__(self, room: int = 100_000) -> None:
        # The context holds each counted prompt plus `room` reply tokens.
        self.room = room
        self.calls: list[tuple[str, dict[str, Any], bytes]] = []
        self.lock = threading.Lock()

    def serve(self, handler: BaseHTTPRequestHandler, payload: dict[str, Any]) -> None:
        with self.lock:
            self.calls.append((handler.path, payload, vars(handler)["raw_body"]))
        if handler.path == "/tokenize":
            count = prompt_tokens(payload, generation=False)
            send(handler, {"count": count, "max_model_len": count + self.room})
            return
        message = {"content": STAY, "reasoning": "Hold the line for now."}
        usage = {
            "prompt_tokens": prompt_tokens(payload, generation=True),
            "completion_tokens": 30,
        }
        send(
            handler,
            {
                "choices": [{"finish_reason": "stop", "message": message}],
                "usage": usage,
            },
        )

    def sent(self, route: str) -> list[tuple[dict[str, Any], bytes]]:
        return [(payload, raw) for path, payload, raw in self.calls if path == route]


def thinking(request: dict[str, Any]) -> dict[str, Any]:
    return request | {"chat_template_kwargs": {"enable_thinking": True}}


def provider_fields(request: dict[str, Any]) -> dict[str, Any]:
    request = {k: v for k, v in request.items() if k not in ("temperature", "top_p")}
    request["max_completion_tokens"] = request.pop("max_tokens")
    return request | {
        "reasoning_effort": "none",
        "tools": TOOLS,
        "documents": DOCUMENTS,
    }


def lowered_limit(request: dict[str, Any]) -> dict[str, Any]:
    return provider_fields(request) | {"max_completion_tokens": 10}


def raised_limit(request: dict[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(provider_fields(request) | {"max_completion_tokens": 200})


def test_edited_request_is_counted_saved_and_sent_byte_for_byte(
    tmp_path: Path,
) -> None:
    model = VllmModel()
    with server(model.serve) as url:
        method = llm.make_system(
            "fake",
            url,
            records="full",
            request_editor=thinking,
            custom_version="thinking-v1",
        )
        result = marl_bgs.evaluate(
            method,
            "random",
            num_episodes=2,
            maps=[0],
            num_envs=2,
            max_steps=2,
            output_dir=tmp_path,
        )
    assert result.run_dir is not None
    generations = model.sent("/v1/chat/completions")
    assert len(generations) == 20
    assert all(
        payload["chat_template_kwargs"] == {"enable_thinking": True}
        for payload, _ in generations + model.sent("/tokenize")
    )
    rows = list(llm.read_calls(result.run_dir / "model_calls"))
    assert sorted(row["request_json"].encode() for row in rows) == sorted(
        raw for _, raw in generations
    )
    assert all(
        row["request_sha256"] == sha256(row["request_json"].encode()).hexdigest()
        for row in rows
    )
    counts = llm.call_summary(result)["completed_games"]["team_a"]
    assert counts["model_calls"] == counts["reasoning_replies"] == 20
    assert counts["output_tokens"] == 600


def test_renamed_fields_and_reasoning_effort_keep_exact_counts(
    game: tuple[Environment, Observations, EnvironmentState],
) -> None:
    model = VllmModel()
    supplied = inputs(game)
    with server(model.serve) as url, Client(url) as client:
        method = make_system(
            "fake",
            client=client,
            server_type="chat",
            token_counter="vllm",
            history_turns=1,
            request_editor=provider_fields,
        )
        assert method.init is not None
        memory = method.init(method.variables, supplied, jax.random.key(1))
        method.apply(method.variables, memory, supplied, jax.random.key(2))
    generations = model.sent("/v1/chat/completions")
    assert len(generations) == 10
    for payload, _ in generations:
        assert not {"max_tokens", "temperature", "top_p"}.intersection(payload)
        assert payload["max_completion_tokens"] == 64
        assert payload["reasoning_effort"] == "none"
    for payload, _ in model.sent("/tokenize"):
        assert payload["chat_template_kwargs"] == {
            "enable_thinking": False,
            "reasoning_effort": "none",
            "documents": DOCUMENTS,
        }
        assert payload["tools"] == TOOLS


@pytest.mark.parametrize(
    "room,editor,fits",
    [
        (63, provider_fields, False),
        (64, provider_fields, True),
        (63, lowered_limit, False),
        (199, raised_limit, False),
        (200, raised_limit, True),
    ],
)
def test_fitting_reserves_the_declared_or_larger_edited_reply_limit(
    game: tuple[Environment, Observations, EnvironmentState],
    room: int,
    editor: Callable[[dict[str, Any]], Mapping[str, Any]],
    fits: bool,
) -> None:
    model = VllmModel(room)
    supplied = inputs(game)
    with server(model.serve) as url, Client(url) as client:
        method = make_system(
            "fake",
            client=client,
            server_type="chat",
            token_counter="vllm",
            request_editor=editor,
        )
        assert method.init is not None
        memory = method.init(method.variables, supplied, jax.random.key(1))
        if fits:
            method.apply(method.variables, memory, supplied, jax.random.key(2))
        else:
            with pytest.raises(ContextLimitError):
                method.apply(method.variables, memory, supplied, jax.random.key(2))
    assert len(model.sent("/v1/chat/completions")) == (10 if fits else 0)


def test_editor_failures_stop_and_only_editors_enter_identity(
    game: tuple[Environment, Observations, EnvironmentState],
) -> None:
    def broken(request: dict[str, Any]) -> dict[str, Any]:
        raise ReplyFormatError("editor bug")

    def not_object(request: dict[str, Any]) -> list[dict[str, Any]]:
        return [request]

    def not_finite(request: dict[str, Any]) -> dict[str, Any]:
        return request | {"temperature": float("nan")}

    def streaming(request: dict[str, Any]) -> dict[str, Any]:
        return request | {"stream": True}

    def other_model(request: dict[str, Any]) -> dict[str, Any]:
        return request | {"model": "other"}

    def two_choices(request: dict[str, Any]) -> dict[str, Any]:
        return request | {"n": 2}

    def true_choices(request: dict[str, Any]) -> dict[str, Any]:
        return request | {"n": True}

    model = FakeModel()
    supplied = inputs(game)
    with server(model.serve) as url, Client(url) as client:
        for editor, error in (
            (broken, RuntimeError),
            (cast(Any, not_object), TypeError),
            (not_finite, TypeError),
            (streaming, ValueError),
            (other_model, ValueError),
            (two_choices, ValueError),
            (true_choices, ValueError),
        ):
            method = make_system(
                "fake", client=client, failure_policy="fallback", request_editor=editor
            )
            assert method.init is not None
            memory = method.init(method.variables, supplied, jax.random.key(1))
            with pytest.raises(error):
                method.apply(method.variables, memory, supplied, jax.random.key(2))
    assert not model.generations()
    with pytest.raises(TypeError, match="callable"):
        make_system("fake", request_editor=cast(Any, "thinking"))

    with Client("http://127.0.0.1:1") as client:
        plain = make_system("fake", client=client)
        edited = make_system("fake", client=client, request_editor=thinking)
        declared = make_system(
            "fake", client=client, request_editor=thinking, custom_version="v1"
        )
    hooks = json.loads(plain.variables.identity_json)["custom_hooks"]
    assert set(hooks) == {"prompt_builder", "reply_parser", "token_counter"}
    hooks = json.loads(edited.variables.identity_json)["custom_hooks"]
    assert hooks["request_editor"]["code_digest"] is not None
    assert plain.variables.method_id != edited.variables.method_id
    assert not edited.variables.custom_identity_declared
    assert declared.variables.custom_identity_declared
