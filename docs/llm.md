# LLM Agents

Turn a served language model into an ordinary team System, play it against anything and read what every model call cost.

Website: https://marl-battlegrounds.com/#llm

## Quick Start

```python
import marl_battlegrounds as marl_bgs
from marl_battlegrounds import llm

system = llm.make_system("my-served-model", "http://127.0.0.1:8000/v1")
result = marl_bgs.evaluate(system, "random", num_episodes=2, maps=[0], num_envs=2,
                           output_dir="runs/llm-example")
print(llm.call_summary(result))
```

Start your model server first; MARL-BGs never starts one or downloads weights. `make_system`
makes no request: it returns a host `System` that asks the model once per living actor per
tick, with only that actor's permitted view as text, and checks each reply against the legal
moves. Use it wherever a System goes: `evaluate`, `run_tournament`, `team`, `battle_client`.

**Watch Out:** each actor gets only its own view. Returning a whole team's actions does not
let actors share private inputs or memory. The model, not the simulator, sets the pace.

## Settings And Endpoint Rules

```python
with llm.Client("http://127.0.0.1:8000/v1", concurrency=16, timeout=60.0) as client:
    system = llm.make_system("my-served-model", client=client, history_turns=2)
```

The default `server_type="vllm"` uses vLLM's Chat Completions and `/tokenize` routes,
with thinking off and temperature 0. `server_type="chat"` sends standard Chat Completions
fields. The endpoint must return plain text in `choices[0].message.content` with
`finish_reason` `"stop"`, take an optional key as a Bearer token and need no query string.

**Watch Out:** `server_type="chat"` needs a trusted `token_counter` for history, or else
`history_turns=0`. Runners close only clients they made; nothing ever stops your server.

| Setting | Default | What It Does |
| --- | --- | --- |
| `history_turns` | `0` | Earlier living turns kept per actor; the oldest whole turns are dropped to fit the context |
| `max_tokens` | `64` | The reply allowance, also reserved when fitting history |
| `failure_policy` | `"stop"` | `"fallback"` plays a checked stay and no combat after a bad reply or failed request, and counts it |
| `records` | `"light"` | `"full"` also keeps each exact request; `"none"` keeps only counters |
| `frame` | `"left"` | Each actor sees its own spawn end on the left; `"world"` keeps the map as it is |
| `Client(concurrency=, timeout=)` | `16`, `60.0` | At most 16 requests at once; each HTTP call gets 60 seconds; no retries |

## The Tested Qwen Recipe

```bash
export QWEN_MODEL_DIR=/path/to/pinned/Qwen3.5-4B
export CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
vllm serve "$QWEN_MODEL_DIR" \
  --host 127.0.0.1 --port 8000 --served-model-name qwen3.5-4b-fp8 \
  --language-model-only --quantization fp8 --dtype bfloat16 \
  --max-model-len 16384 --max-num-seqs 16 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.50 --no-enable-prefix-caching \
  --generation-config vllm --seed 0
```

Use a separate serving environment; the tested one had Python 3.12.13, vLLM 0.30.0, PyTorch
2.13 with CUDA 13.0 and Transformers 5.17.0. vLLM turns the original weights into FP8 as it
loads them. The server stays in the foreground until you stop it.

**Watch Out:** a different vLLM version, model content, quantization or memory setting is a
new recipe that needs its own checks. Pass the pinned `model_revision` in every experiment.

| Model | Pinned Revision |
| --- | --- |
| `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |

To share one GPU, set `JAX_PLATFORMS=cuda,cpu`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`,
`XLA_CLIENT_MEM_FRACTION=0.10` and `XLA_PYTHON_CLIENT_ALLOCATOR=bfc` before Python starts.

## Your Own Prompt And Reply

```python
import json

def my_prompt(actor, masks, history):
    return llm.format_actor_view(actor, masks, reply_instruction="Reply: <move> <combat>")

def my_parser(reply, actor, masks, history):
    words = reply.split()
    if len(words) != 2:
        raise llm.ReplyFormatError("Reply must contain one move and one combat name")
    return llm.parse_action_reply(json.dumps({"move": words[0], "combat": words[1]}), masks)

system = llm.make_system("my-served-model", prompt_builder=my_prompt, reply_parser=my_parser,
                         custom_version="words-v1", custom_settings={"format": "words"})
```

A prompt builder gets one actor's view, its masks and its own history, and returns text. A
parser returns an `ActorAction`. The System still owns routing, history, request limits,
the frame and the legal check. The default reply is JSON such as `{"move": "north",
"combat": "enemy_2_ultimate"}`: `enemy_2` is enemy roster row 2, and `ally_k` counts the
actor's own team, itself included. `llm.legal_action_names(masks)` lists this tick's names.

**Watch Out:** keep both functions pure; fitting history may call them again. Declare a
`custom_version` and `custom_settings`: a resume with changed code or settings is rejected.

From A Checkout, [llm.py](../examples/llm.py) runs the default, a note or this two-word format:

```bash
python examples/llm.py model=my-served-model format=words history_turns=2 output_dir=runs/llm-words
```

## Reasoning Settings

```python
def qwen_thinking(request):
    return request | {"chat_template_kwargs": {"enable_thinking": True}}

with llm.Client("http://127.0.0.1:8000/v1", timeout=600) as client:
    system = llm.make_system(
        "qwen3.5-4b-fp8", client=client, max_tokens=8192, request_editor=qwen_thinking,
        custom_version="qwen-thinking-v1", failure_policy="fallback",
    )
    result = marl_bgs.evaluate(system, "tdm-alpha", num_episodes=2, maps=[47], num_envs=2)
```

A `request_editor` gets the finished request and returns the one to send; the System
counts, saves and sends exactly that. Start vLLM with `--reasoning-parser qwen3` too, and
pass the sampling values Qwen suggests for thinking with `sampling=`. Hosted reasoning models
may refuse fields: an editor can drop them or rename `max_tokens` to `max_completion_tokens`.

**Watch Out:** `max_tokens` covers thinking and answer; a cut-off reply is a failure, so
raise the server's `--max-model-len` too. The editor must keep the model, one choice and no streaming.

## Calls, Failures And Costs

```python
summary = llm.call_summary(result)  # all_attempts, completed_games and by_system
for call in llm.read_calls(result.run_dir):
    print(call["episode_id"], call["team"], call["actor"], call["outcome"], call.get("world_action"))
```

`all_attempts` counts every call, token and failure; `completed_games` counts only the
attempt that saved each game. A call `outcome` is `played`, `abandoned` or `unknown`. To
resume, repeat the call with the same System and settings and `resume_from`: saved games
make no new requests, and unfinished games restart with empty history.

**Watch Out:** report both teams' fallback counts beside win rates. Temperature 0 does not
promise the same fresh answers; saved replays replay the saved choices without the model.

| Records | What Is Kept | Call Files For 32 Games Of 8 Ticks (Fake Model) |
| --- | --- | --- |
| `"none"` | Failure and cost counters only | 0 MB |
| `"light"` | Request hash, reply and checked action | 0.813 MB |
| `"full"` | Also the exact request text | 16.19 MB |

## Play A Field And The Scenarios

```bash
python -m marl_battlegrounds.experiments.llm_prompt output_dir=runs/llm-prompt
JAX_PLATFORMS=cpu python -m marl_battlegrounds.experiments.llm_challenge model=qwen3.5-4b-fp8 \
  model_revision=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a output_dir=runs/llm-field
python -m marl_battlegrounds.experiments.llm_analysis field_dir=runs/llm-field output_dir=runs/llm-analysis
```

`llm_prompt` writes the exact text a model reads on one tick, with no model needed.
`llm_challenge` plays your model as Team A against TDM-Beta, TDM-Gamma and TDM-Alpha on
maps 47-51; `'maps=[]' 'scenario_ids=[1,2,3,4,5,6,7,8]'` plays the scenarios instead.
`llm_analysis` explains how it played, from the saved games and calls.

**Watch Out:** `llm_challenge` saves after every pair of games, and the same command resumes;
it refuses saved games made by a different System. Its default model is the 9B one.

## Next

- [battle_client.md](battle_client.md): play beside or against your LLM.
- [evaluation.md](evaluation.md): evaluation settings, saved results and resume.
- [your_method.md](your_method.md): write a host System that calls a model your own way.
