"""Show exactly what a language-model team reads on one tick of a real game.

Use python -m marl_battlegrounds.experiments.llm_prompt output_dir=/path/prompt.

The stage makes the canonical 5v5 game on one test map with marl_bgs.make and
lets two scripted Systems play it, TDM-Alpha as Team A and TDM-Beta as Team B by
default. It stops at the first tick, from min_tick on, at which a Team A agent
may hit an enemy with its Basic ability. It then builds the text that
llm.make_system would send for that agent:

1. env.policy_inputs gives the agent's permitted view and legal choices.
2. team_on_right and mirror_team_view put the agent's own team on the left,
   the frame every language-model System reads.
3. llm.format_actor_view renders the view and its legal menu.

It writes prompt.txt (that text) and menu.json (the tick, the agent's slot,
whether the view was reflected, the text format version, the text's size and
the legal move and combat names from llm.legal_action_names). No model is
called and no file outside output_dir changes. results/llm_prompt.txt and
results/llm_prompt.json are this stage's output with the default settings on
one RTX 5090; they are the example printed in the paper's supplement.
"""

from __future__ import annotations

import json
from operator import itemgetter
from pathlib import Path
from typing import Any, cast

# Import the optional adapter first for its installation error and parser fix.
# isort: off
from marl_battlegrounds.experiments.hydra import prepare_hydra

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
# isort: on

# The Basic column of a slot's joint target mask; enemies are targets 6 to 10.
_BASIC = 0
_FIRST_ENEMY_TARGET = 6


def first_attack_view(
    map_id: int,
    team_a: str,
    team_b: str,
    seed: int,
    min_tick: int,
    max_ticks: int,
) -> dict[str, Any]:
    """Play until a Team A agent may hit an enemy, and render what it reads.

    Parameters
    ----------
    map_id : int
        Map to play, such as test map 47.
    team_a, team_b : str
        Built-in scripted methods for Team A and Team B, such as "tdm-alpha".
        Each plays as one shared policy for its whole team.
    seed : int
        Seed of jax.random.key; the reset, the Systems' first memory and each
        tick's actions and step use keys split from it, in that order.
    min_tick : int
        First tick that may be shown. Earlier ticks are played, not shown.
    max_ticks : int
        Last tick to search, inclusive; must be at least min_tick.

    Returns
    -------
    dict
        ``prompt`` (the rendered text), ``tick`` (ticks played before it),
        ``slot`` (the Team A agent, 0 to 4, the lowest that qualifies),
        ``reflected`` (True when Team A started on the right, so the view was
        mirrored), ``text_version`` (llm.TEXT_VERSION), ``chars`` and ``lines``
        (the text's size) and ``menu`` (the legal move and combat names).

    Raises
    ------
    ValueError
        min_tick is negative or above max_ticks, or no Team A agent may hit an
        enemy by max_ticks.

    Notes
    -----
    Plays two games side by side on the default JAX device and shows game 0;
    the second game only fills the smallest supported batch. Each tick is one
    call of the compiled step and Systems. With the default settings, most of
    the time is compiling them: about 110 seconds and 3.5 GB of RAM on an
    RTX 5090, or 60 seconds and 3 GB on the CPU. Unit positions can differ
    between devices in their last digits, because floating-point arithmetic
    rounds differently on each.
    """
    import jax
    import numpy as np

    import marl_battlegrounds as marl_bgs
    from marl_battlegrounds import llm
    from marl_battlegrounds.policies.input import mirror_team_view, team_on_right

    if not 0 <= min_tick <= max_ticks:
        raise ValueError("Use 0 <= min_tick <= max_ticks")
    env = marl_bgs.make("tdm", map_id=map_id, num_envs=2, metrics="none")
    system_a = marl_bgs.shared_policy(marl_bgs.policy(team_a))
    system_b = marl_bgs.shared_policy(marl_bgs.policy(team_b))
    key, reset_key, memory_key = jax.random.split(jax.random.key(seed), 3)
    observations, state = env.reset(reset_key)
    memory = marl_bgs.init_systems(system_a, system_b, observations, state, memory_key)
    for tick in range(max_ticks + 1):
        inputs = env.policy_inputs(observations, state, team=0)
        joint = np.asarray(inputs.action_mask.select_target_use_ultimate_joint_mask)
        if tick >= min_tick:
            ready = joint[0, :, _FIRST_ENEMY_TARGET:, _BASIC].any(axis=1)
            if ready.any():
                slot = int(np.flatnonzero(ready)[0])
                # Game 0's row for this agent, from every array of the inputs.
                pick = itemgetter((0, slot))
                actor = jax.tree.map(pick, inputs.actors)
                masks = jax.tree.map(pick, inputs.action_mask)
                reflected = team_on_right(actor)
                view, view_masks = jax.device_get(
                    mirror_team_view(actor, masks, reflected)
                )
                prompt = llm.format_actor_view(view, view_masks)
                return {
                    "prompt": prompt,
                    "tick": tick,
                    "slot": slot,
                    "reflected": bool(reflected),
                    "text_version": llm.TEXT_VERSION,
                    "chars": len(prompt),
                    "lines": prompt.count("\n") + 1,
                    "menu": llm.legal_action_names(view_masks),
                }
        key, action_key, step_key = jax.random.split(key, 3)
        actions, memory, _ = marl_bgs.apply_systems(
            system_a, system_b, memory, observations, state, action_key
        )
        observations, state, _, _, _ = env.step(step_key, state, actions)
    raise ValueError(f"No Team A agent could hit an enemy by tick {max_ticks}")


@hydra.main(version_base="1.3", config_path="configs", config_name="llm_prompt")
def main(config: DictConfig) -> dict[str, Any]:
    """Render one real prompt and write prompt.txt and menu.json.

    Parameters
    ----------
    config : omegaconf.DictConfig
        ``map_id``, ``team_a``, ``team_b``, ``seed``, ``min_tick`` and
        ``max_ticks`` (see first_attack_view) and ``output_dir``. A relative
        output_dir uses the folder the command was started from.

    Returns
    -------
    dict
        menu.json's content: everything first_attack_view returns except the
        prompt text itself.

    Raises
    ------
    ValueError
        As first_attack_view.
    """
    settings = cast(dict[str, Any], OmegaConf.to_container(config, resolve=True))
    output = Path(to_absolute_path(str(Path(settings["output_dir"]).expanduser())))
    found = first_attack_view(
        int(settings["map_id"]),
        str(settings["team_a"]),
        str(settings["team_b"]),
        int(settings["seed"]),
        int(settings["min_tick"]),
        int(settings["max_ticks"]),
    )
    prompt = found.pop("prompt")
    output.mkdir(parents=True, exist_ok=True)
    (output / "prompt.txt").write_text(prompt, encoding="utf-8")
    (output / "menu.json").write_text(json.dumps(found, indent=1) + "\n")
    print(output / "prompt.txt", flush=True)
    return found


if __name__ == "__main__":
    prepare_hydra()
    main()
