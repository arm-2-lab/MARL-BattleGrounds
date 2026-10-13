"""Play saved methods through the installed BattleClient, or compare fixed partners.

Copy this file anywhere after installing MARL-BattleGrounds with its training
extra. Run ``python battle_client.py --help`` for modes and each mode's ``--help``
for paths. The companion guide is docs/battle_client.md in the source
release. No checkout, other example, private file or test module is imported.

Interactive modes save under recordings/MODE by default and block until Exit or
Ctrl-C. The page opens in Agent POV; Oracle View, which shows everything, is one
switch away. Press ? in the page for How to Play and the key list. Ordinary
maps keep BattleClient's defaults unless an explicit option changes them.
Scenario modes keep the complete packaged scenario 5 unchanged. LLM modes use
an already running service; this script starts no server, downloads no weights
and substitutes no mock. Only scenario-partners runs to completion
without a browser. Its exposure labels are declarations, not training audits.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import marl_battlegrounds as marl_bgs
from marl_battlegrounds import llm
from marl_battlegrounds.evaluation.tdm_scenarios import (
    TDM_BETA_SCENARIO_IDS,
    build_tdm_qualification_seed_schedule,
    build_tdm_scenario_specification,
)
from marl_battlegrounds.policies.random_valid import random_policy
from marl_battlegrounds.training import load_system
from marl_battlegrounds.types import ActorAction, SystemInput, SystemOutput

type Tree = Any


def jax_actions(
    variables: Tree, memory: Tree, inputs: SystemInput, keys: Array
) -> SystemOutput:
    """Sample legal actions separately for each owned actor, with empty memory.

    Parameters
    ----------
    variables, memory : object
        Unused parameters and unchanged empty episode memory.
    inputs : SystemInput
        Separate permitted actor rows. active_mask and controlled_mask are
        Boolean (B,5); valid marks real lanes (B,). A missing controlled_mask
        means all active slots. Only each actor's observation and masks are used.
    keys : Array
        One JAX random key per lane, split into five independent actor keys.

    Returns
    -------
    SystemOutput
        int32 (B,5) action heads and unchanged memory. Unowned, inactive and
        invalid rows are neutral zero. No input is changed. Pure JAX work uses
        no files, network request or privileged state.
    """
    del variables

    def split_actors(key: Array) -> Array:
        """Give each of the five actor rows a separate key."""
        return jax.random.split(key, 5)

    actor_keys = jax.vmap(split_actors)(keys)
    actions = jax.vmap(jax.vmap(random_policy))(
        inputs.actors.observation, inputs.action_mask, actor_keys
    )
    owned = inputs.active_mask
    if inputs.controlled_mask is not None:
        owned = owned & inputs.controlled_mask
    selected = owned & inputs.valid[:, None]
    return SystemOutput(
        ActorAction(
            jnp.where(selected, actions.move, 0).astype(jnp.int32),
            jnp.where(selected, actions.select_target, 0).astype(jnp.int32),
            jnp.where(selected, actions.use_ultimate, 0).astype(jnp.int32),
        ),
        memory,
    )


def host_actions(
    variables: Tree, memory: Tree, inputs: SystemInput, keys: Array
) -> SystemOutput:
    """Choose the first legal move and combat pair from each actor's own masks.

    Parameters
    ----------
    variables, memory : object
        Unused parameters and unchanged empty episode memory.
    inputs : SystemInput
        NumPy actor inputs with Boolean active/controlled masks (B,5) and valid
        lanes (B,). A missing controlled_mask means all active slots. Rows stay
        separate; joint target/Ultimate masks keep combat choices compatible.
    keys : Array
        One key per lane; unused by this deterministic teaching method.

    Returns
    -------
    SystemOutput
        int32 (B,5) heads and unchanged memory. Inactive, unowned and invalid
        rows are neutral zero. Inputs are not changed. No network is used;
        choosing the first legal action claims no learned skill.
    """
    del variables, keys
    owned = np.asarray(inputs.active_mask)
    if inputs.controlled_mask is not None:
        owned = owned & np.asarray(inputs.controlled_mask)
    selected = owned & np.asarray(inputs.valid)[:, None]
    moves = np.argmax(inputs.action_mask.move_mask, axis=-1)
    pairs = np.asarray(inputs.action_mask.select_target_use_ultimate_joint_mask)
    pair_index = np.argmax(pairs.reshape((*owned.shape, -1)), axis=-1)
    ultimate_count = pairs.shape[-1]
    return SystemOutput(
        ActorAction(
            cast(Array, np.where(selected, moves, 0).astype(np.int32)),
            cast(
                Array,
                np.where(selected, pair_index // ultimate_count, 0).astype(np.int32),
            ),
            cast(
                Array,
                np.where(selected, pair_index % ultimate_count, 0).astype(np.int32),
            ),
        ),
        memory,
    )


def _live_options(args: argparse.Namespace) -> dict[str, Any]:
    """Return explicit browser/output options without changing game rules.

    args supplies mode, output_dir, port and no_open. The default destination is
    recordings/MODE relative to the caller. BattleClient owns creation and saves.
    """
    return {
        "output_dir": args.output_dir or Path("recordings") / args.mode,
        "port": args.port,
        "open_browser": not args.no_open,
    }


def _ordinary_options(args: argparse.Namespace) -> dict[str, Any]:
    """Add an ordinary map and only explicitly requested rule overrides.

    args.map_id defaults to 0. An omitted max_steps keeps the public 300-step
    default. Scenario modes never call this helper or accept that option.
    """
    options = _live_options(args)
    options["map_id"] = args.map_id
    if args.max_steps is not None:
        options["max_steps"] = args.max_steps
    return options


def simple(args: argparse.Namespace) -> int:
    """Play one Manual and four shared-policy slots against a second saved policy.

    Parameters
    ----------
    args : argparse.Namespace
        policy_a/policy_b are real native artifact folders. map_id defaults to
        0; omitted max_steps keeps 300. output_dir defaults to recordings/simple,
        port to 0 (choose a free port), and no_open to False (open the browser).

    Returns
    -------
    int
        BattleClient exit status after Exit or Ctrl-C. Each source loads once;
        repeated slots share inference where supported, with separate memories.
        Loading errors propagate before serving. The client saves local replays.
    """
    policy_a = load_system(args.policy_a)
    policy_b = load_system(args.policy_b)
    return marl_bgs.battle_client(
        team_a=["manual", policy_a, policy_a, policy_a, policy_a],
        team_b=policy_b,
        **_ordinary_options(args),
    )


def medium(args: argparse.Namespace) -> int:
    """Play two saved sources per team, each assigned three and two roster slots.

    Parameters
    ----------
    args : argparse.Namespace
        a_first/a_second/b_first/b_second are four different native folders.
        map_id defaults to 0; omitted max_steps keeps 300. output_dir defaults to
        recordings/medium, port to 0 and no_open to False.

    Returns
    -------
    int
        Client status after Exit or Ctrl-C. Each source loads once. All ten
        slots are automatic; Run/Pause is available. Local replays are saved.
        Invalid artifact/setup errors propagate before serving.
    """
    a_first, a_second = load_system(args.a_first), load_system(args.a_second)
    b_first, b_second = load_system(args.b_first), load_system(args.b_second)
    return marl_bgs.battle_client(
        team_a=[a_first, a_first, a_first, a_second, a_second],
        team_b=[b_first, b_first, b_first, b_second, b_second],
        **_ordinary_options(args),
    )


def small(args: argparse.Namespace) -> int:
    """Show a 2v3 roster with one Manual actor and two saved sources.

    Parameters
    ----------
    args : argparse.Namespace
        policy_a/policy_b are native folders compatible with the chosen rosters.
        map_id defaults to 0; omitted max_steps keeps 300. output_dir defaults to
        recordings/small, port to 0 and no_open to False.

    Returns
    -------
    int
        Client status after Exit or Ctrl-C. Team A is Mage/Priest; Team B is
        Rogue/Rogue/Priest. Lists follow roster order. Sources load once; bad
        artifact/setup errors propagate. The client saves local replays.
    """
    policy_a = load_system(args.policy_a)
    policy_b = load_system(args.policy_b)
    return marl_bgs.battle_client(
        team_a_roster=("mage", "priest"),
        team_b_roster=("rogue", "rogue", "priest"),
        team_a=["manual", policy_a],
        team_b=[policy_b, policy_b, policy_b],
        **_ordinary_options(args),
    )


def _language_model(args: argparse.Namespace, client: llm.Client) -> marl_bgs.System:
    """Describe one actor-local LLM using the caller-owned open client.

    Parameters
    ----------
    args : argparse.Namespace
        model names the served model; model_revision declares its pinned weight
        revision. llm_records is light (default) or full (include exact requests).
    client : llm.Client
        Open HTTP client. The caller keeps it open through play and closes it.

    Returns
    -------
    marl_bgs.System
        Existing LLM adapter with legal JSON actions, no history, 64 reply tokens
        and stop-on-failure. Construction sends no request. A declared revision
        does not prove the server's weights; verify the separate service recipe.
    """
    return llm.make_system(
        args.model,
        client=client,
        model_revision=args.model_revision,
        history_turns=0,
        failure_policy="stop",
        max_tokens=64,
        records=args.llm_records,
    )


def hard(args: argparse.Namespace) -> int:
    """Assign ten distinct controllers, including six native families and an LLM.

    Parameters
    ----------
    args : argparse.Namespace
        mappo/ippo/ff_mappo/ff_ippo/qmix/pqn_vdn are matching native folders.
        model and model_revision identify the separately running model.
        server_url defaults to http://127.0.0.1:8000/v1 and llm_records to light.
        map_id defaults to 0; omitted max_steps keeps 300. output_dir defaults
        to recordings/hard, port to 0 and no_open to False.

    Returns
    -------
    int
        Client status after Exit or Ctrl-C. A has Manual/MAPPO/IPPO/FF-MAPPO/
        FF-IPPO; B has QMIX/PQN-VDN/our LLM/custom JAX/custom host. Native inputs
        keep saved scale/frame/sharing. Loads, HTTP and setup errors propagate;
        the live client reports turn failures and saves accepted play.

    Notes
    -----
    This owns one HTTP client, with concurrency 1, a 60-second request deadline
    and no retry. It closes that client after play or error. It does not start
    or stop the independent model server or download weights.
    """
    mappo = load_system(args.mappo)
    ippo = load_system(args.ippo)
    ff_mappo = load_system(args.ff_mappo)
    ff_ippo = load_system(args.ff_ippo)
    qmix = load_system(args.qmix)
    pqn_vdn = load_system(args.pqn_vdn)
    custom_jax = marl_bgs.System("Tutorial JAX", jax_actions)
    custom_host = marl_bgs.System("Tutorial Host", host_actions, execution="host")
    with llm.Client(args.server_url, concurrency=1, timeout=60.0) as client:
        language_model = _language_model(args, client)
        return marl_bgs.battle_client(
            team_a=["manual", mappo, ippo, ff_mappo, ff_ippo],
            team_b=[qmix, pqn_vdn, language_model, custom_jax, custom_host],
            **_ordinary_options(args),
        )


def scenario_manual(args: argparse.Namespace) -> int:
    """Play exact scenario 5 with Manual/QMIX/MAPPO in living Team A slots 0/1/4.

    Parameters
    ----------
    args : argparse.Namespace
        qmix/mappo are matching native folders. output_dir defaults to
        recordings/scenario-manual, port to 0 and no_open to False.

    Returns
    -------
    int
        Client status after Exit or Ctrl-C; local recordings are saved. Dead
        slots 2 and 3 keep MAPPO. All five configured slots, the opponent, rules,
        roster, initial state and seed stay fixed. Invalid source errors
        propagate. Human-assisted play is not autonomous ZSC.
    """
    qmix, mappo = load_system(args.qmix), load_system(args.mappo)
    return marl_bgs.battle_client(
        scenario_id=5,
        team_a=["manual", qmix, mappo, mappo, mappo],
        **_live_options(args),
    )


def scenario_llm(args: argparse.Namespace) -> int:
    """Watch exact scenario 5 with LLM/QMIX/MAPPO in living Team A slots 0/1/4.

    Parameters
    ----------
    args : argparse.Namespace
        qmix/mappo are matching native folders. model/model_revision identify
        the separately running model. server_url defaults to
        http://127.0.0.1:8000/v1 and llm_records to light. output_dir defaults
        to recordings/scenario-llm, port to 0 and no_open to False.

    Returns
    -------
    int
        Client status after Exit or Ctrl-C; local recordings are saved. Dead
        slots 2 and 3 retain MAPPO. No Manual slot is configured, so Run works.
        The complete scenario remains fixed. Bad source/setup errors propagate;
        the live client reports turn failures and retains accepted play.

    Notes
    -----
    This closes its own HTTP client after play or error, never the model server.
    Requests use concurrency 1, a 60-second deadline and no retry. No server is
    started and no weights are downloaded.
    """
    qmix, mappo = load_system(args.qmix), load_system(args.mappo)
    with llm.Client(args.server_url, concurrency=1, timeout=60.0) as client:
        language_model = _language_model(args, client)
        return marl_bgs.battle_client(
            scenario_id=5,
            team_a=[language_model, qmix, mappo, mappo, mappo],
            **_live_options(args),
        )


def scenario_partners(args: argparse.Namespace) -> int:
    """Compare two declared partners at the same complete scenario 5 start.

    Parameters
    ----------
    args : argparse.Namespace
        focal/partner_a/partner_b are native folders. exposure_a/exposure_b are
        researcher declarations: familiar, held_out or unknown (default).
        output_dir defaults to runs/scenario-partners, with one child per partner.

    Returns
    -------
    int
        Zero after both autonomous games complete. Normal evaluator tables and
        replays are saved in new run folders, and their paths are printed.
        Artifact, execution and write errors propagate; no training occurs.

    Notes
    -----
    The focal owns Team A slots 0..3; each partner owns slot 4. The shared
    scenario owner supplies complete config/state, seed schedule and opponent
    pressure. Both runs use the same numeric episode ID and random coordinates.
    Metadata retains the full specification and declared exposure; actual method
    IDs come from the existing evaluator. The fixed horizon is unchanged.
    This calls no LLM and opens no browser. Exposure labels alone prove neither
    held-out training exposure nor zero-shot coordination competence.
    """
    scenario = marl_bgs.load_tdm_scenario(5)
    schedule = build_tdm_qualification_seed_schedule()
    specification = build_tdm_scenario_specification(5, schedule)
    seed = schedule.realized_seed_protocols[0]
    assert seed.root_seed is not None
    opponent = "tdm-beta" if 5 in TDM_BETA_SCENARIO_IDS else "tdm-alpha"
    # Scenario recording assigns two numeric episode IDs per packaged scenario.
    episode_id = (scenario.info.scenario_id - 1) * 2 + 1
    focal = load_system(args.focal)
    partners = (load_system(args.partner_a), load_system(args.partner_b))
    output = args.output_dir or Path("runs/scenario-partners")
    for label, partner, exposure in zip(
        ("partner_a", "partner_b"),
        partners,
        (args.exposure_a, args.exposure_b),
        strict=True,
    ):
        combined = marl_bgs.team(focal, partner, slots=[[0, 1, 2, 3], [4]])
        episode = marl_bgs.EpisodeSpec(
            episode_id,
            scenario.config,
            seed_id=seed.episode_seed,
            initial_state=scenario.initial_state,
            metadata={
                "scenario_id": 5,
                "scenario_specification": specification.model_dump(mode="json"),
                "partner_label": label,
                "declared_partner_exposure": exposure,
                "focal_slots": [0, 1, 2, 3],
                "partner_slots": [4],
                "human_assisted": False,
            },
        )
        result = marl_bgs.evaluate_episodes(
            combined,
            opponent,
            [episode],
            seed=seed.root_seed,
            num_envs=1,
            chunk_size=1,
            replay_episodes=[episode_id],
            output_dir=output / label,
        )
        print(label, "Declared Exposure:", exposure)
        print(result.table("episodes"))
        print("Saved Run:", result.run_dir)
        print("Saved Replays:", result.replay_paths)
    print("Declared exposure is not an audit or a claim of ZSC competence.")
    return 0


def _arguments() -> argparse.Namespace:
    """Read one tutorial mode and its explicit paths without loading a method.

    Each mode's --help lists only relevant settings. Fixed scenario modes reject
    map, seed, roster and horizon options through argparse before any loading.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for name, help_text in (
        ("simple", "Manual plus one saved source against another source"),
        ("medium", "Two saved sources per team, split three/two"),
        ("hard", "Ten distinct controllers, including our real LLM adapter"),
        ("small", "A two-agent team against a three-agent team"),
        ("scenario-manual", "Scenario 5 with Manual, QMIX and MAPPO"),
        ("scenario-llm", "Scenario 5 with LLM, QMIX and MAPPO"),
        ("scenario-partners", "Autonomous exact-start partner comparison"),
    ):
        command = modes.add_parser(name, help=help_text, description=help_text)
        command.add_argument(
            "--output-dir", type=Path, help="Parent for new saved files"
        )
        if name == "scenario-partners":
            for option in ("focal", "partner-a", "partner-b"):
                command.add_argument("--" + option, type=Path, required=True)
            for option in ("exposure-a", "exposure-b"):
                command.add_argument(
                    "--" + option,
                    choices=("familiar", "held_out", "unknown"),
                    default="unknown",
                    help="Researcher-declared partner exposure relative to the focal",
                )
            continue
        command.add_argument(
            "--port", type=int, default=0, help="Loopback port; 0 chooses"
        )
        command.add_argument("--no-open", action="store_true", help="Print URL only")
        if not name.startswith("scenario-"):
            command.add_argument("--map-id", type=int, default=0)
            command.add_argument(
                "--max-steps", type=int, help="Explicit ordinary-map horizon override"
            )
        options = (
            ("policy-a", "policy-b")
            if name in ("simple", "small")
            else ("a-first", "a-second", "b-first", "b-second")
            if name == "medium"
            else ("mappo", "ippo", "ff-mappo", "ff-ippo", "qmix", "pqn-vdn")
            if name == "hard"
            else ("qmix", "mappo")
        )
        for option in options:
            command.add_argument(
                "--" + option, type=Path, required=True, help="Native actor/full folder"
            )
        if name in ("hard", "scenario-llm"):
            command.add_argument("--model", required=True, help="Served model name")
            command.add_argument(
                "--model-revision", required=True, help="Verified pinned model revision"
            )
            command.add_argument("--server-url", default="http://127.0.0.1:8000/v1")
            command.add_argument(
                "--llm-records", choices=("light", "full"), default="light"
            )
    return parser.parse_args()


def main() -> int:
    """Run the chosen tutorial through its existing public API owner.

    Returns
    -------
    int
        The client exit code, or zero after the autonomous comparison saves its
        results. Parsing starts no client and loads no weights. Chosen mode
        errors propagate; there is no substitute source or silent retry.
    """
    args = _arguments()
    if args.mode == "simple":
        return simple(args)
    if args.mode == "medium":
        return medium(args)
    if args.mode == "hard":
        return hard(args)
    if args.mode == "small":
        return small(args)
    if args.mode == "scenario-manual":
        return scenario_manual(args)
    if args.mode == "scenario-llm":
        return scenario_llm(args)
    return scenario_partners(args)


if __name__ == "__main__":
    raise SystemExit(main())
