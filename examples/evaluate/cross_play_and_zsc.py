"""Save fixed focal, partner and opponent games, including a local host partner.

Install the hydra extra and run the small CPU example::

    python examples/evaluate/cross_play_and_zsc.py output_dir=artifacts/cross-play

The default plays four games capped at two steps, with no external provider.
Use named focals, partners and opponents mappings to compare built-ins, saved
actors or trusted module:function factories. Python run accepts live Systems.
The focal controls Team A slots 0--3 and its partner controls slot 4. Canonical
5v5 games exchange spawn ends while keeping the focal on Team A. Familiarity is
declared exposure, relative to each focal; missing pairs remain unknown. Neither
these labels nor tiny games prove zero-shot coordination. Use --help or --cfg job.
"""

from __future__ import annotations

import csv
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from jax import Array

    from marl_battlegrounds import Policy, System, SystemInput
    from marl_battlegrounds.types import ActorAction


type Tree = Any


def host_idle(
    variables: Tree, memory: Tree, inputs: SystemInput, keys: Array
) -> tuple[ActorAction, Tree]:
    """Return legal idle actions from a local fake host, with no network calls.

    variables and keys are unused. inputs contains NumPy arrays shaped (B,5)
    for actor masks; valid marks real games and controlled_mask marks this
    member's slots. Zero action heads are legal idle choices. The team helper
    retains only controlled actions. Empty memory is returned unchanged.
    """
    import numpy as np

    from marl_battlegrounds.types import ActorAction

    del variables, keys
    zero = cast("Array", np.zeros(inputs.active_mask.shape, np.int32))
    return ActorAction(zero, zero, zero), memory


def _cell_label(role: str, index: int, name: str) -> str:
    """Make a bounded snake_case folder label; exact names and IDs stay in rows.

    role names the focal, partner or opponent, index preserves population order,
    and name is its display alias. Non-ASCII/punctuation becomes an underscore.
    """
    alias = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:20].rstrip("_")
    return f"{role}_{index:02d}_{alias or 'method'}"


def run(
    output_dir: Path,
    *,
    focals: Mapping[str, System | Policy | str],
    partners: Mapping[str, System | Policy | str],
    opponents: Mapping[str, System | Policy | str],
    familiarity: Mapping[str, Mapping[str, str]] | None = None,
    maps: Sequence[int] = (42,),
    seed_pairs: int = 1,
    max_steps: int = 2,
    num_envs: int = 2,
    seed: int = 17,
) -> list[dict[str, Any]]:
    """Play each fixed combination and save a CSV row for every paired game.

    Parameters
    ----------
    output_dir : Path
        Dedicated study folder. A matching repeat resumes each cell through
        evaluate; it rebinds supplied methods and checks their saved identities.
    focals, partners, opponents : mapping of str to System, Policy or str
        Nonempty named methods. Strings use the shared built-in, actor-folder
        or factory loader. Names only label rows; saved IDs identify methods.
    familiarity : nested mapping, optional
        focal name -> partner name -> familiar, held_out or unknown. Missing
        pairs are unknown. Declare exposure before games; this is not an audit
        of training history. Changed declarations need a new output directory.
    maps : sequence of int, default=(42,)
        Nonempty ordered map IDs, with one spawn pair per map and seed pair.
    seed_pairs, max_steps, num_envs : int, default=1, 2, 2
        Positive pairs per map, per-game step cap and parallel game limit.
        max_steps=2 is only a software demonstration, not a research horizon.
    seed : int, default=17
        Shared evaluation seed, separate from any training seed.

    Returns
    -------
    list of dict
        Per-game rows also written to games.csv. Each row names the methods,
        declared familiarity, exact recorded team IDs, member registration IDs,
        real checkpoint labels when present, map, spawn end and saved run.

    Notes
    -----
    cross_play_settings.json saves the fixed settings and available member evidence.
    Numerical population values are snapshotted once before all comparisons.
    Cell folders contain ordinary evaluator records, including native points
    and full System descriptions. games.csv is a rebuildable view of those
    records. A retry skips durable games; interrupted games restart through
    the existing evaluator. Live host sessions must be supplied again. IDs
    describe known code/parameters; opaque host state is not proved frozen.
    API/provider errors propagate. Use one process per output directory.

    Raises
    ------
    ValueError
        A population is empty, a familiarity label is invalid, a declaration
        changes, or a cell has more than one saved run. The evaluator checks
        game settings and method identity before executing or resuming games.
    """
    import marl_battlegrounds as marl_bgs
    from marl_battlegrounds import load_method
    from marl_battlegrounds.evaluation.recording_identity import (
        normalize_system_registration,
    )
    from marl_battlegrounds.evaluation.system_evaluation import (
        freeze_evaluation_method,
    )

    populations = {"focals": focals, "partners": partners, "opponents": opponents}
    if any(not values for values in populations.values()):
        raise ValueError("Focals, partners and opponents must each contain a method")
    labels = {name: dict(values) for name, values in (familiarity or {}).items()}
    if any(
        focal not in focals
        or any(
            partner not in partners or label not in {"familiar", "held_out", "unknown"}
            for partner, label in values.items()
        )
        for focal, values in labels.items()
    ):
        raise ValueError(
            "Familiarity must name a declared focal and partner with a valid label"
        )
    methods = {
        role: {
            name: freeze_evaluation_method(
                load_method(value) if isinstance(value, str) else value
            )
            for name, value in values.items()
        }
        for role, values in populations.items()
    }
    registrations: dict[str, dict[str, dict[str, Any]]] = {
        role: {
            name: dict(
                zip(
                    ("id", "description"),
                    normalize_system_registration(
                        value, phase="evaluation", frozen=True
                    ),
                    strict=True,
                )
            )
            for name, value in values.items()
        }
        for role, values in methods.items()
    }
    declaration = {
        "members": registrations,
        "member_order": {role: list(values) for role, values in methods.items()},
        "familiarity": labels,
        "maps": list(maps),
        "seed_pairs": seed_pairs,
        "max_steps": max_steps,
        "seed": seed,
        "rosters": "canonical_5v5",
        "focal_slots": [0, 1, 2, 3],
        "partner_slots": [4],
        "spawn_mode": "paired",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    declaration_path = output_dir / "cross_play_settings.json"
    if declaration_path.exists():
        if json.loads(declaration_path.read_text()) != declaration:
            raise ValueError(
                "Cross-play declaration changed; use a new output directory"
            )
    else:
        with declaration_path.open("x") as stream:
            stream.write(json.dumps(declaration, indent=2, allow_nan=False) + "\n")

    rows: list[dict[str, Any]] = []
    for focal_index, (focal_name, focal) in enumerate(methods["focals"].items()):
        for partner_index, (partner_name, partner) in enumerate(
            methods["partners"].items()
        ):
            mixed = marl_bgs.team(focal, partner, slots=[[0, 1, 2, 3], [4]])
            for opponent_index, (opponent_name, opponent) in enumerate(
                methods["opponents"].items()
            ):
                cell = output_dir / "_".join(
                    (
                        _cell_label("focal", focal_index, focal_name),
                        _cell_label("partner", partner_index, partner_name),
                        _cell_label("opponent", opponent_index, opponent_name),
                    )
                )
                saved = list(cell.glob("*/run_details.json"))
                if len(saved) > 1:
                    raise ValueError(f"Expected one saved run in {cell}")
                result = marl_bgs.evaluate(
                    mixed,
                    opponent,
                    num_episodes=2 * len(maps) * seed_pairs,
                    maps=maps,
                    spawn_mode="paired",
                    seed=seed,
                    num_envs=num_envs,
                    max_steps=max_steps,
                    metrics="priority",
                    chunk_size=min(max_steps, 16),
                    output_dir=None if saved else cell,
                    resume_from=saved[0].parent if saved else None,
                )
                assert result.paths is not None
                run_dir = result.paths["run_details"].parent
                loaded = marl_bgs.load_results(run_dir, phase="evaluation", pass_id="1")
                saved_pass = next(iter(loaded.metadata["passes"].values()))
                summary = loaded.head_to_head()
                focal_row = next(
                    index
                    for index, identifier in enumerate(summary["system_id"])
                    if identifier == saved_pass["system_ids"]["team_a"]
                )
                print(
                    f"{focal_name} + {partner_name} vs {opponent_name}: "
                    f"{summary['games'][focal_row]} games; "
                    f"W/D/L: {summary['wins'][focal_row]}/"
                    f"{summary['draws'][focal_row]}/{summary['losses'][focal_row]}; "
                    f"Mean Points: {summary['mean_points_for'][focal_row]:.3f} for, "
                    f"{summary['mean_points_against'][focal_row]:.3f} against; "
                    f"Mean Point Margin: {summary['mean_point_margin'][focal_row]:.3f}"
                )
                games = loaded.table("episodes")
                for index, episode_id in enumerate(games["episode_id"]):
                    episode = saved_pass["episodes"][str(int(episode_id))]
                    row: dict[str, Any] = {
                        "focal": focal_name,
                        "partner": partner_name,
                        "opponent": opponent_name,
                        "familiarity": labels.get(focal_name, {}).get(
                            partner_name, "unknown"
                        ),
                        "team_a_system_id": saved_pass["system_ids"]["team_a"],
                        "team_b_system_id": saved_pass["system_ids"]["team_b"],
                        "spawn_end": "team_a"
                        if episode["spawn_locations"] == 0
                        else "team_b",
                        "run_path": str(run_dir.relative_to(output_dir)),
                    }
                    for role, name in (
                        ("focal", focal_name),
                        ("partner", partner_name),
                        ("opponent", opponent_name),
                    ):
                        member = registrations[role + "s"][name]
                        row[role + "_id"] = member["id"]
                        row[role + "_checkpoint"] = member["description"].get(
                            "checkpoint"
                        )
                    for field in (
                        "run_id",
                        "phase",
                        "pass_id",
                        "episode_id",
                        "seed_id",
                        "map_id",
                        "config_id",
                        "outcome",
                        "team_a_score",
                        "team_b_score",
                    ):
                        value = games[field][index]
                        row[field] = value.item() if hasattr(value, "item") else value
                    rows.append(row)
    with (output_dir / "games.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def launch(
    output_dir: str | Path,
    *,
    focals: dict[str, str] | None = None,
    partners: dict[str, str] | None = None,
    opponents: dict[str, str] | None = None,
    familiarity: dict[str, dict[str, str]] | str | None = None,
    maps: Sequence[int] = (42,),
    seed_pairs: int = 1,
    max_steps: int = 2,
    num_envs: int = 2,
    seed: int = 17,
) -> int:
    """Load declared references once and run every focal/partner/opponent cell.

    output_dir holds the saved matrix; repeating unchanged settings resumes it.
    Named mappings retain their labels. The reserved partner reference local-host
    demonstrates a local idle host System. familiarity is a nested exposure-label
    mapping or its JSON file, never inferred from these games. maps, seed_pairs,
    max_steps, num_envs and seed have the same meanings as run. CPU is the fallback
    when no backend was chosen. Print game location and return zero on success.
    """
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import marl_battlegrounds as marl_bgs

    if isinstance(familiarity, str):
        familiarity = cast(
            dict[str, dict[str, str]], json.loads(Path(familiarity).read_text())
        )
    focals = {"tdm-alpha": "tdm-alpha"} if focals is None else focals
    partners = (
        {"random": "random", "local-host": "local-host"}
        if partners is None
        else partners
    )
    opponents = {"tdm-beta": "tdm-beta"} if opponents is None else opponents
    destination = Path(output_dir)
    rows = run(
        destination,
        focals=focals,
        partners={
            name: marl_bgs.System("Local Idle Host", host_idle, execution="host")
            if reference == "local-host"
            else reference
            for name, reference in partners.items()
        },
        opponents=opponents,
        familiarity=familiarity,
        maps=maps,
        seed_pairs=seed_pairs,
        max_steps=max_steps,
        num_envs=num_envs,
        seed=seed,
    )
    print(f"Saved {len(rows)} games to {destination / 'games.csv'}")
    print(
        "Familiarity is declared exposure. "
        "These games alone do not prove ZSC competence."
    )
    return 0


def main() -> int:
    """Compose packaged YAML, call the Python lesson and return its exit status."""
    from marl_battlegrounds.experiments.hydra import example_main

    return example_main("cross_play_and_zsc", launch)


if __name__ == "__main__":
    raise SystemExit(main())
