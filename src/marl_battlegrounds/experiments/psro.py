"""Run PSRO through the optional Hydra adapter and the public training API.

Use ``python -m marl_battlegrounds.experiments.psro output_dir=/path/run``
after installing marl-battlegrounds[training,hydra]. The oracle is a training
method with its tournament recipe (``method=ippo`` by default) and accepts
every training field override the train stage accepts; ``psro.*`` settings
name the seeded members and the panel and size the population. Run the same
command again to continue a stopped run.

This entry owns only the command: it reads the settings, finds the seeded
members and the panel in an opponent bundle by name (by default the released
Season 0 training opponents, downloaded into ``output_dir/opponents``) and
calls ``marl_battlegrounds.training.train_psro``, which owns the method, its
files and its resume.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

# Import the optional adapter first for its installation error and parser fix.
# isort: off
from marl_battlegrounds.experiments.hydra import prepare_hydra

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
# isort: on


def run(settings: dict[str, Any], *, base_dir: str | Path) -> dict[str, Any]:
    """Resolve one composed psro config and run it.

    Parameters
    ----------
    settings : dict
        The resolved config: the oracle's training fields, the required
        ``output_dir`` and ``psro`` with ``seed_members`` and ``panel`` (bundle
        member names), ``opponent_bundle`` (a bundle folder, or None to
        download the released one) and the population settings
        ``train_psro`` takes (``round_env_steps``, ``payoff_games``,
        ``payoff_maps``, ``payoff_seed``, ``validation_seed``,
        ``confirmation_seed``, ``eval_num_envs``, ``shortlist``).
    base_dir : str or Path
        Folder that relative paths (``output_dir``, ``opponent_bundle``) are
        read from.

    Returns
    -------
    dict
        What ``train_psro`` returns.

    Raises
    ------
    ValueError
        A training field fails the shared parser, a bundle member is unknown,
        or ``train_psro`` refuses the settings.
    """
    from marl_battlegrounds import training
    from marl_battlegrounds.experiments.hydra import resolve_training_config
    from marl_battlegrounds.training.psro import bundle_references

    plain = dict(settings)
    psro = dict(cast(dict[str, Any], plain.pop("psro")))
    base = Path(base_dir).expanduser().resolve()
    output_dir = base / Path(str(plain.pop("output_dir"))).expanduser()
    config = resolve_training_config(plain, base_dir=base)
    if folder := psro.pop("opponent_bundle"):
        bundle = base / Path(str(folder)).expanduser()
    else:
        from marl_battlegrounds.released_models import (
            _released_training_opponents,  # pyright: ignore[reportPrivateUsage]
        )

        address, bundle = _released_training_opponents(output_dir / "opponents")
        print(f"Released Opponents: {address}")
    return training.train_psro(
        config,
        output_dir=output_dir,
        seed_members=bundle_references(bundle, psro.pop("seed_members")),
        panel=bundle_references(bundle, psro.pop("panel")),
        **psro,
    )


@hydra.main(version_base="1.3", config_path="configs", config_name="psro")
def main(config: DictConfig) -> dict[str, Any]:
    """Pass one composed Hydra job to ``run`` from the launch folder.

    Parameters
    ----------
    config : omegaconf.DictConfig
        The packaged ``psro`` config with any command-line overrides.

    Returns
    -------
    dict
        What ``train_psro`` returns.
    """
    settings: dict[str, Any] = OmegaConf.to_container(config, resolve=True)  # pyright: ignore[reportAssignmentType] - the root is a mapping.
    return run(settings, base_dir=get_original_cwd())


if __name__ == "__main__":
    prepare_hydra()
    main()
