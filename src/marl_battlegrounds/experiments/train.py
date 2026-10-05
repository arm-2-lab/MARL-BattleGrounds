"""Train through the optional Hydra adapter and the public training API.

Run ``python -m marl_battlegrounds.experiments.train --help`` after installing
marl-battlegrounds[training,hydra]. Use --cfg job --resolve for settings without
outputs, or preview=true for checked settings and declared output locations.
"""

# Import the adapter first so a missing optional extra has its install command.
# isort: off
from marl_battlegrounds.experiments.hydra import run_training

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

# isort: on


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(config: DictConfig) -> None:
    """Pass one resolved Hydra job to the shared adapter; print its saved paths.

    config comes from packaged presets and ordinary Hydra overrides. Inputs use
    Hydra's original working directory even if a caller changed its chdir flag.
    The adapter and training API own validation and output; this entry owns none.
    """
    run_training(
        config,
        base_dir=get_original_cwd(),
        launch_dir=HydraConfig.get().runtime.output_dir,
    )


if __name__ == "__main__":
    main()
