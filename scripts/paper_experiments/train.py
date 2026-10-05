"""Run the train stage through packaged Hydra settings and the public API.

Use --help for controls. The installed route is
python -m marl_battlegrounds.experiments.study.
"""

from marl_battlegrounds.experiments.hydra import prepare_hydra
from marl_battlegrounds.experiments.study import main

if __name__ == "__main__":
    prepare_hydra()
    main()
