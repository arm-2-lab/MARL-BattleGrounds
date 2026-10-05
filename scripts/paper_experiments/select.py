"""Run the select stage through packaged Hydra settings and the public API.

Use --help for controls. The installed route is
python -m marl_battlegrounds.experiments.select.
"""

from marl_battlegrounds.experiments.hydra import prepare_hydra
from marl_battlegrounds.experiments.select import main

if __name__ == "__main__":
    prepare_hydra()
    main()
