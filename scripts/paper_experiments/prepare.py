"""Run the packaged prepare command with Hydra YAML and the public API.

Use python scripts/paper_experiments/prepare.py --help for settings. The same
installed command is python -m marl_battlegrounds.experiments.prepare.
"""

from marl_battlegrounds.experiments.hydra import prepare_hydra
from marl_battlegrounds.experiments.prepare import main

if __name__ == "__main__":
    prepare_hydra()
    main()
