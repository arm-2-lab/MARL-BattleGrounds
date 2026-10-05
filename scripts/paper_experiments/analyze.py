"""Run the packaged analyze command with Hydra YAML and the public API.

Use python scripts/paper_experiments/analyze.py --help for settings. The same
installed command is python -m marl_battlegrounds.experiments.analyze.
"""

from marl_battlegrounds.experiments.analyze import main
from marl_battlegrounds.experiments.hydra import prepare_hydra

if __name__ == "__main__":
    prepare_hydra()
    main()
