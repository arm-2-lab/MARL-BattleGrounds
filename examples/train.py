"""Launch any built-in learner with packaged Hydra settings and the public API.

Install marl-battlegrounds[training,hydra]. From the checkout run
``python examples/train.py preview=true method=qmix recipe=software_check``.
Use --help or --cfg job --resolve without starting training. The installed
entry ``python -m marl_battlegrounds.experiments.train`` is identical.
"""

from marl_battlegrounds.experiments.train import main

if __name__ == "__main__":
    main()
