"""Evaluate a saved ff_ippo actor through the shared public evaluator.

Install marl-battlegrounds[training,hydra]. Run this file with checkpoint=PATH,
evaluation.output_dir=PATH and optional evaluation=full. Use --help or --cfg job
to inspect settings; --config-name evaluate_resume keeps omitted saved settings.
This script does not train or select a model.
"""

from marl_battlegrounds.experiments.hydra import evaluation_main

if __name__ == "__main__":
    raise SystemExit(evaluation_main("ff_ippo"))
