"""Evaluate a saved mappo actor through the shared public evaluator.

Install marl-battlegrounds[training,hydra]. Run this file with checkpoint=PATH
(a saved actor folder or a portable model file) or a released model name,
for example checkpoint=MARL-BattleGrounds/tdm-season-0-recurrent-mappo@final
(downloaded once into the Hugging Face cache), plus
evaluation.output_dir=PATH and optional evaluation=full. Use --help or --cfg job
to inspect settings; --config-name evaluate_resume keeps omitted saved settings.
This script does not train or select a model.
"""

from marl_battlegrounds.experiments.hydra import evaluation_main

if __name__ == "__main__":
    raise SystemExit(evaluation_main("mappo"))
