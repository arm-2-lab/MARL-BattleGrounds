#!/usr/bin/env bash
# PSRO's short TDM screen: does a PSRO population learn?
#
# Three PSRO runs, one per search seed. Each spends 40,108,032 game steps in all
# (training, payoff and validation games). The oracle is the search's recurrent
# IPPO tournament recipe, the seeded members are the Season 0 training
# opponents, and the validation panel is the Season 0 panel. The comparison is
# with the search's three ippo-reference runs at the same steps.
#
# Usage: psro.sh OPPONENT_BUNDLE OUTPUT_ROOT
#   OPPONENT_BUNDLE  absolute path of the search's checked opponent bundle
#   OUTPUT_ROOT      absolute path; each seed writes OUTPUT_ROOT/seed-SEED/
# The seeds run one after another. Running the script again continues any
# stopped run. Needs marl-battlegrounds[training,hydra] and one GPU.
set -euo pipefail
bundle="$1"
root="$2"
for seed in 12259871 20261001 20261002; do
  python -m marl_battlegrounds.experiments.psro method=ippo recipe=default \
    total_env_steps=40108032 seed="${seed}" \
    psro.opponent_bundle="${bundle}" output_dir="${root}/seed-${seed}"
done
