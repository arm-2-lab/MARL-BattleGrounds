# Evaluation Package

Import public names from `marl_battlegrounds` or `marl_battlegrounds.evaluation`, for example
`from marl_battlegrounds.evaluation import ReplayArtifactV4`. Never import from a module inside
a folder whose name starts with `_`; those paths can change. The files here are the evaluation
core and the modules the guides import by path. `catalog.py`, `combat_metrics.py`,
`host_evidence.py`, `models.py` and `revision.py` also stay here because files that saved
results hash import them by this path. `_metrics/`, `_recording/`, `_replays/` and
`_tournaments/` hold the machinery behind the public names, grouped by topic.
