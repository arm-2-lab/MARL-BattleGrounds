# Contributing To MARL-BattleGrounds

Thank you for helping. Please follow the [Code of Conduct](CODE_OF_CONDUCT.md).
Report security problems privately, as [SECURITY.md](SECURITY.md) explains.

## Your First Pull Request

Fork the repository, clone your fork, install [uv](https://docs.astral.sh/uv/) and run:

```bash
uv sync --locked --extra dev --extra training --extra hydra --extra viz
scripts/dev/format.sh
scripts/dev/check.sh --static-only
JAX_PLATFORMS=cpu uv run --no-sync pytest tests/<folder>/<file>.py
```

1. `uv sync` installs the package and its tools. On an NVIDIA GPU machine, add `--extra cuda13`.
2. `format.sh` formats your files. It may edit them, so look at the diff.
3. `check.sh --static-only` runs the fast checks (Ruff and Pyright).
4. `pytest` runs the tests for the files you changed, on the CPU.

Then open the pull request and fill in its form. GitHub runs the full test suite.

## Where Things Live

| Folder | What It Holds |
|---|---|
| `src/marl_battlegrounds/` | The package |
| `tests/` | The tests, in 14 folders that mirror the package |
| `examples/` | Short runnable lessons, listed in `examples/README.md` |
| `docs/` | Short guides |
| `scripts/dev/` | Contributor tools: checks, generators and benchmarks |
| `scripts/paper_experiments/` | The paper's experiment records |
| `web/visual_debugger/` | The browser code shared by the Replay Viewer and the BattleClient. It keeps this folder name because tools and the wheel build depend on it. |

## Test Rules

- Each test file has exactly one docstring, at the top, saying what the file
  checks. Tests, classes and fixtures have no docstrings.
- Tests never use time limits: no `timeout=`, deadlines or Playwright
  timeouts. If a test hangs, fix the cause.
- Put new tests in small files, so they fit the shards (groups of tests that run
  side by side). Time each new file once and add its time in whole seconds to
  `files` in `scripts/dev/pytest_shard_costs.json`. The routine for rebalancing
  the shards is at the top of `scripts/dev/shard_costs.py`.

## Documentation Rules

- Every Python function and class has a NumPy-style docstring: what each
  argument means, its default, what it returns and which errors it raises.
  Browser code uses JSDoc.
- A changed argument or behavior updates its docstring in the same change.
- Write in American English, with plain words and short sentences.

## Generated Files

Run the command; never edit these files by hand. Add `--check` to compare only.

| File | Command |
|---|---|
| `docs/api.md` | `uv run --no-sync python scripts/dev/export_api_reference.py` |
| `docs/metric_columns.csv` and the generated tables in `docs/metrics.md` | `uv run --no-sync python -m scripts.dev.export_metric_dictionary` |
| Hydra configs in `src/marl_battlegrounds/experiments/configs/` | `uv run --no-sync python scripts/dev/generate_experiment_configs.py` |

## Full Checks Before A Release

| Command | What It Checks |
|---|---|
| `scripts/dev/check.sh` | All Python tests, Ruff and Pyright, on the CPU |
| `scripts/dev/check_frontend.sh` | The browser code. Needs Node.js 24. First run `npm ci` and `npx playwright install chromium` once in `web/visual_debugger/`. |
| `scripts/dev/check_gpu.sh` | A clean commit on an NVIDIA GPU |

Each script explains its options with `--help`.

## Dependencies

Dependabot updates GitHub Actions only. Python packages change by hand. A bump
of jax, jaxlib, numpy, flax, optax, orbax-checkpoint, pydantic, scipy or
flashbax stops saved runs from being extended or saved tournaments from adding
games, so such bumps are deliberate. Before moving to a newer JAX, maintainers
follow the JAX refresh rule at the top of `scripts/dev/publish_models.py`.

## Rules, Commits And License

- A change to the game rules, or to what an agent may see, starts with an
  issue. Open one before you write code, so we can agree on the change first.
- Write each commit subject as `[TAG] Imperative summary`. TAG is FEATURE, FIX,
  TEST, REFACTOR or DOCS. Example: `[FIX] Remove collision ordering bias`.
- Your contributions are licensed under [Apache-2.0](LICENSE), the license of
  this project.
