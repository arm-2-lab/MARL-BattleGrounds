"""Write a separate checkpoint choice and print any missing confirmation calls.

Run with saved completed training data, for example::

    python examples/training_selection.py 'run_dirs=[RUN]' \
        declaration=choice.json output_dir=artifacts/new-choice

choice.json contains a schema-1 named selection declaration. Existing artifact
imports can initialize JAX. Set JAX_PLATFORMS=cpu before starting Python
when you want CPU execution.
This reads actor artifacts and saved game tables; it does not train or execute a
model. Output must be new or empty. Missing confirmations are printed, never run
automatically.
The matching package command is ``python -m marl_battlegrounds
reselect-checkpoint RUN --declaration choice.json --output-dir DIRECTORY``.
"""

from collections.abc import Sequence


def run(
    run_dirs: Sequence[str], *, declaration: str | dict[str, object], output_dir: str
) -> None:
    """Read saved runs, write a separate choice and print missing confirmation calls.

    run_dirs preserves the declared run order. declaration is a rule mapping or
    its JSON path; file-relative inputs keep the selection owner's rules.
    output_dir names a new decision directory. Incomplete confirmation is
    reported, never executed. Source records are unchanged; errors propagate.
    """
    from marl_battlegrounds import training

    decision = training.reselect_checkpoint(
        run_dirs, declaration=declaration, output_dir=output_dir
    )
    print(f"Selection Status: {decision['status']}")
    for record in decision["runs"]:
        print(f"Run: {record['run_id']}; Status: {record['status']}")
        if record["winner"] is not None:
            print(f"Selected Actor: {record['winner']['actor_path']}")
        for request in record["needs_confirmation"]:
            print(request["python"])


def main() -> int:
    """Compose packaged YAML, call the Python lesson and return its exit status."""
    from marl_battlegrounds.experiments.hydra import example_main

    return example_main("training_selection", run)


if __name__ == "__main__":
    raise SystemExit(main())
