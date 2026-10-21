"""Generate ``docs/api.md``: every public name of MARL-BGs on one page.

Run from the repository root on the CPU::

    JAX_PLATFORMS=cpu uv run --no-sync python scripts/dev/export_api_reference.py
    JAX_PLATFORMS=cpu uv run --no-sync python \
        scripts/dev/export_api_reference.py --check

The tool imports the installed package, reads each public module's names (its
``__all__``, or else the names the module defines itself) and writes one table
row per name: name, kind, parameter names, the first sentence of its
docstring and the module that defines it. A last table lists every
``python -m marl_battlegrounds`` command. ``--check`` compares the file on disk
with fresh output and changes nothing. ``--json PATH`` also writes the rows as
JSON for the website's API page, so both come from this one tool.

Signatures are read as parameter names only, never as annotation text, so a
name imported only for type checking cannot break the run and the output is the
same on Python 3.12, 3.13 and 3.14. The run fails if any text holds a home
folder (``/home/`` or ``/Users/``) or an internal project label.
"""

from __future__ import annotations

import argparse
import enum
import importlib
import inspect
import json
import re
import sys
import types
from pathlib import Path
from typing import cast

OUTPUT = Path("docs/api.md")

# (module, title). Researcher API first, then built-in method parts, then
# versioned records and rendering adapters.
MODULES = (
    ("marl_battlegrounds", "Top Level (`import marl_battlegrounds as marl_bgs`)"),
    ("marl_battlegrounds.training", "Training (`marl_bgs.training`)"),
    ("marl_battlegrounds.llm", "Language-Model Systems (`marl_bgs.llm`)"),
    ("marl_battlegrounds.entry", "Leaderboard Entries (`entry`)"),
    ("marl_battlegrounds.baselines.config", "Learner Settings (`baselines.config`)"),
    ("marl_battlegrounds.policies", "Built-In Policies (`policies`)"),
    ("marl_battlegrounds.policies.input", "Policy Inputs (`policies.input`)"),
    ("marl_battlegrounds.tasks", "Tasks, Maps and Scenarios (`tasks`)"),
    ("marl_battlegrounds.types", "Types (`types`)"),
    ("marl_battlegrounds.spaces", "Spaces (`spaces`)"),
    ("marl_battlegrounds.experiments.hydra", "Hydra Helpers (`experiments.hydra`)"),
    (
        "marl_battlegrounds.experiments.callbacks",
        "Hydra Callbacks (`experiments.callbacks`)",
    ),
    ("marl_battlegrounds.baselines.actions", "Action Codec (`baselines.actions`)"),
    ("marl_battlegrounds.baselines.inputs", "Learner Inputs (`baselines.inputs`)"),
    ("marl_battlegrounds.baselines.ppo", "PPO Networks (`baselines.ppo`)"),
    ("marl_battlegrounds.baselines.qmix", "QMIX Networks (`baselines.qmix`)"),
    ("marl_battlegrounds.baselines.pqn", "PQN Networks (`baselines.pqn`)"),
    ("marl_battlegrounds.evaluation", "Evaluation Records (`evaluation`)"),
    ("marl_battlegrounds.rendering", "Rendering Records (`rendering`)"),
)

# Helpers the guides teach that no ``__all__`` exports: import them from
# their module.
MODULE_HELPERS = (
    (
        "marl_battlegrounds.training.config",
        (
            "read_config",
            "config_from_dict",
            "config_to_dict",
            "validate_red_zone_depth",
        ),
    ),
    (
        "marl_battlegrounds.training.validation",
        ("validate_random", "run_slot_diagnostic"),
    ),
    ("marl_battlegrounds.training.analysis", ("select_checkpoint",)),
    (
        "marl_battlegrounds.training.shaping",
        ("resolve_reward", "validate_reward", "reward_adjustments"),
    ),
    ("marl_battlegrounds.evaluation.tournament_assets", ("prepare_tournament_assets",)),
    (
        "marl_battlegrounds.training.learner",
        ("init_learner", "build_ppo_batch", "update_learner", "validate_learner"),
    ),
    (
        "marl_battlegrounds.training.qmix_learner",
        ("init_qmix_learner", "update_qmix_learner", "validate_qmix_learner"),
    ),
    (
        "marl_battlegrounds.training.pqn_learner",
        ("init_pqn_learner", "update_pqn_learner", "validate_pqn_learner"),
    ),
    ("marl_battlegrounds.training.checkpoints", ("save_checkpoint", "export_system")),
)

# First sentences said plainly where the docstring itself must keep an
# internal label (saved runs record those docstrings byte for byte).
DOC_OVERRIDES = {
    "make_ppo_system": "Wrap one PPO actor as a JAX System with sampled actions.",
    "make_recurrent_mappo_system": (
        "Wrap actor parameters as a JAX System, with sampled actions."
    ),
    "make_qmix_system": (
        "Wrap Q-network variables as a JAX System with epsilon-greedy actions."
    ),
    "make_pqn_system": (
        "Wrap PQN network variables as a JAX System with epsilon-greedy actions."
    ),
    "read_calls": "Stream actor rows from a saved run folder, a call folder or a file.",
    "run_slot_diagnostic": (
        "Run the predeclared, frozen two-model slot comparison through evaluation."
    ),
}
INTERNAL = re.compile(
    r"\bM\d+[a-z]?\b|DevClient|\bPacket\b|milestone|\bamendment A\d+", re.IGNORECASE
)
EXIT_CODES = (
    "Every command returns 0 on success, 1 when the work fails for a stated "
    "reason (printed to standard error), 2 for bad arguments and 130 when "
    "interrupted."
)


def first_sentence(doc: str | None) -> str:
    """Return the first sentence of a docstring's first paragraph.

    Parameters
    ----------
    doc : str or None
        A cleaned docstring (``inspect.getdoc``); None or empty gives "".

    Returns
    -------
    str
        The text up to the first ``.``, ``!`` or ``?`` followed by a space or
        the end, with line breaks joined.
    """
    if not doc:
        return ""
    paragraph = " ".join(
        line.strip() for line in doc.strip().split("\n\n")[0].splitlines()
    )
    match = re.search(r"(.+?[.!?])(\s|$)", paragraph)
    return (match.group(1) if match else paragraph).strip()


def kind_of(value: object) -> str:
    """Name what a public object is.

    Parameters
    ----------
    value : object
        A public attribute.

    Returns
    -------
    str
        One of ``module``, ``exception``, ``enum``, ``class``, ``function`` or
        ``constant``.
    """
    if isinstance(value, types.ModuleType):
        return "module"
    if isinstance(value, type):
        if issubclass(value, BaseException):
            return "exception"
        if issubclass(value, enum.Enum):
            return "enum"
        return "class"
    if inspect.isroutine(value) or callable(value):
        return "function"
    return "constant"


def parameter_names(value: object) -> str:
    """List a callable's parameters as names, without annotations or values.

    Parameters
    ----------
    value : object
        A function or class.

    Returns
    -------
    str
        For example ``system, opponent, *, num_episodes, maps=…``: ``*`` marks
        where keyword-only parameters start, ``=…`` marks a default, and
        ``*args``/``**kwargs`` keep their stars. ``self`` and private
        parameters (names starting with ``_``) are left out. Empty when the
        signature cannot be read.
    """
    try:
        if sys.version_info >= (3, 14):
            import annotationlib

            signature = inspect.signature(
                value, annotation_format=annotationlib.Format.STRING
            )  # type: ignore[arg-type]
        else:
            signature = inspect.signature(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, NameError):
        return ""
    parts: list[str] = []
    keyword_marked = False
    for parameter in signature.parameters.values():
        if parameter.name == "self" or parameter.name.startswith("_"):
            continue
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            parts.append(f"*{parameter.name}")
            keyword_marked = True
            continue
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            parts.append(f"**{parameter.name}")
            continue
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY and not keyword_marked:
            parts.append("*")
            keyword_marked = True
        default = "" if parameter.default is inspect.Parameter.empty else "=…"
        parts.append(f"{parameter.name}{default}")
    return ", ".join(parts)


def public_names(module: types.ModuleType) -> list[str]:
    """Return a module's public names in their declared order.

    Parameters
    ----------
    module : types.ModuleType
        An imported public module.

    Returns
    -------
    list of str
        ``__all__`` when present; otherwise the non-underscore names whose
        objects the module defines itself, sorted.
    """
    declared = getattr(module, "__all__", None)
    if declared is not None:
        return list(declared)
    return sorted(
        name
        for name, value in vars(module).items()
        if not name.startswith("_")
        and getattr(value, "__module__", None) == module.__name__
    )


def short_module(name: str) -> str:
    """Return a module path without the package prefix (``marl_bgs`` for the root)."""
    return name.replace("marl_battlegrounds.", "").replace(
        "marl_battlegrounds", "marl_bgs"
    )


def collect() -> list[dict[str, str]]:
    """Import every listed module and return one row per public name.

    Returns
    -------
    list of dict
        Keys ``section``, ``name``, ``kind``, ``parameters``, ``summary`` and
        ``defined_in``. A name listed twice in one section appears once.

    Raises
    ------
    SystemExit
        Generated text holds a home folder (``/home/`` or ``/Users/``) or an
        internal label.
    """
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(section: str, name: str, value: object, fallback_owner: str) -> None:
        if (section, name) in seen:
            return
        seen.add((section, name))
        kind = kind_of(value)
        owner = str(
            value.__name__
            if isinstance(value, types.ModuleType)
            else getattr(value, "__module__", None) or fallback_owner
        )
        summary = DOC_OVERRIDES.get(name) or (
            "" if kind == "constant" else first_sentence(inspect.getdoc(value))
        )
        rows.append(
            {
                "section": section,
                "name": name,
                "kind": kind,
                "parameters": parameter_names(value)
                if kind in ("function", "class", "exception")
                else "",
                "summary": summary,
                "defined_in": short_module(owner),
            }
        )

    for module_name, title in MODULES:
        module = importlib.import_module(module_name)
        for name in public_names(module):
            add(title, name, getattr(module, name), module_name)
    for module_name, names in MODULE_HELPERS:
        module = importlib.import_module(module_name)
        for name in names:
            add(
                f"Module Helpers (`from {module_name} import ...`)",
                name,
                getattr(module, name),
                module_name,
            )
    text = json.dumps(rows)
    # No search for the bare user name: on GitHub's machines it is "runner",
    # an ordinary word here (training.runner). A leaked name sits in a path.
    for forbidden in ("/home/", "/Users/"):
        if forbidden in text:
            raise SystemExit(
                f"refusing to write: generated text contains {forbidden!r}"
            )
    for row in rows:
        if INTERNAL.search(row["summary"]):
            where = f"{row['defined_in']}.{row['name']}"
            raise SystemExit(f"internal label in {where}: {row['summary']!r}")
    return rows


def commands() -> list[tuple[str, str]]:
    """Return every ``python -m marl_battlegrounds`` command with its help line.

    Returns
    -------
    list of (str, str)
        Command words (subcommands included) and their one-line help.
    """
    from marl_battlegrounds._cli import build_parser

    found: list[tuple[str, str]] = []

    def walk(parser: argparse.ArgumentParser, path: str) -> None:
        for action in parser._actions:  # pyright: ignore[reportPrivateUsage]
            if isinstance(action, argparse._SubParsersAction):  # pyright: ignore[reportPrivateUsage]
                subcommands = cast(
                    "argparse._SubParsersAction[argparse.ArgumentParser]", action
                )
                helps = {
                    choice.dest: choice.help or ""
                    for choice in subcommands._choices_actions  # pyright: ignore[reportPrivateUsage]
                }
                for name, child in subcommands.choices.items():
                    found.append((f"{path} {name}", helps.get(name, "")))
                    walk(child, f"{path} {name}")

    walk(build_parser(), "python -m marl_battlegrounds")
    return found


def cell(text: str) -> str:
    """Escape a value for one Markdown table cell (pipes and line breaks)."""
    return text.replace("|", "\\|").replace("\n", " ")


def render(rows: list[dict[str, str]]) -> str:
    """Turn rows and the command list into the Markdown page.

    Parameters
    ----------
    rows : list of dict
        Output of `collect`.

    Returns
    -------
    str
        The complete ``docs/api.md`` text, ending in one newline.
    """
    lines = [
        "# API Reference",
        "",
        "Every public name of MARL-BGs on one page, with the parameter names of each",
        "callable and the first sentence of its docstring. The docstrings hold the",
        "full contracts: `help(marl_bgs.evaluate)` in Python, or your editor's hover.",
        "",
        "Website: https://marl-battlegrounds.com/#api",
        "",
        "Generated by `scripts/dev/export_api_reference.py`; do not edit by hand.",
        "",
    ]
    section = None
    for row in rows:
        if row["section"] != section:
            section = row["section"]
            lines += [
                "",
                f"## {section}",
                "",
                "| Name | Kind | Parameters | What It Does | Defined In |",
                "|---|---|---|---|---|",
            ]
        parameters = f"`{cell(row['parameters'])}`" if row["parameters"] else ""
        cells = (
            f"`{row['name']}`",
            row["kind"],
            parameters,
            cell(row["summary"]),
            f"`{row['defined_in']}`",
        )
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Commands",
        "",
        EXIT_CODES,
        "",
        "| Command | What It Does |",
        "|---|---|",
    ]
    for command, help_text in commands():
        lines.append(f"| `{command}` | {cell(help_text)} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    """Write or check ``docs/api.md``; return 1 when ``--check`` finds it stale."""
    parser = argparse.ArgumentParser(
        description="Generate docs/api.md from the installed package."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare docs/api.md with fresh output; write nothing",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="also write the rows as JSON to this path (for the website)",
    )
    arguments = parser.parse_args()
    rows = collect()
    text = render(rows)
    if arguments.json is not None:
        arguments.json.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n")
    if arguments.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != text:
            print(
                "docs/api.md is stale; rerun scripts/dev/export_api_reference.py",
                file=sys.stderr,
            )
            return 1
        return 0
    OUTPUT.write_text(text)
    print(f"wrote {OUTPUT}: {len(rows)} names, {text.count(chr(10))} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
