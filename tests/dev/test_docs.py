"""Check that the public Markdown guides stay true to the code.

Contracts: every relative link in every tracked Markdown file names a tracked
file or folder, and every absolute link into this repository maps to a tracked
path; the README quick start and the first Python block of every guide in docs/
run on the CPU (except llm.md, which needs a model server, and
battle_client.md, which opens a browser); every other Python block parses,
every public name it reaches through an import (its own or an earlier block's
in the same file) exists, and every keyword it passes to a public callable is
one of that callable's parameters; every ``python -m
marl_battlegrounds.experiments.<stage>`` command and every Hydra example
command composes with ``--cfg job`` (a sweep composes the first choice of each
override, since Hydra refuses ``--cfg`` with ``-m``), other ``python
examples/*.py`` commands print ``--help``, and every ``python -m
marl_battlegrounds`` command parses with the real argument parser without
running; every example (in any folder under examples/) is listed in
examples/README.md by its path inside examples/, and the guide named there
links it; the README repeats performance.md's headline speed sentence
word for word; CITATION.cff gives pyproject.toml's version, and every
repository and website address is written one way; and docs/api.md is current.
Guide code runs in a temporary folder with JAX on the CPU.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import os
import re
import shlex
import subprocess
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
REPOSITORY_URL = "https://github.com/arm-2-lab/MARL-BattleGrounds"
NOT_RUN = {"llm.md", "battle_client.md"}
HYDRA_EXAMPLES = {
    "mappo_training",
    "qmix_training",
    "pqn_training",
    "curriculum",
    "competitive_training",
    "partner_training",
    "own_learner",
    "training_collection",
    "training_distributions",
    "training_continuation",
    "training_selection",
    "training_study",
    "evaluation",
    "evaluation_results",
    "scenario_evaluation",
    "cross_play_and_zsc",
    "population_selection",
    "llm",
    "canonical_tournament",
    "train_against_released_models",
    "train",
    "evaluate_recurrent_ippo",
    "evaluate_recurrent_mappo",
    "evaluate_feedforward_ippo",
    "evaluate_feedforward_mappo",
    "evaluate_pqn_vdn",
    "evaluate_qmix",
}
FENCE = re.compile(r"^```([A-Za-z]*)[^\n]*\n(.*?)^```", re.MULTILINE | re.DOTALL)
LINK = re.compile(r"\]\(([^)\s]+)\)")


def tracked_files() -> set[str]:
    return set(
        subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.split()
    )


TRACKED = tracked_files()
TRACKED_FOLDERS = {
    str(parent)
    for name in TRACKED
    for parent in Path(name).parents
    if str(parent) != "."
}


def is_tracked(path: Path) -> bool:
    relative = os.path.relpath(path.resolve(), ROOT)
    return relative in TRACKED or relative in TRACKED_FOLDERS


def tracked_markdown() -> list[Path]:
    return [ROOT / name for name in sorted(TRACKED) if name.endswith(".md")]


def guides() -> list[Path]:
    return sorted(path for path in DOCS.glob("*.md") if path.name != "api.md")


def blocks(path: Path, languages: tuple[str, ...]) -> list[str]:
    return [
        body
        for language, body in FENCE.findall(path.read_text())
        if language in languages
    ]


def child_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update({"JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": "", "COLUMNS": "100"})
    return env


@pytest.mark.parametrize(
    "path", tracked_markdown(), ids=lambda path: str(path.relative_to(ROOT))
)
def test_links_resolve(path: Path) -> None:
    for target in LINK.findall(path.read_text()):
        if target.startswith(("#", "mailto:")):
            continue
        location = target.split("#", 1)[0]
        if target.startswith(REPOSITORY_URL):
            match = re.match(
                rf"{re.escape(REPOSITORY_URL)}/(?:blob|tree|raw)/main/(.+)", location
            )
            if match is not None:
                assert is_tracked(ROOT / match.group(1)), f"{path.name}: {target}"
            continue
        if re.match(r"^[a-z]+://", target):
            continue
        assert is_tracked(path.parent / location), f"{path.name}: broken link {target}"


def first_python_blocks() -> Iterator[tuple[str, str]]:
    readme = blocks(ROOT / "README.md", ("python",))
    if readme:
        yield "README.md", readme[0]
    for path in guides():
        found = blocks(path, ("python",))
        if found and path.name not in NOT_RUN:
            yield path.name, found[0]


@pytest.mark.parametrize(
    ("name", "code"),
    list(first_python_blocks()),
    ids=lambda value: value if value.endswith(".md") else "",
)
def test_first_python_block_runs_on_cpu(name: str, code: str, tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=child_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"{name}:\n{result.stderr[-3000:]}"


def resolve(tree: ast.Module) -> dict[str, object]:
    names: dict[str, object] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("marl_battlegrounds"):
                    module = importlib.import_module(alias.name)
                    if alias.asname:
                        names[alias.asname] = module
                    else:
                        names[alias.name.split(".")[0]] = importlib.import_module(
                            alias.name.split(".")[0]
                        )
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "marl_battlegrounds"
        ):
            module = importlib.import_module(node.module or "")
            for alias in node.names:
                if hasattr(module, alias.name):
                    names[alias.asname or alias.name] = getattr(module, alias.name)
                else:
                    names[alias.asname or alias.name] = importlib.import_module(
                        f"{node.module}.{alias.name}"
                    )
    return names


def reach(node: ast.expr, names: dict[str, object]) -> object | None:
    if isinstance(node, ast.Name):
        return names.get(node.id)
    if isinstance(node, ast.Attribute):
        owner = reach(node.value, names)
        if owner is None or not inspect.ismodule(owner):
            return None
        assert hasattr(owner, node.attr), f"{owner.__name__} has no {node.attr}"
        return getattr(owner, node.attr)
    return None


def all_python_blocks() -> Iterator[tuple[str, int, str, str]]:
    for path in [ROOT / "README.md", *guides()]:
        earlier: list[str] = []
        for index, code in enumerate(blocks(path, ("python",))):
            yield path.name, index, code, "\n".join(earlier)
            earlier.append(code)


@pytest.mark.parametrize(
    ("name", "index", "code", "earlier"),
    list(all_python_blocks()),
    ids=lambda value: (
        str(value)
        if isinstance(value, int) or (isinstance(value, str) and value.endswith(".md"))
        else ""
    ),
)
def test_python_blocks_use_real_names_and_keywords(
    name: str, index: int, code: str, earlier: str
) -> None:
    tree = ast.parse(code)
    # A guide's later blocks may reuse imports from its earlier blocks.
    names = {**resolve(ast.parse(earlier)), **resolve(tree)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            reach(node, names)
        if not isinstance(node, ast.Call):
            continue
        target = reach(node.func, names)
        if target is None or not callable(target):
            continue
        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):
            continue
        parameters = signature.parameters.values()
        if any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters):
            continue
        allowed = {item.name for item in parameters}
        for keyword in node.keywords:
            if keyword.arg is not None:
                assert keyword.arg in allowed, f"{name} block {index}: {keyword.arg}"


def commands() -> Iterator[tuple[str, list[str]]]:
    for path in [ROOT / "README.md", *guides()]:
        for body in blocks(path, ("bash", "sh", "shell", "console", "")):
            for line in body.replace("\\\n", " ").splitlines():
                words = shlex.split(line.strip().removeprefix("$ "), comments=True)
                while words and re.match(r"^[A-Z_][A-Z0-9_]*=", words[0]):
                    words = words[1:]
                if words[:3] == ["uv", "run", "--no-sync"]:
                    words = words[3:]
                if words and words[0] in ("python", "python3"):
                    yield path.name, words[1:]


def test_marl_battlegrounds_commands_parse() -> None:
    from marl_battlegrounds._cli import build_parser

    parser = build_parser()
    for name, words in commands():
        if words[:2] == ["-m", "marl_battlegrounds"]:
            try:
                parser.parse_args(words[2:])
            except SystemExit as error:
                pytest.fail(f"{name}: python {' '.join(words)} (exit {error.code})")


@pytest.mark.parametrize(
    ("name", "words"),
    [
        item
        for item in commands()
        if item[1][:1] != ["-m"]
        or item[1][1].startswith("marl_battlegrounds.experiments.")
    ],
    ids=lambda value: value if isinstance(value, str) else " ".join(value)[:60],
)
def test_stage_and_example_commands_compose(
    name: str, words: list[str], tmp_path: Path
) -> None:
    if "-m" in words[2:] or "--multirun" in words:
        # Hydra refuses --cfg with a sweep: compose each override's first choice.
        words = [
            word.split(",")[0] if "=" in word and not set("{[(") & set(word) else word
            for word in words[:2]
            + [w for w in words[2:] if w not in ("-m", "--multirun")]
        ]
    if words[0] == "-m":
        command = [sys.executable, *words, "--cfg", "job"]
    else:
        script = ROOT / words[0]
        assert script.exists(), f"{name}: {words[0]}"
        if Path(words[0]).stem in HYDRA_EXAMPLES:
            command = [sys.executable, str(script), *words[1:], "--cfg", "job"]
        else:
            command = [sys.executable, str(script), "--help"]
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=child_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"{name}: {' '.join(words)}\n{result.stderr[-2000:]}"


def test_every_example_is_indexed_and_linked_from_its_guide() -> None:
    index = (ROOT / "examples" / "README.md").read_text()
    examples = sorted(
        Path(name).relative_to("examples").as_posix()
        for name in TRACKED
        if Path(name).is_relative_to("examples")
        and Path(name).suffix in (".py", ".ipynb")
        and not Path(name).name.startswith("_")
    )
    for example in examples:
        rows = [
            line for line in index.splitlines() if line.startswith(f"| `{example}`")
        ]
        assert rows, f"examples/README.md does not list {example}"
        guide = re.search(r"\(\.\./docs/([a-z_]+\.md)", rows[0])
        assert guide is not None, f"examples/README.md row for {example} names no guide"
        text = (DOCS / guide.group(1)).read_text()
        assert f"examples/{example}" in text, (
            f"{guide.group(1)} does not link {example}"
        )


def test_readme_repeats_the_performance_headline() -> None:
    lines = (DOCS / "performance.md").read_text().splitlines()
    start = next(
        index for index, line in enumerate(lines) if line.startswith("Website: ")
    )
    headline = next(line for line in lines[start + 1 :] if line.strip())
    assert headline in (ROOT / "README.md").read_text()


def test_api_reference_is_current() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/dev/export_api_reference.py", "--check"],
        cwd=ROOT,
        env=child_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_versions_and_addresses_agree() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    citation = (ROOT / "CITATION.cff").read_text()
    assert f'version: "{project["version"]}"' in citation
    for name in sorted(TRACKED):
        if not name.endswith((".md", ".cff", ".toml", ".yml", ".yaml")):
            continue
        text = (ROOT / name).read_text()
        for found in re.findall(r"github\.com/[\w.-]+/MARL-BattleGrounds", text, re.I):
            assert f"https://{found}" == REPOSITORY_URL, f"{name}: {found}"
        for found in re.findall(r"[\w.:/-]*marl-battlegrounds\.com", text):
            assert found in (
                "https://marl-battlegrounds.com",
                "marl-battlegrounds.com",
            ), f"{name}: {found}"
