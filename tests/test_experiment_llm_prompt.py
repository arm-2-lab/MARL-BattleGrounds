"""Check the llm_prompt experiment stage without playing its minute-long game.

Contracts: the packaged settings are TDM-Alpha against TDM-Beta on map 47, seed
0, from tick 12 to tick 120; first_attack_view rejects a tick range with a
negative start or a start after its end before making a game; main writes the
prompt text to prompt.txt and everything else first_attack_view returns to
menu.json, and returns that menu; the committed results/llm_prompt.txt and
llm_prompt.json agree on size and format version, the version is the current
llm.TEXT_VERSION, and the text is in the left frame. The full run's evidence is
the committed result itself, made by the stage's documented command.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from marl_battlegrounds import llm
from marl_battlegrounds.experiments import llm_prompt

RESULTS = Path(llm_prompt.__file__).parent / "results"


def test_settings_compose() -> None:
    with initialize_config_module(
        config_module="marl_battlegrounds.experiments.configs", version_base="1.3"
    ):
        config = compose(config_name="llm_prompt", overrides=["output_dir=o"])
    assert (config.map_id, config.team_a, config.team_b) == (
        47,
        "tdm-alpha",
        "tdm-beta",
    )
    assert (config.seed, config.min_tick, config.max_ticks) == (0, 12, 120)


@pytest.mark.parametrize(("min_tick", "max_ticks"), [(-1, 5), (6, 5)])
def test_rejects_bad_tick_range(min_tick: int, max_ticks: int) -> None:
    with pytest.raises(ValueError, match="min_tick"):
        llm_prompt.first_attack_view(
            47, "tdm-alpha", "tdm-beta", 0, min_tick, max_ticks
        )


def test_main_writes_prompt_and_menu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    found: dict[str, Any] = {
        "prompt": "MARL-BGs actor-text-v1; Frame: left\nline two",
        "tick": 12,
        "slot": 0,
        "reflected": False,
        "text_version": "actor-text-v1",
        "chars": 44,
        "lines": 2,
        "menu": {"move": ["stay"], "combat": ["no_combat"]},
    }
    seen: list[tuple[object, ...]] = []

    def fake(*arguments: object) -> dict[str, Any]:
        seen.append(arguments)
        return dict(found)

    monkeypatch.setattr(llm_prompt, "first_attack_view", fake)
    config = OmegaConf.create(
        {
            "map_id": 47,
            "team_a": "tdm-alpha",
            "team_b": "tdm-beta",
            "seed": 0,
            "min_tick": 12,
            "max_ticks": 120,
            "output_dir": str(tmp_path / "out"),
        }
    )
    menu = llm_prompt.main(config)  # pyright: ignore[reportCallIssue] - Hydra passes a given config straight through.
    assert seen == [(47, "tdm-alpha", "tdm-beta", 0, 12, 120)]
    assert (tmp_path / "out" / "prompt.txt").read_text() == found["prompt"]
    saved = json.loads((tmp_path / "out" / "menu.json").read_text())
    assert saved == menu == {key: found[key] for key in found if key != "prompt"}


def test_committed_example_matches_its_menu() -> None:
    prompt = (RESULTS / "llm_prompt.txt").read_text(encoding="utf-8")
    menu = json.loads((RESULTS / "llm_prompt.json").read_text())
    assert menu["text_version"] == llm.TEXT_VERSION
    assert (len(prompt), prompt.count("\n") + 1) == (menu["chars"], menu["lines"])
    assert prompt.startswith(f"MARL-BGs {llm.TEXT_VERSION}; Frame: left\n")
    assert menu["menu"]["move"][0] == "stay"
