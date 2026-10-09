"""Check the speed experiment stage's settings, summary and argument checks.

Contracts: the packaged settings time test map 47 at 32, 64, 128, 512 and 1024
games with 500-tick calls and 5 warm repeats, then train recurrent MAPPO in pure
self-play for 40 updates of 1024 games by 32 ticks with 3 warm-up updates;
speed.md lists the device, one row per batch size and, when training ran, its
rate, compile time and memory in GB; training_speed refuses a warm-up that
leaves no timed update before it trains. No JAX program is compiled.
"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_module

from marl_battlegrounds.experiments import speed


def test_default_settings_compose() -> None:
    with initialize_config_module(
        config_module="marl_battlegrounds.experiments.configs", version_base="1.3"
    ):
        config = compose(config_name="speed", overrides=["output_dir=out"])
    assert config.map_id == 47
    assert list(config.batch_sizes) == [32, 64, 128, 512, 1024]
    assert (config.steps, config.repeats) == (500, 5)
    assert (config.train_method, config.train_num_envs) == ("mappo", 1024)
    assert (config.train_rollout_length, config.train_blocks) == (32, 40)
    assert config.warmup_blocks == 3


def test_summary_lists_rates_and_memory(tmp_path: Path) -> None:
    row = {
        "num_envs": 1024,
        "game_steps_per_second": 50067,
        "agent_steps_per_second": 500667,
        "first_call_seconds": 18.1,
        "warm_spread_percent": 0.2,
    }
    training = {
        "method": "mappo",
        "num_envs": 1024,
        "rollout_length": 32,
        "game_steps_per_second": 29147,
        "warm_median_seconds": 1.1242,
        "first_block_seconds": 33.81,
        "peak_jax_bytes": 5_215_163_904,
        "jax_pool_bytes": 8_994_684_928,
        "peak_ram_bytes": 4_325_797_888,
    }
    speed.write_summary("GPU", [row], training, tmp_path / "speed.md")
    text = (tmp_path / "speed.md").read_text(encoding="utf-8")
    assert "Device: GPU." in text
    assert "| 1,024 | 50,067 | 500,667 | 18.1 s | 0.2% |" in text
    assert "29,147 game steps per second" in text
    assert "Peak JAX memory in use 5.2 GB (allocator pool 9.0 GB); peak RAM 4.3 GB" in (
        text
    )

    speed.write_summary("GPU", [row], None, tmp_path / "skipped.md")
    assert "Training" not in (tmp_path / "skipped.md").read_text(encoding="utf-8")


def test_training_refuses_warmup_without_timed_updates(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="warmup_blocks"):
        speed.training_speed(
            tmp_path / "run",
            method="mappo",
            num_envs=32,
            rollout_length=32,
            blocks=3,
            warmup_blocks=3,
        )
    assert not (tmp_path / "run").exists()
