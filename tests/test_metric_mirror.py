"""Check the catalog's mirror rule and the summary weight of every average.

Contracts: mirror_metric gives every one of the full-metric columns a twin that
exists, describes the same measurement (unit, family, stem, scope, class,
status and roles) and mirrors back to it; only score_difference changes sign;
unknown names raise ValueError. Every column with denominator text has a
summary_weight or a listed reason in SUMMARY_WEIGHT_MISSING, every weight is an
existing count, health, tick or observation column, and the weight of a twin is
the twin of the weight. Named links are checked by hand. On the committed real
fixture (tests/fixtures/metric_weights, made by
scripts/dev/generate_metric_weight_fixture.py with two Hunters on one team,
missing classes, dead agents and effects present at the start), value times
weight equals each linked column's exact saved numerator (the directed amount,
or the numerator named for each hand rule), the participation shares that
divide by all of a team's kills and rescues included (named links pin those
four, because in the fixture's games every kill had a Basic helper, which
cannot tell all kills from Basic kills); Trap break rate divides
by Trap periods, which differ from Trap applications there; and the fixture's
schema version is the current one. The metric dictionary export carries summary_weight.
"""

# pyright: reportPrivateUsage=false
import csv
import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from marl_battlegrounds.evaluation.metric_catalog import (
    _SUMMARY_WEIGHT_RULES,
    DIRECTED_METRICS,
    FULL_METRIC_NAMES,
    METRIC_COLUMNS,
    METRIC_COLUMNS_BY_NAME,
    METRIC_SCHEMA_VERSION,
    SUMMARY_WEIGHT_MISSING,
    ULTIMATE_CLASS_NAMES,
    mirror_metric,
)

FIXTURE = Path(__file__).parent / "fixtures" / "metric_weights"
# Averages whose numerator has no column of its own, so value times weight
# cannot be compared with a saved number.
NO_SAVED_NUMERATOR = {
    "ally_distance_mean",
    "trap_mean_remaining_steps_at_break",
    "mean_agents_per_respawn_wave",
    "focus_fire_concentration",
}


def _prefix(name: str) -> str:
    column = METRIC_COLUMNS_BY_NAME[name]
    return name[: len(name) - len(column.stem)]


def test_every_column_has_a_mirrored_twin() -> None:
    changed_sign = []
    for column in METRIC_COLUMNS:
        twin_name, sign = mirror_metric(column.name)
        twin = METRIC_COLUMNS_BY_NAME[twin_name]
        assert mirror_metric(twin_name) == (column.name, sign)
        assert (
            twin.unit,
            twin.family,
            twin.stem,
            twin.scope,
            twin.required_class_id,
            twin.status_channel,
            twin.subject_role,
            twin.recipient_role,
        ) == (
            column.unit,
            column.family,
            column.stem,
            column.scope,
            column.required_class_id,
            column.status_channel,
            column.subject_role,
            column.recipient_role,
        )
        if sign == -1:
            changed_sign.append(column.name)
    assert changed_sign == ["score_difference"]


@pytest.mark.parametrize(
    ("name", "twin"),
    [
        ("team_a_kills", "team_b_kills"),
        ("episode_length", "episode_length"),
        ("agent_2_to_agent_7_damage_done", "agent_7_to_agent_2_damage_done"),
        (
            "team_a_to_agent_7_basic_applications",
            "team_b_to_agent_2_basic_applications",
        ),
        (
            "agent_0_and_agent_3_ally_distance_mean",
            "agent_5_and_agent_8_ally_distance_mean",
        ),
        ("agent_9_deaths", "agent_4_deaths"),
    ],
)
def test_mirror_examples(name: str, twin: str) -> None:
    assert mirror_metric(name) == (twin, 1)


def test_mirror_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="Unknown metric column"):
        mirror_metric("team_a_policy")


def test_every_average_has_a_weight_or_a_reason() -> None:
    stems = {(column.scope, column.stem) for column in METRIC_COLUMNS}
    assert set(SUMMARY_WEIGHT_MISSING) <= stems
    for column in METRIC_COLUMNS:
        missing = (column.scope, column.stem) in SUMMARY_WEIGHT_MISSING
        if column.denominator is not None:
            assert (column.summary_weight is None) == missing, column.name
        else:
            assert column.summary_weight is None, column.name
        if column.summary_weight is not None:
            weight = METRIC_COLUMNS_BY_NAME[column.summary_weight]
            assert weight.unit not in ("fraction", "ratio"), column.name
            assert weight.summary_weight is None
            twin = METRIC_COLUMNS_BY_NAME[mirror_metric(column.name)[0]]
            assert twin.summary_weight == mirror_metric(column.summary_weight)[0]


@pytest.mark.parametrize(
    ("name", "weight"),
    [
        ("team_a_multi_contributor_kill_fraction", "team_a_kills"),
        ("team_b_trap_break_rate", "team_b_trap_intervals"),
        ("team_a_rescue_rate", "team_a_rescue_opportunities"),
        ("team_a_mage_aura_coverage", "team_a_mage_aura_eligible_steps"),
        ("team_b_ally_distance_mean", "team_b_ally_distance_observations"),
        ("agent_7_death_fraction", "team_b_deaths"),
        ("agent_2_basic_kill_participation", "team_a_kills"),
        ("agent_7_ultimate_kill_participation", "team_b_kills"),
        ("agent_4_basic_rescue_participation", "team_a_rescues"),
        ("agent_9_ultimate_rescue_participation", "team_b_rescues"),
        ("agent_4_effective_healing_fraction", "agent_4_healing_done"),
        ("agent_0_to_agent_6_damage_done_allocation_fraction", "agent_0_damage_done"),
        ("agent_0_to_agent_6_damage_done_fraction", "agent_6_damage_received"),
        ("agent_0_kill_contributions_per_death", None),
        ("agent_0_to_agent_1_mage_aura_coverage", None),
        ("team_a_kills", None),
    ],
)
def test_named_links(name: str, weight: str | None) -> None:
    assert METRIC_COLUMNS_BY_NAME[name].summary_weight == weight


# The exact saved numerator of each hand-written weight rule, by stem, read with
# the same subject prefix; None where the numerator has no column of its own.
NUMERATORS: dict[str, str | None] = {
    "action_acceptance_rate": "actions_accepted",
    "effective_healing_fraction": "effective_healing_done",
    "basic_effective_healing_fraction": "basic_effective_healing_done",
    "ultimate_effective_healing_fraction": "ultimate_effective_healing_done",
    "excess_healing_fraction": "excess_healing",
    "basic_excess_healing_fraction": "basic_excess_healing",
    "ultimate_excess_healing_fraction": "ultimate_excess_healing",
    "priest_healing_received_fraction": "priest_healing_received",
    "regeneration_healing_received_fraction": "regenerated_healing",
    "effective_priest_healing_received_fraction": "effective_priest_healing_received",
    "excess_priest_healing_received_fraction": None,
    "mage_aura_coverage": "mage_aura_covered_steps",
    "warrior_aura_coverage": "warrior_aura_covered_steps",
    "freedom_protection_fraction": "freedom_protected_steps",
    "trap_break_rate": "trap_breaks",
    "rescue_rate": "rescues",
    "basic_rescue_fraction": "basic_rescues",
    "ultimate_rescue_fraction": "ultimate_rescues",
    "solo_kill_fraction": "solo_kills",
    "burst_damage_fraction": "burst_damage",
    "damage_received_fraction": "damage_received",
    "healing_received_fraction": "healing_received",
    "death_fraction": "deaths",
    "dead_step_fraction": "dead_steps",
    "red_zone_kill_participation": "red_zone_kill_contributions",
    "red_zone_death_fraction": "red_zone_deaths",
    "basic_kill_participation": "basic_kill_contributions",
    "ultimate_kill_participation": "ultimate_kill_contributions",
    "basic_rescue_participation": "basic_rescue_contributions",
    "ultimate_rescue_participation": "ultimate_rescue_contributions",
    **{
        stem: stem.removesuffix("_fraction") + "s"
        for stem in (
            "single_contributor_kill_fraction",
            "multi_contributor_kill_fraction",
            "basic_kill_fraction",
            "ultimate_kill_fraction",
            *(f"{name}_basic_kill_fraction" for name in ULTIMATE_CLASS_NAMES),
            *(f"{name}_ultimate_kill_fraction" for name in ULTIMATE_CLASS_NAMES[1:]),
        )
    },
}


def _numerator(name: str) -> str | None:
    # Names the saved column that value times weight must equal, if any.
    column = METRIC_COLUMNS_BY_NAME[name]
    prefix = _prefix(name)
    if (column.scope, column.stem) in _SUMMARY_WEIGHT_RULES:
        stem = NUMERATORS.get(column.stem)
        return None if stem is None else prefix + stem
    for metric in DIRECTED_METRICS:
        if column.stem in (metric.allocation, metric.contribution):
            amount = (
                metric.key
                if column.scope == "source_recipient"
                else metric.source_total
            )
            return prefix + amount
    return None


def test_every_hand_rule_names_its_numerator() -> None:
    stems = {stem for _, stem in _SUMMARY_WEIGHT_RULES}
    assert stems - NO_SAVED_NUMERATOR == set(NUMERATORS)
    for name, column in METRIC_COLUMNS_BY_NAME.items():
        numerator = _numerator(name) if column.summary_weight else None
        if numerator is not None:
            assert numerator in METRIC_COLUMNS_BY_NAME, (name, numerator)


def test_fixture_links_reproduce_their_exact_numerators() -> None:
    record = json.loads((FIXTURE / "fixture.json").read_text())
    assert record["metric_schema_version"] == METRIC_SCHEMA_VERSION
    with (FIXTURE / "full_metrics.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 4
    assert set(FULL_METRIC_NAMES) <= set(rows[0])

    def values(name: str) -> np.ndarray:
        return np.array([float(row[name]) if row[name] else np.nan for row in rows])

    checked: set[str] = set()
    for column in METRIC_COLUMNS:
        numerator = _numerator(column.name) if column.summary_weight else None
        if numerator is None:
            continue
        value = values(column.name)
        weight = values(cast(str, column.summary_weight))
        used = (weight > 0) & np.isfinite(value)
        if not np.any(value[used] != 0):
            continue
        np.testing.assert_allclose(
            value[used] * weight[used],
            values(numerator)[used],
            rtol=2e-6,
            atol=1e-3,
            err_msg=column.name,
        )
        checked.add(column.stem)
    assert len(checked) > 120
    # Hand rules with data in the fixture are all checked, the participation
    # shares that divide by all kills and rescues included.
    assert {
        "basic_kill_participation",
        "ultimate_kill_participation",
        "trap_break_rate",
        "death_fraction",
        "damage_received_fraction",
        "mage_aura_coverage",
    } <= checked
    rate, periods = values("team_a_trap_break_rate"), values("team_a_trap_intervals")
    np.testing.assert_allclose(rate * periods, values("team_a_trap_breaks"), rtol=1e-6)
    assert np.any(periods != values("team_a_hunter_trap_applications"))


def test_dictionary_export_carries_summary_weight() -> None:
    from scripts.dev.export_metric_dictionary import dictionary_csv

    rows = list(csv.DictReader(dictionary_csv().splitlines()))
    by_name = {row["column"]: row for row in rows}
    assert by_name["team_a_multi_contributor_kill_fraction"]["summary_weight"] == (
        "team_a_kills"
    )
    assert by_name["team_a_kills"]["summary_weight"] == ""
    assert by_name["run_id"]["summary_weight"] == ""
