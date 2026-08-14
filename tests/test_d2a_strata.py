"""D2A tabakali gorulmemis-kutle sozlesmesi testleri.

Belgedeki "sabit sira / gizlenmez / null kalir / 1 etmeli" maddelerini kod
duzeyinde bagli tutar.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ml.lodo_generalization import _fixture  # noqa: E402
from ml.unseen_category_strata import (  # noqa: E402
    ANY_COLUMN,
    CROSS_ORDER,
    LABEL_ORDER,
    SCENARIO_ORDER,
    StrataGuardError,
    build_strata,
    check_allocation_invariant,
    measure,
)


def _source():
    return _fixture("openplc", "phase", 6, 90)


def _target():
    """Hedefe kaynakta bulunmayan bir kategori enjekte edilir."""
    target = _fixture("macyste", "phase", 3, 90)
    target.loc[target.index[:40], "flow_state"] = "rare_state"
    return target


def _rows(report, panel=None, schema="physical_proxy", column=ANY_COLUMN):
    return [r for r in report["rows"]
            if r["schema"] == schema and r["column"] == column
            and (panel is None or r["panel"] == panel)]


# ---------------------------------------------------------------------------
# §2 -- tabaka kurulumu ve sabit sira
# ---------------------------------------------------------------------------
def test_fourteen_strata_in_the_frozen_order():
    strata, _ = build_strata(_target())
    assert len(strata) == 14

    keys = [(s["panel"], s["scenario_family"], s["attack_label"]) for s in strata]
    expected = (
        [("scenario", sc, "all") for sc in SCENARIO_ORDER]
        + [("label", "all", lb) for lb in LABEL_ORDER]
        + [("scenario_x_label", sc, lb) for sc, lb in CROSS_ORDER]
    )
    assert keys == expected


def test_cross_cells_are_scenario_major():
    assert CROSS_ORDER[:4] == (("normal", "attack"), ("normal", "non_attack"),
                               ("fault", "attack"), ("fault", "non_attack"))


def test_unknown_scenario_family_halts():
    target = _target()
    target.loc[target.index[:5], "scenario"] = "macyste_sabotage"
    with pytest.raises(StrataGuardError, match="other"):
        build_strata(target)


def test_unexpected_raw_label_halts():
    target = _target()
    target.loc[target.index[:5], "label"] = "suspicious"
    with pytest.raises(StrataGuardError, match="ham etiket"):
        build_strata(target)


def test_missing_label_halts_instead_of_falling_into_non_attack():
    target = _target()
    target.loc[target.index[:5], "label"] = np.nan
    with pytest.raises(StrataGuardError, match="eksik"):
        build_strata(target)


def test_scenario_and_label_panels_partition_the_target():
    target = _target()
    strata, _ = build_strata(target)
    for panel in ("scenario", "label", "scenario_x_label"):
        masks = [s["mask"] for s in strata if s["panel"] == panel]
        stacked = np.vstack([np.asarray(m, dtype=bool) for m in masks])
        assert stacked.sum(axis=0).max() <= 1, f"{panel}: hucreler ortusuyor"
        assert stacked.any(axis=0).all(), f"{panel}: kapsanmayan satir var"


# ---------------------------------------------------------------------------
# §2.3 -- bos hucreler gizlenmez
# ---------------------------------------------------------------------------
def test_empty_cells_are_reported_with_zero_and_null():
    report = measure(_source(), _target(), ["physical_proxy"])
    cross = _rows(report, "scenario_x_label")
    assert len(cross) == 8, "bos hucreler dusurulmus"

    empty = [r for r in cross if r["n_all"] == 0]
    assert empty, "fixture'da yapisal-sifir hucre bekleniyordu"
    for row in empty:
        assert row["n_unseen"] == 0
        assert row["mass_all"] is None      # sifir DEGIL, null
        assert row["enrichment_all"] is None


def test_every_stratum_appears_for_every_schema():
    report = measure(_source(), _target())
    for schema in ("flow", "protocol", "physical_strict", "physical_proxy"):
        assert len(_rows(report, schema=schema)) == 14


# ---------------------------------------------------------------------------
# §3.2 -- allocation degismezligi
# ---------------------------------------------------------------------------
def test_allocation_sums_to_one_within_each_panel():
    report = measure(_source(), _target())
    assert report["allocation_invariant"]["passed"] is True
    assert report["allocation_invariant"]["groups_checked"] > 0

    for panel in ("scenario", "label", "scenario_x_label"):
        values = [r["allocation"] for r in _rows(report, panel)
                  if r["allocation"] is not None]
        assert values
        assert abs(sum(values) - 1.0) < 1e-9


def test_allocation_invariant_catches_a_dropped_cell():
    broken = [
        {"panel": "scenario", "schema": "flow", "column": ANY_COLUMN,
         "allocation": 0.5},
        {"panel": "scenario", "schema": "flow", "column": ANY_COLUMN,
         "allocation": 0.3},
    ]
    with pytest.raises(AssertionError, match="allocation toplami"):
        check_allocation_invariant(broken)


def test_allocation_is_null_when_no_unseen_rows_at_all():
    """Hicbir gorulmemis satir yoksa allocation sifir degil null olmali."""
    source, target = _source(), _fixture("macyste", "phase", 3, 90)
    report = measure(source, target, ["flow"])
    flow_rows = _rows(report, schema="flow", column="flow_state")
    assert all(r["n_unseen"] == 0 for r in flow_rows)
    assert all(r["allocation"] is None for r in flow_rows)


# ---------------------------------------------------------------------------
# §3.1 / §5 -- paydalar ve alan sozlesmesi
# ---------------------------------------------------------------------------
def test_union_rows_have_no_observed_denominator():
    report = measure(_source(), _target(), ["physical_proxy"])
    for row in _rows(report):
        assert row["n_observed"] is None
        assert row["mass_observed"] is None


def test_column_rows_carry_both_denominators():
    report = measure(_source(), _target(), ["physical_proxy"])
    rows = _rows(report, column="flow_state")
    filled = [r for r in rows if r["n_all"] > 0]
    assert filled
    for row in filled:
        assert row["n_observed"] is not None
        assert row["missing_basis"] is not None
        assert row["n_observed"] <= row["n_all"]


def test_run_sensitivity_only_on_union_rows():
    report = measure(_source(), _target(), ["physical_proxy"])
    for row in _rows(report):
        assert row["n_runs"] is not None
    for row in _rows(report, column="modbus_function"):
        assert row["n_runs"] is None
        assert row["run_mass_median"] is None
        assert row["run_mass_min"] is None
        assert row["run_mass_max"] is None


def test_run_spread_brackets_the_median():
    report = measure(_source(), _target(), ["physical_proxy"])
    for row in _rows(report):
        if row["n_runs"]:
            assert row["run_mass_min"] <= row["run_mass_median"] <= row["run_mass_max"]


def test_required_output_fields_present():
    required = {"panel", "scenario_family", "attack_label", "schema", "column",
                "n_all", "n_observed", "n_unseen", "mass_all", "mass_observed",
                "allocation", "overall_mass_all", "enrichment_all", "n_runs",
                "run_mass_median", "run_mass_min", "run_mass_max"}
    report = measure(_source(), _target(), ["physical_proxy"])
    for row in report["rows"]:
        assert required <= set(row)


# ---------------------------------------------------------------------------
# Olcu anlamlari
# ---------------------------------------------------------------------------
def test_mass_and_allocation_are_different_questions():
    """mass = tabaka ICI oran; allocation = toplamin DAGILIMI. Ayni degiller."""
    report = measure(_source(), _target(), ["physical_proxy"])
    rows = [r for r in _rows(report, "scenario_x_label")
            if r["n_unseen"] > 0 and r["n_all"] > 0]
    assert rows
    assert any(abs(r["mass_all"] - r["allocation"]) > 1e-9 for r in rows), \
        "iki olcu ayirt edilemiyor; test kurulumu zayif"


def test_enrichment_is_stratum_mass_over_overall_mass():
    report = measure(_source(), _target(), ["physical_proxy"])
    for row in report["rows"]:
        if row["enrichment_all"] is None or not row["overall_mass_all"]:
            continue
        assert row["enrichment_all"] == pytest.approx(
            row["mass_all"] / row["overall_mass_all"])


def test_overall_mass_is_stratum_independent():
    """overall_* paydasi hedefin TAMAMI uzerinden; tabakadan bagimsiz."""
    report = measure(_source(), _target(), ["physical_proxy"])
    for column in ("flow_state", ANY_COLUMN):
        values = {r["overall_mass_all"] for r in _rows(report, column=column)}
        assert len(values) == 1


def test_unseen_counts_add_up_across_a_partition():
    report = measure(_source(), _target(), ["physical_proxy"])
    total = sum(r["n_unseen"] for r in _rows(report, "scenario_x_label"))
    scenario_total = sum(r["n_unseen"] for r in _rows(report, "scenario"))
    label_total = sum(r["n_unseen"] for r in _rows(report, "label"))
    assert total == scenario_total == label_total
