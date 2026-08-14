"""D3 skor kesikligi sozlesmesi testleri.

Belgedeki tanimlari (plato, taban oran, esitlik) ve karar tablosunu kod
duzeyinde bagli tutar. On-kayitli bir esik sonuctan sonra gevsetilirse burasi
kirilir.
"""
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ml.lodo_generalization import _fixture  # noqa: E402
from ml.score_discreteness_diagnostic import (  # noqa: E402
    DECISION_INTERVAL,
    ENRICHMENT_SUPPORT_MEDIAN,
    KEFF_RATIO_FALSIFY,
    KEFF_RATIO_SUPPORT,
    MIN_EVALUABLE_SEEDS,
    MIN_K_I,
    MIN_N_I,
    MIN_POINT_ROWS,
    SCHEMA,
    TREE_PANEL,
    D3GuardError,
    _spearman,
    check_universe,
    combined_verdict,
    enrichments,
    plateau,
    score_seed,
    support_profile,
)


# ---------------------------------------------------------------------------
# On-kayitli sabitler (belge §2, §4.1, §5)
# ---------------------------------------------------------------------------
def test_frozen_constants():
    assert SCHEMA == "physical_proxy"
    assert DECISION_INTERVAL == (0.1238, 0.8795)
    assert TREE_PANEL == (25, 50, 100, 200)
    assert MIN_K_I == 10
    assert MIN_EVALUABLE_SEEDS == 10
    assert MIN_POINT_ROWS == 50
    assert MIN_N_I == 100
    assert ENRICHMENT_SUPPORT_MEDIAN == 2.0
    assert KEFF_RATIO_SUPPORT == 2.0
    assert KEFF_RATIO_FALSIFY == 1.25


def test_base_mass_at_the_gate_does_not_exceed_one_tenth():
    """Kapinin gerekcesi: K_I = MIN_K_I iken taban kutle 0.10'u ASMAZ."""
    assert 1.0 / MIN_K_I <= 0.10


# ---------------------------------------------------------------------------
# §3.1 -- skor esitligi ve destek
# ---------------------------------------------------------------------------
def test_support_profile_counts_distinct_scores_bitwise():
    scores = np.array([0.2, 0.2, 0.5, 0.5, 0.5, 0.9])
    profile = support_profile(scores)
    assert profile["K"] == 3
    assert profile["n"] == 6
    assert profile["largest_point_mass"] == pytest.approx(0.5)
    assert [p["n"] for p in profile["point_masses"]] == [2, 3, 1]
    assert all("score_hex" in p for p in profile["point_masses"])


def test_interval_counts_use_the_open_interval():
    low, high = DECISION_INTERVAL
    scores = np.array([low, 0.5, 0.6, high])
    profile = support_profile(scores)
    assert profile["K"] == 4
    assert profile["K_I"] == 2          # sinirlar I icinde SAYILMAZ
    assert profile["N_I"] == 2


def test_k_eff_equals_k_only_under_uniform_mass():
    uniform = support_profile(np.array([0.2, 0.4, 0.6, 0.8]))
    assert uniform["K_eff"] == pytest.approx(uniform["K"])

    skewed = support_profile(np.array([0.2] * 97 + [0.4, 0.6, 0.8]))
    assert skewed["K"] == 4
    assert skewed["K_eff"] < 2.0, "etkin destek ham K'yi taklit ediyor"


# ---------------------------------------------------------------------------
# §3.3 -- plato
# ---------------------------------------------------------------------------
def test_plateau_is_right_closed():
    """`>=` karar kurali: sag uc plato ICINDE, sol uc DISINDA."""
    unique = np.array([0.30, 0.40, 0.70, 0.80])
    plate = plateau(unique, 0.55)
    assert plate["L"] == 0.40
    assert plate["R"] == 0.70
    assert plate["width"] == pytest.approx(0.30)


def test_threshold_exactly_on_a_score_keeps_that_score_as_right_bound():
    unique = np.array([0.30, 0.40, 0.70])
    plate = plateau(unique, 0.70)
    assert plate["R"] == 0.70          # S >= tau -> 0.70 pozitif tarafta
    assert plate["L"] == 0.40


def test_plateau_clipped_to_the_decision_interval():
    low, high = DECISION_INTERVAL
    unique = np.array([0.05, 0.95])
    plate = plateau(unique, 0.5)
    assert plate["L"] == 0.05 and plate["R"] == 0.95
    assert plate["clipped"] == [low, high]
    assert plate["width"] == pytest.approx(high - low)


def test_infinite_bounds_are_flagged_not_silently_zero():
    unique = np.array([0.30, 0.40])
    plate = plateau(unique, 0.99)      # tau butun skorlarin uzerinde
    assert plate["R_finite"] is False
    assert plate["L_finite"] is True
    assert plate["R_in_interval"] is False


# ---------------------------------------------------------------------------
# §3.4 -- taban oranlar
# ---------------------------------------------------------------------------
def test_gap_enrichment_is_one_for_an_average_gap():
    low, high = DECISION_INTERVAL
    k_i = 9
    inner = np.linspace(low + 0.01, high - 0.01, k_i)
    profile = support_profile(inner)
    assert profile["K_I"] == k_i
    plate = plateau(profile["unique_scores"], float(inner[4]) - 1e-6)
    enriched = enrichments(profile, plate)
    expected_base = (high - low) / (k_i + 1)
    assert enriched["base_gap"] == pytest.approx(expected_base)
    assert enriched["gap_enrichment"] == pytest.approx(
        plate["width"] / expected_base)


def test_mass_enrichment_is_one_under_equal_mass():
    low, high = DECISION_INTERVAL
    inner = np.linspace(low + 0.01, high - 0.01, 10)
    scores = np.repeat(inner, 20)              # her ayri skora esit kutle
    profile = support_profile(scores)
    plate = plateau(profile["unique_scores"], float(inner[5]))
    enriched = enrichments(profile, plate)
    assert enriched["base_mass_I"] == pytest.approx(1 / 10)
    assert enriched["mass_enrichment"] == pytest.approx(1.0)


def test_mass_enrichment_rises_with_a_heavy_boundary():
    low, high = DECISION_INTERVAL
    inner = np.linspace(low + 0.01, high - 0.01, 10)
    scores = np.concatenate([np.repeat(inner, 20), np.repeat(inner[5], 400)])
    profile = support_profile(scores)
    plate = plateau(profile["unique_scores"], float(inner[5]))
    enriched = enrichments(profile, plate)
    assert enriched["boundary_n"] == 420
    assert enriched["mass_enrichment"] > 2.0


def test_enrichment_is_null_when_boundary_is_outside_the_interval():
    unique = np.array([0.30, 0.40])
    profile = support_profile(unique)
    plate = plateau(profile["unique_scores"], 0.99)
    enriched = enrichments(profile, plate)
    assert enriched["mass_enrichment"] is None
    assert enriched["boundary_n"] is None


# ---------------------------------------------------------------------------
# §5.1 -- birlesik karar tablosu
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("h1,h2,expected", [
    ("destek", "destek", "destek"),
    ("destek", "yanlislandi", "kismi_genislik"),
    ("yanlislandi", "destek", "kismi_kutle"),
    ("yanlislandi", "yanlislandi", "reddedildi"),
    ("underpowered", "destek", "test_edilemedi"),
    ("destek", "underpowered", "test_edilemedi"),
    ("karisik", "karisik", "karisik"),
])
def test_combined_verdict_table(h1, h2, expected):
    assert combined_verdict(h1, h2)["verdict"] == expected


def test_mixed_is_never_counted_as_support():
    for other in ("destek", "yanlislandi", "karisik"):
        assert combined_verdict("karisik", other)["verdict"] != "destek"


# ---------------------------------------------------------------------------
# Spearman yardimcisi
# ---------------------------------------------------------------------------
def test_spearman_directions():
    assert _spearman([25, 50, 100, 200], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert _spearman([25, 50, 100, 200], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert _spearman([25, 50, 100, 200], [1, 1, 1, 1]) is None   # varyans yok


def test_spearman_single_adjacent_swap_reaches_the_gate():
    """n=4'te tek komsu takas rho = 0.8 verir; kapi ulasilabilir olmali."""
    assert _spearman([25, 50, 100, 200], [1, 3, 2, 4]) == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# §7 -- durdurma kosullari
# ---------------------------------------------------------------------------
def test_non_canonical_target_halts():
    target = _fixture("macyste", "phase", 3, 90)
    with pytest.raises(D3GuardError, match="satir"):
        check_universe(target, list(range(42, 62)), SCHEMA)


def test_wrong_seed_universe_halts():
    target = _fixture("macyste", "phase", 3, 90)
    with pytest.raises(D3GuardError):
        check_universe(target, [42, 43], SCHEMA)


def test_wrong_schema_halts():
    target = _fixture("macyste", "phase", 3, 90)
    with pytest.raises(D3GuardError):
        check_universe(target, list(range(42, 62)), "protocol")


# ---------------------------------------------------------------------------
# Skorlama sozlesmesi (yavas -- gercek fit)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_nested_two_hundred_matches_model_bitwise_and_returns_no_metrics():
    """Belge §7.6 invaryanti ve bypass'in performans alani DONDURMEDIGI."""
    source = _fixture("openplc", "phase", 6, 90)
    target = _fixture("macyste", "phase", 3, 90)
    scored = score_seed(source, target, seed=42)      # bit esitligi ici guard

    full = scored["target_panel"][200]["soft"]
    assert full.shape == (len(target),)

    forbidden = {"fpr_normal", "fpr_fault", "recall_event_manipulation",
                 "alarm_load", "auc"}
    assert not (forbidden & set(scored)), "performans metrigi sizmis"
    assert isinstance(scored["guard_reasons"], list)


@pytest.mark.slow
def test_hard_vote_support_cannot_exceed_t_plus_one():
    source = _fixture("openplc", "phase", 6, 90)
    target = _fixture("macyste", "phase", 3, 90)
    scored = score_seed(source, target, seed=42)
    for trees in TREE_PANEL:
        hard = scored["target_panel"][trees]["hard"]
        assert len(np.unique(hard)) <= trees + 1


@pytest.mark.slow
def test_soft_support_is_at_least_hard_support():
    """Saf olmayan yapraklar kafesi genisletebilir, daraltamaz."""
    source = _fixture("openplc", "phase", 6, 90)
    target = _fixture("macyste", "phase", 3, 90)
    scored = score_seed(source, target, seed=42)
    for trees in TREE_PANEL:
        panel = scored["target_panel"][trees]
        assert len(np.unique(panel["soft"])) >= len(np.unique(panel["hard"]))


@pytest.mark.slow
def test_purity_fields_are_bounded():
    source = _fixture("openplc", "phase", 6, 90)
    target = _fixture("macyste", "phase", 3, 90)
    scored = score_seed(source, target, seed=42)
    for trees in TREE_PANEL:
        panel = scored["target_panel"][trees]
        assert 0.0 <= panel["pure_visit_share"] <= 1.0
        assert 0.5 <= panel["mean_leaf_purity"] <= 1.0
