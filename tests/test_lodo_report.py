"""Olay-agirlikli / senaryo-dengeli AUC raporlamasi testleri.

Bootstrap tabakalamasi bir kez YANLIS yapildi: 12 kosu ayrimsiz yeniden
orneklenince senaryo bilesimi de degisiyor ve GA nokta tahminiyle
merkezlenmiyordu. Bu testler o hatanin geri gelmesini engeller.
"""
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ml.lodo_report import (  # noqa: E402
    _auc,
    _bootstrap_over_runs,
    balanced_weights,
)


def test_balanced_weights_give_each_scenario_equal_total_weight():
    families = np.array(["recon"] * 60 + ["normal"] * 6 + ["fault"] * 4)
    weights = balanced_weights(families)
    totals = {f: weights[families == f].sum() for f in np.unique(families)}
    for family, total in totals.items():
        assert total == pytest.approx(1.0), family


def test_balanced_weights_neutralise_composition_skew():
    """Carpik bilesimde dengeli AUC, bilesimden bagimsiz olmali."""
    # recon cok kalabalik ve KOTU ayrisiyor; normal/fault az ve IYI ayrisiyor
    families = np.array(["recon"] * 100 + ["fault"] * 10)
    y = np.array([1] * 50 + [0] * 50 + [1] * 5 + [0] * 5)
    scores = np.concatenate([
        np.full(50, 0.5), np.full(50, 0.5),      # recon: hic ayrim yok
        np.full(5, 0.9), np.full(5, 0.1),        # fault: mukemmel ayrim
    ])
    event_auc = _auc(y, scores)
    balanced_auc = _auc(y, scores, balanced_weights(families))
    # Dengeli olcum, kalabalik ve ayrimsiz recon'un baskisini azaltmali
    assert balanced_auc > event_auc


def _skewed_setup():
    """Bilesimi carpik sentetik veri: recon kosulari cok satirli."""
    runs, families, y, scores = [], [], [], []
    rng = np.random.RandomState(0)
    for rep in range(3):
        # recon: 700 satir/kosu, ayrim yok
        runs += [f"recon-r{rep}"] * 700
        families += ["recon"] * 700
        y += list(rng.randint(0, 2, 700))
        scores += list(rng.uniform(0.4, 0.6, 700))
        # normal: 100 satir/kosu, tumu negatif
        runs += [f"normal-r{rep}"] * 100
        families += ["normal"] * 100
        y += [0] * 100
        scores += list(rng.uniform(0.0, 0.2, 100))
        # manipulation: 100 satir/kosu, tumu pozitif ve iyi ayrisiyor
        runs += [f"manip-r{rep}"] * 100
        families += ["manipulation"] * 100
        y += [1] * 100
        scores += list(rng.uniform(0.8, 1.0, 100))
    return (np.array(runs), np.array(families),
            np.array(y), np.array(scores, dtype=float))


def test_stratified_bootstrap_preserves_scenario_composition():
    """Tabakali bootstrap GA'si nokta tahminini ICERMELI.

    Ayrimsiz bootstrap bilesimi degistirdigi icin GA'yi baska bir yerde
    merkezliyordu; tabakali olan tasarimi (3 kosu x senaryo) korur.
    """
    runs, families, y, scores = _skewed_setup()
    point = _auc(y, scores)
    lo, hi = _bootstrap_over_runs(runs, families, y, scores, None)
    assert lo <= point <= hi, (
        f"GA nokta tahminini icermiyor: {point:.4f} not in [{lo:.4f},{hi:.4f}]")


def test_stratified_bootstrap_holds_for_balanced_metric_too():
    runs, families, y, scores = _skewed_setup()
    weights = balanced_weights(families)
    point = _auc(y, scores, weights)
    lo, hi = _bootstrap_over_runs(runs, families, y, scores, weights)
    assert lo <= point <= hi, (
        f"dengeli GA nokta tahminini icermiyor: {point:.4f} "
        f"not in [{lo:.4f},{hi:.4f}]")


def test_bootstrap_draws_the_same_number_of_runs_per_scenario():
    """Her senaryodan kendi kosu sayisi kadar cekilmeli (bilesim korunur)."""
    runs, families, y, scores = _skewed_setup()
    seen = {}

    original = np.random.RandomState

    class Spy(original):
        def choice(self, a, size=None, replace=True, p=None):
            seen.setdefault(len(a), []).append(size)
            return super().choice(a, size=size, replace=replace, p=p)

    import ml.lodo_report as lr
    lr.np.random.RandomState = Spy
    try:
        _bootstrap_over_runs(runs, families, y, scores, None)
    finally:
        lr.np.random.RandomState = original
    # Her tabakada 3 kosu var ve her cekiliste 3 kosu istenmeli
    assert seen, "choice hic cagrilmadi"
    for pool_size, sizes in seen.items():
        assert pool_size == 3, f"beklenmeyen tabaka buyuklugu: {pool_size}"
        assert set(sizes) == {3}, f"tabaka boyutu korunmadi: {set(sizes)}"


def test_bootstrap_is_deterministic_for_a_given_seed():
    runs, families, y, scores = _skewed_setup()
    first = _bootstrap_over_runs(runs, families, y, scores, None, seed=7)
    second = _bootstrap_over_runs(runs, families, y, scores, None, seed=7)
    assert first == second


def test_single_run_per_scenario_still_returns_an_interval():
    runs = np.array(["a"] * 10 + ["b"] * 10)
    families = np.array(["normal"] * 10 + ["manipulation"] * 10)
    y = np.array([0] * 10 + [1] * 10)
    scores = np.concatenate([np.linspace(0, 0.4, 10), np.linspace(0.6, 1, 10)])
    lo, hi = _bootstrap_over_runs(runs, families, y, scores, None)
    assert not np.isnan(lo) and not np.isnan(hi)
    assert 0.0 <= lo <= hi <= 1.0


def test_auc_returns_nan_for_single_class():
    assert np.isnan(_auc(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3])))
