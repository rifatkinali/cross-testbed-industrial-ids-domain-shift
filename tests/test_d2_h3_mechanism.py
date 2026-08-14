"""D2 §7 -- H3 mekanizma testi sozlesmesi.

Testler belgedeki sabitleri ve karar olcutlerini kod duzeyinde bagli tutar.
Ön-kayitli bir esik sonuctan sonra gevsetilirse burasi kirilir.
"""
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ml.lodo_generalization import _fixture, build_feature_frame  # noqa: E402
from ml.unseen_category_mechanism import (  # noqa: E402
    ALIGNMENT_MIN_SHARE,
    BLOCK_RADIUS,
    H3_SCHEMA,
    MIN_BLOCK_ROWS,
    MIN_SEEDS_PER_TRANSITION,
    OUTSIDE_BANDS,
    REGISTERED_BANDS,
    TRANSITIONS,
    _base_rate_check,
    _verdict,
    alignment,
    band_of,
    block_mask,
    fit_and_score,
    point_mass_inventory,
    signature_strings,
    unseen_signature,
)


# ---------------------------------------------------------------------------
# §7.1 / §7.2 -- on-kayitli sabitler
# ---------------------------------------------------------------------------
def test_preregistered_constants():
    assert BLOCK_RADIUS == 0.005
    assert MIN_BLOCK_ROWS == 50
    assert MIN_SEEDS_PER_TRANSITION == 10
    assert ALIGNMENT_MIN_SHARE == 0.80
    assert H3_SCHEMA == "physical_proxy"


def test_transitions_are_the_gaps_between_registered_bands():
    """Olcum yeri bant ICI degil, bantlar ARASI bosluktur (belge §7.1)."""
    assert set(TRANSITIONS) == {"T1", "T2"}
    assert TRANSITIONS["T1"] == (REGISTERED_BANDS["A"][1], REGISTERED_BANDS["B"][0])
    assert TRANSITIONS["T2"][0] == REGISTERED_BANDS["B"][1]
    assert TRANSITIONS["T2"][1] == 0.88
    # Uc bant iki gecis verir -- ucuncu bir karar bolgesi YOK.
    assert len(TRANSITIONS) == len(REGISTERED_BANDS) - 1


def test_band_of_places_thresholds_descriptively():
    assert band_of(0.30) == "A"
    assert band_of(0.73) == "B"
    assert band_of(0.88) == "C"
    assert band_of(0.60) == OUTSIDE_BANDS      # bosluga dusen esik dislanmaz
    assert band_of(0.99) == OUTSIDE_BANDS


# ---------------------------------------------------------------------------
# Nokta kutlesi ve blok
# ---------------------------------------------------------------------------
def test_point_mass_inventory_is_open_interval_and_sorted():
    scores = np.array([0.49, 0.50, 0.50, 0.50, 0.60, 0.60, 0.72, 0.80])
    inventory = point_mass_inventory(scores, 0.49, 0.72)
    assert [item["score"] for item in inventory] == [0.50, 0.60]
    assert [item["n"] for item in inventory] == [3, 2]
    # Sinir degerleri ACIK aralik disinda kalir.
    assert all(item["score"] not in (0.49, 0.72) for item in inventory)


def test_point_mass_inventory_empty_interval():
    assert point_mass_inventory(np.array([0.1, 0.95]), 0.49, 0.72) == []


def test_full_inventory_is_available_so_choice_is_auditable():
    scores = np.array([0.50] * 3 + [0.60] * 2 + [0.65])
    inventory = point_mass_inventory(scores, 0.49, 0.72)
    assert len(inventory) == 3, "envanterin tamami raporlanmali"


def test_block_mask_radius_is_inclusive_despite_float_representation():
    """Tam sinirdaki satir ikili gosterim hatasi yuzunden dusmemeli.

    |0.705 - 0.700| kayan noktada 0.005000000000000004 cikar; ham `<=` bu
    satiri blok DISINDA birakirdi ve on-kayitli yaricap belirsizlesirdi.
    """
    scores = np.array([0.700, 0.705, 0.706, 0.695, 0.694])
    mask = block_mask(scores, 0.700)
    assert mask.tolist() == [True, True, False, True, False]

    # Yaricap on-kayitli degeriyle ayni kalmali; pay yalniz gosterim icin.
    assert BLOCK_RADIUS == 0.005
    assert block_mask(np.array([0.7061]), 0.700).tolist() == [False]


# ---------------------------------------------------------------------------
# Gorulmemislik imzasi
# ---------------------------------------------------------------------------
def test_unseen_signature_flags_values_absent_from_train_vocabulary():
    train = _fixture("openplc", "phase", 4, 90)
    target = _fixture("macyste", "phase", 3, 90)
    target.loc[target.index[:10], "flow_state"] = "rare_state"

    train_frame = build_feature_frame(train, H3_SCHEMA)
    target_frame = build_feature_frame(target, H3_SCHEMA)
    columns = list(target_frame.attrs["categorical"])

    bits = unseen_signature(train_frame, target_frame, columns)
    assert bits.shape == (len(target_frame), len(columns))

    flow_bit = columns.index("flow_state")
    assert bits[:10, flow_bit].sum() == 10
    assert bits[10:, flow_bit].sum() == 0


def test_signature_bit_order_follows_the_feature_frame_not_a_hardcoded_list():
    target_frame = build_feature_frame(_fixture("macyste", "phase", 3, 90),
                                       H3_SCHEMA)
    assert list(target_frame.attrs["categorical"]) == [
        "flow_state", "modbus_function", "modbus_access",
        "rudder_band", "propulsion_band",
    ]


def test_signature_strings_are_five_bits_for_physical_proxy():
    bits = np.array([[0, 1, 0, 0, 1], [0, 0, 0, 0, 0]], dtype=np.int8)
    assert signature_strings(bits).tolist() == ["01001", "00000"]


# ---------------------------------------------------------------------------
# §7.4 -- hizalanma olcutu
# ---------------------------------------------------------------------------
def test_alignment_requires_eighty_percent():
    layers = [1] * 80 + [2] * 20
    signatures = ["10000"] * 80 + ["01000"] * 20
    result = alignment(layers, signatures)
    assert result["layer_share"] == 0.8
    assert result["aligned"] is True

    layers = [1] * 79 + [2] * 21
    signatures = ["10000"] * 79 + ["01000"] * 21
    assert alignment(layers, signatures)["aligned"] is False


def test_alignment_can_be_carried_by_signature_alone():
    """Tek imza yeterlidir; katman payi dusuk olsa da hizali sayilir."""
    layers = [1] * 85 + [2] * 15
    signatures = ["10000"] * 85 + ["01000"] * 15
    result = alignment(layers, signatures)
    assert result["signature_share"] >= ALIGNMENT_MIN_SHARE
    assert result["aligned"] is True


def test_alignment_on_empty_block():
    result = alignment([], [])
    assert result["n_rows"] == 0
    assert result["aligned"] is False


# ---------------------------------------------------------------------------
# §7.4 -- karar tablosu
# ---------------------------------------------------------------------------
def _transition(evaluable: bool, aligned):
    return {"evaluable": evaluable, "aligned": aligned}


def test_verdict_supported_when_both_transitions_align():
    verdict = _verdict({"T1": _transition(True, True),
                        "T2": _transition(True, True)})
    assert verdict["h3"] == "desteklendi"


def test_verdict_partial_when_only_one_aligns():
    verdict = _verdict({"T1": _transition(True, True),
                        "T2": _transition(True, False)})
    assert verdict["h3"] == "kismi_destek"


def test_verdict_partial_when_one_aligns_and_other_underpowered():
    verdict = _verdict({"T1": _transition(True, True),
                        "T2": _transition(False, None)})
    assert verdict["h3"] == "kismi_destek"


def test_verdict_rejected_when_no_evaluable_transition_aligns():
    verdict = _verdict({"T1": _transition(True, False),
                        "T2": _transition(True, False)})
    assert verdict["h3"] == "reddedildi"


def test_degeneracy_flag_when_alignment_is_carried_by_the_zero_layer():
    """k=0 hizalanmasi olcutu saglar ama H3'un TERSINI gosterir; isaretlenmeli."""
    transitions = {
        "T1": {"evaluable": True, "aligned": True,
               "pooled_alignment": {"dominant_layer": 0}},
        "T2": {"evaluable": True, "aligned": True,
               "pooled_alignment": {"dominant_layer": 0}},
    }
    verdict = _verdict(transitions)
    assert verdict["h3"] == "desteklendi"      # on-kayitli karar DEGISMEZ
    assert verdict["degenerate"] is True
    assert "k=0" in verdict["degenerate_note"]


def test_no_degeneracy_when_alignment_is_carried_by_an_unseen_layer():
    transitions = {
        "T1": {"evaluable": True, "aligned": True,
               "pooled_alignment": {"dominant_layer": 2}},
        "T2": {"evaluable": True, "aligned": True,
               "pooled_alignment": {"dominant_layer": 2}},
    }
    verdict = _verdict(transitions)
    assert verdict["h3"] == "desteklendi"
    assert verdict["degenerate"] is False


def test_base_rate_check_measures_enrichment_not_share():
    """Zenginlesme ~1.0 ise blok hedefin genelinden farksizdir (bilgi yok)."""
    bucket = {"layers": [0] * 99 + [2], "signatures": [], "seeds": [1],
              "base_n": 1000, "base_unseen_n": 10}
    check = _base_rate_check(bucket)
    assert check["block_unseen_share"] == 0.01
    assert check["base_unseen_share"] == 0.01
    assert check["enrichment"] == 1.0
    assert "GORULDUKTEN SONRA" in check["note"]

    bucket["layers"] = [0] * 90 + [2] * 10          # 10x yigilma
    assert _base_rate_check(bucket)["enrichment"] == 10.0


def test_verdict_untested_when_every_transition_is_underpowered():
    """Iki gecis de desteksizse H3 ne reddedilir ne desteklenir."""
    verdict = _verdict({"T1": _transition(False, None),
                        "T2": _transition(False, None)})
    assert verdict["h3"] == "test_edilemedi"
    assert verdict["aligned"] == []


# ---------------------------------------------------------------------------
# Teshis amacli skorlama
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_fit_and_score_returns_scores_and_threshold_but_no_metrics():
    """Bypass yalniz skorlara erisim icindir; performans alani DONMEZ."""
    source = _fixture("openplc", "phase", 6, 90)
    target = _fixture("macyste", "phase", 3, 90)
    fitted = fit_and_score(source, target, seed=42)

    assert fitted["scores"].shape == (len(target),)
    assert 0.0 <= fitted["threshold"] <= 1.0
    assert fitted["split_basis"] == "run"

    forbidden = {"fpr_normal", "fpr_fault", "recall_event_manipulation",
                 "recall_event_recon", "alarm_load", "recall_episode_manipulation"}
    assert not (forbidden & set(fitted)), "performans metrigi sizmis"


@pytest.mark.slow
def test_fit_and_score_preserves_guard_reasons():
    """Guard nedenleri her seed icin KORUNUR (belge §7); sessizce yutulmaz."""
    source = _fixture("openplc", "phase", 6, 90)
    target = _fixture("macyste", "phase", 3, 90)
    fitted = fit_and_score(source, target, seed=42)
    assert isinstance(fitted["guard_reasons"], list)
    # Fixture'da OpenPLC tarafinda 'fault' yok -> hedefteki fault sinifi icin
    # bilesim guard'i konusmali.
    assert fitted["guard_reasons"], "guard sessiz kalmis"
