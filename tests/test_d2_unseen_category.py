"""D2 gorulmemis kategori teshisi -- on-kayit sozlesmesi testleri.

Belge: `DENEY-D2-Gorulmemis-Kategori-Teshisi.md`. Testler belgedeki
"zorunludur / durdurur / sayilmaz" maddelerini FONKSIYON duzeyinde bagli tutar;
belge ile kod ayrisirsa test kirilir.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ml.lodo_generalization import (  # noqa: E402
    SCHEMA_NAMES,
    _fixture,
    build_feature_frame,
)
from ml.unseen_category_diagnostic import (  # noqa: E402
    BAND_COLUMNS,
    BAND_MISSING,
    BAND_NOT_A_WRITE,
    MISSING_SOURCE,
    SEED_UNIVERSE,
    RenderGuardError,
    guard_class,
    missing_mask,
    primary_panel,
    render_precondition,
    seed_panel,
)


def _source():
    return _fixture("openplc", "phase", 6, 90)


def _target():
    return _fixture("macyste", "phase", 3, 90)


def _column(panel, schema, column):
    for item in panel:
        if item["schema"] != schema:
            continue
        for col in item["columns"]:
            if col["column"] == column:
                return col
    raise AssertionError(f"kolon bulunamadi: {schema}/{column}")


# ---------------------------------------------------------------------------
# §3.2 -- eksik-veri gostergeleri
# ---------------------------------------------------------------------------
def test_missing_source_table_covers_every_categorical_column():
    """Belge §3.2 tablosu, kodun urettigi HER kategorik kolonu kapsamali."""
    for schema in SCHEMA_NAMES:
        frame = build_feature_frame(_target(), schema)
        for column in frame.attrs["categorical"]:
            assert column in MISSING_SOURCE, (
                f"{schema}/{column} icin eksiklik kaynagi tanimsiz; "
                "belge §3.2 ile kod ayrismis"
            )


def test_band_na_is_not_treated_as_missing():
    """`_band()`'in urettigi 'na' GERCEK gozlemdir, eksiklik degildir."""
    assert BAND_MISSING == "unknown"
    assert BAND_NOT_A_WRITE == "na"
    for column in BAND_COLUMNS:
        assert MISSING_SOURCE[column] == "band_unknown"

    target = _target()
    frame = build_feature_frame(target, "physical_proxy")
    for column in BAND_COLUMNS:
        mask = missing_mask(target, frame, column)
        not_a_write = (frame[column].astype(str) == BAND_NOT_A_WRITE).to_numpy()
        assert not (mask & not_a_write).any(), (
            f"{column}: 'na' eksik sayilmis; belge §3.2 bunu yasaklar")


def test_missing_is_derived_from_raw_nan_not_from_a_literal():
    """Eksiklik literali surume baglidir; maske ham NaN'dan TURETILMELIDIR.

    pandas 3.0'da `.astype(str)` eksikligi korur ve `build_feature_frame()`'in
    `.fillna("na")` satiri devreye girer -> literal "na". Daha eski surumlerde
    ayni deger "nan" olurdu. Maske hicbir literale bagli olmamalidir.
    """
    target = _target()
    target.loc[target.index[:10], "flow_state"] = np.nan
    frame = build_feature_frame(target, "flow")

    mask = missing_mask(target, frame, "flow_state")
    assert mask.sum() == 10
    assert set(np.flatnonzero(mask)) == set(range(10))

    # Eksik satirlar DUSMEZ; bir literale toplanir (adi surume gore degisir).
    literal = set(frame["flow_state"].astype(str).to_numpy()[mask])
    assert len(literal) == 1
    assert len(frame) == len(target)


# ---------------------------------------------------------------------------
# §3.1 -- render on-kosulu
# ---------------------------------------------------------------------------
def test_render_guard_passes_on_matched_frames():
    report = render_precondition(_source(), _target())
    assert report["passed"]
    assert not report["render_mismatch"]
    assert not report["dtype_class_mismatch"]


def test_render_guard_halts_on_float_int_rendering_difference():
    """Belgenin asil korktugu regresyon: ayni deger '3.0' ve '3' olarak render."""
    source, target = _source(), _target()
    source["modbus_function"] = source["modbus_function"].astype("float64")
    target["modbus_function"] = target["modbus_function"].astype("int64")
    with pytest.raises(RenderGuardError) as excinfo:
        render_precondition(source, target)
    mismatch = excinfo.value.report["render_mismatch"]
    assert any(item["column"] == "modbus_function" for item in mismatch)
    # Iki taraf da SAYISAL; durduran sey dtype degil, GOSTERIM farki.
    assert not excinfo.value.report["dtype_class_mismatch"]


def test_render_guard_halts_on_dtype_class_mismatch():
    """Sayisal <-> string-benzeri gecisi durdurur (ortak ham deger kalmasa da)."""
    source, target = _source(), _target()
    target["modbus_function"] = target["modbus_function"].astype(str)
    with pytest.raises(RenderGuardError) as excinfo:
        render_precondition(source, target)
    classes = excinfo.value.report["dtype_class_mismatch"]
    assert any(item["column"] == "modbus_function" for item in classes)


def test_within_class_dtype_difference_warns_but_does_not_halt():
    """`str` <-> `object` montaj etkisidir; render'i degistirmez, durdurmaz."""
    source, target = _source(), _target()
    source["modbus_access"] = source["modbus_access"].astype("object")
    target["modbus_access"] = target["modbus_access"].astype("str")
    if str(source["modbus_access"].dtype) == str(target["modbus_access"].dtype):
        pytest.skip("bu pandas surumunde str/object ayrimi gozlenmiyor")
    report = render_precondition(source, target)
    assert report["passed"]
    assert any(w["column"] == "modbus_access"
               for w in report["within_class_dtype_warnings"])


def test_guard_failure_produces_no_ratio():
    """Durdurma hâlinde cagiran taraf hicbir oran GORMEZ."""
    source, target = _source(), _target()
    target["modbus_function"] = target["modbus_function"].astype(str)
    with pytest.raises(RenderGuardError) as excinfo:
        render_precondition(source, target)
    payload = str(excinfo.value.report)
    assert "unseen_row_mass" not in payload
    assert "unseen_value_ratio" not in payload


# ---------------------------------------------------------------------------
# §4 -- birincil panel
# ---------------------------------------------------------------------------
def test_unseen_value_is_detected_and_listed():
    source, target = _source(), _target()
    target.loc[target.index[:10], "flow_state"] = "rare_state"
    column = _column(primary_panel(source, target, ["flow"]), "flow", "flow_state")
    assert "rare_state" in column["unseen_values"]
    assert column["unseen_row_n"] == 10
    assert column["unseen_unique_n"] == 1


def test_two_denominators_are_reported_and_differ_when_sentinels_present():
    """Belge §4.2: kutle IKI paydayla; gozlenmis payda daha kucuk olmali."""
    source, target = _source(), _target()
    source.loc[source.index[:5], "flow_state"] = np.nan     # 'nan' iki alanda da var
    target.loc[target.index[:10], "flow_state"] = "rare_state"
    target.loc[target.index[10:30], "flow_state"] = np.nan

    column = _column(primary_panel(source, target, ["flow"]), "flow", "flow_state")
    assert column["unseen_values"] == ["rare_state"]   # eksik literali gorulmemis DEGIL
    assert column["missing_literal_is_unseen"] is False
    assert column["target_row_n"] == len(target)
    assert column["target_observed_row_n"] == len(target) - 20
    assert column["unseen_row_mass"] < column["unseen_row_mass_observed"]


def test_target_only_missing_literal_is_flagged():
    """Eksik literali yalniz hedefte varsa YAPISAL farktir; sessiz gecmez."""
    source, target = _source(), _target()
    target.loc[target.index[:10], "flow_state"] = np.nan
    column = _column(primary_panel(source, target, ["flow"]), "flow", "flow_state")
    assert column["missing_literal_is_unseen"] is True
    assert column["missing_literals_observed"]
    assert set(column["missing_literals_observed"]) <= set(column["unseen_values"])


def test_empty_frame_is_rejected_not_reported_as_zero():
    """Bos alan 'gorulmemis kategori yok' anlamina GELMEZ."""
    with pytest.raises(ValueError, match="BOS"):
        primary_panel(_source(), _target().iloc[0:0], ["flow"])
    with pytest.raises(ValueError, match="BOS"):
        primary_panel(_source().iloc[0:0], _target(), ["flow"])


def test_union_mass_is_between_max_column_and_sum():
    """Belge §8: sema birlesim kutlesi kolon kutlelerinin TOPLAMI degildir."""
    source, target = _source(), _target()
    target.loc[target.index[:10], "flow_state"] = "rare_state"
    target.loc[target.index[5:20], "modbus_access"] = "WRITE | EXOTIC"
    panel = primary_panel(source, target, ["protocol"])
    schema = panel[0]
    masses = [c["unseen_row_mass"] for c in schema["columns"]]
    assert max(masses) <= schema["unseen_any_mass"] <= sum(masses)


def test_union_has_no_observed_denominator():
    """Kolonlarin gostergeleri farkli oldugu icin birlesimde ikinci payda YOK."""
    panel = primary_panel(_source(), _target(), ["physical_proxy"])
    assert "unseen_any_mass_observed" not in panel[0]


def test_primary_panel_is_split_free():
    """Panelde seed/bolme yok: satir sirasi degisince sonuc DEGISMEZ."""
    source, target = _source(), _target()
    shuffled = source.sample(frac=1.0, random_state=7).reset_index(drop=True)
    base = primary_panel(source, target, ["protocol"])
    other = primary_panel(shuffled, target, ["protocol"])
    for a, b in zip(base[0]["columns"], other[0]["columns"]):
        assert a == b


def test_primary_panel_covers_documented_columns_per_schema():
    """Belge §3 tablosu ile uretilen kolonlar birebir ayni olmali."""
    expected = {
        "flow": ["flow_state"],
        "protocol": ["flow_state", "modbus_function", "modbus_access"],
        "physical_strict": ["flow_state", "modbus_function", "modbus_access",
                            "rudder_band"],
        "physical_proxy": ["flow_state", "modbus_function", "modbus_access",
                           "rudder_band", "propulsion_band"],
    }
    panel = primary_panel(_source(), _target(), list(SCHEMA_NAMES))
    for item in panel:
        assert [c["column"] for c in item["columns"]] == expected[item["schema"]]


def test_required_output_fields_present():
    """Belge §4.3: mutlak sayilar olmadan yalniz oran yeterli degildir."""
    required = {"source_unique_n", "target_unique_n", "unseen_unique_n",
                "unseen_values", "unseen_value_ratio", "unseen_row_n",
                "target_row_n", "unseen_row_mass", "target_observed_row_n",
                "unseen_row_mass_observed", "missing_basis",
                "missing_literals_observed"}
    panel = primary_panel(_source(), _target(), ["physical_proxy"])
    for column in panel[0]["columns"]:
        assert required <= set(column)


# ---------------------------------------------------------------------------
# §5 -- ikincil seed paneli
# ---------------------------------------------------------------------------
def test_seed_universe_is_the_preregistered_one():
    assert SEED_UNIVERSE == tuple(range(42, 62))
    assert len(SEED_UNIVERSE) == 20


def test_seed_panel_runs_every_seed_without_exclusion():
    result = seed_panel(_source(), _target(), ["flow"], seeds=range(42, 52))
    assert len(result["per_seed"]) == 10
    assert all(entry["split_basis"] == "run" for entry in result["per_seed"])
    valid = {"kusur_a_validationda_yok", "kusur_b_trainde_yok",
             "iki_tarafta_var", "bilinmiyor"}
    assert all(entry["guard_class"] in valid for entry in result["per_seed"])


def test_seed_vocabulary_comes_from_train_side_only():
    """Yalniz TEK kosuda gecen bir deger, o kosu validation'a dusunce gorulmemis olur.

    Bu, H2'nin mekanizmasidir: sozluk bolmeye bagli degisir.
    """
    source, target = _source(), _target()
    lone_run = sorted(source["run_id"].astype(str).unique())[0]
    mask = source["run_id"].astype(str) == lone_run
    source.loc[mask, "flow_state"] = "lone_state"
    target.loc[target.index[:10], "flow_state"] = "lone_state"

    result = seed_panel(source, target, ["flow"], seeds=range(42, 62))
    counts = [entry["schemas"][0]["columns"][0]["unseen_row_n"]
              for entry in result["per_seed"]]
    assert min(counts) == 0, "hicbir seed'de kosu train'e dusmemis"
    assert max(counts) == 10, "hicbir seed'de kosu validation'a dusmemis"

    spread = result["summary"][0]["columns"][0]["unseen_row_n"]
    assert spread["max"] > spread["min"]
    assert spread["n_seeds"] == 20


def test_seed_summary_reports_median_min_max():
    result = seed_panel(_source(), _target(), ["flow"], seeds=range(42, 47))
    spread = result["summary"][0]["columns"][0]["unseen_row_mass"]
    assert set(spread) == {"median", "min", "max", "n_seeds"}
    assert spread["min"] <= spread["median"] <= spread["max"]


def test_guard_class_detects_missing_class_on_each_side():
    source = _source()
    source["label"] = source["label"].astype(str)
    runs = sorted(source["run_id"].astype(str).unique())
    fault_run = runs[0]
    source.loc[source["run_id"].astype(str) == fault_run, "label"] = "fault"

    all_idx = np.arange(len(source))
    is_fault_run = (source["run_id"].astype(str) == fault_run).to_numpy()
    tr_only = all_idx[~is_fault_run]
    va_only = all_idx[is_fault_run]

    # fault yalniz validation'da -> train'de yok (Kusur B)
    assert guard_class(source, tr_only, va_only) == "kusur_b_trainde_yok"
    # fault yalniz train'de -> validation'da yok (Kusur A)
    assert guard_class(source, va_only, tr_only) == "kusur_a_validationda_yok"


def test_seed_panel_target_is_always_the_full_target_frame():
    target = _target()
    result = seed_panel(_source(), target, ["flow"], seeds=range(42, 45))
    for entry in result["per_seed"]:
        assert entry["schemas"][0]["target_row_n"] == len(target)
