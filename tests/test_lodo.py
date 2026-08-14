"""LODO iskeleti sozlesme/smoke testleri (sklearn GEREKMEZ; dummy model).

Uzun harici kampanyadan ONCE sema ve etiket sozlesmesini dogrular.
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
    DOMAIN_IDENTITY_BLOCKLIST,
    SCHEMA_NAMES,
    _dummy_factory,
    _episode_hits,
    _epoch_seconds,
    _fixture,
    _n_pump3000,
    _n_rpm1200,
    _n_rudder,
    _rate_per_run,
    _split_by_run,
    _threshold_at_fpr,
    alarm_load,
    assert_no_leakage,
    build_feature_frame,
    detection_latency,
    episode_bootstrap_ci,
    run_lodo,
    run_lodo_multiseed,
    scenario_family,
    validate_readiness,
    validate_split,
)


def _stamped_gt(runs, first="baseline", second="unauthorized_manipulation"):
    """Gercek ground-truth.json bicimi: faz adi + baslangic zaman damgasi."""
    return {
        run: {"phases": [
            {"phase": first, "timestamp": "2023-11-14T22:13:20+00:00"},
            {"phase": second, "timestamp": "2023-11-14T22:13:38+00:00"},
        ]}
        for run in runs
    }


@pytest.mark.unit
def test_leakage_guard_catches_domain_identity():
    with pytest.raises(AssertionError):
        assert_no_leakage(pd.DataFrame({"pkts_toserver": [1], "dest_port": [502]}))


@pytest.mark.unit
def test_physical_normalization_contract():
    assert _n_rudder(35) == pytest.approx(1.0)
    assert _n_rpm1200(1200) == pytest.approx(1.0)   # propulsion proxy: rpm/1200
    assert _n_pump3000(3000) == pytest.approx(1.0)  # hidrolik pompa, engine DEGIL


@pytest.mark.unit
@pytest.mark.parametrize("schema", SCHEMA_NAMES)
def test_no_domain_identity_or_encoding_leak(schema):
    frame = build_feature_frame(_fixture("macyste", "phase", 3, 90), schema)
    assert not DOMAIN_IDENTITY_BLOCKLIST.intersection(frame.columns)
    # modbus_quantity (encoding sizintisi) hicbir semada olmamali
    assert "modbus_quantity" not in frame.columns
    assert "modbus_quantity_norm" not in frame.columns
    # ham adres/port/engine hicbir semada olmamali
    for banned in ("modbus_address", "dest_port", "engine_rpm", "rudder_angle",
                   "throttle_command_left", "load_cmd"):
        assert banned not in frame.columns


@pytest.mark.unit
def test_load_cmd_never_maps_to_propulsion():
    # load_cmd jenerator yuku; propulsion_demand'a KARISMAMALI
    frame = build_feature_frame(_fixture("openplc", "phase", 3, 90), "physical_proxy")
    assert "load_cmd" not in frame.columns


@pytest.mark.unit
def test_strict_vs_proxy_schema():
    strict = build_feature_frame(_fixture("macyste", "phase", 3, 90), "physical_strict")
    proxy = build_feature_frame(_fixture("macyste", "phase", 3, 90), "physical_proxy")
    # strict: yalniz gercek ortak fiziksel (rudder) + present + rate
    assert "rudder_norm" in strict.columns and "rudder_norm_present" in strict.columns
    assert "rudder_rate" in strict.columns
    assert "propulsion_demand_norm" not in strict.columns
    # proxy: strict + propulsion proxy
    assert "propulsion_demand_norm" in proxy.columns
    assert "propulsion_demand_norm_present" in proxy.columns


@pytest.mark.unit
def test_missing_field_is_present_flag_not_zero():
    # OpenPLC'de throttle yok -> macyste rol cozumunde propulsion present=0 olmali
    df = _fixture("openplc", "phase", 3, 90).drop(columns=["rpm_command"])
    frame = build_feature_frame(df, "physical_proxy")
    assert (frame["propulsion_demand_norm_present"] == 0).all()


@pytest.mark.unit
def test_readiness_refuses_short_and_run_level():
    ready, reasons = validate_readiness(_fixture("macyste", "run", 1, 5), None)
    assert not ready and reasons


@pytest.mark.unit
def test_ready_data_with_per_run_ground_truth():
    good = _fixture("macyste", "phase", 3, 90)
    gt = {g: {"phases": [{"name": "warmup"}, {"name": "act"}]} for g in good["run_id"].unique()}
    ready, reasons = validate_readiness(good, gt)
    assert ready, reasons


@pytest.mark.unit
def test_contract_mode_refuses_headline():
    r = run_lodo(_fixture("openplc", "phase", 3, 90),
                 _fixture("macyste", "run", 1, 5),
                 "physical_proxy", ground_truth=None, model_factory=_dummy_factory)
    assert r["mode"] == "contract"
    assert "fpr_normal" not in r


@pytest.mark.unit
def test_headline_metrics_are_scenario_based():
    good = _fixture("macyste", "phase", 3, 90)
    gt = {g: {"phases": [{"name": "act"}]} for g in good["run_id"].unique()}
    r = run_lodo(_fixture("openplc", "phase", 4, 90), good, "physical_proxy",
                 ground_truth=gt, model_factory=_dummy_factory, force_headline=True)
    # saldiri turu scenario'dan secilir; normal/fault FPR ayri
    assert "fpr_normal" in r and "fpr_fault" in r
    assert "recall_event_manipulation" in r and "recall_event_recon" in r


@pytest.mark.unit
def test_bootstrap_episode_level_bounded():
    m, lo, hi = episode_bootstrap_ci([1, 1, 0, 1, 0], n_boot=500)
    assert 0.0 <= lo <= m <= hi <= 1.0


# --------------------------------------------------------------------------
# Regresyon: iskeletin testlerin YAKALAMADIGI hatalari
# --------------------------------------------------------------------------
@pytest.mark.unit
def test_rate_uses_only_real_rudder_samples():
    """Araya baska Modbus olaylari girdiginde hiz sahte olarak sifirlanmamali.

    Tum satirlarda diff() alinirsa NaN->fillna(0) yuzunden hiz her yerde 0 olur
    ve fiziksel bant icindeki hizli dumen kacislari gorunmez hale gelir.
    """
    df = pd.DataFrame({
        "run_id": ["r1"] * 5,
        "timestamp": [f"2023-11-14T22:13:{20 + i}+00:00" for i in range(5)],
    })
    # 0., 2., 4. satirda dumen ornegi; 1. ve 3. satir baska register olayi
    values = pd.Series([0.0, np.nan, 0.2, np.nan, 0.4], index=df.index)
    rate = _rate_per_run(df, values)
    # ornekler arasi 2 saniye, 0.2 birim -> 0.1 birim/s
    assert rate.iloc[2] == pytest.approx(0.1)
    assert rate.iloc[4] == pytest.approx(0.1)
    assert rate.iloc[0] == 0.0          # ilk ornegin oncesi yok
    assert (rate.iloc[[1, 3]] == 0.0).all()   # orneksiz satir; present=0 ayirt eder


@pytest.mark.unit
def test_mixed_precision_timestamps_are_all_parsed():
    """pandas format= verilmezse bicimi ILK elemandan cikarir.

    Ayni sutunda '...:20.500+00:00' ile '...:22+00:00' bulunabilir (ground-truth
    tam saniyeye denk dustugunde kesir kismi hic yazilmaz). Format sabitlenmezse
    uyusmayan satirlar sessizce NaT olur ve o satirlar zaman ekseninden dusar.
    """
    mixed = pd.Series([
        "2026-07-26T02:43:20.500000+00:00",
        "2026-07-26T02:43:22+00:00",
        "2026-07-26T02:43:24.125+00:00",
    ])
    seconds = _epoch_seconds(mixed)
    assert seconds.notna().all(), "karisik hassasiyette satir NaT oldu"
    assert seconds.iloc[1] - seconds.iloc[0] == pytest.approx(1.5)
    assert seconds.iloc[2] - seconds.iloc[1] == pytest.approx(2.125)


@pytest.mark.unit
def test_rate_is_computed_across_mixed_precision_timestamps():
    df = pd.DataFrame({
        "run_id": ["r1"] * 3,
        "timestamp": ["2026-07-26T02:43:20+00:00",
                      "2026-07-26T02:43:22.500000+00:00",
                      "2026-07-26T02:43:25+00:00"],
    })
    values = pd.Series([0.0, 0.25, 0.5], index=df.index)
    rate = _rate_per_run(df, values)
    assert rate.iloc[1] == pytest.approx(0.1)   # 0.25 / 2.5 s
    assert rate.iloc[2] == pytest.approx(0.1)


@pytest.mark.unit
def test_rudder_rate_survives_sparse_sampling_in_fixture():
    frame = build_feature_frame(_fixture("macyste", "phase", 3, 90), "physical_strict")
    assert (frame["rudder_rate"] != 0).any()


@pytest.mark.unit
def test_episode_boundaries_come_from_ground_truth_phases():
    """Olay CSV'sinde 'phase' kolonu YOKTUR; sinirlar ground-truth'tan gelmeli."""
    df = pd.DataFrame({
        "run_id": ["r1"] * 4,
        "timestamp": ["2023-11-14T22:13:25+00:00", "2023-11-14T22:13:30+00:00",
                      "2023-11-14T22:13:40+00:00", "2023-11-14T22:13:45+00:00"],
    })
    hits, basis = _episode_hits(df, np.array([0, 0, 1, 0]), _stamped_gt(["r1"]))
    assert basis == "ground_truth_phase"
    assert sorted(hits) == [0, 1]       # baseline kacirildi, manipulation yakalandi


@pytest.mark.unit
def test_episode_falls_back_when_ground_truth_has_no_timestamps():
    df = pd.DataFrame({
        "run_id": ["r1"] * 2,
        "phase": ["warmup", "act"],
        "timestamp": ["2023-11-14T22:13:25+00:00", "2023-11-14T22:13:40+00:00"],
    })
    _, basis = _episode_hits(df, np.array([0, 1]),
                             {"r1": {"phases": [{"name": "warmup"}, {"name": "act"}]}})
    assert basis == "phase_column"


@pytest.mark.unit
def test_event_recall_mask_requires_scenario_and_attack_label():
    """Saldiri kosusunun normal baslangici recall'a KARISMAMALI."""
    good = _fixture("macyste", "phase", 3, 90)
    r = run_lodo(_fixture("openplc", "phase", 4, 90), good, "physical_proxy",
                 ground_truth=_stamped_gt(good["run_id"].unique()),
                 model_factory=_dummy_factory, force_headline=True)
    # fixture: kosu basina 20 satirin ilk 4'u warmup/normal -> 16 saldiri satiri
    assert r["n_manipulation_attack_rows"] == 3 * 16
    assert r["n_recon_attack_rows"] == 3 * 16
    # FPR gruplarina saldiri satiri girmez
    assert r["n_normal"] == 3 * 20 and r["n_fault"] == 3 * 20


@pytest.mark.unit
@pytest.mark.parametrize("max_fpr", [0.01, 0.05, 0.1])
def test_threshold_holds_target_fpr_with_tied_scores(max_fpr):
    """Agac toplulugu ayrik skor uretir; `>=` ile quantile tek basina yetmez."""
    proba = np.array([0.0] * 90 + [0.4] * 10)
    y = np.zeros(100, dtype=int)
    thr = _threshold_at_fpr(proba, y, max_fpr)
    assert float((proba >= thr).mean()) <= max_fpr


@pytest.mark.unit
def test_threshold_still_reaches_target_when_scores_are_continuous():
    rng = np.random.RandomState(0)
    proba = rng.uniform(0, 1, 1000)
    thr = _threshold_at_fpr(proba, np.zeros(1000, int), 0.05)
    achieved = float((proba >= thr).mean())
    assert achieved <= 0.05 and achieved > 0.03   # muhafazakar ama asiri degil


@pytest.mark.unit
def test_left_right_throttle_is_row_wise_role_coalesce():
    """Sol/sag telgraf ayri sunuculardan gelir; ayni satirda bulunmazlar.

    Bu yuzden birlestirme SENKRON L/R ortalamasi degil, satir-bazli
    role-coalesce'tir: her satir hangi tarafi tasiyorsa o kullanilir.
    """
    df = _fixture("macyste", "phase", 1, 90)
    both = df[["throttle_command_left", "throttle_command_right"]].notna().sum(axis=1)
    assert (both <= 1).all(), "fixture gercek yakalamayi taklit etmiyor"
    frame = build_feature_frame(df, "physical_proxy")
    carried = both == 1
    assert (frame.loc[carried, "propulsion_demand_norm_present"] == 1).all()
    assert (frame.loc[~carried, "propulsion_demand_norm_present"] == 0).all()


@pytest.mark.unit
def test_scenario_family_maps_real_campaign_vocabulary():
    """Kampanya '<kaynak>_<aile>' yazar; yalin esitlik tum maskeleri bosaltir."""
    got = list(scenario_family([
        "macyste_manipulation", "macyste_recon", "macyste_normal",
        "macyste_fault", "manipulation", "MaCySTe_Normal", "openplc_baseline",
    ]))
    assert got == ["manipulation", "recon", "normal", "fault",
                   "manipulation", "normal", "other"]


@pytest.mark.unit
def test_unmapped_scenario_is_refused_not_silently_dropped():
    df = _fixture("macyste", "phase", 3, 90).copy()
    df["scenario"] = df["scenario"].replace({"macyste_recon": "weird_name"})
    ready, reasons = validate_readiness(df, _stamped_gt(df["run_id"].unique()))
    assert not ready
    assert any("eslenemedi" in r and "weird_name" in r for r in reasons)


@pytest.mark.unit
def test_headline_masks_hit_with_prefixed_scenario_names():
    """Regresyon: gercek adlarla metrikler null donmemeli."""
    good = _fixture("macyste", "phase", 3, 90)
    assert good["scenario"].str.startswith("macyste_").all()
    r = run_lodo(_fixture("openplc", "phase", 4, 90), good, "physical_proxy",
                 ground_truth=_stamped_gt(good["run_id"].unique()),
                 model_factory=_dummy_factory, force_headline=True)
    assert r["recall_event_manipulation"] is not None
    assert r["recall_event_recon"] is not None
    assert r["fpr_normal"] is not None and r["fpr_fault"] is not None
    assert r["scenario_map"]["macyste_manipulation"] == "manipulation"


# --------------------------------------------------------------------------
# Contract'in TRAIN tarafi: esik guvenilir bir bolmeden mi geliyor
# --------------------------------------------------------------------------
@pytest.mark.unit
def test_row_level_split_is_detected_and_refused():
    """Egitimde tek kosu varsa satir-bazli bolme devreye girer -- sessiz kalmamali."""
    df = _fixture("openplc", "phase", 1, 90).copy()
    df["run_id"] = "tek-kosu"
    tr_idx, va_idx, basis = _split_by_run(df, 0.3, 42)
    assert basis == "row"
    reasons = validate_split(df, tr_idx, va_idx, basis, 0.01)
    assert any("KOSU bazli degil" in r for r in reasons)
    assert any("ayni kosulari paylasiyor" in r for r in reasons)


@pytest.mark.unit
def test_run_based_split_keeps_train_and_validation_disjoint():
    df = _fixture("openplc", "phase", 4, 90)
    tr_idx, va_idx, basis = _split_by_run(df, 0.3, 42)
    assert basis == "run"
    tr_runs = set(df.iloc[tr_idx]["run_id"])
    va_runs = set(df.iloc[va_idx]["run_id"])
    assert tr_runs and va_runs and not (tr_runs & va_runs)


@pytest.mark.unit
def test_validation_without_negatives_is_refused():
    df = _fixture("openplc", "phase", 4, 90).copy()
    df["label"] = "attack"
    tr_idx, va_idx, basis = _split_by_run(df, 0.3, 42)
    reasons = validate_split(df, tr_idx, va_idx, basis, 0.01)
    assert any("normal satir yok" in r for r in reasons)


@pytest.mark.unit
def test_validation_too_small_for_target_fpr_is_refused():
    """Hedef FPR 0.01 icin en az 100 negatif gerekir; 44 ile cozulemez."""
    df = _fixture("openplc", "phase", 4, 90)
    tr_idx, va_idx, basis = _split_by_run(df, 0.3, 42)
    strict = validate_split(df, tr_idx, va_idx, basis, 0.01)
    assert any("1/max_fpr" in r for r in strict)
    # Ayni bolme, gevsek hedefte kabul edilir
    assert validate_split(df, tr_idx, va_idx, basis, 0.5) == []


def _split_with_fault_runs(n_val_fault, n_train_fault, n_reps=6):
    """(kare, train_idx, val_idx): taraflara N BAGIMSIZ fault kosusu yerlestir.

    Bolme deterministik olmadigi icin kosular elle ayrilir; bu testler bolme
    algoritmasini degil BILESIM mantigini dogrular. Her tarafta fault disinda
    yeterli 'normal' kosusu da kalir (guard normal'i de denetler).
    """
    frame = _fixture("openplc", "phase", n_reps, 90).copy()
    runs = sorted(frame["run_id"].astype(str).unique())
    half = len(runs) // 2
    val_runs, train_runs = runs[:half], runs[half:]
    assert n_val_fault < half and n_train_fault < half, "normal kosusu kalmali"
    fault_runs = set(val_runs[:n_val_fault]) | set(train_runs[:n_train_fault])

    run_col = frame["run_id"].astype(str).to_numpy()
    labels = frame["label"].astype(str).to_numpy()
    labels[np.isin(run_col, sorted(fault_runs))] = "fault"
    frame["label"] = labels
    return (frame,
            np.where(np.isin(run_col, train_runs))[0],
            np.where(np.isin(run_col, val_runs))[0])


@pytest.mark.unit
def test_composition_is_measured_in_runs_not_rows():
    """Binlerce satirlik TEK fault kosusu yeterli ornek SAYILMAMALI.

    Satirlar tek bir olayin korele tekrarlaridir; bagimsiz gozlem sayisi
    kosu sayisidir. OpenPLC'de 3563 fault satiri yalniz 2 kosuda.
    """
    train, tr_idx, va_idx = _split_with_fault_runs(1, 1)
    test = _fixture("macyste", "phase", 3, 90)
    max_fpr = 0.5                      # satir esigi = ceil(1/0.5) = 2
    n_rows = int((train.iloc[va_idx]["label"].astype(str) == "fault").sum())
    # Satir-bazli bir kontrol bunu GECIRIRDI: 20 satir >> 2 satir esigi.
    assert n_rows >= 20 > int(np.ceil(1 / max_fpr))
    reasons = validate_split(train, tr_idx, va_idx, "run", max_fpr, test)
    # Kosu-bazli kontrol yine de reddeder: bagimsiz gozlem sayisi 1.
    assert any("bagimsiz 'fault' kosusu" in r for r in reasons), reasons
    assert any("satir sayisi cok olsa da" in r for r in reasons)


@pytest.mark.unit
def test_missing_fault_in_validation_is_refused():
    """Validation'da sinif yoksa esik o sinifi gormeden secilir."""
    train, tr_idx, va_idx = _split_with_fault_runs(0, 2)
    test = _fixture("macyste", "phase", 3, 90)
    reasons = validate_split(train, tr_idx, va_idx, "run", 0.5, test)
    assert any("VALIDATION'da hic 'fault' kosusu yok" in r for r in reasons)


@pytest.mark.unit
def test_missing_fault_in_train_is_refused():
    """Train'de sinif yoksa model onu HIC ogrenmemis olur -- daha agir kusur.

    OpenPLC'de gercekten olusuyor: seed 43/46/48 iki fault kosusunu da
    validation'a atiyor ve model fault gormeden egitiliyor.
    """
    train, tr_idx, va_idx = _split_with_fault_runs(2, 0)
    test = _fixture("macyste", "phase", 3, 90)
    reasons = validate_split(train, tr_idx, va_idx, "run", 0.5, test)
    assert any("TRAIN'de hic 'fault' kosusu yok" in r for r in reasons)


@pytest.mark.unit
def test_two_independent_runs_each_side_satisfies_composition():
    train, tr_idx, va_idx = _split_with_fault_runs(2, 2)
    test = _fixture("macyste", "phase", 3, 90)
    reasons = validate_split(train, tr_idx, va_idx, "run", 0.5, test)
    assert not any("fault" in r for r in reasons), reasons


@pytest.mark.unit
def test_composition_is_skipped_when_no_test_df_is_given():
    """Geriye uyum: test_df verilmezse bilesim denetlenmez."""
    train, tr_idx, va_idx = _split_with_fault_runs(1, 1)
    reasons = validate_split(train, tr_idx, va_idx, "run", 0.5)
    assert not any("fault" in r for r in reasons)


@pytest.mark.unit
def test_run_lodo_refuses_when_fault_composition_is_inadequate():
    """Uctan uca: bilesim eksikligi headline'i engellemeli."""
    good = _fixture("macyste", "phase", 3, 90)           # fault iceriyor
    r = run_lodo(_fixture("openplc", "phase", 6, 90), good, "physical_proxy",
                 ground_truth=_stamped_gt(good["run_id"].unique()),
                 model_factory=_dummy_factory)
    assert r["mode"] == "contract"
    assert any("'fault'" in x for x in r["refusal_reasons"])


@pytest.mark.unit
def test_run_lodo_reports_split_basis_and_refuses_on_bad_split():
    train = _fixture("openplc", "phase", 1, 90).copy()
    train["run_id"] = "tek-kosu"
    good = _fixture("macyste", "phase", 3, 90)
    r = run_lodo(train, good, "physical_proxy",
                 ground_truth=_stamped_gt(good["run_id"].unique()),
                 model_factory=_dummy_factory)
    assert r["split_basis"] == "row"
    assert r["mode"] == "contract"
    assert any("KOSU bazli degil" in x for x in r["refusal_reasons"])


# --------------------------------------------------------------------------
# Faz 1 zorunlu metrikleri: tespit gecikmesi ve alarm yuku
# --------------------------------------------------------------------------
@pytest.mark.unit
def test_detection_latency_measured_from_first_attack_event():
    df = pd.DataFrame({
        "run_id": ["r1"] * 4,
        "timestamp": ["2023-11-14T22:13:40+00:00", "2023-11-14T22:13:41+00:00",
                      "2023-11-14T22:13:43+00:00", "2023-11-14T22:13:45+00:00"],
    })
    out = detection_latency(df, np.array([0, 0, 1, 0]), _stamped_gt(["r1"]))
    assert out["n_detected"] == 1 and out["n_missed"] == 0
    assert out["median_sec"] == pytest.approx(3.0)   # :40 -> :43


@pytest.mark.unit
def test_missed_episode_excluded_from_latency_but_counted():
    """Kacan episode sonsuz gecikme demektir; ortalamayi kirletmemeli."""
    df = pd.DataFrame({
        "run_id": ["r1"] * 2 + ["r2"] * 2,
        "timestamp": ["2023-11-14T22:13:40+00:00", "2023-11-14T22:13:42+00:00",
                      "2023-11-14T22:13:40+00:00", "2023-11-14T22:13:44+00:00"],
    })
    out = detection_latency(df, np.array([0, 1, 0, 0]), _stamped_gt(["r1", "r2"]))
    assert out["n_detected"] == 1 and out["n_missed"] == 1
    assert out["median_sec"] == pytest.approx(2.0)


@pytest.mark.unit
def test_alarm_load_counts_every_row_not_only_attack_phases():
    df = pd.DataFrame({
        "run_id": ["r1"] * 3 + ["r2"] * 3,
        "timestamp": ["2023-11-14T22:13:00+00:00", "2023-11-14T22:13:30+00:00",
                      "2023-11-14T22:14:00+00:00"] * 2,
    })
    load = alarm_load(df, np.array([1, 0, 0, 1, 1, 0]))
    assert load["n_runs"] == 2
    assert load["alarms_per_run_mean"] == pytest.approx(1.5)
    assert load["alarms_per_run_max"] == 2
    assert load["observed_hours"] == pytest.approx(120 / 3600.0, rel=1e-3)
    assert load["alarms_per_hour"] == pytest.approx(90.0, rel=1e-3)


@pytest.mark.unit
def test_headline_result_carries_latency_and_alarm_load():
    good = _fixture("macyste", "phase", 3, 90)
    r = run_lodo(_fixture("openplc", "phase", 4, 90), good, "physical_proxy",
                 ground_truth=_stamped_gt(good["run_id"].unique()),
                 model_factory=_dummy_factory, force_headline=True)
    assert r["latency_manipulation"] is not None
    assert r["alarm_load"]["n_runs"] == good["run_id"].nunique()


# --------------------------------------------------------------------------
# Coklu seed
# --------------------------------------------------------------------------
@pytest.mark.unit
def test_seed_changes_the_run_split():
    df = _fixture("openplc", "phase", 6, 90)
    splits = {tuple(sorted(set(df.iloc[_split_by_run(df, 0.3, s)[1]]["run_id"])))
              for s in (1, 2, 3, 4, 5)}
    assert len(splits) > 1, "seed bolmeyi etkilemiyor"


@pytest.mark.unit
def test_seed_is_forwarded_to_factories_that_accept_it():
    seen = []

    def spy_factory(numeric, categorical, seed=42):
        seen.append(seed)
        return _dummy_factory(numeric, categorical)

    run_lodo(_fixture("openplc", "phase", 4, 90),
             _fixture("macyste", "phase", 3, 90), "physical_strict",
             model_factory=spy_factory, seed=1234)
    assert seen == [1234]


@pytest.mark.unit
def test_legacy_factory_without_seed_still_works():
    def old_factory(numeric, categorical):
        return _dummy_factory(numeric, categorical)

    r = run_lodo(_fixture("openplc", "phase", 4, 90),
                 _fixture("macyste", "phase", 3, 90), "physical_strict",
                 model_factory=old_factory, seed=7)
    assert r["seed"] == 7


@pytest.mark.unit
def test_multiseed_reports_spread_not_just_best_seed():
    good = _fixture("macyste", "phase", 3, 90)
    out = run_lodo_multiseed(
        _fixture("openplc", "phase", 4, 90), good, "physical_proxy",
        ground_truth=_stamped_gt(good["run_id"].unique()),
        seeds=(1, 2, 3), model_factory=_dummy_factory, force_headline=True)
    assert out["seeds"] == [1, 2, 3]
    assert len(out["per_seed"]) == 3
    assert [r["seed"] for r in out["per_seed"]] == [1, 2, 3]
    for metric, stats in out["summary"].items():
        assert stats["min"] <= stats["mean"] <= stats["max"], metric


@pytest.mark.unit
def test_pump_rpm_never_reaches_the_common_schema():
    for schema in SCHEMA_NAMES:
        frame = build_feature_frame(_fixture("macyste", "phase", 2, 90), schema)
        for banned in ("hydraulic_pump_rpm_left", "hydraulic_pump_rpm_right"):
            assert banned not in frame.columns
