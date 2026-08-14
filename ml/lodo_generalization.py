#!/usr/bin/env python3
r"""
lodo_generalization.py - LODO (Leave-One-Domain-Out) genelleme iskeleti

BU BIR SOZLESME / DOGRULAMA ARACIDIR, "headline sonuc" ureticisi DEGILDIR.
Amac: uzun harici kampanyadan (MaCySTe vb.) ONCE sema ve etiket hatalarini
yakalamak. Veri hazir degilse (kisa/tek-tekrar/kosu-duzeyi etiket) araç
METRIK URETMEYI REDDEDER (contract mode).

Fiziksel eşleme sozlesmesi (kullanici teyitli):
  * OpenPLC rpm_command/1200  <->  MaCySTe throttle_command(0-1)
    Bunlar BIREBIR ayni fiziksel degisken DEGILDIR:
      - rpm_command  : ana makine devir hedefi (ballast_v6.st: 0..1200)
      - throttle_cmd : motor telgrafindan istenen guc orani (0..1)
    Bu yuzden rol adi "propulsion_demand_norm" (proxy); tezde
    "normalized propulsion-demand proxy" denir, "ayni fiziksel degisken" DENMEZ.
  * OpenPLC load_cmd KULLANILMAZ: jenerator elektrik yuku (gen_load/bus_freq);
    propulsion throttle degildir.
  * rudder_angle/35: gercek dumen geri bildirimi -> birebir ortak (strict).
  * MaCySTe sol/sag throttle tek role ROLE-COALESCE edilir. Bu SENKRON L/R
    ORTALAMASI DEGILDIR: sol ve sag telgraf ayri sunuculardan (macyste-engine-l
    / -r) gelir, ayni satirda bulunmazlar. Islem satir-bazlidir -- "o satir
    hangi tarafi tasiyorsa o". Gercek zaman-pencereli L/R ortalamasi ayri bir
    on-isleme adimi olurdu; bu iskelette YOK. Sol-sag farki ortak semaya
    girmez (OpenPLC'de karsiligi yok).
  * rpm1/rpm2 = hidrolik pompa RPM'i -> hydraulic_pump_rpm_* (yan analiz),
    propulsion ozelliklerinden ve ana makine RPM'inden ayri tutulur.

Iki fiziksel sema varyanti:
  physical_strict : yalniz birebir ortak fiziksel roller (rudder + zaman ozelligi)
  physical_proxy  : strict + propulsion_demand_norm (rpm_command<->throttle proxy)
Boylece kotu sonuç, gercek domain basarisizligi mi yoksa RPM-hedefi/guc-komutu
farki mi -> ayirt edilebilir.

Diger sozlesme kurallari:
  * Ham adres/port/register modele GIRMEZ (sert blocklist + allowlist).
  * modbus_quantity ANA semadan cikarilir: float-cifti vs tek-register encoding
    farkini ele verebilir (ayri encoding-hassasiyet deneyi olarak raporlanmali).
  * Eksik semantik alan 0 YAPILMAZ: <role> + <role>_present gosterimi
    ("alan yok" ile "gercek deger 0" ayrilir).
  * Fiziksel bant tek basina yetmez (35 derece icinde de saldiri olabilir):
    zaman ozelligi olarak degisim hizi (rudder_rate) eklenir. Hiz YALNIZ ardisik
    gercek dumen ornekleri arasinda hesaplanir (araya giren Modbus olaylari
    sahte "hiz=0" uretmesin diye).
  * Komut-geri bildirim farki ORTAK SEMAYA GIRMEZ. MaCySTe rudder_command
    dogrudan dumen acisi emri; OpenPLC heading_cmd rota hedefidir ve kontrol
    mantigi bunu +-15 derece dumen hareketine cevirir -- birebir eslesmez.
    Yan deney olarak (yalniz-MaCySTe teshis; physical_strict'e ve ana LODO
    kiyasina GIRMEZ, ayri raporlanir):
        rudder_tracking_error = (rudder_command - rudder_angle_actual) / 35
  * Esik yalniz OpenPLC VALIDATION bolumunden secilir (train tahmininden DEGIL)
    ve `pred = proba >= thr` altinda hedef FPR'yi AMPIRIK olarak saglar
    (quantile tek basina beraberlikler yuzunden garanti vermez).
  * Saldiri TURU 'scenario', saldiri ANI 'label' ile maskelenir: event recall
    maskesi (scenario AILESI in {manipulation, recon}) & (label==attack).
    Senaryo adi once aileye indirgenir ('macyste_manipulation' ->
    'manipulation'); eslesmeyen ad 'other' olur ve contract guard REDDEDER --
    aksi halde butun maskeler bos kalip metrikler sessizce null doner.
  * Bootstrap kosu/episode bazinda; normal FPR ve fault FPR ayri raporlanir.
    Episode sinirlari ONCE ground-truth faz zaman damgalarindan kurulur
    (olay CSV'sinde 'phase' kolonu yoktur); kullanilan kaynak sonuca
    'episode_basis' olarak yazilir.
  * ground-truth tek dosya ya da run_id->ground-truth eslemesi olabilir.

Kullanim (veri hazir OLDUGUNDA):
    python ml/lodo_generalization.py \
        --train data/openplc_all.csv \
        --test  captures/macyste/<dir>/events.csv \
        --ground-truth captures/macyste/<dir>/ground-truth.json \
        --schema all

Yapisal dogrulama (veri gerekmez, sklearn gerekmez):
    python ml/lodo_generalization.py --self-test
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import statistics
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. Sert domain-kimligi blocklist -- bu kolonlar modele ASLA girmez.
# ---------------------------------------------------------------------------
DOMAIN_IDENTITY_BLOCKLIST = frozenset({
    "source_dataset", "sensor", "register_space", "macyste_decoded",
    "src_ip", "dest_ip", "dst_ip", "sip", "dip",
    "src_port", "dest_port", "dst_port", "src_port_role",
    "modbus_address", "modbus_quantity", "modbus_quantity_norm",  # encoding sizintisi
    "run_id", "scenario", "phase", "label", "attack_type",
    "alert_signature", "alert_sid", "timestamp", "_order",
    # POZISYONLA cozulmus, ham (domaine gore anlami degisen) register alanlari:
    "engine_rpm", "ballast_level", "ballast_setpoint", "rpm_command",
    "heading_cmd", "heading", "rudder_angle", "gen_load", "bus_freq", "load_cmd",
    "hydraulic_pump_rpm_left", "hydraulic_pump_rpm_right",  # MaCySTe: pompa RPM'i
    "throttle_command_left", "throttle_command_right",  # ham; role coalesce edilir
})


# ---------------------------------------------------------------------------
# 1. Normalizasyon (testbed-bagimsiz fiziksel olcek)
# ---------------------------------------------------------------------------
def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _n_rudder(v):      # dumen acisi -> 35 derece = 1.0
    return _num(v) / 35.0


def _n_rpm1200(v):     # ana makine devir hedefi 0..1200 -> 0..1 (propulsion proxy)
    return _num(v) / 1200.0


def _n_unit(v):        # throttle telgrafi zaten 0..1
    return _num(v)


def _n_pump3000(v):    # hidrolik pompa RPM -> 0..1 (ANA MAKINE RPM'i DEGIL; yan analiz)
    return _num(v) / 3000.0


# ROLE_SPECS[source] = [(ham_kolonlar, kanonik_rol, normalize_fn), ...]
# Birden fazla ham kolon -> tek role ROLE-COALESCE (satir-bazli). Sol/sag ayrimi
# girmez; senkron L/R ortalamasi DEGILDIR (bkz. modul basligi).
ROLE_SPECS = {
    "openplc": [
        (["rudder_angle"], "rudder_norm", _n_rudder),
        (["rpm_command"], "propulsion_demand_norm", _n_rpm1200),  # <-- proxy
    ],
    "macyste": [
        (["rudder_angle"], "rudder_norm", _n_rudder),
        (["throttle_command_left", "throttle_command_right"],
         "propulsion_demand_norm", _n_unit),                      # sol+sag coalesce
    ],
}

# Yan-analiz rolleri (ANA semaya girmez): yalniz-MaCySTe fiziksel alanlar.
SIDE_ROLE_SPECS = {
    "macyste": [
        (["hydraulic_pump_rpm_left"], "hydraulic_pump_rpm_left", _n_pump3000),
        (["hydraulic_pump_rpm_right"], "hydraulic_pump_rpm_right", _n_pump3000),
    ],
}

STRICT_ROLES = ["rudder_norm"]                          # birebir ortak fiziksel
PROXY_ROLES = ["rudder_norm", "propulsion_demand_norm"]  # strict + proxy

# Fiziksel bantlar NORMALIZE role gore (adrese gore DEGIL).
RUDDER_SAFE = (-1.0, 1.0)    # |dumen| <= 35 derece
PROP_SAFE = (0.0, 1.0)       # propulsion demand 0..1

BASE_FLOW_NUM = ["pkts_toserver", "pkts_toclient", "bytes_toserver", "bytes_toclient"]
BASE_FLOW_CAT = ["flow_state"]
PROTO_CAT = ["modbus_function", "modbus_access"]

SCHEMA_NAMES = ["flow", "protocol", "physical_strict", "physical_proxy"]


def _schema_roles(schema):
    return {"physical_strict": STRICT_ROLES, "physical_proxy": PROXY_ROLES}.get(schema, [])


def _uses_protocol(schema):
    return schema in ("protocol", "physical_strict", "physical_proxy")


# ---------------------------------------------------------------------------
# 2. Yardimcilar: rol cozumu, bant, zaman-hizi
# ---------------------------------------------------------------------------
# Senaryo AILELERI. Kampanya kosucusu senaryoyu '<kaynak>_<aile>' olarak yazar
# (ör. 'macyste_manipulation'); fixture'lar yalin ad kullanir. Yalin esitlik
# aranirsa GERCEK veride butun maskeler bos kalir ve tum metrikler sessizce
# null doner -- bu sessiz basarisizlik acikca engellenir.
SCENARIO_FAMILIES = ("normal", "fault", "manipulation", "recon")


def scenario_family(values):
    """Senaryo adlarini aileye indirge; eslesmeyen ad 'other' olur (dusurulmez)."""
    families = []
    for raw in values:
        name = str(raw).strip().lower()
        match = "other"
        for candidate in SCENARIO_FAMILIES:
            if name == candidate or name.endswith("_" + candidate):
                match = candidate
                break
        families.append(match)
    return np.asarray(families, dtype=object)


_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _epoch_seconds(values):
    """Zaman damgasi -> epoch saniyesi (pandas cozunurlugunden BAGIMSIZ).

    `.astype("int64")` KULLANILMAZ: pandas surumune ve girdiye gore ns ya da us
    dondurur; sabit bir 1e9 bolenle olcek 1000 kat kayar (hiz ve faz pencereleri
    sessizce bozulur). NaT -> NaN.
    """
    ts = pd.to_datetime(values, format="ISO8601", errors="coerce", utc=True)
    return (ts - _EPOCH).dt.total_seconds()


def _band(value, lo, hi, is_write):
    if not is_write:
        return "na"
    v = _num(value)
    if v != v:
        return "unknown"
    return "in_band" if lo <= v <= hi else "out_of_band"


def _resolve_roles(df, source, roles_wanted):
    specs = ROLE_SPECS.get(source, ROLE_SPECS["openplc"])
    values, present = {}, {}
    for raw_cols, role, fn in specs:
        if role not in roles_wanted:
            continue
        cols = [c for c in raw_cols if c in df.columns]
        if not cols:
            values[role] = pd.Series(np.nan, index=df.index)
            present[role] = pd.Series(0, index=df.index)
            continue
        stacked = pd.concat([df[c].map(fn) for c in cols], axis=1)
        # ROLE-COALESCE: satir-bazli, NaN-atlayan ortalama. Sol/sag ayni satirda
        # bulunmadigi icin pratikte "mevcut olan taraf" secilir; ikisi birden
        # varsa ortalamalari alinir. Senkron zaman-pencereli L/R ortalamasi DEGIL.
        v = stacked.mean(axis=1)
        values[role] = v
        present[role] = v.notna().astype(int)
    for role in roles_wanted:                          # eksik rol -> yok isareti
        values.setdefault(role, pd.Series(np.nan, index=df.index))
        present.setdefault(role, pd.Series(0, index=df.index))
    return values, present


def _rate_per_run(df, values):
    """Kosu icinde zaman-sirali degisim hizi (fiziksel bant tek basina yetmez).

    diff YALNIZ gercek dumen ORNEKLERI arasinda alinir. Araya baska Modbus
    olaylari (rol degeri NaN) girdiginde tum-satir diff'i NaN uretir; fillna(0)
    bunu sahte olarak "hiz = 0" yapar ve hizli dumen kacislarini gizler.
    Orneksiz satirlarin hizi 0'dir; bunu <role>_present bayragi ayirt eder.
    """
    rate = pd.Series(0.0, index=df.index)
    if "run_id" not in df.columns or "timestamp" not in df.columns:
        return rate
    t = _epoch_seconds(df["timestamp"])
    sampled = values.notna() & t.notna()
    for _, idx in df.groupby(df["run_id"].astype(str)).groups.items():
        idx = [i for i in idx if sampled.loc[i]]
        if len(idx) < 2:
            continue
        order = t.loc[idx].sort_values().index
        dv = values.loc[order].diff()
        dt = t.loc[order].diff().replace(0, np.nan)
        rate.loc[order] = (dv / dt).fillna(0.0).to_numpy()
    return rate


# ---------------------------------------------------------------------------
# 3. Feature frame (allowlist; ham adres/port/quantity GIRMEZ)
# ---------------------------------------------------------------------------
def build_feature_frame(df: pd.DataFrame, schema: str) -> pd.DataFrame:
    if schema not in SCHEMA_NAMES:
        raise ValueError(f"bilinmeyen sema: {schema}")
    source = str(df["source_dataset"].iloc[0]) if "source_dataset" in df.columns else "openplc"
    numeric, categorical = [], []
    out = pd.DataFrame(index=df.index)

    # flow (ortak)
    for c in BASE_FLOW_NUM:
        out[c] = pd.to_numeric(df.get(c), errors="coerce").fillna(0.0) if c in df.columns else 0.0
        numeric.append(c)
    out["flow_state"] = df["flow_state"].astype(str) if "flow_state" in df.columns else "unknown"
    categorical.append("flow_state")

    is_write = pd.to_numeric(df.get("modbus_function"), errors="coerce").isin([5, 6, 15, 16]) \
        if "modbus_function" in df.columns else pd.Series(False, index=df.index)

    if _uses_protocol(schema):
        out["modbus_function"] = df.get("modbus_function", "na").astype(str)
        out["modbus_access"] = df.get("modbus_access", "na").astype(str)
        categorical += PROTO_CAT
        # modbus_quantity BILINCLI olarak DAHIL EDILMEZ (encoding sizintisi).

    roles = _schema_roles(schema)
    if roles:
        values, present = _resolve_roles(df, source, roles)
        for role in roles:
            out[role] = values[role].fillna(0.0)               # deger (imputed)
            out[f"{role}_present"] = present[role]             # "alan var mi" isareti
            numeric += [role, f"{role}_present"]
        # zaman ozelligi: dumen degisim hizi
        out["rudder_rate"] = _rate_per_run(df, values["rudder_norm"])
        numeric.append("rudder_rate")
        # fiziksel bantlar (role gore)
        out["rudder_band"] = [_band(v, *RUDDER_SAFE, w)
                              for v, w in zip(values["rudder_norm"], is_write)]
        categorical.append("rudder_band")
        if "propulsion_demand_norm" in roles:
            out["propulsion_band"] = [_band(v, *PROP_SAFE, w)
                                      for v, w in zip(values["propulsion_demand_norm"], is_write)]
            categorical.append("propulsion_band")

    frame = out[numeric + categorical].copy()
    for c in numeric:
        frame[c] = pd.to_numeric(frame[c], errors="coerce").fillna(0.0)
    for c in categorical:
        frame[c] = frame[c].astype(str).fillna("na")
    frame.attrs["numeric"] = numeric
    frame.attrs["categorical"] = categorical
    assert_no_leakage(frame)
    return frame


def _schema_spec(schema):
    fr = build_feature_frame(_fixture("openplc", "phase", 3, 90), schema)
    return {"numeric": fr.attrs["numeric"], "categorical": fr.attrs["categorical"]}


def assert_no_leakage(frame: pd.DataFrame) -> None:
    leaked = DOMAIN_IDENTITY_BLOCKLIST.intersection(frame.columns)
    if leaked:
        raise AssertionError(f"SIZINTI: domain-kimligi kolonlari: {sorted(leaked)}")


# ---------------------------------------------------------------------------
# 4. Hazir-olma guard'lari (contract)
# ---------------------------------------------------------------------------
MIN_REPEATS = 3
MIN_DURATION_SEC = 60.0


def _gt_has_phases(ground_truth, run_id=None):
    if not ground_truth:
        return False
    if run_id is not None and isinstance(ground_truth, dict) and run_id in ground_truth:
        return bool(ground_truth[run_id].get("phases"))
    return bool(ground_truth.get("phases")) if isinstance(ground_truth, dict) else False


def validate_readiness(test_df, ground_truth):
    reasons = []
    runs = test_df["run_id"].astype(str).unique() if "run_id" in test_df.columns else []
    if not any(_gt_has_phases(ground_truth, r) for r in (runs if len(runs) else [None])):
        reasons.append("ground-truth fazlari yok (episode sinirlari faz-bazli olmali; "
                       "uzun kampanyada run_id->ground-truth eslemesi gerekir)")
    if "label" in test_df.columns and "run_id" in test_df.columns:
        per_run = test_df.groupby("run_id")["label"].nunique()
        if len(per_run) and (per_run <= 1).all():
            reasons.append("etiketler kosu-duzeyi gorunuyor; fault kosusunun saglikli "
                           "baslangici da 'fault' olabilir (faz-bazli etiket gerekli)")
    if "scenario" in test_df.columns:
        raw_scen = test_df["scenario"].astype(str)
        fam = pd.Series(scenario_family(raw_scen), index=test_df.index)
        unmapped = sorted(set(raw_scen[fam == "other"]))
        if unmapped:
            # Aksi halde butun maskeler bos kalir ve metrikler sessizce null doner.
            reasons.append(
                f"senaryo adlari {SCENARIO_FAMILIES} ailelerinden birine "
                f"eslenemedi: {unmapped}"
            )
        if "run_id" in test_df.columns:
            reps = fam.groupby(fam).apply(
                lambda s: test_df.loc[s.index, "run_id"].nunique()
            )
            thin = reps[reps < MIN_REPEATS]
            if len(thin):
                reasons.append(f"senaryo basina < {MIN_REPEATS} tekrar: {thin.to_dict()}")
    if "timestamp" in test_df.columns and "run_id" in test_df.columns:
        ts = pd.to_datetime(test_df["timestamp"], format="ISO8601",
                            errors="coerce", utc=True)
        dur = ts.groupby(test_df["run_id"]).agg(lambda s: (s.max() - s.min()).total_seconds())
        short = dur[dur < MIN_DURATION_SEC]
        if len(short):
            reasons.append(f"kosu suresi < {MIN_DURATION_SEC:.0f}s: "
                           f"{ {k: round(v, 1) for k, v in short.to_dict().items()} }")
    return (len(reasons) == 0), reasons


# ---------------------------------------------------------------------------
# 5. Episode-bazli bootstrap
# ---------------------------------------------------------------------------
def episode_bootstrap_ci(episode_hits, n_boot=2000, alpha=0.05, seed=42):
    a = np.asarray(episode_hits, dtype=float)
    if len(a) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    boots = [a[rng.randint(0, len(a), len(a))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(a.mean()), float(lo), float(hi))


# ---------------------------------------------------------------------------
# 6. Model fabrikasi (handle_unknown='ignore' -- gorulmemis kategori yutar)
# ---------------------------------------------------------------------------
def default_model_factory(numeric, categorical, seed=42):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    pre = ColumnTransformer([
        ("num", "passthrough", numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return Pipeline([("pre", pre),
                     ("clf", RandomForestClassifier(n_estimators=200,
                                                    random_state=seed))])


def _build_model(factory, numeric, categorical, seed):
    """Seed'i KABUL EDEN fabrikaya ilet; kabul etmeyen eski fabrikalari kirma."""
    if "seed" in inspect.signature(factory).parameters:
        return factory(numeric, categorical, seed=seed)
    return factory(numeric, categorical)


# ---------------------------------------------------------------------------
# 7. LODO kosusu
# ---------------------------------------------------------------------------
def _split_by_run(df, val_frac=0.3, seed=42):
    """(train_idx, val_idx, bolme_kaynagi) dondur; esik YALNIZ val'dan secilir.

    Kaynak 'run' olmalidir. 'row' yedegi GUVENILMEZDIR: ayni kosunun satirlari
    hem train'e hem validation'a duser, esik iyimser cikar (etiket sizintisi).
    Sessizce kullanilmaz -- validate_split() bunu reddeder.
    """
    n = len(df)
    if "run_id" in df.columns:
        # np.asarray: pandas .unique() ExtensionArray dondurebilir; RandomState
        # .shuffle bunun uzerinde uyari uretir.
        runs = np.asarray(df["run_id"].astype(str).unique(), dtype=object)
        if len(runs) >= 2:
            rng = np.random.RandomState(seed)
            rng.shuffle(runs)
            cut = max(1, min(len(runs) - 1, int(len(runs) * (1 - val_frac))))
            tr_runs = set(runs[:cut])
            mask = df["run_id"].astype(str).isin(tr_runs).to_numpy()
            return np.where(mask)[0], np.where(~mask)[0], "run"
    idx = np.arange(n)
    np.random.RandomState(seed).shuffle(idx)
    cut = int(n * (1 - val_frac))
    return idx[:cut], idx[cut:], "row"


def validate_split(train_df, tr_idx, va_idx, split_basis, max_fpr, test_df=None):
    """Esigin guvenilir secilebildigini dogrula (contract'in train tarafi).

    validate_readiness yalniz TEST alanina bakar; esigin nereden geldigi de en
    az onun kadar kritiktir: kotu bir train/validation bolmesi iyimser esik
    uretir ve butun headline metrikleri sessizce sisirir.

    test_df verilirse BILESIM de denetlenir: test setinde bulunan her negatif
    etiket sinifi (ornegin 'fault') validation'da da yeterli sayida temsil
    edilmelidir. Aksi halde o sinif icin raporlanan FPR kalibre edilmemistir.
    """
    reasons = []
    if split_basis != "run":
        reasons.append(
            "train/validation bolmesi KOSU bazli degil (satir-bazli rastgele "
            "bolme): ayni kosunun satirlari her iki tarafa dusuyor, esik "
            "iyimser cikar; egitim verisinde >= 2 run_id gerekir"
        )
    if len(va_idx) == 0:
        reasons.append("validation bolumu bos; esik OpenPLC validation'dan secilemez")
        return reasons
    if "run_id" in train_df.columns:
        tr_runs = set(train_df.iloc[tr_idx]["run_id"].astype(str))
        va_runs = set(train_df.iloc[va_idx]["run_id"].astype(str))
        shared = sorted(tr_runs & va_runs)
        if shared:
            reasons.append(f"train ve validation ayni kosulari paylasiyor: {shared[:5]}")
    if "label" in train_df.columns:
        yva = train_df.iloc[va_idx]["label"].astype(str).to_numpy()
        n_neg = int((yva != "attack").sum())
        # Hedef FPR'yi ampirik cozebilmek icin en az 1/max_fpr negatif gerekir;
        # daha azinda quantile hedef orani ayirt edemez.
        need = int(np.ceil(1.0 / max_fpr)) if max_fpr > 0 else 0
        if n_neg == 0:
            reasons.append("validation'da normal satir yok; esik secilemez "
                           "(varsayilan 0.5'e dusulur, bu bir sonuc DEGILDIR)")
        elif n_neg < need:
            reasons.append(f"validation negatif sayisi {n_neg} < {need} (=1/max_fpr); "
                           f"hedef FPR {max_fpr} bu ornek buyuklugunde cozulemez")
    reasons += _validate_class_composition(train_df, tr_idx, va_idx, test_df)
    return reasons


# Bir etiket sinifinin "yeterli ornek" sayildigi asgari BAGIMSIZ KOSU sayisi.
# SATIR sayisi olcut DEGILDIR: binlerce satirlik TEK bir fault kosusu yeterli
# ornek gibi sunulamaz, cunku o satirlar tek bir olayin korele tekrarlaridir.
MIN_CLASS_RUNS_PER_SIDE = 2


def _validate_class_composition(train_df, tr_idx, va_idx, test_df):
    """Test setindeki her negatif sinif, train VE validation'da temsil edilmeli.

    Iki ayri kusuru birlikte yakalar:

    1. Validation'da sinif YOKSA esik o sinifi hic gormeden secilir; raporlanan
       o-sinif FPR'si kalibre edilmemis olur.
    2. Train'de sinif YOKSA model o sinifi hic ogrenmemis olur -- daha da agir
       bir kusur; "fault'u attack'tan ayirt ediyor" iddiasi bos kalir.

    Olcut BAGIMSIZ KOSU sayisidir. OpenPLC'de 3563 fault satiri 20 kosudan
    yalniz 2'sinde (kosu 19 ve 20); satir bazli bir esik bu iki kosuyu "3563
    ornek" gibi gosterir, oysa bagimsiz gozlem sayisi 2'dir.
    """
    reasons = []
    if test_df is None or "label" not in test_df.columns:
        return reasons
    if "label" not in train_df.columns or "run_id" not in train_df.columns:
        return reasons

    test_neg = {lab for lab in test_df["label"].astype(str).unique()
                if lab != "attack"}
    tr_part, va_part = train_df.iloc[tr_idx], train_df.iloc[va_idx]

    def runs_with(part, label):
        labels = part["label"].astype(str)
        return set(part.loc[labels == label, "run_id"].astype(str))

    for label in sorted(test_neg):
        tr_runs = runs_with(tr_part, label)
        va_runs = runs_with(va_part, label)
        total = len(tr_runs | va_runs)
        if not va_runs:
            reasons.append(
                f"test setinde '{label}' var ama VALIDATION'da hic '{label}' "
                f"kosusu yok; esik '{label}' trafigi gormeden secilir ve "
                f"raporlanan '{label}' FPR'si kalibre edilmemis olur"
            )
        if not tr_runs:
            reasons.append(
                f"test setinde '{label}' var ama TRAIN'de hic '{label}' kosusu "
                f"yok; model bu sinifi hic ogrenmemis olur"
            )
        if va_runs and len(va_runs) < MIN_CLASS_RUNS_PER_SIDE:
            reasons.append(
                f"validation'da yalniz {len(va_runs)} bagimsiz '{label}' kosusu "
                f"var (< {MIN_CLASS_RUNS_PER_SIDE}); satir sayisi cok olsa da "
                f"bagimsiz gozlem sayisi yetersiz "
                f"(egitim verisinde toplam {total} '{label}' kosusu var)"
            )
        if tr_runs and len(tr_runs) < MIN_CLASS_RUNS_PER_SIDE:
            reasons.append(
                f"train'de yalniz {len(tr_runs)} bagimsiz '{label}' kosusu var "
                f"(< {MIN_CLASS_RUNS_PER_SIDE}); bu sinif icin ogrenme tek "
                f"olaya dayanir (egitim verisinde toplam {total} kosu)"
            )
    return reasons


def _threshold_at_fpr(proba, y, max_fpr):
    """pred = (proba >= thr) icin hedef FPR'yi AMPIRIK olarak saglayan esik.

    Salt quantile yetmez: agac toplulugu ayrik olasilik uretir ve esit degerler
    (ozellikle 0.0) yiginlasir. `>=` karsilastirmasi bu yigini tumden pozitif
    yaptigi icin gerceklesen FPR hedefi asabilir. Bu yuzden quantile bir
    BASLANGIC noktasidir; esik, kosul saglanana dek ayrik degerler uzerinde
    yukari itilir.
    """
    proba = np.asarray(proba, dtype=float)
    neg = proba[np.asarray(y) == 0]
    if len(neg) == 0:
        return 0.5
    start = float(np.quantile(neg, 1 - max_fpr))
    for candidate in np.unique(neg[neg >= start]):
        if float((neg >= candidate).mean()) <= max_fpr:
            return float(candidate)
    # Hicbir gozlenen deger hedefi tutturamiyor -> en buyugun de ustune cik
    # (validation'da FPR = 0; esigin muhafazakar oldugu raporlanmalidir).
    return float(np.nextafter(float(neg.max()), np.inf))


def run_lodo(train_df, test_df, schema, ground_truth=None,
             model_factory=None, max_fpr=0.01, force_headline=False,
             val_frac=0.3, seed=42):
    # Contract iki tarafli: TEST alani hazir mi, ve esik guvenilir bir
    # train/validation bolmesinden mi geliyor.
    tr_idx, va_idx, split_basis = _split_by_run(train_df, val_frac, seed)
    reasons = validate_readiness(test_df, ground_truth)[1]
    reasons = reasons + validate_split(train_df, tr_idx, va_idx,
                                       split_basis, max_fpr, test_df)
    ready = not reasons
    result = {"schema": schema, "seed": seed, "ready": ready,
              "split_basis": split_basis, "refusal_reasons": reasons}

    spec = _schema_spec(schema)
    factory = model_factory or default_model_factory
    model = _build_model(factory, spec["numeric"], spec["categorical"], seed)

    Xtr = build_feature_frame(train_df.iloc[tr_idx], schema)
    ytr = (train_df.iloc[tr_idx]["label"].astype(str) == "attack").astype(int).to_numpy()
    model.fit(Xtr, ytr)

    if not (ready or force_headline):
        result.update(mode="contract", structural_ok=True,
                      note="headline REDDEDILDI; nedenler giderilmeli. Yapisal "
                           "kontroller (sema, sizinti, train/val bolme, fit) gecti.")
        return result

    # Esik YALNIZ validation'dan
    Xva = build_feature_frame(train_df.iloc[va_idx], schema)
    yva = (train_df.iloc[va_idx]["label"].astype(str) == "attack").astype(int).to_numpy()
    thr = _threshold_at_fpr(_proba(model, Xva), yva, max_fpr)

    Xte = build_feature_frame(test_df, schema)
    pred = (_proba(model, Xte) >= thr).astype(int)
    # Saldiri TURU scenario'dan, saldiri ANI label'dan gelir. Ikisi birlikte
    # kullanilmalidir: bir manipulation kosusunun baslangic/bitis bolumleri
    # normaldir; yalniz scenario ile maskelenirse bunlar recall'a karisir ve
    # gercek recall'i asagi ceker.
    raw_scen = test_df["scenario"].astype(str)
    scen = scenario_family(raw_scen)
    lab = (test_df["label"].astype(str).to_numpy() if "label" in test_df.columns
           else np.full(len(test_df), "", dtype=object))
    is_attack = (lab == "attack")

    result["mode"] = "headline" if ready else "forced"
    result["threshold"] = round(float(thr), 4)
    # Ham ad -> aile eslemesi denetlenebilir olsun; 'other' sessizce dusmesin.
    result["scenario_map"] = {
        raw: fam for raw, fam in sorted(set(zip(raw_scen, scen)))
    }
    for grp in ("normal", "fault"):
        m = (scen == grp) & ~is_attack          # simetri: saldiri satiri FPR'ye girmez
        result[f"fpr_{grp}"] = round(float(pred[m].mean()), 4) if m.any() else None
        result[f"n_{grp}"] = int(m.sum())
    for atk in ("manipulation", "recon"):
        m = (scen == atk) & is_attack
        result[f"n_{atk}_attack_rows"] = int(m.sum())
        if not m.any():
            result[f"recall_event_{atk}"] = None
            result[f"recall_episode_{atk}"] = None
            continue
        result[f"recall_event_{atk}"] = round(float(pred[m].mean()), 4)
        hits, basis = _episode_hits(test_df[m], pred[m], ground_truth)
        mm, lo, hi = episode_bootstrap_ci(hits, seed=seed)
        result[f"recall_episode_{atk}"] = {"mean": round(mm, 3),
                                           "ci95": [round(lo, 3), round(hi, 3)],
                                           "n_episodes": len(hits),
                                           "episode_basis": basis}
        result[f"latency_{atk}"] = detection_latency(test_df[m], pred[m],
                                                     ground_truth)
    # Operasyonel yuk: tum test alani uzerinden (yalniz saldiri fazlari degil).
    result["alarm_load"] = alarm_load(test_df, pred)
    return result


SUMMARY_METRICS = ("fpr_normal", "fpr_fault",
                   "recall_event_manipulation", "recall_event_recon")


def run_lodo_multiseed(train_df, test_df, schema, ground_truth=None,
                       seeds=(42,), **kwargs):
    """Ayni protokolu birden fazla seed ile kosur ve yayilimi ozetler.

    Tek seed'lik bir sonuc, bolme ve model rastgeleliginden gelen degiskenligi
    gizler. Faz 1 protokolu coklu seed ister; ozet MIN/MAX'i de tasir ki
    "en iyi seed" secilerek rapor edilmesin.
    """
    runs = [run_lodo(train_df, test_df, schema, ground_truth, seed=int(s), **kwargs)
            for s in seeds]
    summary = {}
    for metric in SUMMARY_METRICS:
        vals = [r[metric] for r in runs if r.get(metric) is not None]
        if vals:
            summary[metric] = {"mean": round(statistics.fmean(vals), 4),
                               "min": round(min(vals), 4),
                               "max": round(max(vals), 4)}
    for atk in ("manipulation", "recon"):
        vals = [r[f"recall_episode_{atk}"]["mean"] for r in runs
                if r.get(f"recall_episode_{atk}")]
        if vals:
            summary[f"recall_episode_{atk}"] = {"mean": round(statistics.fmean(vals), 3),
                                                "min": round(min(vals), 3),
                                                "max": round(max(vals), 3)}
    return {"schema": schema,
            "seeds": [int(s) for s in seeds],
            "ready": all(r["ready"] for r in runs),
            "mode": runs[0].get("mode") if runs else None,
            "summary": summary,
            "per_seed": runs}


def _proba(model, X):
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    return np.asarray(model.predict(X), dtype=float)


def _phase_windows(ground_truth, run_id):
    """Bir kosu icin (baslangic_zamani, faz_adi) listesi -- ground-truth'tan.

    Gercek ground-truth.json fazlari zaman damgasi tasir; olay CSV'sinde 'phase'
    kolonu YOKTUR. Episode sinirlari bu yuzden once ground-truth'tan kurulur.
    """
    gt = ground_truth
    if isinstance(gt, dict) and run_id in gt and isinstance(gt[run_id], dict):
        gt = gt[run_id]
    phases = gt.get("phases") if isinstance(gt, dict) else None
    if not isinstance(phases, list):
        return []
    windows = []
    for entry in phases:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("phase", entry.get("name", ""))).strip()
        stamp = pd.to_datetime(entry.get("timestamp"), format="ISO8601",
                               errors="coerce", utc=True)
        if name and not pd.isna(stamp):
            windows.append(((stamp - _EPOCH).total_seconds(), name))
    windows.sort()
    return windows


def _episode_keys(sub_df, ground_truth):
    """(episode anahtari, sinirlarin KAYNAGI) dondur."""
    if "run_id" in sub_df.columns:
        run = sub_df["run_id"].astype(str)
    else:
        run = pd.Series("all", index=sub_df.index)

    if ground_truth and "timestamp" in sub_df.columns:
        t = _epoch_seconds(sub_df["timestamp"])
        keys = pd.Series(index=sub_df.index, dtype=object)
        resolved = True
        for run_id, idx in run.groupby(run).groups.items():
            windows = _phase_windows(ground_truth, run_id)
            if not windows:
                resolved = False
                break
            edges = np.array([w[0] for w in windows], dtype=float)
            names = [w[1] for w in windows]
            rows = t.loc[idx].to_numpy(dtype=float)
            # NaT -> -inf: hicbir faza dusmez, "pre-phase" olarak ayrilir.
            rows = np.where(np.isnan(rows), -np.inf, rows)
            pos = np.searchsorted(edges, rows, side="right") - 1
            keys.loc[idx] = [
                f"{run_id}|{names[p]}" if p >= 0 else f"{run_id}|pre-phase"
                for p in pos
            ]
        if resolved:
            return keys, "ground_truth_phase"

    if "phase" in sub_df.columns:
        return run + "|" + sub_df["phase"].astype(str), "phase_column"
    return run, "run_id"


def _episode_hits(sub_df, pred, ground_truth):
    key, basis = _episode_keys(sub_df, ground_truth)
    p = pd.Series(pred, index=sub_df.index)
    hits = [1 if p.loc[list(idx)].max() == 1 else 0
            for _, idx in key.groupby(key).groups.items()]
    return hits, basis


def detection_latency(sub_df, pred, ground_truth):
    """Episode basina tespit gecikmesi (saniye).

    Gecikme, episode'un ILK saldiri-etiketli olayindan ilk pozitif tahmine
    kadar gecen suredir. Ground-truth faz damgasi degil ilk gozlenen olay
    referans alinir: faz damgasi ile ilk trafik arasindaki olu zaman tespit
    performansi degildir.

    Yakalanmayan episode gecikmeye GIRMEZ (sonsuz olurdu) -- recall'da zaten 0
    sayilir. Kacan episode sayisi ayrica raporlanir ki gecikme ortalamasi
    "yalniz kolay episode'lar" uzerinden okunmasin.
    """
    if "timestamp" not in sub_df.columns or len(sub_df) == 0:
        return None
    key, _ = _episode_keys(sub_df, ground_truth)
    t = _epoch_seconds(sub_df["timestamp"])
    p = pd.Series(pred, index=sub_df.index)
    latencies, missed = [], 0
    for _, idx in key.groupby(key).groups.items():
        idx = list(idx)
        times, hit = t.loc[idx], (p.loc[idx] == 1)
        if not hit.any() or times.isna().all():
            missed += 1
            continue
        start, first = times.min(), times[hit].min()
        if pd.notna(start) and pd.notna(first):
            latencies.append(float(first - start))
    if not latencies:
        return {"n_detected": 0, "n_missed": missed}
    return {
        "median_sec": round(float(statistics.median(latencies)), 3),
        "mean_sec": round(float(statistics.fmean(latencies)), 3),
        "max_sec": round(float(max(latencies)), 3),
        "n_detected": len(latencies),
        "n_missed": missed,
    }


def alarm_load(test_df, pred):
    """Operasyonel alarm yuku: kosu basina ve saat basina pozitif sayisi.

    Recall/FPR'den ayri raporlanir: %1 FPR bile uzun bir kosuda operatore
    tasinamaz sayida alarm uretebilir. TUM test satirlari sayilir (yalniz
    saldiri fazlari degil) -- operatorun gordugu yuk budur.
    """
    if "run_id" not in test_df.columns or len(test_df) == 0:
        return None
    p = pd.Series(pred, index=test_df.index)
    runs = test_df["run_id"].astype(str)
    per_run = p.groupby(runs).sum()
    load = {
        "alarms_per_run_mean": round(float(per_run.mean()), 3),
        "alarms_per_run_max": int(per_run.max()),
        "n_runs": int(len(per_run)),
    }
    if "timestamp" in test_df.columns:
        t = _epoch_seconds(test_df["timestamp"])
        span = t.groupby(runs).agg(lambda s: s.max() - s.min())
        hours = float(span.sum()) / 3600.0
        if hours > 0:
            load["alarms_per_hour"] = round(float(per_run.sum()) / hours, 3)
            load["observed_hours"] = round(hours, 4)
    return load


# ---------------------------------------------------------------------------
# 8. CLI + self-test
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="LODO genelleme iskeleti (sozlesme araci)")
    ap.add_argument("--train")
    ap.add_argument("--test")
    ap.add_argument("--ground-truth")
    ap.add_argument("--schema", choices=SCHEMA_NAMES + ["all"], default="all")
    ap.add_argument("--max-fpr", type=float, default=0.01)
    ap.add_argument("--seeds", default="42",
                    help="virgulle ayrilmis seed listesi (ör. 42,43,44); bolme, "
                         "model ve bootstrap ayni seed'i kullanir")
    ap.add_argument("--force-headline", action="store_true",
                    help="veri hazir olmasa BILE metrik uret (ONERILMEZ)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()
    if not (args.train and args.test):
        ap.error("--train ve --test gerekli (ya da --self-test)")

    train_df = pd.read_csv(args.train, low_memory=False)
    test_df = pd.read_csv(args.test, low_memory=False)
    gt = None
    if args.ground_truth and os.path.exists(args.ground_truth):
        with open(args.ground_truth, encoding="utf-8") as f:
            gt = json.load(f)

    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    schemas = SCHEMA_NAMES if args.schema == "all" else [args.schema]
    print("=" * 70)
    print(f"LODO GENELLEME  (train=OpenPLC, test=harici domain)  seeds={seeds}")
    print("=" * 70)
    for sc in schemas:
        out = run_lodo_multiseed(train_df, test_df, sc, gt, seeds=seeds,
                                 max_fpr=args.max_fpr,
                                 force_headline=args.force_headline)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        if not out["ready"] and not args.force_headline:
            print(f"[CONTRACT] '{sc}': HEADLINE REDDEDILDI:")
            for why in out["per_seed"][0]["refusal_reasons"]:
                print("   - " + why)
    return 0


def _dummy_factory(numeric, categorical, seed=42):
    class _M:
        def fit(self, X, y):
            return self

        def predict(self, X):
            c = numeric[0] if numeric else None
            if not c:
                return np.zeros(len(X), int)
            return (pd.to_numeric(X[c], errors="coerce").fillna(0) > 0).astype(int).to_numpy()
    return _M()


def _self_test():
    print("[self-test] yapisal dogrulama (sklearn'suz)...")
    ok = True

    try:
        assert_no_leakage(pd.DataFrame({"pkts_toserver": [1], "dest_port": [502]}))
        print("  [FAIL] sizinti guard yakalamadi"); ok = False
    except AssertionError:
        print("  [OK] sizinti guard dest_port'u yakaladi")

    assert abs(_n_rpm1200(1200) - 1.0) < 1e-9 and abs(_n_rudder(35) - 1.0) < 1e-9
    assert abs(_n_pump3000(3000) - 1.0) < 1e-9
    print("  [OK] normalizasyon: rpm/1200=1.0, rudder/35=1.0, pompa/3000=1.0")

    df = _fixture("openplc", "phase", 3, 90)
    for sc in SCHEMA_NAMES:
        fr = build_feature_frame(df, sc)
        assert not DOMAIN_IDENTITY_BLOCKLIST.intersection(fr.columns), sc
        assert "modbus_quantity_norm" not in fr.columns
    print("  [OK] 4 sema; adres/port/quantity/domain kolonu YOK")

    # proxy semada propulsion var, strict'te YOK
    strict = build_feature_frame(_fixture("macyste", "phase", 3, 90), "physical_strict")
    proxy = build_feature_frame(_fixture("macyste", "phase", 3, 90), "physical_proxy")
    assert "propulsion_demand_norm" not in strict.columns
    assert "propulsion_demand_norm" in proxy.columns
    assert "rudder_norm" in strict.columns and "rudder_norm_present" in strict.columns
    assert "rudder_rate" in strict.columns
    print("  [OK] strict=yalniz rudder(+present+rate); proxy=+propulsion_demand")

    short = _fixture("macyste", "run", 1, 5)
    ready, reasons = validate_readiness(short, None)
    assert not ready and reasons
    print(f"  [OK] contract guard kisa/kosu-etiketli veriyi reddetti ({len(reasons)} neden)")

    r = run_lodo(_fixture("openplc", "phase", 3, 90), short, "physical_proxy",
                 ground_truth=None, model_factory=_dummy_factory)
    assert r["mode"] == "contract" and "fpr_normal" not in r
    print("  [OK] hazir-olmayan veride headline metrik URETILMEDI")

    # headline yolu (dummy): scenario+label maskesi, val esigi, episode kaynagi
    good = _fixture("macyste", "phase", 3, 90)
    gt = {g: {"phases": [{"name": "warmup"}, {"name": "act"}]}
          for g in good["run_id"].unique()}
    r2 = run_lodo(_fixture("openplc", "phase", 4, 90), good, "physical_proxy",
                  ground_truth=gt, model_factory=_dummy_factory, force_headline=True)
    assert "fpr_normal" in r2 and "recall_event_manipulation" in r2
    # maske scenario VE label -> manipulation kosusunun 20 satirindan 16'si
    assert r2["n_manipulation_attack_rows"] == 3 * 16
    print("  [OK] headline yolu: FPR + recall; maske = scenario & label==attack")

    # episode sinirlari: zaman damgali ground-truth -> faz penceresi
    stamped = {
        g: {"phases": [
            {"phase": "baseline", "timestamp": "2023-11-14T22:13:20+00:00"},
            {"phase": "unauthorized_manipulation",
             "timestamp": "2023-11-14T22:13:38+00:00"},
        ]}
        for g in good["run_id"].unique()
    }
    r3 = run_lodo(_fixture("openplc", "phase", 4, 90), good, "physical_proxy",
                  ground_truth=stamped, model_factory=_dummy_factory,
                  force_headline=True)
    assert r3["recall_episode_manipulation"]["episode_basis"] == "ground_truth_phase"
    assert r2["recall_episode_manipulation"]["episode_basis"] == "phase_column"
    print("  [OK] episode sinirlari ground-truth fazlarindan (ground_truth kullaniliyor)")

    # rudder_rate YALNIZ gercek dumen ornekleri arasinda
    rate_fr = build_feature_frame(good, "physical_strict")
    assert (rate_fr["rudder_rate"] != 0).any(), "seyrek ornekte hiz tumden sifir"
    print("  [OK] rudder_rate seyrek orneklemede sifirlanmiyor")

    # esik: `>=` altinda hedef FPR ampirik olarak saglanir (beraberlikli skor)
    tied = np.array([0.0] * 90 + [0.4] * 10)
    thr_t = _threshold_at_fpr(tied, np.zeros(100, int), 0.01)
    assert float((tied >= thr_t).mean()) <= 0.01
    print(f"  [OK] esik beraberlikte de hedef FPR'yi tutuyor (thr={thr_t:.4g})")

    m, lo, hi = episode_bootstrap_ci([1, 1, 0, 1, 0], n_boot=500)
    assert 0 <= lo <= m <= hi <= 1
    print(f"  [OK] episode bootstrap: recall={m:.2f} GA95=[{lo:.2f},{hi:.2f}]")

    # train tarafi contract: satir-bazli bolme sessizce kullanilamaz
    solo = _fixture("openplc", "phase", 1, 90).copy()
    solo["run_id"] = "tek-kosu"
    ti, vi, basis = _split_by_run(solo, 0.3, 42)
    assert basis == "row"
    assert any("KOSU bazli degil" in x for x in validate_split(solo, ti, vi, basis, 0.01))
    ti, vi, basis = _split_by_run(_fixture("openplc", "phase", 4, 90), 0.3, 42)
    assert basis == "run"
    print("  [OK] train/validation bolmesi kosu bazli degilse REDDEDILIYOR")

    # tespit gecikmesi ve alarm yuku (Faz 1 zorunlu metrikleri)
    lat_df = pd.DataFrame({
        "run_id": ["r1"] * 3,
        "timestamp": ["2023-11-14T22:13:40+00:00", "2023-11-14T22:13:42+00:00",
                      "2023-11-14T22:13:45+00:00"],
    })
    lat_gt = {"r1": {"phases": [{"phase": "act",
                                 "timestamp": "2023-11-14T22:13:38+00:00"}]}}
    lat = detection_latency(lat_df, np.array([0, 1, 0]), lat_gt)
    assert lat["median_sec"] == 2.0 and lat["n_missed"] == 0
    load = alarm_load(lat_df, np.array([0, 1, 0]))
    assert load["n_runs"] == 1 and load["alarms_per_run_mean"] == 1.0
    print(f"  [OK] tespit gecikmesi={lat['median_sec']}s; alarm/kosu={load['alarms_per_run_mean']}")

    # coklu seed: ozet min<=mean<=max tasir ("en iyi seed" secilemez)
    ms = run_lodo_multiseed(_fixture("openplc", "phase", 4, 90), good,
                            "physical_proxy", ground_truth=stamped, seeds=(1, 2, 3),
                            model_factory=_dummy_factory, force_headline=True)
    assert [r["seed"] for r in ms["per_seed"]] == [1, 2, 3]
    for stats in ms["summary"].values():
        assert stats["min"] <= stats["mean"] <= stats["max"]
    print(f"  [OK] coklu seed {ms['seeds']}: ozet yayilimi (min/mean/max) raporlaniyor")

    print("[self-test] " + ("TUM KONTROLLER GECTI" if ok else "BASARISIZ"))
    return 0 if ok else 1


def _fixture(source, label_mode, n_runs, dur):
    """Sentetik kosu verisi.

    MaCySTe tarafi GERCEK yakalamayi taklit eder: her satir her alani TASIMAZ.
    Dumen ornekleri araya giren baska Modbus olaylariyla seyrekleşir ve sol/sag
    throttle AYRI sunuculardan gelir (asla ayni satirda degildir). Fixture bunu
    yansitmazsa rudder_rate ve role-coalesce hatalari testlerden kacar.
    """
    # Senaryo adlari GERCEK kampanya vokabuleridir ('<kaynak>_<aile>'); yalin ad
    # kullanilirsa aile eslemesi test edilmeden kalir.
    scen = {"openplc": ["baseline", "modbus_write"],
            "macyste": ["macyste_normal", "macyste_manipulation",
                        "macyste_recon", "macyste_fault"]}[source]
    macyste = source == "macyste"
    rows = []
    for run in range(n_runs):
        for si, sc in enumerate(scen):
            base_lab = {"baseline": "normal", "macyste_normal": "normal",
                        "macyste_fault": "fault"}.get(sc, "attack")
            for k in range(20):
                t = 1_700_000_000 + run * 1000 + k * (dur / 20)
                # kosu-modu: tum satir base_lab; faz-modu: ilk 4 warmup=normal
                lab = base_lab if label_mode == "run" else (base_lab if k > 3 else "normal")
                # Dumen sabit durmaz; saldiri aninda sicrar. Sabit deger
                # verilirse rudder_rate her zaman 0 olur ve hiz hatalari kacar.
                rudder = (50 if lab == "attack" else 5) + (k % 3)
                row = {
                    "source_dataset": source, "run_id": f"{source}-{sc}-{run}",
                    "scenario": sc, "label": lab,
                    "phase": "act" if k > 3 else "warmup",
                    "timestamp": pd.Timestamp(t, unit="s", tz="UTC").isoformat(),
                    "pkts_toserver": k, "pkts_toclient": k,
                    "bytes_toserver": 60 * k, "bytes_toclient": 60 * k,
                    "flow_state": "established",
                    "modbus_function": 6 if base_lab == "attack" else 3,
                    "modbus_access": "WRITE" if base_lab == "attack" else "READ",
                    "modbus_quantity": 2 if macyste else 1,
                    "modbus_address": 3 + si,
                    "dest_port": 502,
                }
                if macyste:
                    # Dumen her 2 olayda bir orneklenir; arasi baska register.
                    row["rudder_angle"] = rudder if k % 2 == 0 else np.nan
                    # Sol/sag telgraf ayri sunucu -> ayri satir.
                    side = "left" if k % 4 == 1 else ("right" if k % 4 == 3 else None)
                    row["throttle_command_left"] = np.nan
                    row["throttle_command_right"] = np.nan
                    if side:
                        row[f"throttle_command_{side}"] = (
                            1.0 if base_lab == "attack" else 0.45
                        )
                    row["hydraulic_pump_rpm_left"] = 2500
                    row["hydraulic_pump_rpm_right"] = 2500
                else:
                    row["rudder_angle"] = rudder
                    row["rpm_command"] = 1100 if base_lab == "attack" else 600
                rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    raise SystemExit(main())
