#!/usr/bin/env python3
"""D2 -- gorulmemis kategori teshisi (salt-okunur gozlemci).

Ön-kayit: `DENEY-D2-Gorulmemis-Kategori-Teshisi.md`. Bu modul o belgeyi
uygular; metrik tanimini, paydalari ve durdurma kosullarini SONUCA GORE
degistirmez.

Modul model hattina DOKUNMAZ:

  - `handle_unknown="ignore"` aynen kalir; burada bir encoder fit EDILMEZ,
  - hedef alan sozlugu egitim sozlugune eklenmez,
  - dondurulmus v0.4 sonuclari yeniden hesaplanmaz.

Olculen iki buyukluk AYRI tutulur (belge §4):

  sozluk kapsama acigi   -> CESIT: hedefteki kac ayri degerin karsiligi yok
  gorulmemis satir kutlesi -> FREKANS: hedef satirlarin kaci etkileniyor

Kutle IKI paydayla raporlanir. Kategorikler `.astype(str)` ile uretildigi icin
eksik degerler dusmez, literal bir kategoriye ("nan") donusur ve iki alanda da
yaygindir; tek paydali bir kutle "sozlukler ortusuyor" gibi okunurken aslinda
"alan zaten yok" demek olur (belge §3.2).

Bu modul belge §3.1, §3.2, §4 ve §5'i kapsar. §7'deki H3 mekanizma testi
(skor nokta kutleleri ve gorulmemislik imzalari) AYRI bir adimdir; model fit'i
ve teshis amacli contract bypass'i gerektirir, burada yer ALMAZ.
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from ml.lodo_generalization import (
    SCHEMA_NAMES,
    _split_by_run,
    build_feature_frame,
)

# Ön-kayitli seed evreni: sonuclar gorulmeden sabitlendi (belge §5).
SEED_UNIVERSE = tuple(range(42, 62))
VAL_FRAC = 0.3

# Eksikligin NEREDEN geldigi (belge §3.2). Eksik satirlar kategori DEGERIDIR;
# sozlukten dusurulmez, yalniz "gozlenmis satir" paydasinin disinda birakilir.
#
# Eksiklik bir LITERAL TAHMINIYLE degil, ham NaN maskesinden TURETILIR. Literal
# pandas surumune baglidir: pandas 3.0'da `str` dtype `.astype(str)` sirasinda
# eksikligi korur ve `build_feature_frame()`'in `.fillna("na")` satiri devreye
# girer -> "na". Daha eski surumlerde ayni deger "nan" olarak render edilirdi.
# Sabit bir literal yazmak bu farkta sessizce yanlis olurdu.
#
# DIKKAT: band kolonlarindaki "na" eksiklik SAYILMAZ. `_band()` onu "satir bir
# yazma islemi degil" anlaminda uretir; bu gercek bir gozlemdir. Band tarafinda
# eksiklik gostergesi "unknown"dir (yazma var, deger cozulemedi).
MISSING_SOURCE = {
    "flow_state": "ham_nan",
    "modbus_function": "ham_nan",
    "modbus_access": "ham_nan",
    "rudder_band": "band_unknown",
    "propulsion_band": "band_unknown",
}

BAND_MISSING = "unknown"        # yazma var, deger cozulemedi -> EKSIK
BAND_NOT_A_WRITE = "na"         # satir yazma degil -> GERCEK gozlem

# `_band()`'in uretebilecegi butun literal degerler (belge §3.1, 3. madde).
BAND_LITERALS = {"na", "unknown", "in_band", "out_of_band"}
BAND_COLUMNS = ("rudder_band", "propulsion_band")

# Dogrudan ham kolondan string'e cevrilen kategorikler.
RAW_BACKED_COLUMNS = ("flow_state", "modbus_function", "modbus_access")


class RenderGuardError(RuntimeError):
    """Render on-kosulu tutmadi; teshis SONUC URETMEDEN durur (belge §3.1)."""

    def __init__(self, report: dict[str, Any]):
        self.report = report
        super().__init__(_format_guard_failure(report))


# ---------------------------------------------------------------------------
# §3.1 -- render on-kosulu
# ---------------------------------------------------------------------------
def _dtype_class(dtype) -> str:
    """dtype'i iki sinifa indir: durdurma yalniz SINIF degisiminde tetiklenir.

    Sinif ici fark (ornegin `str` <-> `object`) durdurmaz: pandas ayni kolonu
    tek dosyalik `read_csv`'de `str`, cok dosyalik `concat`'te `object`
    birakabilir ve bu render'i degistirmez. Sinif degisimi (`float64` <->
    `int64` gibi sayisal/string-benzeri gecisi degil, sayisal <-> string) ise
    gosterimi degistirebilecegi icin durdurur.
    """
    return "sayisal" if pd.api.types.is_numeric_dtype(dtype) else "string-benzeri"


def _canon_key(value) -> tuple:
    """Ham degeri alanlar arasi karsilastirilabilir bir anahtara indir."""
    if value is None:
        return ("null",)
    if isinstance(value, float) and value != value:      # NaN
        return ("null",)
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return ("num", float(value))
    return ("str", str(value))


def _render_map(series: pd.Series) -> dict[tuple, set[str]]:
    """ham deger -> `build_feature_frame()`'in urettigi string gosterim(ler).

    Ayni kanonik anahtarin birden fazla gosterime dusmesi tek basina hata
    degildir; alanlar ARASI fark aranir.
    """
    pairs = pd.DataFrame({"raw": series, "rendered": series.astype(str)})
    pairs = pairs.drop_duplicates()
    out: dict[tuple, set[str]] = {}
    for raw, rendered in zip(pairs["raw"], pairs["rendered"]):
        out.setdefault(_canon_key(raw), set()).add(str(rendered))
    return out


def render_precondition(source_df: pd.DataFrame, target_df: pd.DataFrame,
                        schemas: Sequence[str] = SCHEMA_NAMES) -> dict[str, Any]:
    """Belge §3.1. Tutmazsa `RenderGuardError` firlatir; oran HESAPLANMAZ.

    Durdurma kosullari yalniz ikidir:
      1. Semantik olarak ayni ham deger iki alanda farkli string'e donusuyor.
      2. Bir kategorik kolonun dtype SINIFI iki alanda farkli.
    Sinif ici dtype farki UYARI olarak kaydedilir, durdurmaz.
    """
    dtypes: dict[str, dict[str, str]] = {}
    warnings: list[dict[str, str]] = []
    render_mismatch: list[dict[str, Any]] = []
    class_mismatch: list[dict[str, str]] = []

    for column in RAW_BACKED_COLUMNS:
        if column not in source_df.columns or column not in target_df.columns:
            continue
        s_dtype, t_dtype = source_df[column].dtype, target_df[column].dtype
        s_class, t_class = _dtype_class(s_dtype), _dtype_class(t_dtype)
        dtypes[column] = {"source": str(s_dtype), "target": str(t_dtype),
                          "source_class": s_class, "target_class": t_class}
        if s_class != t_class:
            class_mismatch.append({"column": column,
                                   "source": str(s_dtype), "target": str(t_dtype),
                                   "source_class": s_class, "target_class": t_class})
        elif str(s_dtype) != str(t_dtype):
            warnings.append({"column": column, "source": str(s_dtype),
                             "target": str(t_dtype),
                             "note": "sinif ici dtype farki; render'i degistirmez"})

        s_map, t_map = _render_map(source_df[column]), _render_map(target_df[column])
        for key in sorted(set(s_map) & set(t_map)):
            if s_map[key] != t_map[key]:
                render_mismatch.append({
                    "column": column,
                    "raw": key[1] if len(key) > 1 else None,
                    "source_rendered": sorted(s_map[key]),
                    "target_rendered": sorted(t_map[key]),
                })

    # Turetilmis band kategorileri: literal sozluk iki alanda da ayni olmali.
    band_vocab: dict[str, dict[str, list[str]]] = {}
    for schema in schemas:
        s_frame = build_feature_frame(source_df, schema)
        t_frame = build_feature_frame(target_df, schema)
        for column in BAND_COLUMNS:
            if column not in s_frame.columns or column not in t_frame.columns:
                continue
            s_vals = set(s_frame[column].astype(str).unique())
            t_vals = set(t_frame[column].astype(str).unique())
            band_vocab[column] = {"source": sorted(s_vals), "target": sorted(t_vals)}
            stray = (s_vals | t_vals) - BAND_LITERALS
            if stray:
                render_mismatch.append({
                    "column": column,
                    "raw": None,
                    "source_rendered": sorted(s_vals),
                    "target_rendered": sorted(t_vals),
                    "note": f"_band() literal sozlugu disinda deger: {sorted(stray)}",
                })

    report = {
        "passed": not (render_mismatch or class_mismatch),
        "raw_dtypes": dtypes,
        "band_vocabulary": band_vocab,
        "within_class_dtype_warnings": warnings,
        "render_mismatch": render_mismatch,
        "dtype_class_mismatch": class_mismatch,
    }
    if not report["passed"]:
        raise RenderGuardError(report)
    return report


def _format_guard_failure(report: dict[str, Any]) -> str:
    parts = ["D2 render on-kosulu TUTMADI; teshis sonuc uretmeden durdu."]
    for item in report.get("dtype_class_mismatch", []):
        parts.append(
            f"  dtype SINIFI farkli: {item['column']} "
            f"kaynak={item['source']} ({item['source_class']}) "
            f"hedef={item['target']} ({item['target_class']})"
        )
    for item in report.get("render_mismatch", []):
        note = item.get("note", "")
        parts.append(
            f"  gosterim farkli: {item['column']} ham={item['raw']!r} "
            f"kaynak={item['source_rendered']} hedef={item['target_rendered']}"
            + (f" -- {note}" if note else "")
        )
    parts.append("  Ayni kosuda coercion/normalizasyon UYGULANMAZ (belge §3.1).")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# §4 -- olcum cekirdegi
# ---------------------------------------------------------------------------
def missing_mask(raw_df: pd.DataFrame, frame: pd.DataFrame,
                 column: str) -> np.ndarray:
    """Eksik satir maskesi -- literal TAHMIN EDILMEZ, kaynagindan turetilir.

    Band kolonlari `_band()` tarafindan uretilir ve NaN tasimaz; orada eksiklik
    "unknown" literalidir. Ham kolondan gelen kategorikler icin eksiklik ham
    NaN maskesidir; render literali (surume gore "na" ya da "nan") olcut
    DEGILDIR.
    """
    if MISSING_SOURCE[column] == "band_unknown":
        return (frame[column].astype(str) == BAND_MISSING).to_numpy()
    if column not in raw_df.columns:
        # Kolon hic yok -> build_feature_frame sabit dolgu koyar; hicbir satir
        # gozlenmis sayilmaz.
        return np.ones(len(frame), dtype=bool)
    return raw_df[column].isna().to_numpy()


def _column_measures(src_values: pd.Series, tgt_values: pd.Series,
                     column: str, tgt_missing: np.ndarray) -> dict[str, Any]:
    """Bir kolon icin belge §4.1-§4.3 alanlari."""
    v_src = set(src_values.astype(str).unique())
    v_tgt = set(tgt_values.astype(str).unique())
    unseen = v_tgt - v_src

    tgt_str = tgt_values.astype(str)
    unseen_mask = tgt_str.isin(unseen)
    observed_mask = ~pd.Series(tgt_missing, index=tgt_str.index)

    target_row_n = int(len(tgt_str))
    observed_row_n = int(observed_mask.sum())
    unseen_row_n = int(unseen_mask.sum())

    # Eksik satirlarin fiilen hangi literale dustugu denetlenebilir kalsin;
    # bu alan olmasaydi "nan" varsayimi sessizce yanlis kalirdi.
    missing_literals = sorted(set(tgt_str[pd.Series(tgt_missing,
                                                    index=tgt_str.index)]))

    # |V_tgt| = 0 -> oran null; sifir KABUL EDILMEZ (belge §4.1).
    value_ratio = (round(len(unseen) / len(v_tgt), 6) if v_tgt else None)
    row_mass = (round(unseen_row_n / target_row_n, 6) if target_row_n else None)
    row_mass_observed = (round(unseen_row_n / observed_row_n, 6)
                         if observed_row_n else None)

    return {
        "column": column,
        "missing_basis": MISSING_SOURCE[column],
        "missing_literals_observed": missing_literals,
        "source_unique_n": len(v_src),
        "target_unique_n": len(v_tgt),
        "unseen_unique_n": len(unseen),
        "unseen_values": sorted(unseen),
        "unseen_value_ratio": value_ratio,
        "unseen_row_n": unseen_row_n,
        "target_row_n": target_row_n,
        "unseen_row_mass": row_mass,
        "target_observed_row_n": observed_row_n,
        "unseen_row_mass_observed": row_mass_observed,
        # Eksik satirlarin literali kaynak tarafta hic gorulmemisse bu YAPISAL
        # bir farktir ve `unseen_row_mass_observed` 1'i asabilir; sessiz kalmasin.
        "missing_literal_is_unseen": bool(set(missing_literals) & unseen),
        "_unseen_mask": unseen_mask,          # birlesim icin; JSON'a girmez
    }


def _schema_measures(src_frame: pd.DataFrame, tgt_frame: pd.DataFrame,
                     raw_target: pd.DataFrame, schema: str) -> dict[str, Any]:
    columns = list(tgt_frame.attrs["categorical"])
    per_column, masks = [], []
    for column in columns:
        item = _column_measures(src_frame[column], tgt_frame[column], column,
                                missing_mask(raw_target, tgt_frame, column))
        masks.append(item.pop("_unseen_mask"))
        per_column.append(item)

    target_row_n = int(len(tgt_frame))
    if masks:
        any_mask = masks[0].copy()
        for mask in masks[1:]:
            any_mask = any_mask | mask
        any_n = int(any_mask.sum())
    else:
        any_n = 0
    return {
        "schema": schema,
        "columns": per_column,
        "unseen_any_n": any_n,
        "target_row_n": target_row_n,
        # Birlesim icin IKINCI payda tanimlanmaz: kolonlarin eksik-veri
        # gostergeleri farkli oldugundan sema duzeyinde ortak bir "gozlenmis
        # satir" kumesi anlamli degildir (belge §4.2).
        "unseen_any_mass": (round(any_n / target_row_n, 6) if target_row_n else None),
    }


def primary_panel(source_df: pd.DataFrame, target_df: pd.DataFrame,
                  schemas: Sequence[str] = SCHEMA_NAMES) -> list[dict[str, Any]]:
    """Belge §4: bolmeden BAGIMSIZ panel.

    Sozluk tum kaynak frame'inden kurulur. Icinde seed, bolme, model skoru ve
    esik YOKTUR; errata'daki bolme kararsizligi bu panele tasinamaz.
    """
    _reject_empty(source_df, "kaynak")
    _reject_empty(target_df, "hedef")
    out = []
    for schema in schemas:
        src_frame = build_feature_frame(source_df, schema)
        tgt_frame = build_feature_frame(target_df, schema)
        out.append(_schema_measures(src_frame, tgt_frame, target_df, schema))
    return out


def _reject_empty(frame: pd.DataFrame, side: str) -> None:
    """Bos frame sessizce "sifir gorulmemis" gibi raporlanmasin."""
    if len(frame) == 0:
        raise ValueError(
            f"{side} frame BOS; D2 sonuc uretemez. Bos bir alan 'gorulmemis "
            "kategori yok' anlamina GELMEZ -- girdi yolunu denetleyin "
            "(belge §3, kanonik yollar tablosu)."
        )


# ---------------------------------------------------------------------------
# §5 -- ikincil, bolmeye duyarli seed paneli
# ---------------------------------------------------------------------------
def guard_class(source_df: pd.DataFrame, tr_idx, va_idx,
                class_label: str = "fault") -> str:
    """Errata'daki performanstan BAGIMSIZ guard sinifi (belge §5).

    Siniflar betimseldir; hicbiri yeni headline ya da tercih edilen seed alt
    kumesi yapamaz.
    """
    if "label" not in source_df.columns or "run_id" not in source_df.columns:
        return "bilinmiyor"
    labels = source_df["label"].astype(str)
    runs = source_df["run_id"].astype(str)
    tr_runs = set(runs.iloc[tr_idx][labels.iloc[tr_idx] == class_label])
    va_runs = set(runs.iloc[va_idx][labels.iloc[va_idx] == class_label])
    if not va_runs:
        return "kusur_a_validationda_yok"
    if not tr_runs:
        return "kusur_b_trainde_yok"
    return "iki_tarafta_var"


def seed_panel(source_df: pd.DataFrame, target_df: pd.DataFrame,
               schemas: Sequence[str] = SCHEMA_NAMES,
               seeds: Iterable[int] = SEED_UNIVERSE) -> dict[str, Any]:
    """Belge §5: sozluk YALNIZ o seed'in train parcasindan kurulur.

    Hedef her seed'de tum hedef frame'dir. Hicbir seed performansa ya da
    teshis degerine gore dislanmaz.
    """
    _reject_empty(source_df, "kaynak")
    _reject_empty(target_df, "hedef")
    seeds = [int(s) for s in seeds]
    tgt_frames = {sc: build_feature_frame(target_df, sc) for sc in schemas}

    per_seed = []
    for seed in seeds:
        tr_idx, va_idx, basis = _split_by_run(source_df, VAL_FRAC, seed)
        train_df = source_df.iloc[tr_idx]
        entry = {"seed": seed, "split_basis": basis,
                 "guard_class": guard_class(source_df, tr_idx, va_idx),
                 "train_run_n": int(pd.Series(train_df["run_id"].astype(str)).nunique())
                 if "run_id" in train_df.columns else None,
                 "schemas": []}
        for schema in schemas:
            src_frame = build_feature_frame(train_df, schema)
            entry["schemas"].append(
                _schema_measures(src_frame, tgt_frames[schema], target_df, schema))
        per_seed.append(entry)

    return {"seeds": seeds, "per_seed": per_seed,
            "summary": _summarise_seeds(per_seed, schemas)}


_SUMMARISED = ("unseen_value_ratio", "unseen_row_mass", "unseen_row_mass_observed",
               "unseen_unique_n", "unseen_row_n")


def _spread(values: list) -> dict[str, Any] | None:
    """ortanca [min, max] -- alt kume secilmedigi gorunur kalsin diye."""
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return {"median": round(float(statistics.median(clean)), 6),
            "min": round(float(min(clean)), 6),
            "max": round(float(max(clean)), 6),
            "n_seeds": len(clean)}


def _summarise_seeds(per_seed: list[dict[str, Any]],
                     schemas: Sequence[str]) -> list[dict[str, Any]]:
    summary = []
    for si, schema in enumerate(schemas):
        columns = [c["column"] for c in per_seed[0]["schemas"][si]["columns"]]
        per_column = []
        for ci, column in enumerate(columns):
            item = {"column": column}
            for metric in _SUMMARISED:
                item[metric] = _spread(
                    [s["schemas"][si]["columns"][ci][metric] for s in per_seed])
            per_column.append(item)
        summary.append({
            "schema": schema,
            "columns": per_column,
            "unseen_any_mass": _spread(
                [s["schemas"][si]["unseen_any_mass"] for s in per_seed]),
        })
    return summary


# ---------------------------------------------------------------------------
# Girdi yukleme (belge §3: kanonik yollar)
# ---------------------------------------------------------------------------
DEFAULT_SOURCE = "attack/dataset.csv"
DEFAULT_TARGET = "captures/macyste/macyste-20260726T113328Z/*/events-v0.4.csv"


def load_frame(pattern: str) -> tuple[pd.DataFrame, list[str]]:
    """Tek dosya ya da glob; hedef tarafta `events-v0.4.csv` KULLANILIR.

    `events.csv` v0.4 ONCESI turetmedir ve ayni dizinlerde durur. Yanlis dosya
    setini secmek belgenin R1 revizyonuna yol acmisti; yollar bu yuzden acikca
    raporlanir.
    """
    paths = sorted(_glob.glob(pattern)) if any(ch in pattern for ch in "*?[") \
        else [pattern]
    if not paths:
        raise FileNotFoundError(f"girdi bulunamadi: {pattern}")
    frames = [pd.read_csv(p, low_memory=False) for p in paths]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0], paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print_primary(panel: list[dict[str, Any]]) -> None:
    print("BIRINCIL PANEL -- bolmeden BAGIMSIZ (belge §4)")
    print("sozluk tum kaynak frame'inden; seed/bolme/esik YOK\n")
    header = (f"{'sema':16s} {'kolon':18s} {'|V_src|':>7s} {'|V_tgt|':>7s} "
              f"{'|U|':>4s} {'cesit':>7s} {'kutle':>8s} {'kutle(gozl.)':>13s}")
    print(header)
    print("-" * len(header))
    for schema in panel:
        for column in schema["columns"]:
            def fmt(value):
                return "  null" if value is None else f"{value:.4f}"
            print(f"{schema['schema']:16s} {column['column']:18s} "
                  f"{column['source_unique_n']:7d} {column['target_unique_n']:7d} "
                  f"{column['unseen_unique_n']:4d} "
                  f"{fmt(column['unseen_value_ratio']):>7s} "
                  f"{fmt(column['unseen_row_mass']):>8s} "
                  f"{fmt(column['unseen_row_mass_observed']):>13s}")
        any_mass = schema["unseen_any_mass"]
        print(f"{schema['schema']:16s} {'[BIRLESIM]':18s} "
              f"{'':7s} {'':7s} {'':4s} {'':>7s} "
              f"{(('%.4f' % any_mass) if any_mass is not None else 'null'):>8s} "
              f"{'(tanimsiz)':>13s}")
        print()
    print("cesit = unseen_value_ratio;  kutle = unseen_row_mass;")
    print("kutle(gozl.) = eksik-veri gostergesi olmayan satirlar uzerinden.")
    print("Birlesim icin ikinci payda TANIMSIZDIR (belge §4.2).")


def _print_unseen_values(panel: list[dict[str, Any]]) -> None:
    print("\nGORULMEMIS DEGERLER (acik liste; oran tek basina yeterli degil)")
    for schema in panel:
        for column in schema["columns"]:
            if column["unseen_values"]:
                flag = ("  [EKSIK LITERALI GORULMEMIS]"
                        if column["missing_literal_is_unseen"] else "")
                print(f"  {schema['schema']:16s} {column['column']:18s} "
                      f"{column['unseen_values']}{flag}")


def _print_seed_panel(panel: dict[str, Any]) -> None:
    print("\n\nIKINCIL PANEL -- bolmeye duyarli seed evreni (belge §5)")
    print(f"seed evreni {panel['seeds'][0]}-{panel['seeds'][-1]} "
          f"({len(panel['seeds'])} seed); sozluk yalniz train parcasindan")
    counts: dict[str, int] = {}
    for entry in panel["per_seed"]:
        counts[entry["guard_class"]] = counts.get(entry["guard_class"], 0) + 1
    print("guard siniflari (betimsel; hicbiri alt kume secmez): "
          + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print()
    header = (f"{'sema':16s} {'kolon':18s} {'kutle ortanca':>14s} "
              f"{'[min':>9s} {'max]':>9s} {'cesit ortanca':>14s}")
    print(header)
    print("-" * len(header))
    for schema in panel["summary"]:
        for column in schema["columns"]:
            mass, ratio = column["unseen_row_mass"], column["unseen_value_ratio"]
            if mass is None:
                print(f"{schema['schema']:16s} {column['column']:18s} "
                      f"{'null':>14s}")
                continue
            print(f"{schema['schema']:16s} {column['column']:18s} "
                  f"{mass['median']:14.4f} {mass['min']:9.4f} {mass['max']:9.4f} "
                  f"{(ratio['median'] if ratio else float('nan')):14.4f}")
        print()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="D2 gorulmemis kategori teshisi (on-kayit: "
                    "DENEY-D2-Gorulmemis-Kategori-Teshisi.md)")
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help="kaynak alan (OpenPLC) CSV")
    parser.add_argument("--target", default=DEFAULT_TARGET,
                        help="hedef alan (MaCySTe) CSV ya da glob; "
                             "events-v0.4.csv KULLANILIR")
    parser.add_argument("--schema", choices=list(SCHEMA_NAMES) + ["all"],
                        default="all")
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEED_UNIVERSE),
                        help="ikincil panel seed evreni (on-kayitli: 42-61)")
    parser.add_argument("--skip-seed-panel", action="store_true",
                        help="yalniz birincil paneli uret")
    parser.add_argument("--out", type=Path,
                        default=Path("ml/ml_out/d2_unseen_category.json"))
    args = parser.parse_args(argv)

    schemas = list(SCHEMA_NAMES) if args.schema == "all" else [args.schema]
    source_df, source_paths = load_frame(args.source)
    target_df, target_paths = load_frame(args.target)

    print(f"kaynak : {args.source}  ({len(source_df)} satir, "
          f"{len(source_paths)} dosya)")
    print(f"hedef  : {args.target}  ({len(target_df)} satir, "
          f"{len(target_paths)} dosya)\n")

    try:
        guard = render_precondition(source_df, target_df, schemas)
    except RenderGuardError as exc:
        print(exc)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"render_precondition": exc.report, "halted": True,
             "note": "Belge §3.1: guard basarisizligi yayimlanir, oran "
                     "URETILMEZ. Duzeltme icin belge revize edilip yeniden "
                     "dondurulmalidir."},
            indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[DURDU] guard raporu -> {args.out}")
        return 2

    print("render on-kosulu: GECTI")
    for warning in guard["within_class_dtype_warnings"]:
        print(f"  [uyari] {warning['column']}: kaynak={warning['source']} "
              f"hedef={warning['target']} -- {warning['note']}")
    print()

    panel = primary_panel(source_df, target_df, schemas)
    _print_primary(panel)
    _print_unseen_values(panel)

    payload: dict[str, Any] = {
        "preregistration": "DENEY-D2-Gorulmemis-Kategori-Teshisi.md",
        "inputs": {"source": args.source, "source_files": source_paths,
                   "source_rows": int(len(source_df)),
                   "target": args.target, "target_files": target_paths,
                   "target_rows": int(len(target_df))},
        "render_precondition": guard,
        "primary_panel": panel,
    }

    if not args.skip_seed_panel:
        seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
        secondary = seed_panel(source_df, target_df, schemas, seeds)
        _print_seed_panel(secondary)
        payload["seed_panel"] = secondary

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\nsonuc -> {args.out}")
    print("H3 (belge §7) bu modulde YOK; ayri bir adimdir.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
