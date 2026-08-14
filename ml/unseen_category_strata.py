#!/usr/bin/env python3
"""D2A -- tabakali gorulmemis-kategori kutlesi (betimsel ayristirma).

Ön-kayit: `DENEY-D2A-Tabakali-Gorulmemis-Kutle.md`. D2'nin toplam sonucu
goruldukten SONRA, tabakali sayilar hesaplanmadan ONCE dondurulmustur.

D2A yonlu bir hipotez testi DEGILDIR. Amaci, sonucu gorup en carpici tabakayi
secmeyi engelleyen on-tanimli bir ayristirmadir: 14 tabakanin (4 senaryo +
2 etiket + 8 capraz hucre) TAMAMI, bos olanlar dahil, sabit sirada yayimlanir.

Iki olcu birbirinin yerine KULLANILAMAZ (belge §3.2):

  mass_*      -> "bu tabakadaki satirlarin ne kadari gorulmemis?"
  allocation* -> "butun gorulmemis satirlar nereye dusuyor?"

D2'nin yukleme, render guard, eksiklik maskesi ve sozluk olcum yollari AYNEN
yeniden kullanilir; ikinci bir kategori ya da eksiklik tanimi uretilmez.

Nedensellik OLCMEZ. Bir saldiri alt kumesinde kutle yuksek ciksa bile bu
yalniz birlikte gorulme bulgusudur (belge §6).
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from ml.lodo_generalization import SCHEMA_NAMES, build_feature_frame, scenario_family
from ml.unseen_category_diagnostic import (
    DEFAULT_SOURCE,
    DEFAULT_TARGET,
    MISSING_SOURCE,
    RenderGuardError,
    _reject_empty,
    load_frame,
    missing_mask,
    render_precondition,
)

# Belge §2.1 / §2.2 / §2.3 -- sabit sira; sonuclara gore DEGISTIRILMEZ.
SCENARIO_ORDER = ("normal", "fault", "manipulation", "recon")
LABEL_ORDER = ("attack", "non_attack")
CROSS_ORDER = tuple((scenario, label)
                    for scenario in SCENARIO_ORDER
                    for label in LABEL_ORDER)       # senaryo-oncelikli

ALLOWED_RAW_LABELS = {"normal", "fault", "attack"}
ANY_COLUMN = "__any__"

# allocation toplaminin 1 etmesi YUVARLAMA ONCESI aranir (belge §3.2).
_ALLOCATION_TOLERANCE = 1e-9


class StrataGuardError(RuntimeError):
    """Belge §2.1/§2.2 durdurma kosulu; tabakali sayi URETILMEZ."""


# ---------------------------------------------------------------------------
# Tabakalar
# ---------------------------------------------------------------------------
def build_strata(target_df: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """14 tabakayi sabit sirada kur; bilinmeyen senaryo/etiket DURDURUR."""
    raw_scenario = target_df["scenario"].astype(str)
    families = scenario_family(raw_scenario)
    if "other" in set(families):
        unknown = sorted({raw for raw, fam in zip(raw_scenario, families)
                          if fam == "other"})
        raise StrataGuardError(
            f"senaryo ailesi 'other' cikti: {unknown}. Belge §2.1: analiz "
            "sonuc uretmeden durur; sessizce bir aileye atanmaz."
        )

    if "label" not in target_df.columns or target_df["label"].isna().any():
        raise StrataGuardError(
            "hedefte eksik `label` var. Belge §2.2: sessizce `non_attack` "
            "icine atilmaz; analiz durur."
        )
    raw_labels = target_df["label"].astype(str)
    stray = sorted(set(raw_labels) - ALLOWED_RAW_LABELS)
    if stray:
        raise StrataGuardError(
            f"beklenmeyen ham etiket: {stray}. Izin verilen kume "
            f"{sorted(ALLOWED_RAW_LABELS)} (belge §2.2)."
        )

    is_attack = (raw_labels == "attack").to_numpy()
    strata: list[dict[str, Any]] = []

    for scenario in SCENARIO_ORDER:                      # panel 1: senaryo
        strata.append({"panel": "scenario", "scenario_family": scenario,
                       "attack_label": "all",
                       "mask": (families == scenario)})
    for label in LABEL_ORDER:                            # panel 2: etiket
        mask = is_attack if label == "attack" else ~is_attack
        strata.append({"panel": "label", "scenario_family": "all",
                       "attack_label": label, "mask": mask})
    for scenario, label in CROSS_ORDER:                  # panel 3: capraz
        side = is_attack if label == "attack" else ~is_attack
        strata.append({"panel": "scenario_x_label", "scenario_family": scenario,
                       "attack_label": label,
                       "mask": (families == scenario) & side})

    scenario_map = {raw: fam for raw, fam in
                    sorted(set(zip(raw_scenario, families)))}
    return strata, scenario_map


# ---------------------------------------------------------------------------
# Olcum
# ---------------------------------------------------------------------------
def _ratio(numerator: int, denominator: int) -> float | None:
    """Payda sifirsa `null`; SIFIR KABUL EDILMEZ (belge §3.1)."""
    if not denominator:
        return None
    return numerator / denominator


def _run_sensitivity(run_ids: pd.Series, stratum_mask: np.ndarray,
                     unseen_any: np.ndarray) -> dict[str, Any]:
    """Belge §4 -- olay-agirlikli sonucu tek bir kosu mu suruklyor?

    Yeni bir headline URETMEZ; kosular esit agirlikli bir sonuc degildir.
    """
    runs = run_ids[stratum_mask]
    if runs.empty:
        return {"n_runs": 0, "run_mass_median": None,
                "run_mass_min": None, "run_mass_max": None}
    per_run = []
    for _, index in runs.groupby(runs).groups.items():
        positions = run_ids.index.get_indexer(index)
        per_run.append(float(unseen_any[positions].mean()))
    return {"n_runs": int(runs.nunique()),
            "run_mass_median": float(statistics.median(per_run)),
            "run_mass_min": float(min(per_run)),
            "run_mass_max": float(max(per_run))}


def measure(source_df: pd.DataFrame, target_df: pd.DataFrame,
            schemas: Sequence[str] = SCHEMA_NAMES) -> dict[str, Any]:
    """Belge §3 -- butun paneller, butun tabakalar, butun kolonlar."""
    _reject_empty(source_df, "kaynak")
    _reject_empty(target_df, "hedef")
    strata, scenario_map = build_strata(target_df)

    run_ids = (target_df["run_id"].astype(str).reset_index(drop=True)
               if "run_id" in target_df.columns else None)
    total_rows = int(len(target_df))
    rows: list[dict[str, Any]] = []

    for schema in schemas:
        source_frame = build_feature_frame(source_df, schema)
        target_frame = build_feature_frame(target_df, schema)
        columns = list(target_frame.attrs["categorical"])

        # Gorulmemislik: deger KAYNAK SOZLUGUN disinda mi (D2 §4 ile ayni).
        unseen_masks: dict[str, np.ndarray] = {}
        missing_masks: dict[str, np.ndarray] = {}
        for column in columns:
            vocabulary = set(source_frame[column].astype(str).unique())
            unseen_masks[column] = (~target_frame[column].astype(str)
                                    .isin(vocabulary)).to_numpy()
            missing_masks[column] = missing_mask(target_df, target_frame, column)

        unseen_any = np.zeros(total_rows, dtype=bool)
        for column in columns:
            unseen_any |= unseen_masks[column]

        overall = {column: int(unseen_masks[column].sum()) for column in columns}
        overall[ANY_COLUMN] = int(unseen_any.sum())

        for stratum in strata:
            mask = np.asarray(stratum["mask"], dtype=bool)
            n_all = int(mask.sum())
            base = {"panel": stratum["panel"],
                    "scenario_family": stratum["scenario_family"],
                    "attack_label": stratum["attack_label"],
                    "schema": schema, "n_all": n_all}

            for column in columns:
                n_unseen = int((mask & unseen_masks[column]).sum())
                n_observed = int((mask & ~missing_masks[column]).sum())
                overall_mass = _ratio(overall[column], total_rows)
                mass_all = _ratio(n_unseen, n_all)
                rows.append({
                    **base,
                    "column": column,
                    "missing_basis": MISSING_SOURCE[column],
                    "n_observed": n_observed,
                    "n_unseen": n_unseen,
                    "mass_all": mass_all,
                    "mass_observed": _ratio(n_unseen, n_observed),
                    "allocation": _ratio(n_unseen, overall[column]),
                    "overall_mass_all": overall_mass,
                    "enrichment_all": (mass_all / overall_mass
                                       if mass_all is not None and overall_mass
                                       else None),
                    # Kosu duyarliligi YALNIZ birlesim satirlarinda (belge §5).
                    "n_runs": None, "run_mass_median": None,
                    "run_mass_min": None, "run_mass_max": None,
                })

            n_unseen_any = int((mask & unseen_any).sum())
            overall_mass_any = _ratio(overall[ANY_COLUMN], total_rows)
            mass_any_all = _ratio(n_unseen_any, n_all)
            union = {
                **base,
                "column": ANY_COLUMN,
                "missing_basis": None,
                # Sema birlesimi icin yapay bir "observed" paydasi
                # TANIMLANMAZ (belge §3.1).
                "n_observed": None,
                "n_unseen": n_unseen_any,
                "mass_all": mass_any_all,
                "mass_observed": None,
                "allocation": _ratio(n_unseen_any, overall[ANY_COLUMN]),
                "overall_mass_all": overall_mass_any,
                "enrichment_all": (mass_any_all / overall_mass_any
                                   if mass_any_all is not None and overall_mass_any
                                   else None),
            }
            union.update(_run_sensitivity(run_ids, mask, unseen_any)
                         if run_ids is not None else
                         {"n_runs": None, "run_mass_median": None,
                          "run_mass_min": None, "run_mass_max": None})
            rows.append(union)

    invariants = check_allocation_invariant(rows)
    return {"preregistration": "DENEY-D2A-Tabakali-Gorulmemis-Kutle.md",
            "target_rows": total_rows,
            "scenario_map": scenario_map,
            "strata_order": [(s["panel"], s["scenario_family"],
                              s["attack_label"]) for s in strata],
            "rows": rows,
            "allocation_invariant": invariants}


# ---------------------------------------------------------------------------
# Degismezlik denetimi (belge §3.2)
# ---------------------------------------------------------------------------
def check_allocation_invariant(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Birbirini dislayan tabakalarin allocation toplami 1 etmeli.

    Uc panel AYRI AYRI denetlenir. Toplam gorulmemis sifirsa allocation
    tumden `null`'dir ve o grup denetim disi kalir.
    """
    groups: dict[tuple, list[float | None]] = {}
    for row in rows:
        key = (row["panel"], row["schema"], row["column"])
        groups.setdefault(key, []).append(row["allocation"])

    failures = []
    checked = 0
    for (panel, schema, column), values in sorted(groups.items()):
        if all(value is None for value in values):
            continue
        total = sum(value for value in values if value is not None)
        checked += 1
        if abs(total - 1.0) > _ALLOCATION_TOLERANCE:
            failures.append({"panel": panel, "schema": schema,
                             "column": column, "sum": total})
    if failures:
        raise AssertionError(
            "allocation toplami 1 etmiyor -- tabakalar birbirini dislamiyor "
            f"ya da bir hucre dusmus: {failures[:5]}"
        )
    return {"groups_checked": checked, "tolerance": _ALLOCATION_TOLERANCE,
            "passed": True}


# ---------------------------------------------------------------------------
# Sunum
# ---------------------------------------------------------------------------
def _fmt(value: float | None, width: int = 8, digits: int = 4) -> str:
    return f"{'null':>{width}}" if value is None else f"{value:>{width}.{digits}f}"


def _print_panel(rows: Sequence[dict[str, Any]], schema: str,
                 column: str, title: str) -> None:
    print(f"\n{title}  (sema {schema}, kolon {column})")
    header = (f"  {'tabaka':30s} {'n_all':>7s} {'n_unseen':>9s} "
              f"{'mass_all':>9s} {'mass_obs':>9s} {'alloc':>8s} {'zengin':>8s} "
              f"{'kosu':>5s}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in rows:
        if row["schema"] != schema or row["column"] != column:
            continue
        label = f"{row['scenario_family']} x {row['attack_label']}"
        print(f"  {label:30s} {row['n_all']:7d} {row['n_unseen']:9d} "
              f"{_fmt(row['mass_all'], 9)} {_fmt(row['mass_observed'], 9)} "
              f"{_fmt(row['allocation'])} {_fmt(row['enrichment_all'])} "
              f"{(row['n_runs'] if row['n_runs'] is not None else '-'):>5}")


def _print_report(report: dict[str, Any]) -> None:
    rows = report["rows"]
    print(f"D2A TABAKALI GORULMEMIS KUTLE -- {report['target_rows']} hedef satir")
    print("14 tabakanin TAMAMI sabit sirada; bos hucreler gizlenmedi.\n")
    print("ham senaryo -> aile eslemesi:")
    for raw, family in report["scenario_map"].items():
        print(f"  {raw:26s} -> {family}")

    for panel, title in (("scenario", "PANEL 1 -- senaryo marjinleri"),
                         ("label", "PANEL 2 -- etiket marjinleri"),
                         ("scenario_x_label", "PANEL 3 -- senaryo x etiket")):
        subset = [row for row in rows if row["panel"] == panel]
        _print_panel(subset, "physical_proxy", ANY_COLUMN, title + " [BIRLESIM]")

    print("\n\nKOLON DUZEYI (physical_proxy; kolon sonuclari semadan bagimsizdir)")
    for column in ("modbus_function", "modbus_access", "flow_state",
                   "rudder_band", "propulsion_band"):
        subset = [row for row in rows
                  if row["panel"] == "scenario_x_label" and row["column"] == column]
        if subset:
            _print_panel(subset, "physical_proxy", column,
                         f"senaryo x etiket -- {column}")

    invariant = report["allocation_invariant"]
    print(f"\n\nallocation degismezligi: {invariant['groups_checked']} grup "
          f"denetlendi, hepsi 1.0 (tolerans {invariant['tolerance']})")
    print("mass_* = tabaka ICI yaygınlik;  alloc = toplam kutlenin DAGILIMI.")
    print("zengin = tabaka kutlesi / hedefin tamamindaki kutle (betimsel; "
          "p-degeri ya da etki buyuklugu DEGIL).")
    print("\nBu tablo BIRLIKTE GORULME gosterir. Nedensel iddia icin kodlama "
          "mudahalesi ve karsi-olgusal karsilastirma iceren AYRI bir "
          "on-kayit gerekir (belge §6).")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="D2A tabakali gorulmemis-kategori kutlesi (on-kayit: "
                    "DENEY-D2A-Tabakali-Gorulmemis-Kutle.md)")
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--schema", choices=list(SCHEMA_NAMES) + ["all"],
                        default="all")
    parser.add_argument("--out", type=Path,
                        default=Path("ml/ml_out/d2a_unseen_category_strata.json"))
    args = parser.parse_args(argv)

    schemas = list(SCHEMA_NAMES) if args.schema == "all" else [args.schema]
    source_df, source_paths = load_frame(args.source)
    target_df, target_paths = load_frame(args.target)
    print(f"kaynak : {args.source}  ({len(source_df)} satir)")
    print(f"hedef  : {args.target}  ({len(target_df)} satir, "
          f"{len(target_paths)} dosya)\n")

    # D2'nin render guard'i AYNEN uygulanir (belge §1).
    try:
        guard = render_precondition(source_df, target_df, schemas)
    except RenderGuardError as exc:
        print(exc)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"render_precondition": exc.report,
                                        "halted": True}, indent=2,
                                       ensure_ascii=False), encoding="utf-8")
        return 2
    print("render on-kosulu: GECTI")

    try:
        report = measure(source_df, target_df, schemas)
    except StrataGuardError as exc:
        print(f"\n[DURDU] {exc}")
        return 2

    report["render_precondition"] = guard
    report["inputs"] = {"source": args.source, "source_files": source_paths,
                        "target": args.target, "target_files": target_paths}
    _print_report(report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\nsonuc -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
