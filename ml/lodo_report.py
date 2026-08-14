#!/usr/bin/env python3
"""LODO sonuclarini olay-agirlikli VE senaryo-dengeli olarak raporlar.

Ana sonuc olay-agirliklidir ve DEGISTIRILMEZ. Bu arac ek analiz uretir:

  * Olay-agirlikli AUC : her satir esit agirlik. Test setinin bilesimine
    duyarlidir -- mevcut kampanyada recon satirlarin %66'sidir cunku tarama
    saniyede diger senaryolardan ~5 kat fazla olay uretir.
  * Senaryo-dengeli AUC: her senaryo esit agirlik alir (satir agirligi
    1/n_senaryo). Bilesim carpikligindan bagimsizdir.
  * Senaryo-bazli AUC  : her saldiri senaryosu, normal senaryo satirlarina
    KARSI ayri ayri. En yorumlanabilir olan.

Guven araligi kosu (run) bazinda bootstrap ile hesaplanir. Bagimsiz birim
kosudur; satir bazinda bootstrap ayni kosunun binlerce korele satirini
bagimsiz sayar ve araligi yapay olarak daraltir.

Seed yayilimi ayrica raporlanir: 3 seed ile parametrik bir GA anlamli olmaz,
bu yuzden min/mean/max verilir ve n acikca yazilir.

Kullanim:
    python ml/lodo_report.py --train attack/dataset.csv \
        --test <birlesik events-v0.4.csv> --seeds 42,43,44
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.lodo_generalization import (  # noqa: E402
    SCHEMA_NAMES,
    _proba,
    _schema_spec,
    _split_by_run,
    _threshold_at_fpr,
    build_feature_frame,
    default_model_factory,
    scenario_family,
)

ATTACK_FAMILIES = ("manipulation", "recon")
N_BOOT = 2000
ALPHA = 0.05
# Bu genislikten dar bir kosu-bootstrap araligi GENELLEME araligi olarak
# okunamaz: tekrarlar betiklenmis bir senaryonun neredeyse birebir kopyasi
# oldugu icin aralik yalnizca tekrar gurultusunu olcer. Olculdu: 3 tekrarin
# AUC yayilimi < 0.001.
DEGENERATE_CI_WIDTH = 0.01


def _auc(y, scores, weights=None) -> float:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores, sample_weight=weights))


def balanced_weights(families: np.ndarray) -> np.ndarray:
    """Her senaryo esit toplam agirlik alsin: w_i = 1 / n_senaryo(i)."""
    weights = np.ones(len(families), dtype=float)
    for family in np.unique(families):
        mask = families == family
        weights[mask] = 1.0 / float(mask.sum())
    return weights


def _bootstrap_over_runs(runs, families, y, scores, weights, seed=42):
    """Senaryo icinde TABAKALI, kosu bazinda bootstrap.

    Bagimsiz birim kosudur, satir DEGIL: satir bazinda bootstrap ayni kosunun
    binlerce korele satirini bagimsiz sayar ve araligi yapay olarak daraltir.

    Tabakalama ZORUNLU: 12 kosu ayrimsiz yeniden orneklenirse senaryo BILESIMI
    de degisir (recon kosulari ~7700 satir, digerleri ~1100). O zaman bootstrap
    dagilimi orijinal bilesimin AUC'sini degil, bilesim degiskenligiyle karisik
    bir seyi tahmin eder -- olculdu: olay-agirlikli nokta tahmini 0.322 iken
    ayrimsiz GA [0.289, 0.502] cikiyor ve ~0.395'te merkezleniyor.

    Tasarim 3 kosu x 4 senaryo oldugu icin her senaryo icinde kendi kosu sayisi
    kadar yeniden ornekleme yapilir; bilesim korunur.
    """
    values = []
    strata = {}
    for family in np.unique(families):
        runs_in = np.unique(runs[families == family])
        if len(runs_in):
            strata[family] = runs_in
    if not strata:
        return (float("nan"), float("nan"))
    index_by_run = {run: np.where(runs == run)[0] for run in np.unique(runs)}
    rng = np.random.RandomState(seed)
    for _ in range(N_BOOT):
        picked = []
        for runs_in in strata.values():
            picked.extend(rng.choice(runs_in, size=len(runs_in), replace=True))
        idx = np.concatenate([index_by_run[r] for r in picked])
        value = _auc(y[idx], scores[idx],
                     weights[idx] if weights is not None else None)
        if value == value:            # NaN degilse
            values.append(value)
    if not values:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(values, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return (float(lo), float(hi))


def analyse(train_df, test_df, schema, seeds, max_fpr=0.01):
    families = scenario_family(test_df["scenario"].astype(str))
    labels = test_df["label"].astype(str).to_numpy()
    y = (labels == "attack").astype(int)
    runs = test_df["run_id"].astype(str).to_numpy()
    weights = balanced_weights(families)

    spec = _schema_spec(schema)
    Xte = build_feature_frame(test_df, schema)

    per_seed = []
    for seed in seeds:
        tr_idx, va_idx, split_basis = _split_by_run(train_df, 0.3, seed)
        model = default_model_factory(spec["numeric"], spec["categorical"], seed)
        model.fit(build_feature_frame(train_df.iloc[tr_idx], schema),
                  (train_df.iloc[tr_idx]["label"].astype(str) == "attack")
                  .astype(int).to_numpy())
        Xva = build_feature_frame(train_df.iloc[va_idx], schema)
        yva = (train_df.iloc[va_idx]["label"].astype(str) == "attack") \
            .astype(int).to_numpy()
        proba_va = _proba(model, Xva)
        threshold = _threshold_at_fpr(proba_va, yva, max_fpr)
        scores = _proba(model, Xte)

        entry = {
            "seed": int(seed),
            "split_basis": split_basis,
            "threshold": round(float(threshold), 4),
            "auc_validation_openplc": round(_auc(yva, proba_va), 4),
            "auc_event_weighted": round(_auc(y, scores), 4),
            "auc_scenario_balanced": round(_auc(y, scores, weights), 4),
            "per_scenario_auc": {},
            "scores": scores,
        }
        # Senaryo-bazli: her saldiri senaryosu NORMAL senaryoya karsi
        normal_mask = families == "normal"
        for family in ATTACK_FAMILIES:
            mask = (families == family) & (y == 1)
            pair = mask | normal_mask
            if pair.sum() and mask.any() and normal_mask.any():
                entry["per_scenario_auc"][family] = round(
                    _auc(y[pair], scores[pair]), 4)
        per_seed.append(entry)

    def spread(key):
        values = [e[key] for e in per_seed if e[key] == e[key]]
        if not values:
            return None
        return {"mean": round(float(np.mean(values)), 4),
                "min": round(float(min(values)), 4),
                "max": round(float(max(values)), 4),
                "n_seeds": len(values)}

    # GA: seed'ler arasi ortalama skorla, kosu bazinda bootstrap
    mean_scores = np.mean([e["scores"] for e in per_seed], axis=0)
    ev_lo, ev_hi = _bootstrap_over_runs(runs, families, y, mean_scores, None)
    ba_lo, ba_hi = _bootstrap_over_runs(runs, families, y, mean_scores, weights)

    result = {
        "schema": schema,
        "seeds": [int(s) for s in seeds],
        "composition": {family: int((families == family).sum())
                        for family in np.unique(families)},
        "auc_validation_openplc": spread("auc_validation_openplc"),
        "auc_event_weighted": spread("auc_event_weighted"),
        "auc_scenario_balanced": spread("auc_scenario_balanced"),
        "auc_event_weighted_ci95_runs": [round(ev_lo, 4), round(ev_hi, 4)],
        "auc_scenario_balanced_ci95_runs": [round(ba_lo, 4), round(ba_hi, 4)],
        "per_scenario_auc": {
            family: {
                "mean": round(float(np.mean([
                    e["per_scenario_auc"][family] for e in per_seed
                    if family in e["per_scenario_auc"]])), 4),
                "min": round(float(min(e["per_scenario_auc"][family]
                                       for e in per_seed
                                       if family in e["per_scenario_auc"])), 4),
                "max": round(float(max(e["per_scenario_auc"][family]
                                       for e in per_seed
                                       if family in e["per_scenario_auc"])), 4),
            }
            for family in ATTACK_FAMILIES
            if any(family in e["per_scenario_auc"] for e in per_seed)
        },
        "per_seed": [{k: v for k, v in e.items() if k != "scores"}
                     for e in per_seed],
    }

    # Tekrar bazinda AUC: her saldiri kosusu TUM normal satirlara karsi.
    # Gercek tekrar yayilimini ACIKCA gosterir; GA'nin neden dar oldugu
    # gizlenmez.
    normal_mask = families == "normal"
    per_replicate = {}
    for family in ATTACK_FAMILIES:
        values = {}
        for run in sorted(set(runs[families == family])):
            mask = (runs == run) & (y == 1)
            pair = mask | normal_mask
            if mask.any() and normal_mask.any():
                values[run] = round(_auc(y[pair], mean_scores[pair]), 4)
        if values:
            spread_values = list(values.values())
            per_replicate[family] = {
                "per_run": values,
                "mean": round(float(np.mean(spread_values)), 4),
                "range": [round(min(spread_values), 4),
                          round(max(spread_values), 4)],
                "range_width": round(max(spread_values) - min(spread_values), 5),
                "n_runs": len(values),
            }
    result["per_replicate_auc"] = per_replicate

    widths = {
        "auc_event_weighted": round(ev_hi - ev_lo, 5),
        "auc_scenario_balanced": round(ba_hi - ba_lo, 5),
    }
    result["ci_widths"] = widths
    result["ci_is_replicate_noise_only"] = bool(
        max(widths.values()) < DEGENERATE_CI_WIDTH)
    result["ci_interpretation"] = (
        "Kosu-bootstrap araligi COK DAR: tekrarlar betiklenmis senaryonun "
        "neredeyse birebir kopyasi oldugu icin aralik yalnizca TEKRAR "
        "gurultusunu olcer. GENELLEME araligi DEGILDIR; tek testbed cifti, tek "
        "saldiri uygulamasi ve tek istemci betigi kaynakli belirsizligi "
        "KAPSAMAZ. Bu belirsizlik nitel olarak raporlanmalidir."
        if result["ci_is_replicate_noise_only"] else
        "Kosu-bootstrap araligi anlamli genislikte; tekrarlar arasi degiskenlik "
        "gozlenebilir duzeyde."
    )
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="LODO ek analizi: olay-agirlikli + senaryo-dengeli AUC")
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--max-fpr", type=float, default=0.01)
    parser.add_argument("--schema", choices=SCHEMA_NAMES + ["all"], default="all")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    train_df = pd.read_csv(args.train, low_memory=False)
    test_df = pd.read_csv(args.test, low_memory=False)
    schemas = SCHEMA_NAMES if args.schema == "all" else [args.schema]

    results = [analyse(train_df, test_df, sc, seeds, args.max_fpr)
               for sc in schemas]

    print("TEST SETI BILESIMI (olay sayisi)")
    for family, count in results[0]["composition"].items():
        total = sum(results[0]["composition"].values())
        print(f"  {family:14s} {count:6d}  (%{100 * count / total:.1f})")
    print()
    header = (f"{'sema':16s} {'val-AUC':9s} {'AUC olay':9s} "
              f"{'GA95(kosu)':18s} {'AUC dengeli':11s} {'GA95(kosu)':18s} "
              f"{'manip':7s} {'recon':7s}")
    print(header)
    print("-" * len(header))
    for item in results:
        ev, ba = item["auc_event_weighted"], item["auc_scenario_balanced"]
        evc, bac = (item["auc_event_weighted_ci95_runs"],
                    item["auc_scenario_balanced_ci95_runs"])
        ps = item["per_scenario_auc"]
        print(f"{item['schema']:16s} "
              f"{item['auc_validation_openplc']['mean']:<9.4f} "
              f"{ev['mean']:<9.4f} [{evc[0]:.3f},{evc[1]:.3f}]      "
              f"{ba['mean']:<11.4f} [{bac[0]:.3f},{bac[1]:.3f}]      "
              f"{ps.get('manipulation', {}).get('mean', float('nan')):<7.4f} "
              f"{ps.get('recon', {}).get('mean', float('nan')):<7.4f}")
    print()
    print("TEKRAR BAZINDA AUC (her saldiri kosusu, tum normal satirlara karsi)")
    for item in results:
        parts = []
        for family, info in item.get("per_replicate_auc", {}).items():
            parts.append(f"{family}={info['mean']:.4f} "
                         f"(yayilim {info['range_width']:.5f}, n={info['n_runs']})")
        print(f"  {item['schema']:16s} " + "  ".join(parts))
    print()
    print(f"seed sayisi = {len(seeds)} ({seeds}); seed yayilimi min/max olarak "
          "verilir -- bu kadar az seed ile parametrik GA anlamli olmaz.")
    print("GA95(kosu) = senaryo icinde TABAKALI kosu bootstrap "
          "(3 kosu x 4 senaryo);")
    print("             bagimsiz birim kosudur ve senaryo bilesimi korunur.")
    if any(item["ci_is_replicate_noise_only"] for item in results):
        print()
        print("!! UYARI: kosu-bootstrap araliklari COK DAR "
              "(genislik < 0.01).")
        print("   Tekrarlar betiklenmis senaryonun neredeyse birebir "
              "kopyasidir; aralik yalnizca")
        print("   TEKRAR gurultusunu olcer. GENELLEME araligi DEGILDIR -- tek "
              "testbed cifti, tek")
        print("   saldiri uygulamasi ve tek istemci betigi belirsizligini "
              "KAPSAMAZ.")

    if args.out:
        args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False)
                            + "\n", encoding="utf-8")
        print(f"\nJSON: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
