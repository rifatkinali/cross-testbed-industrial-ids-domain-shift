#!/usr/bin/env python3
"""D2 §7 -- H3 mekanizma testi: platolari gorulmemis-kutle katmanlari aciklar mi?

Ön-kayit: `DENEY-D2-Gorulmemis-Kategori-Teshisi.md` §7. Olcum yeri, esik
bantlari degil bantlar ARASI bosluklardir (§7.1); asgari destek esikleri
sonuclar gorulmeden sabitlenmistir (§7.2); karar olcutleri §7.4'tedir.

Neden bantlar arasi: errata §5'teki `0.29-0.49`, `0.72-0.74`, `0.88` degerleri
ESIKLERIN DUSTUGU bantlardir, platolar arasi sinirlar degil. Metrigin atladigi
yer bantlarin arasidir. Uc bant iki gecis verir.

Bu modul H3 kararini ESIK DEGERINDEN ayirir: nokta kutlesi seed'in skor
dagiliminin ozelligidir ve esigin nereye dustugunden bagimsiz olculur.
Errata'da gecersiz sayilan esik-bagimli metrikler karara GIRMEZ; esikler
yalniz betimsel olarak raporlanir.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from ml.lodo_generalization import (
    _build_model,
    _proba,
    _schema_spec,
    _split_by_run,
    _threshold_at_fpr,
    build_feature_frame,
    default_model_factory,
    validate_split,
)
from ml.unseen_category_diagnostic import (
    SEED_UNIVERSE,
    VAL_FRAC,
    _reject_empty,
    guard_class,
    load_frame,
)

H3_SCHEMA = "physical_proxy"

# §7.1 -- olcum yeri: bantlar ARASI bosluklar (acik aralik).
TRANSITIONS: dict[str, tuple[float, float]] = {
    "T1": (0.49, 0.72),      # A bandi ile B bandi arasinda
    "T2": (0.74, 0.88),      # B bandi ile C bandi arasinda
}

# §7.1 -- blok yaricapi; on-kayitli, sonuctan sonra genisletilmez. Kesin bir
# Random Forest skor kuantumu DEGILDIR (saf olmayan yapraklar 1/200'den kucuk
# fark uretebilir); 200 agacli recetede tek bir agacin saf-yaprak katkisindan
# kucuk olmayan bir komsuluk secimidir.
BLOCK_RADIUS = 0.005

# §7.2 -- asgari destek. %80 olcutu kucuk paydada anlamsizdir.
MIN_BLOCK_ROWS = 50
MIN_SEEDS_PER_TRANSITION = 10

# §7.4 -- hizalanma olcutu.
ALIGNMENT_MIN_SHARE = 0.80

# §7.3 -- yalniz BETIMSEL: esiklerin errata §5 bantlarina gore konumu. Karara
# girmez. C bandi errata'da tek nokta olarak verildigi icin blok yaricapiyla
# ayni genislikte alinir.
REGISTERED_BANDS: dict[str, tuple[float, float]] = {
    "A": (0.29, 0.49),
    "B": (0.72, 0.74),
    "C": (0.88 - BLOCK_RADIUS, 0.88 + BLOCK_RADIUS),
}
OUTSIDE_BANDS = "outside_registered_bands"


# ---------------------------------------------------------------------------
# Teshis amacli skorlama (belge §7 girisi)
# ---------------------------------------------------------------------------
def fit_and_score(source_df: pd.DataFrame, target_df: pd.DataFrame, seed: int,
                  schema: str = H3_SCHEMA, max_fpr: float = 0.01) -> dict[str, Any]:
    """v0.4 hattinin split/fit/esik adimlarini AYNEN tekrarlar (salt okunur).

    `run_lodo()` guard'i OpenPLC'de hicbir seed'i headline icin gecerli saymaz
    ve erken doner; H3 bu erken donusu YALNIZ TESHIS AMACIYLA asar. Burada
    hicbir performans metrigi (FPR, recall, alarm yuku) hesaplanmaz ve
    dondurulmez -- yalniz esik ve ham hedef skorlari. Bu islem yeni bir
    headline uretmez ve errata'daki gecersizlik kararini degistirmez.

    Guard nedenleri her seed icin KORUNUR ve raporlanir.
    """
    tr_idx, va_idx, split_basis = _split_by_run(source_df, VAL_FRAC, seed)
    guard_reasons = validate_split(source_df, tr_idx, va_idx, split_basis,
                                   max_fpr, target_df)

    spec = _schema_spec(schema)
    model = _build_model(default_model_factory, spec["numeric"],
                         spec["categorical"], seed)

    train_part = source_df.iloc[tr_idx]
    train_frame = build_feature_frame(train_part, schema)
    y_train = (train_part["label"].astype(str) == "attack").astype(int).to_numpy()
    model.fit(train_frame, y_train)

    val_part = source_df.iloc[va_idx]
    val_frame = build_feature_frame(val_part, schema)
    y_val = (val_part["label"].astype(str) == "attack").astype(int).to_numpy()
    threshold = _threshold_at_fpr(_proba(model, val_frame), y_val, max_fpr)

    target_frame = build_feature_frame(target_df, schema)
    scores = np.asarray(_proba(model, target_frame), dtype=float)

    return {
        "seed": int(seed),
        "split_basis": split_basis,
        "guard_class": guard_class(source_df, tr_idx, va_idx),
        "guard_reasons": guard_reasons,
        "threshold": float(threshold),
        "scores": scores,
        "train_frame": train_frame,
        "target_frame": target_frame,
    }


# ---------------------------------------------------------------------------
# Gorulmemislik imzasi (belge §7)
# ---------------------------------------------------------------------------
def unseen_signature(train_frame: pd.DataFrame, target_frame: pd.DataFrame,
                     columns: Sequence[str]) -> np.ndarray:
    """Bit `1` = deger, ilgili seed'in OpenPLC TRAIN sozlugunde yok.

    Bit sirasi `physical_proxy` kolon sirasidir; `build_feature_frame()`'in
    urettigi sira kullanilir, elle sabitlenmez.
    """
    bits = np.zeros((len(target_frame), len(columns)), dtype=np.int8)
    for j, column in enumerate(columns):
        vocabulary = set(train_frame[column].astype(str).unique())
        seen = target_frame[column].astype(str).isin(vocabulary).to_numpy()
        bits[:, j] = (~seen).astype(np.int8)
    return bits


def signature_strings(bits: np.ndarray) -> np.ndarray:
    return np.array(["".join(str(b) for b in row) for row in bits], dtype=object)


# ---------------------------------------------------------------------------
# Nokta kutleleri ve bloklar (belge §7.1)
# ---------------------------------------------------------------------------
def point_mass_inventory(scores: np.ndarray, low: float,
                         high: float) -> list[dict[str, Any]]:
    """Acik aralikta gozlenen HER ayrik skor ve satir sayisi (buyukten kucuge).

    Envanterin tamami raporlanir; en buyugun secimi boylece denetlenebilir
    kalir (belge §7.1).
    """
    inside = scores[(scores > low) & (scores < high)]
    if inside.size == 0:
        return []
    values, counts = np.unique(inside, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    return [{"score": float(values[i]), "n": int(counts[i])} for i in order]


# Kayan nokta payi: |0.705 - 0.700| ikili gosterimde 0.005000000000000004
# cikar ve tam sinirdaki satir blok DISINDA kalirdi. Yaricap on-kayitli 0.005
# olarak DEGISMEDI; yalniz karsilastirma gosterim hatasindan bagimsiz kilindi.
# Bu pay olcum olcegine gore ihmal edilebilir (1e-12 << 0.005).
_FLOAT_SLACK = 1e-12


def block_mask(scores: np.ndarray, centre: float,
               radius: float = BLOCK_RADIUS) -> np.ndarray:
    return np.abs(scores - centre) <= radius + _FLOAT_SLACK


def band_of(threshold: float) -> str:
    for name, (low, high) in REGISTERED_BANDS.items():
        if low <= threshold <= high:
            return name
    return OUTSIDE_BANDS


# ---------------------------------------------------------------------------
# Hizalanma (belge §7.4)
# ---------------------------------------------------------------------------
def alignment(layers: Sequence[int], signatures: Sequence[str]) -> dict[str, Any]:
    """Bloktaki satirlarin en az %80'i tek bir katmana ya da tek bir imzaya mi ait?"""
    n = len(layers)
    if n == 0:
        return {"n_rows": 0, "aligned": False, "layer_share": None,
                "signature_share": None, "dominant_layer": None,
                "dominant_signature": None}
    layer_counts = Counter(int(k) for k in layers)
    sig_counts = Counter(str(s) for s in signatures)
    dominant_layer, layer_n = layer_counts.most_common(1)[0]
    dominant_sig, sig_n = sig_counts.most_common(1)[0]
    layer_share, sig_share = layer_n / n, sig_n / n
    return {
        "n_rows": int(n),
        "layer_share": round(layer_share, 6),
        "signature_share": round(sig_share, 6),
        "dominant_layer": int(dominant_layer),
        "dominant_signature": dominant_sig,
        "aligned": bool(max(layer_share, sig_share) >= ALIGNMENT_MIN_SHARE),
        "layer_histogram": {str(k): int(v) for k, v in sorted(layer_counts.items())},
        "signature_histogram": dict(sorted(sig_counts.items(),
                                           key=lambda kv: -kv[1])[:10]),
    }


# ---------------------------------------------------------------------------
# Kosu
# ---------------------------------------------------------------------------
def run_h3(source_df: pd.DataFrame, target_df: pd.DataFrame,
           seeds: Iterable[int] = SEED_UNIVERSE,
           max_fpr: float = 0.01) -> dict[str, Any]:
    """Belge §7'nin tamami. Karar evreni seed evreninin TAMAMIDIR."""
    _reject_empty(source_df, "kaynak")
    _reject_empty(target_df, "hedef")
    seeds = [int(s) for s in seeds]

    per_seed: list[dict[str, Any]] = []
    pooled: dict[str, dict[str, Any]] = {
        name: {"layers": [], "signatures": [], "seeds": [],
               "base_n": 0, "base_unseen_n": 0}
        for name in TRANSITIONS
    }

    for seed in seeds:
        fitted = fit_and_score(source_df, target_df, seed, H3_SCHEMA, max_fpr)
        scores = fitted["scores"]
        columns = list(fitted["target_frame"].attrs["categorical"])
        bits = unseen_signature(fitted["train_frame"], fitted["target_frame"],
                                columns)
        layers = bits.sum(axis=1).astype(int)
        signatures = signature_strings(bits)

        values, counts = np.unique(scores, return_counts=True)
        entry: dict[str, Any] = {
            "seed": seed,
            "guard_class": fitted["guard_class"],
            "guard_reasons": fitted["guard_reasons"],
            "threshold": round(fitted["threshold"], 6),
            "threshold_band": band_of(fitted["threshold"]),
            "signature_columns": columns,
            "layer_counts": {str(k): int(v) for k, v in
                             sorted(Counter(layers.tolist()).items())},
            "signature_counts": {str(k): int(v) for k, v in
                                 sorted(Counter(signatures.tolist()).items(),
                                        key=lambda kv: -kv[1])},
            "distinct_scores_total": int(len(values)),
            "largest_point_mass_share": round(float(counts.max() / len(scores)), 6),
            "distinct_scores_per_layer": {
                str(k): int(len(np.unique(scores[layers == k])))
                for k in sorted(set(layers.tolist()))
            },
            "transitions": {},
        }

        for name, (low, high) in TRANSITIONS.items():
            inventory = point_mass_inventory(scores, low, high)
            item: dict[str, Any] = {
                "interval": [low, high],
                "point_mass_inventory": inventory[:20],
                "point_mass_n_distinct": len(inventory),
            }
            if not inventory:
                item.update(chosen=None, block_rows=0, qualifies=False,
                            reason="aralikta skor yok")
            else:
                chosen = inventory[0]
                mask = block_mask(scores, chosen["score"])
                block_rows = int(mask.sum())
                item.update(chosen=chosen, block_rows=block_rows,
                            qualifies=bool(block_rows >= MIN_BLOCK_ROWS))
                if item["qualifies"]:
                    item["block_alignment"] = alignment(layers[mask],
                                                        signatures[mask])
                    pooled[name]["layers"].extend(layers[mask].tolist())
                    pooled[name]["signatures"].extend(signatures[mask].tolist())
                    pooled[name]["seeds"].append(seed)
                    # Taban oran AYNI seed kumesinden toplanir ki blok payiyla
                    # karsilastirilabilir olsun (post-hoc kontrol, asagida).
                    pooled[name]["base_n"] += int(len(layers))
                    pooled[name]["base_unseen_n"] += int((layers > 0).sum())
                else:
                    item["reason"] = (f"blok {block_rows} satir "
                                      f"< MIN_BLOCK_ROWS={MIN_BLOCK_ROWS}")
            entry["transitions"][name] = item

        per_seed.append(entry)

    # ---- gecis kararlari (belge §7.2 ve §7.4)
    transition_results: dict[str, Any] = {}
    for name in TRANSITIONS:
        qualifying = pooled[name]["seeds"]
        evaluable = len(qualifying) >= MIN_SEEDS_PER_TRANSITION
        result: dict[str, Any] = {
            "interval": list(TRANSITIONS[name]),
            "qualifying_seeds": qualifying,
            "qualifying_seed_n": len(qualifying),
            "min_seeds_required": MIN_SEEDS_PER_TRANSITION,
            "evaluable": evaluable,
        }
        if evaluable:
            result["pooled_alignment"] = alignment(pooled[name]["layers"],
                                                   pooled[name]["signatures"])
            result["aligned"] = result["pooled_alignment"]["aligned"]
            result["status"] = "hizali" if result["aligned"] else "hizasiz"
            result["post_hoc_base_rate_check"] = _base_rate_check(pooled[name])
        else:
            result["aligned"] = None
            result["status"] = "underpowered"
            result["note"] = ("ne destek ne ret; esik sonuctan sonra "
                              "dusurulmez (belge §7.2)")
        transition_results[name] = result

    verdict = _verdict(transition_results)
    return {
        "preregistration": "DENEY-D2-Gorulmemis-Kategori-Teshisi.md §7",
        "schema": H3_SCHEMA,
        "seeds": seeds,
        "constants": {"BLOCK_RADIUS": BLOCK_RADIUS,
                      "MIN_BLOCK_ROWS": MIN_BLOCK_ROWS,
                      "MIN_SEEDS_PER_TRANSITION": MIN_SEEDS_PER_TRANSITION,
                      "ALIGNMENT_MIN_SHARE": ALIGNMENT_MIN_SHARE,
                      "TRANSITIONS": {k: list(v) for k, v in TRANSITIONS.items()}},
        "threshold_bands_descriptive": _threshold_band_summary(per_seed),
        "transitions": transition_results,
        "verdict": verdict,
        "per_seed": per_seed,
    }


def _base_rate_check(bucket: dict[str, Any]) -> dict[str, Any]:
    """POST-HOC kontrol -- sonuclar goruldukten SONRA eklendi, karara GIRMEZ.

    §7.4'teki hizalanma olcutu "tek bir katman ya da tek bir imza" der ama
    katmanin HANGISI oldugunu sormaz. Gorulmemis satirlar hedefte cok seyrekse
    her blok taban oran geregi `k=0` agirlikli cikar ve olcut mekanizmadan
    BAGIMSIZ olarak saglanir. Bu alan, hizalanmanin taban orandan mi yoksa
    gercek bir yigilmadan mi geldigini gorunur kilar.

    zenginlesme ~ 1.0  -> blok, hedefin genelinden farksiz (bilgi yok)
    zenginlesme >> 1.0 -> gorulmemis satirlar blokta yigiliyor (mekanizma)
    """
    block_rows = len(bucket["layers"])
    block_unseen = sum(1 for k in bucket["layers"] if k > 0)
    base_share = (bucket["base_unseen_n"] / bucket["base_n"]
                  if bucket["base_n"] else None)
    block_share = block_unseen / block_rows if block_rows else None
    enrichment = (round(block_share / base_share, 4)
                  if base_share else None)
    return {
        "note": "SONUCLAR GORULDUKTEN SONRA eklendi; §7.4 kararina GIRMEZ.",
        "base_unseen_share": round(base_share, 6) if base_share is not None else None,
        "block_unseen_share": round(block_share, 6) if block_share is not None else None,
        "block_unseen_rows": block_unseen,
        "block_rows": block_rows,
        "enrichment": enrichment,
    }


def _verdict(transitions: dict[str, Any]) -> dict[str, Any]:
    """Belge §7.4 karar olcutleri -- sonuctan sonra gevsetilmez."""
    evaluable = [name for name, item in transitions.items() if item["evaluable"]]
    aligned = [name for name in evaluable if transitions[name]["aligned"]]

    def _finish(result: dict[str, Any]) -> dict[str, Any]:
        result.update(evaluable=evaluable, aligned=aligned)
        result.update(_degeneracy(transitions, aligned))
        return result

    if not evaluable:
        return _finish({"h3": "test_edilemedi",
                        "explanation": "her iki gecis de underpowered; H3 ne "
                                       "reddedilmis ne desteklenmis sayilir"})
    if not aligned:
        return _finish({"h3": "reddedildi",
                        "explanation": "degerlendirilebilir gecislerin hicbirinde "
                                       f"%{ALIGNMENT_MIN_SHARE:.0%} hizalanma yok"})
    if len(aligned) == len(TRANSITIONS):
        return _finish({"h3": "desteklendi",
                        "explanation": "iki gecisin ikisinde de hizalanma"})
    return _finish({"h3": "kismi_destek",
                    "explanation": "yalniz bir geciste hizalanma; mekanizmanin "
                                   "genel aciklamasi SAYILMAZ"})


def _degeneracy(transitions: dict[str, Any], aligned: list[str]) -> dict[str, Any]:
    """Hizalanma `k=0` katmanindan mi geliyor? POST-HOC uyari, karar DEGIL.

    §7.4 "tek bir katman" der, katmanin hangisi oldugunu sormaz. Hizalanmayi
    tasiyan katman `k=0` ise -- yani blok agirlikla GORULMEMIS KATEGORI
    TASIMAYAN satirlardan olusuyorsa -- olcut saglanmis gorunur ama H3'un
    iddiasinin TERSI gozlenmis olur. Bu durumda karar dejeneredir.
    """
    if not aligned:
        return {"degenerate": False}
    dominants = []
    for name in aligned:
        pooled = transitions[name].get("pooled_alignment") or {}
        if "dominant_layer" in pooled:
            dominants.append(pooled["dominant_layer"])
    if not dominants or not all(layer == 0 for layer in dominants):
        return {"degenerate": False}
    return {
        "degenerate": True,
        "degenerate_note": (
            "Hizalanmayi tasiyan katman k=0'dir: bloklar agirlikla "
            "GORULMEMIS KATEGORI TASIMAYAN satirlardan olusuyor. Olcut "
            "saglanmis gorunur, fakat gozlenen sey H3'un iddiasinin TERSIDIR. "
            "Karar on-kayitli haliyle KAYDEDILIR; olcut sonuca gore "
            "degistirilmez. Bkz. belge §10 R3."
        ),
    }


def _threshold_band_summary(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    bands: dict[str, list[int]] = {}
    for entry in per_seed:
        bands.setdefault(entry["threshold_band"], []).append(entry["seed"])
    thresholds = [entry["threshold"] for entry in per_seed]
    return {"by_band": {k: sorted(v) for k, v in sorted(bands.items())},
            "threshold_min": round(min(thresholds), 6),
            "threshold_max": round(max(thresholds), 6),
            "note": "yalniz betimsel; H3 karari esik DEGERINE bagli degildir"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print_report(report: dict[str, Any]) -> None:
    print(f"H3 MEKANIZMA TESTI -- sema {report['schema']}, "
          f"{len(report['seeds'])} seed\n")

    bands = report["threshold_bands_descriptive"]
    print("esik konumlari (BETIMSEL; karara girmez)")
    print(f"  aralik: {bands['threshold_min']:.4f} - {bands['threshold_max']:.4f}")
    for band, seeds in bands["by_band"].items():
        print(f"  {band:26s} n={len(seeds):2d}  {seeds}")
    print()

    print("GECISLER")
    for name, item in report["transitions"].items():
        low, high = item["interval"]
        print(f"  {name}  ({low} , {high})  nitelikli seed "
              f"{item['qualifying_seed_n']}/{item['min_seeds_required']} "
              f"-> {item['status'].upper()}")
        if item["evaluable"]:
            pooled = item["pooled_alignment"]
            print(f"      havuz {pooled['n_rows']} satir x seed; "
                  f"katman payi {pooled['layer_share']:.4f} "
                  f"(k={pooled['dominant_layer']}), "
                  f"imza payi {pooled['signature_share']:.4f} "
                  f"({pooled['dominant_signature']})")
            print(f"      olcut %{ALIGNMENT_MIN_SHARE:.0%} -> "
                  f"{'HIZALI' if item['aligned'] else 'HIZASIZ'}")
            check = item.get("post_hoc_base_rate_check", {})
            if check:
                print(f"      [post-hoc] blokta gorulmemis pay "
                      f"{check['block_unseen_share']:.5f} vs taban "
                      f"{check['base_unseen_share']:.5f} -> zenginlesme "
                      f"{check['enrichment']}")
        else:
            print(f"      {item['note']}")
    print()

    verdict = report["verdict"]
    print(f"KARAR: H3 = {verdict['h3'].upper()}")
    print(f"  {verdict['explanation']}")
    print(f"  degerlendirilebilir gecisler: {verdict['evaluable'] or '(yok)'}")
    print(f"  hizali gecisler: {verdict['aligned'] or '(yok)'}")
    if verdict.get("degenerate"):
        print()
        print("  *** DEJENERE KARAR -- POST-HOC UYARI ***")
        print(textwrap.fill(verdict["degenerate_note"], width=76,
                            initial_indent="  ", subsequent_indent="  "))


def main(argv: Sequence[str] | None = None) -> int:
    from ml.unseen_category_diagnostic import DEFAULT_SOURCE, DEFAULT_TARGET

    parser = argparse.ArgumentParser(
        description="D2 §7 H3 mekanizma testi (on-kayit: "
                    "DENEY-D2-Gorulmemis-Kategori-Teshisi.md)")
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEED_UNIVERSE))
    parser.add_argument("--max-fpr", type=float, default=0.01)
    parser.add_argument("--out", type=Path,
                        default=Path("ml/ml_out/d2_h3_mechanism.json"))
    args = parser.parse_args(argv)

    source_df, source_paths = load_frame(args.source)
    target_df, target_paths = load_frame(args.target)
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]

    print(f"kaynak : {args.source}  ({len(source_df)} satir)")
    print(f"hedef  : {args.target}  ({len(target_df)} satir, "
          f"{len(target_paths)} dosya)")
    print(f"seed   : {len(seeds)} adet\n")

    report = run_h3(source_df, target_df, seeds, args.max_fpr)
    report["inputs"] = {"source": args.source, "source_files": source_paths,
                        "target": args.target, "target_files": target_paths}
    _print_report(report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\nsonuc -> {args.out}")
    print("Performans metrikleri (FPR/recall) URETILMEDI ve YAYIMLANMAZ; "
          "bypass yalniz skorlara erisim icindir.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
