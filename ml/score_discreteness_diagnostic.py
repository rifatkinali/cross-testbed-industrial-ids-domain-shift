#!/usr/bin/env python3
"""D3 -- esik platolarinin skor kesikligi teshisi.

Ön-kayit: `DENEY-D3-Esik-Platolari-Skor-Kesikligi.md`. Kod belge donduruktan
SONRA yazilmistir; H1-H4 esikleri, karar araligi `I`, seed evreni, `T` paneli,
esitlik tanimi ve asgari destek sayilari buradan DEGISTIRILEMEZ.

D3'un adayi D2'ninkinden farklidir: gorulmemis kategori degil, skor
dagiliminin KENDI kesikligi. Iki olgu ayri tanimlanir ve ayri yanlislanir
(belge §3.3):

  w(tau)   -> platonun GENISLIGI      (skor destegindeki bosluk)
  n(R)/N   -> sicramanin YUKSEKLIGI   (sinirdaki nokta kutlesi)

Her ikisi de kendi taban oranina karsi olculur. D2'nin dersi: ham pay tek
basina karar olcutu degildir.

D3 performans metrigi (FPR, recall, precision, F1, alarm yuku) URETMEZ; mevcut
guard'lari yalniz ham skor, esik ve agac-yaprak ciktisina erisim icin asar.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
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
    DEFAULT_SOURCE,
    DEFAULT_TARGET,
    SEED_UNIVERSE,
    VAL_FRAC,
    guard_class,
    load_frame,
)

# --- belge §2: degistirilmeyecek analiz evreni -----------------------------
SCHEMA = "physical_proxy"
DECISION_INTERVAL = (0.1238, 0.8795)          # I = [a, b] (Errata §9)
MAX_FPR = 0.01
CANONICAL_TARGET_ROWS = 34949
CANONICAL_TARGET_RUNS = 12

# --- belge §4.1: agac sayisi paneli (ic ice) -------------------------------
TREE_PANEL = (25, 50, 100, 200)

# --- belge §5: on-kayitli karar esikleri -----------------------------------
MIN_K_I = 10                    # degerlendirilebilirlik kapisi (bkz. §5 notu)
MIN_EVALUABLE_SEEDS = 10
MIN_POINT_ROWS = 50             # "buyuk blok sicramasi" asgari satir destegi
MIN_N_I = 100
ENRICHMENT_SUPPORT_MEDIAN = 2.0
SUPPORT_SHARE = 0.75
FALSIFY_SHARE = 0.50
RHO_SUPPORT_MIN = 0.8
RHO_WIDTH_MAX = -0.8
KEFF_RATIO_SUPPORT = 2.0
KEFF_RATIO_FALSIFY = 1.25


class D3GuardError(RuntimeError):
    """Belge §7 durdurma kosulu; tolerans/yuvarlama/seed cikarma UYGULANMAZ."""


# ---------------------------------------------------------------------------
# Skor destegi (belge §3.1 - §3.2)
# ---------------------------------------------------------------------------
def _hex_scores(values: Iterable[float]) -> list[str]:
    return [float(v).hex() for v in values]


def support_profile(scores: np.ndarray,
                    interval: tuple[float, float] = DECISION_INTERVAL) -> dict[str, Any]:
    """Ayri skor sayisi, nokta kutleleri, bosluklar ve etkin destek.

    Esitlik BIT duzeyindedir: `np.unique` float64 bit gosterimine gore ayirir;
    yuvarlama, histogram kutusu ya da +-epsilon komsulugu KULLANILMAZ.
    """
    low, high = interval
    values, counts = np.unique(scores, return_counts=True)
    n = int(len(scores))
    inside = (values > low) & (values < high)

    masses = counts / n
    k_eff = float(np.exp(-np.sum(masses * np.log(masses))))

    return {
        "n": n,
        "K": int(len(values)),
        "K_I": int(inside.sum()),
        "N_I": int(counts[inside].sum()),
        "support_ratio": float(len(values) / n),
        "K_eff": k_eff,
        "unique_scores": values,
        "unique_counts": counts,
        # Butun envanter yayimlanir; yalniz en buyukler degil (belge §3.2).
        "point_masses": [
            {"score": float(v), "score_hex": float(v).hex(),
             "n": int(c), "mass": float(c / n)}
            for v, c in zip(values, counts)
        ],
        "gaps": [float(b - a) for a, b in zip(values[:-1], values[1:])],
        "largest_point_mass": float(counts.max() / n),
        "widest_gap": (float(np.max(np.diff(values))) if len(values) > 1 else None),
    }


def plateau(scores_unique: np.ndarray, threshold: float,
            interval: tuple[float, float] = DECISION_INTERVAL) -> dict[str, Any]:
    """Belge §3.3 -- yhat(t) = 1[S >= t] vektorunun degismedigi en genis aralik.

    P(tau) = (L, R] kesisim I. `>=` kurali nedeniyle SAG uc plato icindedir.
    ACIK aralik (L, R) icinde hedef skor yoktur; R'nin kendisi bir hedef
    skorudur ve P(tau) icindedir.
    """
    low, high = interval
    below = scores_unique[scores_unique < threshold]
    at_or_above = scores_unique[scores_unique >= threshold]
    left = float(below.max()) if below.size else float("-inf")
    right = float(at_or_above.min()) if at_or_above.size else float("inf")

    clipped_low = max(left, low)
    clipped_high = min(right, high)
    width = max(0.0, clipped_high - clipped_low)

    return {
        "L": left, "R": right,
        "L_hex": (left.hex() if math.isfinite(left) else None),
        "R_hex": (right.hex() if math.isfinite(right) else None),
        "L_finite": bool(math.isfinite(left)),
        "R_finite": bool(math.isfinite(right)),
        "R_in_interval": bool(math.isfinite(right) and low < right < high),
        "clipped": [clipped_low, clipped_high],
        "width": float(width),
    }


def enrichments(profile: dict[str, Any], plate: dict[str, Any],
                interval: tuple[float, float] = DECISION_INTERVAL) -> dict[str, Any]:
    """Belge §3.4 -- bosluk ve kutle, KENDI taban oranlarina karsi.

    Taban, `I` genisliginin K_I+1 boslugu ve kutlenin K_I ayri skora esit
    dagilmasidir. Bunlar p-degeri degildir.
    """
    low, high = interval
    k_i, n_i = profile["K_I"], profile["N_I"]

    base_gap = ((high - low) / (k_i + 1)) if k_i >= 0 else None
    gap_enrichment = (plate["width"] / base_gap) if base_gap else None

    boundary_n = None
    boundary_mass_i = base_mass_i = mass_enrichment = None
    if plate["R_in_interval"]:
        match = profile["unique_scores"] == plate["R"]
        boundary_n = int(profile["unique_counts"][match].sum())
        if n_i:
            boundary_mass_i = boundary_n / n_i
        if k_i:
            base_mass_i = 1.0 / k_i
        if boundary_mass_i is not None and base_mass_i:
            mass_enrichment = boundary_mass_i / base_mass_i

    return {"base_gap": base_gap, "gap_enrichment": gap_enrichment,
            "boundary_n": boundary_n, "boundary_mass_I": boundary_mass_i,
            "base_mass_I": base_mass_i, "mass_enrichment": mass_enrichment}


# ---------------------------------------------------------------------------
# Skorlama (belge §2, §4.1, §4.2)
# ---------------------------------------------------------------------------
def _tree_panel_scores(estimators, matrix: np.ndarray,
                       panel: Sequence[int]) -> dict[int, dict[str, Any]]:
    """Ic ice ilk-T agac skorlari + sert-oy karsi-olgusali, tek gecerde.

    Toplama SIRALI ve `estimators_` sirasindadir; `predict_proba` da ayni
    sirayla toplar, bu yuzden T=200 bit duzeyinde esitlenir. Bu esitlik
    `n_jobs = 1` kosuluna baglidir (belge §4.1 notu).
    """
    n_rows = matrix.shape[0]
    soft = np.zeros(n_rows, dtype=float)
    hard = np.zeros(n_rows, dtype=float)
    pure_visits = 0
    purity_sum = 0.0
    out: dict[int, dict[str, Any]] = {}

    for index, tree in enumerate(estimators, start=1):
        proba = tree.predict_proba(matrix)
        if proba.shape[1] != 2:
            raise D3GuardError(
                f"agac {index} iki sinif dondurmedi (shape={proba.shape}); "
                "yaprak olasiligi p_ij tanimsiz kalir"
            )
        leaf = proba[:, 1]
        soft += leaf
        hard += (leaf >= 0.5).astype(float)      # esitlikte 1 (belge §4.2)
        pure_visits += int(np.count_nonzero((leaf == 0.0) | (leaf == 1.0)))
        purity_sum += float(np.maximum(leaf, 1.0 - leaf).sum())
        if index in panel:
            out[index] = {
                "soft": soft / index,
                "hard": hard / index,
                "pure_visit_share": pure_visits / (n_rows * index),
                "mean_leaf_purity": purity_sum / (n_rows * index),
            }
    return out


def score_seed(source_df: pd.DataFrame, target_df: pd.DataFrame,
               seed: int) -> dict[str, Any]:
    """Teshis amacli bypass: split/fit/esik adimlari v0.4 ile ayni.

    Performans metrigi HESAPLANMAZ ve DONMEZ; yalniz esik, ham skorlar ve
    agac-yaprak ciktisi.
    """
    tr_idx, va_idx, split_basis = _split_by_run(source_df, VAL_FRAC, seed)
    guard_reasons = validate_split(source_df, tr_idx, va_idx, split_basis,
                                   MAX_FPR, target_df)

    spec = _schema_spec(SCHEMA)
    model = _build_model(default_model_factory, spec["numeric"],
                         spec["categorical"], seed)

    train_part = source_df.iloc[tr_idx]
    train_frame = build_feature_frame(train_part, SCHEMA)
    y_train = (train_part["label"].astype(str) == "attack").astype(int).to_numpy()
    model.fit(train_frame, y_train)

    classifier = model.named_steps["clf"]
    if getattr(classifier, "n_jobs", None) not in (None, 1):
        raise D3GuardError(
            f"clf.n_jobs = {classifier.n_jobs}; paralel birikim kayan nokta "
            "toplama sirasini degistirir ve T=200 bit esitligini bozar "
            "(belge §4.1 notu). D3 yalniz n_jobs=1 ile kosar."
        )
    if classifier.n_estimators != TREE_PANEL[-1]:
        raise D3GuardError(
            f"model recetesi {classifier.n_estimators} agac; dondurulmus v0.4 "
            f"recetesi {TREE_PANEL[-1]} agac (belge §7.4)."
        )

    validation_part = source_df.iloc[va_idx]
    validation_frame = build_feature_frame(validation_part, SCHEMA)
    y_validation = (validation_part["label"].astype(str) == "attack") \
        .astype(int).to_numpy()
    target_frame = build_feature_frame(target_df, SCHEMA)

    # Tekrarlanabilirlik guard'i (belge §7.5): ayni model iki kez skorlanir.
    reference = np.asarray(_proba(model, target_frame), dtype=float)
    repeat = np.asarray(_proba(model, target_frame), dtype=float)
    if not np.array_equal(reference.view(np.int64), repeat.view(np.int64)):
        raise D3GuardError("ayni modelin iki hedef skorlamasi bit duzeyinde "
                           "eslesmiyor; toleransla birlestirme YAPILMAZ")

    preprocessor = model.named_steps["pre"]
    target_matrix = preprocessor.transform(target_frame)
    validation_matrix = preprocessor.transform(validation_frame)

    target_panel = _tree_panel_scores(classifier.estimators_, target_matrix,
                                      TREE_PANEL)
    validation_panel = _tree_panel_scores(classifier.estimators_,
                                          validation_matrix, TREE_PANEL)

    full = target_panel[TREE_PANEL[-1]]["soft"]
    if not np.array_equal(full.view(np.int64), reference.view(np.int64)):
        raise D3GuardError(
            "T=200 ic ice skoru ana model skoruyla bit duzeyinde eslesmiyor "
            "(belge §7.6)"
        )

    return {
        "seed": int(seed),
        "split_basis": split_basis,
        "guard_class": guard_class(source_df, tr_idx, va_idx),
        "guard_reasons": guard_reasons,
        "train_runs": sorted(set(train_part["run_id"].astype(str))),
        "validation_runs": sorted(set(validation_part["run_id"].astype(str))),
        "target_panel": target_panel,
        "validation_panel": validation_panel,
        "y_validation": y_validation,
        "target_index": list(target_frame.index),
    }


# ---------------------------------------------------------------------------
# Seed basina paneller
# ---------------------------------------------------------------------------
def analyse_seed(scored: dict[str, Any]) -> dict[str, Any]:
    low, high = DECISION_INTERVAL
    entry: dict[str, Any] = {
        "seed": scored["seed"],
        "guard_class": scored["guard_class"],
        "guard_reasons": scored["guard_reasons"],
        "split_basis": scored["split_basis"],
        "train_runs": scored["train_runs"],
        "validation_runs": scored["validation_runs"],
        "panels": {},
    }

    for trees in TREE_PANEL:
        target = scored["target_panel"][trees]
        validation = scored["validation_panel"][trees]
        # tau YALNIZ OpenPLC validation'dan (belge §7.9).
        threshold = _threshold_at_fpr(validation["soft"],
                                      scored["y_validation"], MAX_FPR)

        panel: dict[str, Any] = {"T": trees,
                                 "threshold": float(threshold),
                                 "threshold_hex": float(threshold).hex(),
                                 "threshold_in_interval":
                                     bool(low <= threshold <= high),
                                 "pure_visit_share": target["pure_visit_share"],
                                 "mean_leaf_purity": target["mean_leaf_purity"]}

        for kind in ("soft", "hard"):
            profile = support_profile(target[kind])
            if kind == "hard" and profile["K"] > trees + 1:
                raise D3GuardError(
                    f"sert-oy panelinde K_hard={profile['K']} > T+1={trees + 1} "
                    "(belge §7.7); uygulama hatasi kabul edilir"
                )
            plate = plateau(profile["unique_scores"], float(threshold))
            enrich = enrichments(profile, plate)
            panel[kind] = {
                **{key: value for key, value in profile.items()
                   if key not in ("unique_scores", "unique_counts")},
                "plateau": plate,
                **enrich,
            }
        entry["panels"][str(trees)] = panel

    return entry


# ---------------------------------------------------------------------------
# Hipotez kararlari (belge §5)
# ---------------------------------------------------------------------------
def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Sira korelasyonu; baglar ortalama sirayla. n=4 icin kaba ama on-kayitli."""
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = average
            i = j + 1
        return out

    rx, ry = ranks(list(xs)), ranks(list(ys))
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx)
                    * sum((b - my) ** 2 for b in ry))
    return (num / den) if den else None


def _share(values: Sequence[bool]) -> float:
    return (sum(1 for v in values if v) / len(values)) if values else 0.0


def _decide(median_value: float | None, share_above: float,
            support_median: float, falsify_median: float,
            n_evaluable: int) -> str:
    """destek / karisik / yanlislandi / underpowered -- esikler gevsetilmez."""
    if n_evaluable < MIN_EVALUABLE_SEEDS:
        return "underpowered"
    if median_value is None:
        return "underpowered"
    if median_value >= support_median and share_above >= SUPPORT_SHARE:
        return "destek"
    if median_value <= falsify_median and share_above <= FALSIFY_SHARE:
        return "yanlislandi"
    return "karisik"


def decide_hypotheses(per_seed: Sequence[dict[str, Any]]) -> dict[str, Any]:
    full = str(TREE_PANEL[-1])

    # --- H1: bosluk zenginlesmesi
    h1_rows = []
    for entry in per_seed:
        soft = entry["panels"][full]["soft"]
        plate = soft["plateau"]
        if (entry["panels"][full]["threshold_in_interval"]
                and soft["K_I"] >= MIN_K_I
                and plate["L_finite"] and plate["R_finite"]):
            h1_rows.append({"seed": entry["seed"],
                            "gap_enrichment": soft["gap_enrichment"],
                            "width": plate["width"]})
    h1_values = [r["gap_enrichment"] for r in h1_rows
                 if r["gap_enrichment"] is not None]
    h1 = {
        "evaluable_seeds": [r["seed"] for r in h1_rows],
        "n_evaluable": len(h1_rows),
        "median_gap_enrichment": (statistics.median(h1_values)
                                  if h1_values else None),
        "share_above_one": _share([v > 1 for v in h1_values]),
    }
    h1["status"] = _decide(h1["median_gap_enrichment"], h1["share_above_one"],
                           ENRICHMENT_SUPPORT_MEDIAN, 1.0, h1["n_evaluable"])

    # --- H2: sinir nokta kutlesi zenginlesmesi
    h2_rows = []
    for entry in per_seed:
        soft = entry["panels"][full]["soft"]
        plate = soft["plateau"]
        if (entry["panels"][full]["threshold_in_interval"]
                and soft["K_I"] >= MIN_K_I
                and plate["L_finite"] and plate["R_finite"]
                and soft["N_I"] >= MIN_N_I and plate["R_in_interval"]):
            h2_rows.append({"seed": entry["seed"],
                            "mass_enrichment": soft["mass_enrichment"],
                            "boundary_n": soft["boundary_n"]})
    h2_values = [r["mass_enrichment"] for r in h2_rows
                 if r["mass_enrichment"] is not None]
    h2 = {
        "evaluable_seeds": [r["seed"] for r in h2_rows],
        "n_evaluable": len(h2_rows),
        "median_mass_enrichment": (statistics.median(h2_values)
                                   if h2_values else None),
        # Destek payi HEM zenginlesme HEM asgari satir destegi ister.
        "share_above_one": _share([
            (r["mass_enrichment"] or 0) > 1 and (r["boundary_n"] or 0) >= MIN_POINT_ROWS
            for r in h2_rows]),
        "share_enrichment_above_one": _share([v > 1 for v in h2_values]),
    }
    h2["status"] = _decide(h2["median_mass_enrichment"], h2["share_above_one"],
                           ENRICHMENT_SUPPORT_MEDIAN, 1.0, h2["n_evaluable"])

    # --- H3: agac sayisi destegi genisletir
    h3_rows = []
    for entry in per_seed:
        trees = list(TREE_PANEL)
        k_eff = [entry["panels"][str(t)]["soft"]["K_eff"] for t in trees]
        widths = [entry["panels"][str(t)]["soft"]["plateau"]["width"]
                  for t in trees]
        rho_support = _spearman(trees, k_eff)
        rho_width = _spearman(trees, widths)
        h3_rows.append({"seed": entry["seed"], "rho_support": rho_support,
                        "rho_width": rho_width})
    support_rhos = [r["rho_support"] for r in h3_rows
                    if r["rho_support"] is not None]
    width_rhos = [r["rho_width"] for r in h3_rows if r["rho_width"] is not None]
    both_correct = _share([
        (r["rho_support"] is not None and r["rho_support"] >= RHO_SUPPORT_MIN)
        and (r["rho_width"] is not None and r["rho_width"] <= RHO_WIDTH_MAX)
        for r in h3_rows])
    median_support = statistics.median(support_rhos) if support_rhos else None
    median_width = statistics.median(width_rhos) if width_rhos else None
    if len(h3_rows) < MIN_EVALUABLE_SEEDS:
        h3_status = "underpowered"
    elif (median_support is not None and median_width is not None
            and median_support >= RHO_SUPPORT_MIN
            and median_width <= RHO_WIDTH_MAX
            and both_correct >= SUPPORT_SHARE):
        h3_status = "destek"
    elif (median_support is not None and median_width is not None
            and median_support <= 0 and median_width >= 0):
        h3_status = "yanlislandi"
    else:
        h3_status = "karisik"
    h3 = {"n_evaluable": len(h3_rows), "median_rho_support": median_support,
          "median_rho_width": median_width, "share_both_correct": both_correct,
          "per_seed": h3_rows, "status": h3_status}

    # --- H4: saf olmayan yapraklar sert-oy kafesini genisletir
    ratios = []
    for entry in per_seed:
        panel = entry["panels"][full]
        hard_eff = panel["hard"]["K_eff"]
        ratios.append(panel["soft"]["K_eff"] / hard_eff if hard_eff else None)
    clean = [r for r in ratios if r is not None]
    median_ratio = statistics.median(clean) if clean else None
    share_gt_one = _share([r > 1 for r in clean])
    share_gt_falsify = _share([r > KEFF_RATIO_FALSIFY for r in clean])
    if len(clean) < MIN_EVALUABLE_SEEDS:
        h4_status = "underpowered"
    elif (median_ratio is not None and median_ratio >= KEFF_RATIO_SUPPORT
            and share_gt_one >= SUPPORT_SHARE):
        h4_status = "destek"
    elif (median_ratio is not None and median_ratio <= KEFF_RATIO_FALSIFY
            and share_gt_falsify <= FALSIFY_SHARE):
        h4_status = "yanlislandi"
    else:
        h4_status = "karisik"
    h4 = {"n_evaluable": len(clean), "median_keff_ratio": median_ratio,
          "share_above_one": share_gt_one,
          "share_above_falsify": share_gt_falsify, "status": h4_status}

    return {"H1": h1, "H2": h2, "H3": h3, "H4": h4,
            "combined": combined_verdict(h1["status"], h2["status"])}


def combined_verdict(h1_status: str, h2_status: str) -> dict[str, str]:
    """Belge §5.1 karar tablosu. `karisik` destek SAYILMAZ."""
    if h1_status == "underpowered" or h2_status == "underpowered":
        return {"verdict": "test_edilemedi",
                "explanation": "ana karar test edilemedi (underpowered)"}
    if h1_status == "destek" and h2_status == "destek":
        return {"verdict": "destek",
                "explanation": "skor kesikligi hem plato genisligini hem "
                               "sicrama yuksekligini acikliyor"}
    if h1_status == "destek":
        return {"verdict": "kismi_genislik",
                "explanation": "genis duz bolgeler var; buyuk metrik "
                               "sicramalari ACIKLANMADI"}
    if h2_status == "destek":
        return {"verdict": "kismi_kutle",
                "explanation": "agir skor bloklari var; genis esik platolari "
                               "ACIKLANMADI"}
    if h1_status == "yanlislandi" and h2_status == "yanlislandi":
        return {"verdict": "reddedildi",
                "explanation": "skor-kesikligi aciklamasi bu olcutlerle "
                               "reddedildi"}
    return {"verdict": "karisik",
            "explanation": "H1/H2 karisik; destek SAYILMAZ"}


# ---------------------------------------------------------------------------
# Durdurma kosullari (belge §7)
# ---------------------------------------------------------------------------
def check_universe(target_df: pd.DataFrame, seeds: Sequence[int],
                   schema: str) -> None:
    if len(target_df) != CANONICAL_TARGET_ROWS:
        raise D3GuardError(
            f"hedef {len(target_df)} satir; kanonik evren "
            f"{CANONICAL_TARGET_ROWS} satir (belge §7.1)")
    runs = target_df["run_id"].astype(str).nunique() \
        if "run_id" in target_df.columns else 0
    if runs != CANONICAL_TARGET_RUNS:
        raise D3GuardError(f"hedef {runs} kosu; kanonik evren "
                           f"{CANONICAL_TARGET_RUNS} kosu (belge §7.1)")
    if list(seeds) != list(SEED_UNIVERSE):
        raise D3GuardError("seed listesi tam 42-61 degil (belge §7.2)")
    if schema != SCHEMA:
        raise D3GuardError(f"sema {schema}; D3 yalniz {SCHEMA} (belge §7.3)")


def run_d3(source_df: pd.DataFrame, target_df: pd.DataFrame,
           seeds: Sequence[int] = SEED_UNIVERSE) -> dict[str, Any]:
    check_universe(target_df, seeds, SCHEMA)

    per_seed, reference_index = [], None
    for seed in seeds:
        scored = score_seed(source_df, target_df, int(seed))
        if reference_index is None:
            reference_index = scored["target_index"]
        elif scored["target_index"] != reference_index:
            raise D3GuardError("hedef satir sirasi seed'ler arasinda degisti "
                               "(belge §7.8)")
        per_seed.append(analyse_seed(scored))

    return {
        "preregistration": "DENEY-D3-Esik-Platolari-Skor-Kesikligi.md",
        "schema": SCHEMA,
        "decision_interval": list(DECISION_INTERVAL),
        "tree_panel": list(TREE_PANEL),
        "constants": {"MIN_K_I": MIN_K_I,
                      "MIN_EVALUABLE_SEEDS": MIN_EVALUABLE_SEEDS,
                      "MIN_POINT_ROWS": MIN_POINT_ROWS, "MIN_N_I": MIN_N_I,
                      "ENRICHMENT_SUPPORT_MEDIAN": ENRICHMENT_SUPPORT_MEDIAN,
                      "SUPPORT_SHARE": SUPPORT_SHARE,
                      "FALSIFY_SHARE": FALSIFY_SHARE,
                      "RHO_SUPPORT_MIN": RHO_SUPPORT_MIN,
                      "RHO_WIDTH_MAX": RHO_WIDTH_MAX,
                      "KEFF_RATIO_SUPPORT": KEFF_RATIO_SUPPORT,
                      "KEFF_RATIO_FALSIFY": KEFF_RATIO_FALSIFY},
        "seeds": [int(s) for s in seeds],
        "per_seed": per_seed,
        "hypotheses": decide_hypotheses(per_seed),
    }


# ---------------------------------------------------------------------------
# Sunum
# ---------------------------------------------------------------------------
def _spread(values: Sequence[float]) -> str:
    clean = [v for v in values if v is not None]
    if not clean:
        return "null"
    return (f"{statistics.median(clean):.4f} "
            f"[{min(clean):.4f}, {max(clean):.4f}]")


def _print_report(report: dict[str, Any]) -> None:
    full = str(TREE_PANEL[-1])
    per_seed = report["per_seed"]
    low, high = report["decision_interval"]
    print(f"D3 SKOR KESIKLIGI -- sema {report['schema']}, "
          f"{len(per_seed)} seed, I = [{low}, {high}]\n")

    print(f"T=200 gercek skorlar (ortanca [min, max], n={len(per_seed)})")
    for field in ("K", "K_I", "N_I", "K_eff"):
        print(f"  {field:16s} {_spread([e['panels'][full]['soft'][field] for e in per_seed])}")
    print(f"  {'w(tau)':16s} "
          f"{_spread([e['panels'][full]['soft']['plateau']['width'] for e in per_seed])}")
    print(f"  {'base_gap':16s} {_spread([e['panels'][full]['soft']['base_gap'] for e in per_seed])}")
    print(f"  {'gap_enrichment':16s} "
          f"{_spread([e['panels'][full]['soft']['gap_enrichment'] for e in per_seed])}")
    print(f"  {'boundary n(R)':16s} "
          f"{_spread([e['panels'][full]['soft']['boundary_n'] for e in per_seed])}")
    print(f"  {'mass_enrichment':16s} "
          f"{_spread([e['panels'][full]['soft']['mass_enrichment'] for e in per_seed])}")
    print(f"  {'pure_visit_share':16s} "
          f"{_spread([e['panels'][full]['pure_visit_share'] for e in per_seed])}")
    print(f"  {'mean_leaf_purity':16s} "
          f"{_spread([e['panels'][full]['mean_leaf_purity'] for e in per_seed])}")

    print("\nAGAC SAYISI PANELI (gercek / sert-oy)")
    header = (f"  {'T':>4s} {'K_soft':>16s} {'K_eff_soft':>18s} "
              f"{'K_hard':>10s} {'K_eff_hard':>18s} {'w(tau)':>18s}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for trees in TREE_PANEL:
        key = str(trees)
        print(f"  {trees:4d} "
              f"{_spread([e['panels'][key]['soft']['K'] for e in per_seed]):>16s} "
              f"{_spread([e['panels'][key]['soft']['K_eff'] for e in per_seed]):>18s} "
              f"{_spread([e['panels'][key]['hard']['K'] for e in per_seed]):>10s} "
              f"{_spread([e['panels'][key]['hard']['K_eff'] for e in per_seed]):>18s} "
              f"{_spread([e['panels'][key]['soft']['plateau']['width'] for e in per_seed]):>18s}")

    hypotheses = report["hypotheses"]
    print("\nHIPOTEZLER")
    h1, h2 = hypotheses["H1"], hypotheses["H2"]
    print(f"  H1 (bosluk)  degerlendirilebilir {h1['n_evaluable']}/"
          f"{MIN_EVALUABLE_SEEDS}  ortanca zenginlesme "
          f"{h1['median_gap_enrichment']}  >1 payi {h1['share_above_one']:.2f}"
          f"  -> {h1['status'].upper()}")
    print(f"  H2 (kutle)   degerlendirilebilir {h2['n_evaluable']}/"
          f"{MIN_EVALUABLE_SEEDS}  ortanca zenginlesme "
          f"{h2['median_mass_enrichment']}  destek payi "
          f"{h2['share_above_one']:.2f}  -> {h2['status'].upper()}")
    h3, h4 = hypotheses["H3"], hypotheses["H4"]
    print(f"  H3 (agac)    ortanca rho_support {h3['median_rho_support']}  "
          f"rho_width {h3['median_rho_width']}  iki yon dogru payi "
          f"{h3['share_both_correct']:.2f}  -> {h3['status'].upper()}")
    print(f"  H4 (yaprak)  ortanca K_eff soft/hard {h4['median_keff_ratio']}  "
          f">1 payi {h4['share_above_one']:.2f}  -> {h4['status'].upper()}")

    combined = hypotheses["combined"]
    print(f"\nBIRLESIK KARAR: {combined['verdict'].upper()}")
    print(f"  {combined['explanation']}")
    print("\nD3 performans metrigi URETMEDI. Bir nokta kutlesinin cok satir "
          "tasimasi\nbagimsiz olay sayisi gibi okunamaz (belge §8).")


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="D3 skor kesikligi teshisi (on-kayit: "
                    "DENEY-D3-Esik-Platolari-Skor-Kesikligi.md)")
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--out", type=Path,
                        default=Path("ml/ml_out/d3_score_discreteness.json"))
    args = parser.parse_args(argv)

    source_df, source_paths = load_frame(args.source)
    target_df, target_paths = load_frame(args.target)
    print(f"kaynak : {args.source}  ({len(source_df)} satir)")
    print(f"hedef  : {args.target}  ({len(target_df)} satir, "
          f"{len(target_paths)} dosya)\n")

    try:
        report = run_d3(source_df, target_df, SEED_UNIVERSE)
    except D3GuardError as exc:
        print(f"[DURDU] {exc}")
        print("Tolerans, yuvarlama, seed cikarma veya alternatif model "
              "UYGULANMAZ (belge §7).")
        return 2

    report["inputs"] = {"source": args.source, "source_files": source_paths,
                        "target": args.target, "target_files": target_paths}
    _print_report(report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_jsonable(report), indent=2,
                                   ensure_ascii=False), encoding="utf-8")
    print(f"\nsonuc -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
