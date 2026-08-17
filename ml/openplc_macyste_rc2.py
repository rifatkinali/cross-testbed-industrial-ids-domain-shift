#!/usr/bin/env python3
"""Run the preregistered OpenPLC--MaCySTe RC2 transfer experiment.

The module intentionally does not reuse the historical RC1 report entry point.
It enforces the RC2 input hashes, removes complete fault-run contexts from the
main model, selects thresholds only on source validation normals, and evaluates
the eight OpenPLC fault runs as a separate unseen benign stress test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.lodo_generalization import (  # noqa: E402
    _proba,
    _threshold_at_fpr,
    build_feature_frame,
    default_model_factory,
    scenario_family,
)


DEFAULT_SPEC = (
    ROOT / "research_release" / "openplc-macyste-transfer-v2" / "preregistration.json"
)
RESULT_SCHEMA = "maritime.openplc-macyste-rc2-result/v1"


class ContractError(ValueError):
    """Raised when an input or computation differs from the frozen contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def load_spec(path: Path = DEFAULT_SPEC) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    _require(
        spec.get("schema") == "maritime.openplc-macyste-rc2-preregistration/v1",
        "unsupported preregistration schema",
    )
    return spec


def prepare_openplc(
    spec: dict[str, Any], *, root: Path = ROOT
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Verify and split the frozen OpenPLC candidate into main and fault pools."""
    source_spec = spec["inputs"]["openplc_candidate"]
    source_path = root / source_spec["path"]
    provenance_path = root / source_spec["provenance_record"]
    _require(source_path.is_file(), f"OpenPLC input missing: {source_path}")
    _require(provenance_path.is_file(), f"provenance record missing: {provenance_path}")
    _require(sha256_file(source_path) == source_spec["sha256"], "OpenPLC hash mismatch")
    _require(
        sha256_file(provenance_path) == source_spec["provenance_sha256"],
        "OpenPLC provenance hash mismatch",
    )

    raw = pd.read_csv(source_path, low_memory=False)
    _require(len(raw) == source_spec["rows"], "OpenPLC raw row count mismatch")
    required = {"run_id", "label", "scenario", "event_type"}
    _require(required.issubset(raw.columns), "OpenPLC required columns missing")
    raw["run_id"] = pd.to_numeric(raw["run_id"], errors="raise").astype(int)

    accepted_types = set(spec["main_model"]["event_types"])
    accepted = raw[raw["event_type"].astype(str).isin(accepted_types)].copy()
    fault_runs = sorted(
        accepted.loc[accepted["label"].astype(str) == "fault", "run_id"].unique().tolist()
    )
    _require(
        fault_runs == spec["main_model"]["excluded_fault_run_ids"],
        "fault-run set differs from preregistration",
    )

    main_spec = spec["main_model"]
    main = accepted[
        ~accepted["run_id"].isin(fault_runs)
        & accepted["label"].astype(str).isin(main_spec["labels"])
    ].copy()
    counts = main["label"].astype(str).value_counts().to_dict()
    _require(len(main) == main_spec["expected_rows"], "main-model row count mismatch")
    _require(
        counts == {
            "normal": main_spec["expected_normal_rows"],
            "attack": main_spec["expected_attack_rows"],
        },
        "main-model label counts mismatch",
    )
    _require(
        sorted(main["run_id"].unique().tolist()) == main_spec["expected_run_ids"],
        "main-model run set mismatch",
    )
    _require(not main["run_id"].isin(fault_runs).any(), "fault-run context leaked into main")
    _require(set(main["label"].astype(str)) == {"normal", "attack"}, "main label leak")
    _require(set(main["event_type"].astype(str)) <= accepted_types, "main event-type leak")

    stress_spec = spec["openplc_fault_stress"]
    stress = accepted[
        accepted["run_id"].isin(stress_spec["run_ids"])
        & (accepted["label"].astype(str) == stress_spec["label"])
    ].copy()
    cells = set(zip(stress["run_id"].astype(int), stress["scenario"].astype(str)))
    _require(len(stress) == stress_spec["expected_rows"], "fault-stress row count mismatch")
    _require(
        sorted(stress["run_id"].unique().tolist()) == stress_spec["run_ids"],
        "fault-stress run set mismatch",
    )
    _require(
        sorted(stress["scenario"].astype(str).unique().tolist())
        == stress_spec["scenarios"],
        "fault-stress scenario set mismatch",
    )
    _require(
        len(cells) == stress_spec["expected_run_scenario_cells"],
        "fault-stress run-scenario cell count mismatch",
    )

    audit = {
        "raw_rows": len(raw),
        "accepted_flow_modbus_rows": len(accepted),
        "main_rows": len(main),
        "main_label_counts": counts,
        "main_runs": sorted(main["run_id"].unique().tolist()),
        "excluded_fault_runs": fault_runs,
        "fault_stress_rows": len(stress),
        "fault_stress_runs": sorted(stress["run_id"].unique().tolist()),
        "fault_stress_run_scenario_cells": len(cells),
    }
    return main.reset_index(drop=True), stress.reset_index(drop=True), audit


def load_macyste(
    spec: dict[str, Any], *, root: Path = ROOT
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Verify every derived MaCySTe run and return the deterministic combination."""
    target_spec = spec["inputs"]["macyste_target"]
    campaign = root / target_spec["path"]
    manifest_path = campaign / target_spec["derived_manifest"]
    _require(manifest_path.is_file(), f"MaCySTe manifest missing: {manifest_path}")
    _require(
        sha256_file(manifest_path) == target_spec["manifest_sha256"],
        "MaCySTe manifest hash mismatch",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = sorted(manifest["runs"], key=lambda item: item["run"])
    _require(len(records) == target_spec["runs"], "MaCySTe run count mismatch")

    frames: list[pd.DataFrame] = []
    file_audit: list[dict[str, Any]] = []
    for record in records:
        path = campaign / record["run"] / record["output"]
        _require(path.is_file(), f"MaCySTe derived events missing: {record['run']}")
        actual_hash = sha256_file(path)
        _require(actual_hash == record["output_sha256"], f"MaCySTe hash mismatch: {record['run']}")
        frame = pd.read_csv(path, low_memory=False)
        _require(len(frame) == record["rows"], f"MaCySTe row mismatch: {record['run']}")
        _require(
            set(frame["run_id"].astype(str)) == {f"macyste-20260726T113328Z-{record['run']}"},
            f"MaCySTe run_id mismatch: {record['run']}",
        )
        frames.append(frame)
        file_audit.append({"run": record["run"], "rows": len(frame), "sha256": actual_hash})

    target = pd.concat(frames, ignore_index=True)
    target_eval = spec["macyste_evaluation"]
    _require(len(target) == target_spec["rows"], "MaCySTe combined row count mismatch")
    _require(
        set(target["event_type"].astype(str)) <= set(target_eval["event_types"]),
        "MaCySTe contains non-preregistered event types",
    )
    families = scenario_family(target["scenario"].astype(str))
    family_counts = Counter(families.tolist())
    run_family_counts = {
        family: int(target.loc[families == family, "run_id"].nunique())
        for family in sorted(set(families))
    }
    _require(set(family_counts) == {"normal", "fault", "manipulation", "recon"}, "MaCySTe scenario family mismatch")
    _require(run_family_counts == {"fault": 3, "manipulation": 3, "normal": 3, "recon": 3}, "MaCySTe repeat count mismatch")
    _require(set(target["label"].astype(str)) == {"normal", "fault", "attack"}, "MaCySTe label set mismatch")
    return target, {
        "rows": len(target),
        "runs": int(target["run_id"].nunique()),
        "family_event_counts": dict(sorted(family_counts.items())),
        "family_run_counts": run_family_counts,
        "derived_files": file_audit,
    }


def split_main_by_run(
    main: pd.DataFrame, spec: dict[str, Any], seed: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Create and validate the preregistered 70/30 run-level split."""
    main_spec = spec["main_model"]
    runs = np.asarray(sorted(main["run_id"].astype(int).unique()), dtype=int)
    rng = np.random.RandomState(seed)
    rng.shuffle(runs)
    cut = max(
        1,
        min(
            len(runs) - 1,
            int(len(runs) * (1.0 - float(main_spec["validation_fraction"]))),
        ),
    )
    train_runs = set(int(run) for run in runs[:cut])
    validation_runs = set(int(run) for run in runs[cut:])
    _require(not train_runs & validation_runs, f"seed {seed}: run overlap")
    tr_idx = np.flatnonzero(main["run_id"].isin(train_runs).to_numpy())
    va_idx = np.flatnonzero(main["run_id"].isin(validation_runs).to_numpy())
    minimum = int(main_spec["minimum_runs_per_label_per_side"])
    for side, idx in (("train", tr_idx), ("validation", va_idx)):
        side_df = main.iloc[idx]
        for label in main_spec["labels"]:
            count = int(side_df.loc[side_df["label"].astype(str) == label, "run_id"].nunique())
            _require(count >= minimum, f"seed {seed}: {side} has only {count} {label} runs")
    validation_normal = int((main.iloc[va_idx]["label"].astype(str) == "normal").sum())
    _require(
        validation_normal >= main_spec["minimum_validation_normal_events"],
        f"seed {seed}: insufficient validation normals",
    )
    return tr_idx, va_idx, {
        "train_runs": sorted(train_runs),
        "validation_runs": sorted(validation_runs),
        "train_rows": len(tr_idx),
        "validation_rows": len(va_idx),
        "validation_normal_rows": validation_normal,
    }


def auc(y: Iterable[int], scores: Iterable[float], weights: Iterable[float] | None = None) -> float:
    from sklearn.metrics import roc_auc_score

    y_array = np.asarray(y, dtype=int)
    score_array = np.asarray(scores, dtype=float)
    _require(len(np.unique(y_array)) == 2, "AUC requires both classes")
    sample_weight = None if weights is None else np.asarray(weights, dtype=float)
    return float(roc_auc_score(y_array, score_array, sample_weight=sample_weight))


def balanced_weights(families: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(families), dtype=float)
    for family in np.unique(families):
        mask = families == family
        weights[mask] = 1.0 / float(mask.sum())
    return weights


PRIMARY_METRICS = (
    "event_weighted_auc",
    "scenario_balanced_auc",
    "manipulation_vs_normal_auc",
    "recon_vs_normal_auc",
)


def primary_metrics(y: np.ndarray, scores: np.ndarray, families: np.ndarray) -> dict[str, float]:
    output = {
        "event_weighted_auc": auc(y, scores),
        "scenario_balanced_auc": auc(y, scores, balanced_weights(families)),
    }
    normal = families == "normal"
    for family in ("manipulation", "recon"):
        attack = (families == family) & (y == 1)
        pair = normal | attack
        output[f"{family}_vs_normal_auc"] = auc(y[pair], scores[pair])
    return output


def scripted_run_sensitivity_intervals(
    runs: np.ndarray,
    families: np.ndarray,
    y: np.ndarray,
    scores: np.ndarray,
    *,
    replicates: int,
    seed: int,
    quantiles: Sequence[float],
) -> dict[str, list[float]]:
    """Scenario-stratified run bootstrap on the same ensemble score and metrics."""
    strata = {
        family: np.unique(runs[families == family])
        for family in sorted(np.unique(families))
    }
    _require(all(len(run_names) >= 2 for run_names in strata.values()), "bootstrap stratum too small")
    index_by_run = {run: np.flatnonzero(runs == run) for run in np.unique(runs)}
    rng = np.random.RandomState(seed)
    values = {metric: [] for metric in PRIMARY_METRICS}
    for _ in range(replicates):
        picked: list[str] = []
        for run_names in strata.values():
            picked.extend(rng.choice(run_names, size=len(run_names), replace=True).tolist())
        idx = np.concatenate([index_by_run[run] for run in picked])
        sample = primary_metrics(y[idx], scores[idx], families[idx])
        for metric, value in sample.items():
            values[metric].append(value)
    lo_q, hi_q = (float(quantiles[0]), float(quantiles[1]))
    return {
        metric: [float(np.quantile(items, lo_q)), float(np.quantile(items, hi_q))]
        for metric, items in values.items()
    }


def median_min_max(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    _require(len(array) > 0 and np.isfinite(array).all(), "invalid summary values")
    return {
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "n": int(len(array)),
    }


def _threshold_metrics(
    y: np.ndarray, scores: np.ndarray, families: np.ndarray, threshold: float
) -> dict[str, float]:
    pred = scores >= threshold
    masks = {
        "normal_fpr": (families == "normal") & (y == 0),
        "fault_fpr": (families == "fault") & (y == 0),
        "manipulation_recall": (families == "manipulation") & (y == 1),
        "recon_recall": (families == "recon") & (y == 1),
    }
    _require(all(mask.any() for mask in masks.values()), "target threshold metric mask empty")
    return {name: float(pred[mask].mean()) for name, mask in masks.items()}


def _fault_stress_metrics(stress: pd.DataFrame, scores: np.ndarray, threshold: float) -> dict[str, float]:
    pred = scores >= threshold
    frame = stress[["run_id", "scenario"]].copy()
    frame["pred"] = pred.astype(float)
    cells = frame.groupby(["run_id", "scenario"], sort=True)["pred"].mean()
    return {
        "run_scenario_balanced_false_positive_rate": float(cells.mean()),
        "event_weighted_false_positive_rate": float(pred.mean()),
    }


def run_schema(
    main: pd.DataFrame,
    stress: pd.DataFrame,
    target: pd.DataFrame,
    spec: dict[str, Any],
    schema: str,
) -> dict[str, Any]:
    main_spec = spec["main_model"]
    seeds = [int(seed) for seed in main_spec["seeds"]]
    _require(schema in main_spec["schemas"], f"schema not preregistered: {schema}")
    target_families = scenario_family(target["scenario"].astype(str))
    target_y = (target["label"].astype(str).to_numpy() == "attack").astype(int)
    target_runs = target["run_id"].astype(str).to_numpy()
    target_features = build_feature_frame(target, schema)
    stress_features = build_feature_frame(stress, schema)

    target_scores: list[np.ndarray] = []
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        tr_idx, va_idx, split_audit = split_main_by_run(main, spec, seed)
        train = main.iloc[tr_idx]
        validation = main.iloc[va_idx]
        train_features = build_feature_frame(train, schema)
        model = default_model_factory(
            train_features.attrs["numeric"], train_features.attrs["categorical"], seed
        )
        train_y = (train["label"].astype(str).to_numpy() == "attack").astype(int)
        model.fit(train_features, train_y)

        validation_features = build_feature_frame(validation, schema)
        validation_y = (
            validation["label"].astype(str).to_numpy() == "attack"
        ).astype(int)
        validation_scores = _proba(model, validation_features)
        threshold = _threshold_at_fpr(
            validation_scores, validation_y, float(main_spec["max_source_validation_fpr"])
        )
        validation_normal = validation_y == 0
        validation_fpr = float((validation_scores[validation_normal] >= threshold).mean())
        _require(
            validation_fpr <= float(main_spec["max_source_validation_fpr"]),
            f"seed {seed}: threshold violates source validation FPR",
        )

        scores = _proba(model, target_features)
        fault_scores = _proba(model, stress_features)
        target_scores.append(scores)
        per_seed.append(
            {
                "seed": seed,
                "split": split_audit,
                "threshold": float(threshold),
                "source_validation_auc": auc(validation_y, validation_scores),
                "source_validation_fpr": validation_fpr,
                "macyste_descriptive_threshold_metrics": _threshold_metrics(
                    target_y, scores, target_families, threshold
                ),
                "openplc_fault_stress": _fault_stress_metrics(stress, fault_scores, threshold),
            }
        )

    _require([item["seed"] for item in per_seed] == seeds, "seed loss or reordering")
    ensemble_scores = np.mean(np.vstack(target_scores), axis=0)
    point = primary_metrics(target_y, ensemble_scores, target_families)
    eval_spec = spec["macyste_evaluation"]
    intervals = scripted_run_sensitivity_intervals(
        target_runs,
        target_families,
        target_y,
        ensemble_scores,
        replicates=int(eval_spec["bootstrap_replicates"]),
        seed=int(eval_spec["bootstrap_seed"]),
        quantiles=eval_spec["interval_quantiles"],
    )

    threshold_names = eval_spec["descriptive_threshold_metrics"]
    fault_names = (
        "run_scenario_balanced_false_positive_rate",
        "event_weighted_false_positive_rate",
    )
    return {
        "schema": schema,
        "source_validation_auc": median_min_max(
            item["source_validation_auc"] for item in per_seed
        ),
        "source_validation_fpr": median_min_max(
            item["source_validation_fpr"] for item in per_seed
        ),
        "macyste_primary_ensemble": {
            metric: {
                "point": point[metric],
                "scripted_run_sensitivity_interval": intervals[metric],
            }
            for metric in PRIMARY_METRICS
        },
        "macyste_descriptive_threshold_metrics": {
            name: median_min_max(
                item["macyste_descriptive_threshold_metrics"][name] for item in per_seed
            )
            for name in threshold_names
        },
        "openplc_fault_stress": {
            name: median_min_max(item["openplc_fault_stress"][name] for item in per_seed)
            for name in fault_names
        },
        "per_seed": per_seed,
    }


def run_experiment(spec_path: Path = DEFAULT_SPEC, *, root: Path = ROOT) -> dict[str, Any]:
    spec = load_spec(spec_path)
    implementation_path = Path(__file__).resolve()
    main, stress, source_audit = prepare_openplc(spec, root=root)
    target, target_audit = load_macyste(spec, root=root)
    schemas = [
        run_schema(main, stress, target, spec, schema)
        for schema in spec["main_model"]["schemas"]
    ]
    return {
        "schema": RESULT_SCHEMA,
        "version": spec["version"],
        "preregistration": {
            "path": str(spec_path.relative_to(root)).replace("\\", "/")
            if spec_path.is_relative_to(root)
            else str(spec_path),
            "sha256": sha256_file(spec_path),
            "frozen_on": spec["frozen_on"],
        },
        "implementation": {
            "path": str(implementation_path.relative_to(root)).replace("\\", "/"),
            "sha256": sha256_file(implementation_path),
        },
        "source_audit": source_audit,
        "target_audit": target_audit,
        "seeds": spec["main_model"]["seeds"],
        "interpretation": {
            "primary_score": spec["macyste_evaluation"]["primary_score"],
            "uncertainty": spec["macyste_evaluation"]["uncertainty_claim"],
            "threshold_metrics": "descriptive_only_not_headline",
            "fault_stress": "unseen_benign_descriptive_only",
        },
        "results": schemas,
    }


def _rounded(value: Any, digits: int = 6) -> Any:
    if isinstance(value, dict):
        return {key: _rounded(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded(item, digits) for item in value]
    if isinstance(value, float):
        return round(value, digits)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_experiment(args.spec.resolve())
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(_rounded(result), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except (ContractError, FileNotFoundError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"[FAILED] {exc}", file=sys.stderr)
        return 2
    print(f"[PASSED] RC2 preregistered result: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
