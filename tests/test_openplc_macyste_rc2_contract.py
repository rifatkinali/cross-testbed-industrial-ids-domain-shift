from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ml.openplc_macyste_rc2 import (
    load_macyste,
    prepare_openplc,
    primary_metrics,
    split_main_by_run,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = (
    ROOT / "research_release" / "openplc-macyste-transfer-v2" / "preregistration.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_rc2_preregistration_matches_frozen_inputs() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    source_spec = spec["inputs"]["openplc_candidate"]
    source_path = ROOT / source_spec["path"]

    assert sha256(source_path) == source_spec["sha256"]
    assert sha256(ROOT / source_spec["provenance_record"]) == source_spec[
        "provenance_sha256"
    ]

    accepted_types = set(spec["main_model"]["event_types"])
    fault_runs: set[int] = set()
    accepted_rows: list[dict[str, str]] = []
    raw_rows = 0
    with source_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            raw_rows += 1
            if row["event_type"] not in accepted_types:
                continue
            accepted_rows.append(row)
            if row["label"] == "fault":
                fault_runs.add(int(row["run_id"]))

    assert raw_rows == source_spec["rows"]
    assert sorted(fault_runs) == spec["openplc_fault_stress"]["run_ids"]
    assert sorted(fault_runs) == spec["main_model"]["excluded_fault_run_ids"]

    main = [
        row
        for row in accepted_rows
        if int(row["run_id"]) not in fault_runs
        and row["label"] in spec["main_model"]["labels"]
    ]
    main_counts = Counter(row["label"] for row in main)
    assert len(main) == spec["main_model"]["expected_rows"]
    assert main_counts == {
        "normal": spec["main_model"]["expected_normal_rows"],
        "attack": spec["main_model"]["expected_attack_rows"],
    }
    assert sorted({int(row["run_id"]) for row in main}) == spec["main_model"][
        "expected_run_ids"
    ]
    assert {row["label"] for row in main} == {"normal", "attack"}

    stress = [
        row
        for row in accepted_rows
        if int(row["run_id"]) in fault_runs and row["label"] == "fault"
    ]
    stress_spec = spec["openplc_fault_stress"]
    cells = {(int(row["run_id"]), row["scenario"]) for row in stress}
    assert len(stress) == stress_spec["expected_rows"]
    assert len({run for run, _ in cells}) == stress_spec["expected_runs"]
    assert sorted({scenario for _, scenario in cells}) == stress_spec["scenarios"]
    assert len(cells) == stress_spec["expected_run_scenario_cells"]


def test_rc2_preregistration_matches_macyste_manifest() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    target_spec = spec["inputs"]["macyste_target"]
    campaign = ROOT / target_spec["path"]
    manifest_path = campaign / target_spec["derived_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert sha256(manifest_path) == target_spec["manifest_sha256"]
    assert len(manifest["runs"]) == target_spec["runs"]
    assert sum(run["rows"] for run in manifest["runs"]) == target_spec["rows"]
    assert Counter(run["run"].split("-r", 1)[0] for run in manifest["runs"]) == {
        "normal": 3,
        "fault": 3,
        "manipulation": 3,
        "recon": 3,
    }
    event_types: Counter[str] = Counter()
    for run in manifest["runs"]:
        output = campaign / run["run"] / run["output"]
        assert sha256(output) == run["output_sha256"]
        with output.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == run["rows"]
        event_types.update(row["event_type"] for row in rows)
    assert set(event_types) == set(spec["macyste_evaluation"]["event_types"])
    assert sum(event_types.values()) == target_spec["rows"]


def test_rc2_contract_freezes_all_seeds_and_disallows_threshold_headlines() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))

    assert spec["main_model"]["seeds"] == list(range(42, 62))
    assert spec["main_model"]["split_unit"] == "run_id"
    assert spec["macyste_evaluation"]["primary_score"] == (
        "arithmetic_mean_of_20_seed_scores_per_event"
    )
    assert spec["macyste_evaluation"]["event_types"] == ["flow", "modbus"]
    assert spec["macyste_evaluation"]["bootstrap_replicates"] == 2000
    assert spec["macyste_evaluation"]["bootstrap_seed"] == 20260817
    assert spec["macyste_evaluation"]["interval_quantiles"] == [0.025, 0.975]
    assert spec["macyste_evaluation"]["threshold_metrics_allowed_in_headline"] is False
    assert spec["publication"]["rc1_remains_private_draft"] is True


def test_rc2_runtime_contract_accepts_every_preregistered_split() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    main, stress, source_audit = prepare_openplc(spec)
    target, target_audit = load_macyste(spec)

    assert source_audit["main_rows"] == 18_941
    assert source_audit["fault_stress_runs"] == [19, 20, 23, 24, 25, 26, 27, 28]
    assert len(stress) == 3_337
    assert target_audit["runs"] == 12
    assert len(target) == 34_949
    for seed in spec["main_model"]["seeds"]:
        train_idx, validation_idx, audit = split_main_by_run(main, spec, seed)
        assert len(train_idx) + len(validation_idx) == len(main)
        assert set(audit["train_runs"]).isdisjoint(audit["validation_runs"])
        assert len(audit["train_runs"]) == 12
        assert len(audit["validation_runs"]) == 6
        assert audit["validation_normal_rows"] >= 100


def test_rc2_primary_metrics_use_one_score_vector() -> None:
    families = np.array(
        ["normal", "normal", "fault", "fault", "manipulation", "recon"],
        dtype=object,
    )
    y = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])

    metrics = primary_metrics(y, scores, families)

    assert metrics == {
        "event_weighted_auc": 1.0,
        "scenario_balanced_auc": 1.0,
        "manipulation_vs_normal_auc": 1.0,
        "recon_vs_normal_auc": 1.0,
    }
