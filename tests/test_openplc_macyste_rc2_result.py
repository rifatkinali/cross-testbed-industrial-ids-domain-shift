from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = (
    ROOT
    / "docs"
    / "sonuclar"
    / "OPENPLC-MACYSTE-RC2-ONKAYITLI-SONUC-2026-08-17.json"
)
SPEC_PATH = (
    ROOT / "research_release" / "openplc-macyste-transfer-v2" / "preregistration.json"
)
IMPLEMENTATION_PATH = ROOT / "ml" / "openplc_macyste_rc2.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rc2_frozen_result_has_exact_provenance_and_complete_seeds() -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert result["schema"] == "maritime.openplc-macyste-rc2-result/v1"
    assert result["version"] == "1.0.0-rc2"
    assert result["preregistration"]["sha256"] == sha256(SPEC_PATH)
    assert result["implementation"]["sha256"] == sha256(IMPLEMENTATION_PATH)
    assert result["seeds"] == list(range(42, 62))
    assert result["source_audit"]["main_runs"] == [
        3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22
    ]
    assert result["source_audit"]["excluded_fault_runs"] == [
        19, 20, 23, 24, 25, 26, 27, 28
    ]
    assert result["source_audit"]["fault_stress_run_scenario_cells"] == 24
    assert result["target_audit"]["runs"] == 12
    assert result["interpretation"]["threshold_metrics"] == (
        "descriptive_only_not_headline"
    )

    for schema in result["results"]:
        assert [entry["seed"] for entry in schema["per_seed"]] == list(range(42, 62))
        assert all(entry["source_validation_fpr"] <= 0.01 for entry in schema["per_seed"])
        for metric in schema["macyste_primary_ensemble"].values():
            lo, hi = metric["scripted_run_sensitivity_interval"]
            assert lo <= metric["point"] <= hi


def test_rc2_frozen_primary_values_do_not_regress() -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    observed = {
        item["schema"]: (
            item["source_validation_auc"]["median"],
            item["macyste_primary_ensemble"]["event_weighted_auc"]["point"],
            item["macyste_primary_ensemble"]["scenario_balanced_auc"]["point"],
            item["macyste_primary_ensemble"]["manipulation_vs_normal_auc"]["point"],
            item["macyste_primary_ensemble"]["recon_vs_normal_auc"]["point"],
        )
        for item in result["results"]
    }
    assert observed == {
        "flow": (0.723309, 0.321893, 0.393842, 0.50171, 0.288435),
        "protocol": (0.904289, 0.318365, 0.38929, 0.49165, 0.281867),
        "physical_strict": (0.93617, 0.32889, 0.384612, 0.465171, 0.299061),
        "physical_proxy": (0.968221, 0.400285, 0.45404, 0.534719, 0.373295),
    }


def test_rc2_fault_stress_is_kept_separate_and_descriptive() -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    cell_balanced = {
        item["schema"]: item["openplc_fault_stress"][
            "run_scenario_balanced_false_positive_rate"
        ]["median"]
        for item in result["results"]
    }
    assert cell_balanced == {
        "flow": 0.0,
        "protocol": 0.0,
        "physical_strict": 0.0,
        "physical_proxy": 0.000568,
    }
    assert result["interpretation"]["fault_stress"] == (
        "unseen_benign_descriptive_only"
    )
