#!/usr/bin/env python3
"""Maritime-Lab veri sürümünü sayım ve SHA-256 değerleriyle dondur."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
MODEL_EVENT_TYPES = {"flow", "modbus"}
DEFAULT_REPORTS = (
    "ml/ml_out/detection_report.json",
    "ml/ml_out/benchmark_results.json",
    "ml/ml_out/loso_results.json",
    "ml/ml_out/fault_report.json",
    "ml/ml_out/marsim_benchmark.json",
    "ml/ml_out/radarpwn_benchmark.json",
    "ml/ml_out/too2025_integration_report.json",
)
DEFAULT_EXTERNAL_PROFILES = (
    ("lemay", "lemay_all.csv"),
    ("marsim", "ml/ml_out/marsim.csv"),
    ("radarpwn", "ml/ml_out/radarpwn.csv"),
    ("too2025-independent", "ml/ml_out/too2025.csv"),
)
DEFAULT_SOURCE_FILES = (
    "attack/build_dataset.py",
    "attack/run_scenarios.py",
    "ml/features.py",
    "ml/modeling.py",
    "ml/evaluate_detection.py",
    "ml/benchmark_marsim.py",
    "ml/benchmark_radarpwn.py",
    "ml/evaluate_too2025.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def csv_shape(path: Path) -> tuple[int, int]:
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as stream:
        reader = csv.reader(stream)
        header = next(reader, [])
        return sum(1 for _ in reader), len(header)


def dataset_statistics(path: Path) -> dict[str, Any]:
    labels: Counter[str] = Counter()
    scenarios: Counter[str] = Counter()
    event_types: Counter[str] = Counter()
    sensors: Counter[str] = Counter()
    model_labels: Counter[str] = Counter()
    model_scenarios: Counter[str] = Counter()
    runs: set[str] = set()
    first_timestamp = ""
    last_timestamp = ""
    columns: list[str] = []

    with path.open(newline="", encoding="utf-8-sig", errors="replace") as stream:
        reader = csv.DictReader(stream)
        columns = list(reader.fieldnames or [])
        required = {
            "timestamp",
            "run_id",
            "label",
            "scenario",
            "sensor",
            "event_type",
        }
        missing = required - set(columns)
        if missing:
            raise ValueError(f"veri seti kolonları eksik: {sorted(missing)}")
        for row in reader:
            label = row["label"]
            scenario = row["scenario"]
            event_type = row["event_type"]
            labels[label] += 1
            scenarios[scenario] += 1
            event_types[event_type] += 1
            sensors[row["sensor"]] += 1
            runs.add(row["run_id"])
            timestamp = row["timestamp"]
            if timestamp:
                first_timestamp = (
                    timestamp
                    if not first_timestamp
                    else min(first_timestamp, timestamp)
                )
                last_timestamp = max(last_timestamp, timestamp)
            if event_type in MODEL_EVENT_TYPES:
                model_labels[label] += 1
                model_scenarios[scenario] += 1

    def run_key(value: str) -> tuple[int, int | str]:
        return (0, int(value)) if value.isdigit() else (1, value)

    return {
        "rows": sum(labels.values()),
        "columns": len(columns),
        "column_names": columns,
        "run_ids": sorted(runs, key=run_key),
        "run_count": len(runs),
        "timestamp_min": first_timestamp,
        "timestamp_max": last_timestamp,
        "label_counts": dict(sorted(labels.items())),
        "scenario_counts": dict(sorted(scenarios.items())),
        "event_type_counts": dict(sorted(event_types.items())),
        "sensor_counts": dict(sorted(sensors.items())),
        "model_input": {
            "event_types": sorted(MODEL_EVENT_TYPES),
            "rows": sum(model_labels.values()),
            "label_counts": dict(sorted(model_labels.items())),
            "scenario_counts": dict(sorted(model_scenarios.items())),
        },
    }


def _git_state(root: Path) -> dict[str, Any]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        return {
            "head": head,
            "worktree_clean": not status,
            "changed_path_count": len(status),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"head": "", "worktree_clean": False, "changed_path_count": None}


def _package_versions(names: tuple[str, ...]) -> dict[str, str]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def build_snapshot(
    root: Path,
    *,
    version: str,
    dataset: Path,
    labels: Path,
) -> dict[str, Any]:
    dataset_path = root / dataset
    labels_path = root / labels
    if not dataset_path.is_file() or not labels_path.is_file():
        raise ValueError("ana dataset veya labels dosyası bulunamadı")

    reports = [
        file_record(root / relative, root)
        for relative in DEFAULT_REPORTS
        if (root / relative).is_file()
    ]
    sources = [
        file_record(root / relative, root)
        for relative in DEFAULT_SOURCE_FILES
        if (root / relative).is_file()
    ]
    external_profiles = []
    for name, relative in DEFAULT_EXTERNAL_PROFILES:
        path = root / relative
        if not path.is_file():
            continue
        rows, columns = csv_shape(path)
        external_profiles.append({
            "name": name,
            **file_record(path, root),
            "rows": rows,
            "columns": columns,
        })

    return {
        "schema": "maritime-lab-dataset-snapshot/v1",
        "dataset_name": "Maritime-Lab OT IDS Dataset",
        "dataset_version": version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "source": "synthetic-lab-only-no-real-vessel-traffic",
            "attestation_scope": ["primary_dataset", "ground_truth"],
            "synthetic_lab_data": True,
            "live_systems_contacted": False,
            "real_vessel_data_in_open_release": False,
            "git": _git_state(root),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(
                ("numpy", "pandas", "scapy", "scikit-learn", "xgboost")
            ),
        },
        "primary_dataset": {
            **file_record(dataset_path, root),
            "statistics": dataset_statistics(dataset_path),
        },
        "ground_truth": file_record(labels_path, root),
        "evaluation_reports": reports,
        "external_profiles": external_profiles,
        "source_files": sources,
        "notes": [
            "Hashes identify the exact local artifacts used for v0.3.",
            "External profiles have different sampling units and must not be concatenated.",
            "A dirty git worktree means this is a logical data snapshot, not a release commit.",
        ],
    }


def verify_snapshot(
    root: Path,
    manifest_path: Path,
    *,
    data_only: bool = False,
) -> tuple[int, list[str]]:
    try:
        snapshot = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"snapshot bulunamadı: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"geçersiz snapshot JSON: {exc.msg}") from exc

    records = [
        snapshot.get("primary_dataset", {}),
        snapshot.get("ground_truth", {}),
    ]
    if not data_only:
        records.extend([
            *snapshot.get("evaluation_reports", []),
            *snapshot.get("external_profiles", []),
            *snapshot.get("source_files", []),
        ])
    checked = 0
    failures = []
    for record in records:
        relative = record.get("path")
        expected = record.get("sha256")
        if not relative or not expected:
            failures.append(f"eksik path/hash kaydı: {record!r}")
            continue
        path = root / str(relative)
        if not path.is_file():
            failures.append(f"dosya bulunamadı: {relative}")
            continue
        actual = sha256(path)
        checked += 1
        if actual != expected:
            failures.append(
                f"SHA-256 uyuşmuyor: {relative} "
                f"(beklenen={expected}, bulunan={actual})"
            )
    return checked, failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Veri seti sürümünü istatistik ve SHA-256 manifestiyle dondur."
    )
    parser.add_argument("--version", default="v0.3")
    parser.add_argument("--dataset", type=Path, default=Path("attack/dataset.csv"))
    parser.add_argument("--labels", type=Path, default=Path("attack/labels.csv"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("DATASET-SNAPSHOT-v0.3.json"),
    )
    parser.add_argument(
        "--verify",
        type=Path,
        help="yeni snapshot üretme; verilen manifestteki hash'leri doğrula",
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="yalnız ana veri ve ground-truth hash'lerini doğrula",
    )
    args = parser.parse_args()
    if args.verify is not None:
        manifest_path = (
            args.verify
            if args.verify.is_absolute()
            else ROOT / args.verify
        )
        checked, failures = verify_snapshot(
            ROOT,
            manifest_path,
            data_only=args.data_only,
        )
        if failures:
            for failure in failures:
                print(f"[HATA] {failure}")
            raise SystemExit(1)
        print(f"[OK] snapshot doğrulandı: {checked} dosya")
        return
    snapshot = build_snapshot(
        ROOT,
        version=args.version,
        dataset=args.dataset,
        labels=args.labels,
    )
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    stats = snapshot["primary_dataset"]["statistics"]
    print(
        f"[OK] {args.version}: {stats['rows']} satır, "
        f"{stats['run_count']} koşu -> {output}"
    )


if __name__ == "__main__":
    main()
