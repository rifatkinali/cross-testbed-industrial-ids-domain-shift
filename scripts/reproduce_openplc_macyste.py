"""Reproduce the frozen OpenPLC--MaCySTe analysis from derived target events."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = "macyste-20260726T113328Z"
CAMPAIGN = ROOT / "data" / "macyste" / CAMPAIGN_ID
FROZEN = ROOT / "results"
EXPECTED_DERIVED_MANIFEST_SHA256 = (
    "aa03a6fa04c279be88e7e4784129886f93a2f561ce6c730c7d336bdfc36ee8c8"
)
EXPECTED_LIMITATIONS_SHA256 = (
    "0380e9662501e59fcc92651d52ba235eb4c39e13841b27d0b19838ab31429b46"
)
EXPECTED_COMBINED_SHA256 = (
    "908bfad7e3d05e1b1256e9ccfa37cf37a7523c09e5b1936c1df095e4e5c8b232"
)
EXPECTED_ROWS = 34_949
EXPECTED_RUNS = (
    "fault-r01",
    "fault-r02",
    "fault-r03",
    "manipulation-r01",
    "manipulation-r02",
    "manipulation-r03",
    "normal-r01",
    "normal-r02",
    "normal-r03",
    "recon-r01",
    "recon-r02",
    "recon-r03",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(name: str, command: Sequence[str]) -> None:
    print(f"[RUN] {name}", flush=True)
    env = {**os.environ, "PYTHONUTF8": "1"}
    completed = subprocess.run(list(command), cwd=ROOT, env=env, check=False)
    if completed.returncode:
        raise RuntimeError(f"{name} failed with exit code {completed.returncode}")
    print(f"[PASSED] {name}", flush=True)


def normalize_host_paths(value: Any) -> Any:
    """Remove the only expected host-specific value from diagnostic JSON."""
    if isinstance(value, dict):
        return {key: normalize_host_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_host_paths(item) for item in value]
    if isinstance(value, str) and value.replace("\\", "/").endswith(
        "/macyste-events-v0.4-combined.csv"
    ):
        return "<TARGET_COMBINED_CSV>"
    return value


def first_difference(expected: Any, actual: Any, path: str = "$") -> str | None:
    if type(expected) is not type(actual):
        return f"{path}: type {type(expected).__name__} != {type(actual).__name__}"
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            return f"{path}: keys {sorted(expected)} != {sorted(actual)}"
        for key in sorted(expected):
            difference = first_difference(expected[key], actual[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{path}: length {len(expected)} != {len(actual)}"
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            difference = first_difference(left, right, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if expected != actual:
        return f"{path}: {expected!r} != {actual!r}"
    return None


def verify_and_combine_derived(campaign: Path, combined_path: Path) -> dict[str, Any]:
    derived_path = campaign / "derived-manifest-v0.4.json"
    limitations_path = campaign / "ACCEPTED-LIMITATIONS.json"
    raw_manifest_path = campaign / "manifest.json"
    for path in (derived_path, limitations_path, raw_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"required campaign metadata missing: {path}")
    if sha256_file(derived_path) != EXPECTED_DERIVED_MANIFEST_SHA256:
        raise ValueError("derived manifest differs from frozen campaign")
    if sha256_file(limitations_path) != EXPECTED_LIMITATIONS_SHA256:
        raise ValueError("accepted limitations differ from frozen campaign")

    derived = json.loads(derived_path.read_text(encoding="utf-8"))
    by_run = {item["run"]: item for item in derived["runs"]}
    if tuple(sorted(by_run)) != EXPECTED_RUNS:
        raise ValueError("derived campaign run set differs from frozen contract")

    frames: list[pd.DataFrame] = []
    event_files: list[dict[str, Any]] = []
    for run_name in EXPECTED_RUNS:
        record = by_run[run_name]
        path = campaign / run_name / record["output"]
        if not path.is_file():
            raise FileNotFoundError(f"derived event file missing: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != record["output_sha256"]:
            raise ValueError(f"derived event hash mismatch: {run_name}")
        frame = pd.read_csv(path)
        if len(frame) != record["rows"]:
            raise ValueError(f"derived event row count mismatch: {run_name}")
        frames.append(frame)
        event_files.append(
            {"run": run_name, "rows": len(frame), "sha256": actual_hash}
        )

    combined = pd.concat(frames, ignore_index=True)
    if len(combined) != EXPECTED_ROWS:
        raise ValueError("combined target row count differs from frozen contract")
    combined.to_csv(combined_path, index=False)
    combined_hash = sha256_file(combined_path)
    if combined_hash != EXPECTED_COMBINED_SHA256:
        raise ValueError("combined target hash differs from frozen contract")
    return {
        "runs": len(event_files),
        "rows": len(combined),
        "sha256": combined_hash,
    }


def reproduce(work: Path, *, skip_tests: bool = False) -> dict[str, Any]:
    work.mkdir(parents=True, exist_ok=False)
    combined = work / "macyste-events-v0.4-combined.csv"
    report = verify_and_combine_derived(CAMPAIGN, combined)
    python = sys.executable

    if not skip_tests:
        run(
            "academic_contract_tests",
            [
                python,
                "-m",
                "pytest",
                "-q",
                "tests/test_lodo.py",
                "tests/test_lodo_report.py",
                "tests/test_d2_unseen_category.py",
                "tests/test_d2_h3_mechanism.py",
                "tests/test_d2a_strata.py",
                "tests/test_d3_score_discreteness.py",
            ],
        )
    run(
        "dataset_snapshot",
        [
            python,
            "-m",
            "ml.snapshot_dataset",
            "--verify",
            "DATASET-SNAPSHOT-v0.3.json",
            "--data-only",
        ],
    )

    outputs = {
        "lodo-results.json": [
            python,
            "-m",
            "ml.lodo_report",
            "--train",
            "attack/dataset.csv",
            "--test",
            str(combined),
            "--seeds",
            "42,43,44",
            "--schema",
            "all",
            "--out",
            str(work / "lodo-results.json"),
        ],
        "d2-unseen-category.json": [
            python,
            "-m",
            "ml.unseen_category_diagnostic",
            "--source",
            "attack/dataset.csv",
            "--target",
            str(combined),
            "--out",
            str(work / "d2-unseen-category.json"),
        ],
        "d2-h3-mechanism.json": [
            python,
            "-m",
            "ml.unseen_category_mechanism",
            "--source",
            "attack/dataset.csv",
            "--target",
            str(combined),
            "--out",
            str(work / "d2-h3-mechanism.json"),
        ],
        "d2a-strata.json": [
            python,
            "-m",
            "ml.unseen_category_strata",
            "--source",
            "attack/dataset.csv",
            "--target",
            str(combined),
            "--out",
            str(work / "d2a-strata.json"),
        ],
        "d3-score-discreteness.json": [
            python,
            "-m",
            "ml.score_discreteness_diagnostic",
            "--source",
            "attack/dataset.csv",
            "--target",
            str(combined),
            "--out",
            str(work / "d3-score-discreteness.json"),
        ],
    }
    for name, command in outputs.items():
        run(name.removesuffix(".json"), command)

    verified_outputs = 0
    for name in outputs:
        expected = normalize_host_paths(
            json.loads((FROZEN / name).read_text(encoding="utf-8"))
        )
        actual = normalize_host_paths(
            json.loads((work / name).read_text(encoding="utf-8"))
        )
        difference = first_difference(expected, actual)
        if difference is not None:
            raise ValueError(f"reproduced output semantic mismatch: {name}: {difference}")
        verified_outputs += 1
    report["verified_outputs"] = verified_outputs
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument(
        "--keep-output",
        type=Path,
        help="Keep reproduced files in this new directory instead of a temporary directory.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.keep_output:
            output = args.keep_output.resolve()
            if output.exists():
                raise ValueError(f"output already exists: {output}")
            report = reproduce(output, skip_tests=args.skip_tests)
        else:
            with tempfile.TemporaryDirectory(prefix="openplc-macyste-") as temp:
                report = reproduce(Path(temp) / "reproduced", skip_tests=args.skip_tests)
    except (
        FileExistsError,
        FileNotFoundError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"[FAILED] {exc}", file=sys.stderr)
        return 2
    print(
        "[PASSED] derived-data reproduction: "
        f"{report['runs']} runs, {report['rows']} rows, "
        f"{report['verified_outputs']} exact outputs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
