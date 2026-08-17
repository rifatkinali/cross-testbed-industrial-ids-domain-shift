"""Reproduce the preregistered RC2 result and require exact semantic equality."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
FROZEN_RESULT = (
    ROOT
    / "docs"
    / "sonuclar"
    / "OPENPLC-MACYSTE-RC2-ONKAYITLI-SONUC-2026-08-17.json"
)


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


def run(command: Sequence[str]) -> None:
    env = {**os.environ, "PYTHONUTF8": "1"}
    completed = subprocess.run(list(command), cwd=ROOT, env=env, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: {' '.join(command)}"
        )


def reproduce(*, skip_tests: bool = False) -> dict[str, int]:
    if not skip_tests:
        run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_openplc_macyste_rc2_contract.py",
                "tests/test_openplc_macyste_rc2_result.py",
            ]
        )
    with tempfile.TemporaryDirectory(prefix="openplc-macyste-rc2-") as temp:
        output = Path(temp) / "rc2-result.json"
        run(
            [
                sys.executable,
                "-m",
                "ml.openplc_macyste_rc2",
                "--out",
                str(output),
            ]
        )
        expected = json.loads(FROZEN_RESULT.read_text(encoding="utf-8"))
        actual = json.loads(output.read_text(encoding="utf-8"))
        difference = first_difference(expected, actual)
        if difference is not None:
            raise ValueError(f"reproduced RC2 result differs: {difference}")
    return {
        "contract_and_result_tests": 0 if skip_tests else 8,
        "schemas": len(expected["results"]),
        "models": sum(len(item["per_seed"]) for item in expected["results"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-tests", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = reproduce(skip_tests=args.skip_tests)
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        print(f"[FAILED] {exc}", file=sys.stderr)
        return 2
    print(
        "[PASSED] exact RC2 reproduction: "
        f"{report['schemas']} schemas, {report['models']} models, "
        f"{report['contract_and_result_tests']} tests"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
