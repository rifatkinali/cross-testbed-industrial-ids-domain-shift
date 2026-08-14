"""Reproduce and freeze the thesis-critical OpenPLC -> MaCySTe result chain."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN = Path(
    "captures/macyste/macyste-20260726T113328Z"
)
DEFAULT_OUTPUT = Path("academic_release/v0.5.0")
SCHEMAS = ("flow", "protocol", "physical_strict", "physical_proxy")
SEEDS = "42,43,44"
EXPECTED_ROWS = 34_949
EXPECTED_RAW_ARTIFACTS = 144
EXPECTED_RAW_MANIFEST_SHA256 = (
    "5f292140c6137eef955d93f1714aad54bd83ddfeab426f8b78071c7899540475"
)
EXPECTED_DERIVED_MANIFEST_SHA256 = (
    "aa03a6fa04c279be88e7e4784129886f93a2f561ce6c730c7d336bdfc36ee8c8"
)
EXPECTED_LIMITATIONS_SHA256 = (
    "0380e9662501e59fcc92651d52ba235eb4c39e13841b27d0b19838ab31429b46"
)
EXPECTED_COMBINED_SHA256 = (
    "908bfad7e3d05e1b1256e9ccfa37cf37a7523c09e5b1936c1df095e4e5c8b232"
)
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
EXPECTED_HEADLINE = {
    "flow": (0.9543, 0.3220, 0.3940, 0.5017, 0.2889),
    "protocol": (0.9759, 0.3186, 0.3895, 0.4917, 0.2824),
    "physical_strict": (0.9813, 0.3289, 0.3845, 0.4649, 0.2994),
    "physical_proxy": (0.9842, 0.4001, 0.4595, 0.5440, 0.3675),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def run(name: str, command: Sequence[str]) -> None:
    print(f"[RUN] {name}", flush=True)
    env = {**os.environ, "PYTHONUTF8": "1"}
    completed = subprocess.run(list(command), cwd=ROOT, env=env, check=False)
    if completed.returncode:
        raise RuntimeError(f"{name} failed with exit code {completed.returncode}")
    print(f"[PASSED] {name}", flush=True)


def verify_raw_artifacts(campaign: Path, manifest: dict[str, Any]) -> int:
    checked = 0
    for run_record in manifest["runs"]:
        for relative, record in run_record["artifacts"].items():
            path = campaign.joinpath(*relative.replace("\\", "/").split("/"))
            if not path.is_file():
                raise FileNotFoundError(f"raw campaign artifact missing: {path}")
            if path.stat().st_size != record["bytes"]:
                raise ValueError(f"raw artifact size mismatch: {path}")
            if sha256_file(path) != record["sha256"]:
                raise ValueError(f"raw artifact hash mismatch: {path}")
            checked += 1
    return checked


def verify_and_combine_campaign(
    campaign: Path,
    combined_path: Path,
) -> dict[str, Any]:
    raw_manifest_path = campaign / "manifest.json"
    derived_manifest_path = campaign / "derived-manifest-v0.4.json"
    limitation_path = campaign / "ACCEPTED-LIMITATIONS.json"
    for path in (raw_manifest_path, derived_manifest_path, limitation_path):
        if not path.is_file():
            raise FileNotFoundError(f"required campaign file missing: {path}")

    raw = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    derived = json.loads(derived_manifest_path.read_text(encoding="utf-8"))
    if raw["campaign_id"] != campaign.name:
        raise ValueError("campaign id does not match directory name")
    if not raw["complete"] or raw["runs_completed"] != 12:
        raise ValueError("raw campaign is incomplete")
    raw_manifest_hash = sha256_file(raw_manifest_path)
    derived_manifest_hash = sha256_file(derived_manifest_path)
    limitation_hash = sha256_file(limitation_path)
    if raw_manifest_hash != EXPECTED_RAW_MANIFEST_SHA256:
        raise ValueError("raw manifest differs from frozen campaign")
    if derived_manifest_hash != EXPECTED_DERIVED_MANIFEST_SHA256:
        raise ValueError("derived manifest differs from frozen campaign")
    if limitation_hash != EXPECTED_LIMITATIONS_SHA256:
        raise ValueError("accepted limitations differ from frozen campaign")
    if raw_manifest_hash != derived["raw_manifest_sha256"]:
        raise ValueError("derived manifest does not bind the raw manifest")

    raw_artifacts_checked = verify_raw_artifacts(campaign, raw)
    if raw_artifacts_checked != EXPECTED_RAW_ARTIFACTS:
        raise ValueError("raw artifact count differs from frozen campaign")
    by_run = {item["run"]: item for item in derived["runs"]}
    if tuple(sorted(by_run)) != EXPECTED_RUNS:
        raise ValueError("derived campaign run set differs from frozen contract")

    frames: list[pd.DataFrame] = []
    event_files: list[dict[str, Any]] = []
    for run_name in EXPECTED_RUNS:
        record = by_run[run_name]
        path = campaign / run_name / record["output"]
        actual_hash = sha256_file(path)
        if actual_hash != record["output_sha256"]:
            raise ValueError(f"derived event hash mismatch: {run_name}")
        frame = pd.read_csv(path, low_memory=False)
        if len(frame) != record["rows"]:
            raise ValueError(f"derived event row mismatch: {run_name}")
        frames.append(frame)
        event_files.append(
            {
                "run": run_name,
                "rows": len(frame),
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": actual_hash,
            }
        )

    combined = pd.concat(frames, ignore_index=True)
    if len(combined) != EXPECTED_ROWS:
        raise ValueError(
            f"combined target has {len(combined)} rows, expected {EXPECTED_ROWS}"
        )
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(combined_path, index=False)
    combined_hash = sha256_file(combined_path)
    if combined_hash != EXPECTED_COMBINED_SHA256:
        raise ValueError("combined target differs from frozen campaign")
    return {
        "campaign_id": raw["campaign_id"],
        "raw_manifest_sha256": raw_manifest_hash,
        "derived_manifest_sha256": derived_manifest_hash,
        "accepted_limitations_sha256": limitation_hash,
        "raw_artifacts_checked": raw_artifacts_checked,
        "event_files": event_files,
        "combined_rows": len(combined),
        "combined_sha256": combined_hash,
    }


def assert_close(actual: float, expected: float, label: str) -> None:
    if round(float(actual), 4) != round(float(expected), 4):
        raise ValueError(f"{label}: expected {expected:.4f}, got {actual:.4f}")


def verify_headline(results_path: Path) -> list[dict[str, Any]]:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if [item["schema"] for item in results] != list(SCHEMAS):
        raise ValueError("LODO schema order differs from frozen contract")
    rows: list[dict[str, Any]] = []
    for item in results:
        schema = item["schema"]
        expected = EXPECTED_HEADLINE[schema]
        actual = (
            item["auc_validation_openplc"]["mean"],
            item["auc_event_weighted"]["mean"],
            item["auc_scenario_balanced"]["mean"],
            item["per_scenario_auc"]["manipulation"]["mean"],
            item["per_scenario_auc"]["recon"]["mean"],
        )
        for index, (value, frozen) in enumerate(zip(actual, expected)):
            assert_close(value, frozen, f"{schema}[{index}]")
        rows.append(
            {
                "schema": schema,
                "openplc_validation_auc": actual[0],
                "macyste_event_auc": actual[1],
                "macyste_scenario_balanced_auc": actual[2],
                "macyste_manipulation_auc": actual[3],
                "macyste_recon_auc": actual[4],
            }
        )
    return rows


def verify_diagnostics(output: Path) -> dict[str, Any]:
    d2 = json.loads((output / "d2-unseen-category.json").read_text("utf-8"))
    d2_h3 = json.loads((output / "d2-h3-mechanism.json").read_text("utf-8"))
    d2a = json.loads((output / "d2a-strata.json").read_text("utf-8"))
    d3 = json.loads((output / "d3-score-discreteness.json").read_text("utf-8"))

    protocol = next(
        item for item in d2["primary_panel"] if item["schema"] == "protocol"
    )
    if protocol["unseen_any_n"] != 201:
        raise ValueError("D2 unseen row count differs from frozen result")
    if round(protocol["unseen_any_mass"], 6) != 0.005751:
        raise ValueError("D2 unseen mass differs from frozen result")
    if not d2_h3["verdict"]["degenerate"]:
        raise ValueError("D2 H3 must remain explicitly degenerate")
    if len(d2a["strata_order"]) != 14 or len(d2a["rows"]) != 238:
        raise ValueError("D2A must publish all 14 preregistered strata")
    if d3["hypotheses"]["combined"]["verdict"] != "kismi_genislik":
        raise ValueError("D3 combined verdict differs from frozen result")
    return {
        "d2_unseen_rows": protocol["unseen_any_n"],
        "d2_unseen_mass": protocol["unseen_any_mass"],
        "d2_h3": "test_edilemedi_dejenere",
        "d2a_strata": len(d2a["strata_order"]),
        "d2a_detail_rows": len(d2a["rows"]),
        "d3_combined": d3["hypotheses"]["combined"]["verdict"],
    }


def write_table(output: Path, rows: list[dict[str, Any]]) -> None:
    csv_path = output / "lodo-main-table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| Şema | OpenPLC val AUC | MaCySTe olay AUC | "
        "MaCySTe dengeli AUC | Manipulation | Recon |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['schema']}` "
            f"| {row['openplc_validation_auc']:.4f} "
            f"| {row['macyste_event_auc']:.4f} "
            f"| {row['macyste_scenario_balanced_auc']:.4f} "
            f"| {row['macyste_manipulation_auc']:.4f} "
            f"| {row['macyste_recon_auc']:.4f} |"
        )
    (output / "lodo-main-table.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_svg(output: Path, rows: list[dict[str, Any]]) -> None:
    width, height = 920, 520
    left, top, chart_w, chart_h = 90, 55, 770, 360
    colors = ("#17324D", "#D97706", "#0F766E")
    series = (
        ("OpenPLC validation", "openplc_validation_auc"),
        ("MaCySTe event-weighted", "macyste_event_auc"),
        ("MaCySTe scenario-balanced", "macyste_scenario_balanced_auc"),
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#F8FAFC"/>',
        '<text x="90" y="30" font-family="Arial" font-size="20" '
        'font-weight="700" fill="#0F172A">OpenPLC → MaCySTe LODO transfer</text>',
    ]
    for tick in range(0, 11, 2):
        value = tick / 10
        y = top + chart_h - value * chart_h
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" '
            f'y2="{y:.1f}" stroke="#CBD5E1" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end" '
            f'font-family="Arial" font-size="12" fill="#475569">{value:.1f}</text>'
        )
    chance_y = top + chart_h - 0.5 * chart_h
    parts.append(
        f'<line x1="{left}" y1="{chance_y:.1f}" x2="{left + chart_w}" '
        f'y2="{chance_y:.1f}" stroke="#991B1B" stroke-width="2" '
        'stroke-dasharray="7 5"/>'
    )
    parts.append(
        f'<text x="{left + chart_w}" y="{chance_y - 7:.1f}" '
        'text-anchor="end" font-family="Arial" font-size="11" '
        'fill="#991B1B">chance = 0.5</text>'
    )
    group_w = chart_w / len(rows)
    bar_w = 38
    for group, row in enumerate(rows):
        center = left + group_w * (group + 0.5)
        for index, (_, key) in enumerate(series):
            value = float(row[key])
            x = center + (index - 1) * (bar_w + 8) - bar_w / 2
            y = top + chart_h - value * chart_h
            h = value * chart_h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" '
                f'height="{h:.1f}" rx="3" fill="{colors[index]}"/>'
            )
            parts.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}" '
                f'text-anchor="middle" font-family="Arial" font-size="11" '
                f'fill="#0F172A">{value:.3f}</text>'
            )
        parts.append(
            f'<text x="{center:.1f}" y="{top + chart_h + 26}" '
            f'text-anchor="middle" font-family="Arial" font-size="12" '
            f'fill="#0F172A">{row["schema"]}</text>'
        )
    for index, (label, _) in enumerate(series):
        x = 115 + index * 230
        parts.append(
            f'<rect x="{x}" y="472" width="15" height="15" '
            f'fill="{colors[index]}"/>'
        )
        parts.append(
            f'<text x="{x + 23}" y="484" font-family="Arial" '
            f'font-size="12" fill="#334155">{label}</text>'
        )
    parts.append("</svg>")
    (output / "lodo-main-figure.svg").write_text(
        "\n".join(parts) + "\n",
        encoding="utf-8",
    )


def write_manifest(
    output: Path,
    campaign_report: dict[str, Any],
    diagnostic_summary: dict[str, Any],
) -> None:
    output_names = (
        "lodo-results.json",
        "lodo-main-table.csv",
        "lodo-main-table.md",
        "lodo-main-figure.svg",
        "d2-unseen-category.json",
        "d2-h3-mechanism.json",
        "d2a-strata.json",
        "d3-score-discreteness.json",
    )
    source_names = (
        "ml/features.py",
        "ml/lodo_generalization.py",
        "ml/lodo_report.py",
        "ml/unseen_category_diagnostic.py",
        "ml/unseen_category_mechanism.py",
        "ml/unseen_category_strata.py",
        "ml/score_discreteness_diagnostic.py",
        "scripts/freeze_academic_results.py",
        "requirements.txt",
        "requirements-dev.txt",
    )
    document_names = (
        "BULGU-lodo-v04-cross-testbed.md",
        "DENEY-D2-Gorulmemis-Kategori-Teshisi.md",
        "BULGU-d2-gorulmemis-kategori.md",
        "DENEY-D2A-Tabakali-Gorulmemis-Kutle.md",
        "DENEY-D3-Esik-Platolari-Skor-Kesikligi.md",
        "BULGU-d3-skor-kesikligi.md",
    )
    manifest = {
        "schema": "maritime.academic-freeze/v1",
        "version": "v0.5.0",
        "status": "passed",
        "primary_decision": "gate_a_no",
        "portable_reproduction": "pending_external_campaign_archive",
        "training": {
            "path": "attack/dataset.csv",
            "sha256": sha256_file(ROOT / "attack/dataset.csv"),
            "snapshot": "DATASET-SNAPSHOT-v0.3.json",
            "snapshot_sha256": sha256_file(ROOT / "DATASET-SNAPSHOT-v0.3.json"),
        },
        "target": campaign_report,
        "analysis": {
            "seeds": [42, 43, 44],
            "max_fpr": 0.01,
            "schemas": list(SCHEMAS),
            "diagnostic_summary": diagnostic_summary,
        },
        "source_files": {
            path: sha256_file(ROOT / path) for path in source_names
        },
        "protocol_and_finding_documents": {
            path: sha256_file(ROOT / path) for path in document_names
        },
        "outputs": {
            name: {
                "bytes": (output / name).stat().st_size,
                "sha256": sha256_file(output / name),
            }
            for name in output_names
        },
        "claim_boundary": [
            "single OpenPLC to MaCySTe component-integration testbed pair",
            "not cross-vendor evidence",
            "not real-vessel or field validation",
            "run-bootstrap intervals measure scripted replicate noise only",
            "raw campaign is gitignored and needs an external archive for portable replay",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tez-kritik LODO, D2 ve D3 sonuçlarını tek kapıda yeniden üret."
    )
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    campaign = resolve(ROOT, args.campaign)
    output = resolve(ROOT, args.output)
    output.mkdir(parents=True, exist_ok=True)
    scratch = ROOT / ".codex-validation" / "academic-freeze"
    scratch.mkdir(parents=True, exist_ok=True)
    combined = scratch / "macyste-events-v0.4-combined.csv"
    python = sys.executable

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
    campaign_report = verify_and_combine_campaign(campaign, combined)
    print(
        f"[PASSED] campaign integrity: "
        f"{campaign_report['raw_artifacts_checked']} raw artifacts, "
        f"{campaign_report['combined_rows']} derived rows"
    )

    commands = (
        (
            "lodo",
            [
                python,
                "-m",
                "ml.lodo_report",
                "--train",
                "attack/dataset.csv",
                "--test",
                str(combined),
                "--seeds",
                SEEDS,
                "--schema",
                "all",
                "--out",
                str(output / "lodo-results.json"),
            ],
        ),
        (
            "d2",
            [
                python,
                "-m",
                "ml.unseen_category_diagnostic",
                "--source",
                "attack/dataset.csv",
                "--target",
                str(combined),
                "--out",
                str(output / "d2-unseen-category.json"),
            ],
        ),
        (
            "d2_h3",
            [
                python,
                "-m",
                "ml.unseen_category_mechanism",
                "--source",
                "attack/dataset.csv",
                "--target",
                str(combined),
                "--out",
                str(output / "d2-h3-mechanism.json"),
            ],
        ),
        (
            "d2a",
            [
                python,
                "-m",
                "ml.unseen_category_strata",
                "--source",
                "attack/dataset.csv",
                "--target",
                str(combined),
                "--out",
                str(output / "d2a-strata.json"),
            ],
        ),
        (
            "d3",
            [
                python,
                "-m",
                "ml.score_discreteness_diagnostic",
                "--source",
                "attack/dataset.csv",
                "--target",
                str(combined),
                "--out",
                str(output / "d3-score-discreteness.json"),
            ],
        ),
    )
    for name, command in commands:
        run(name, command)

    rows = verify_headline(output / "lodo-results.json")
    diagnostic_summary = verify_diagnostics(output)
    write_table(output, rows)
    write_svg(output, rows)
    write_manifest(output, campaign_report, diagnostic_summary)
    shutil.copyfile(
        output / "manifest.json",
        scratch / "academic-freeze-manifest.json",
    )
    print(f"[PASSED] academic freeze: {output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
