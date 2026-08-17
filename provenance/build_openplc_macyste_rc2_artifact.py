"""Build the standalone preregistered OpenPLC--MaCySTe RC2 artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "research_release" / "openplc-macyste-transfer-v2"
ARCHIVE_ROOT = "cross-testbed-industrial-ids-domain-shift-v1.0.0-rc2"
DEFAULT_OUTPUT = ROOT / "output" / "research-artifact" / ARCHIVE_ROOT
CAMPAIGN_ID = "macyste-20260726T113328Z"
CAMPAIGN = ROOT / "captures" / "macyste" / CAMPAIGN_ID
RUNS = (
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


STATIC_FILES: tuple[tuple[str, str], ...] = (
    ("research_release/openplc-macyste-transfer-v2/README.md", "README.md"),
    ("research_release/openplc-macyste-transfer-v2/RELEASE-NOTES.md", "RELEASE-NOTES.md"),
    ("research_release/openplc-macyste-transfer-v2/CITATION.cff", "CITATION.cff"),
    ("research_release/openplc-macyste-transfer-v2/release-spec.json", "release-spec.json"),
    ("research_release/openplc-macyste-transfer-v2/RELEASE-CHECKS.json", "RELEASE-CHECKS.json"),
    ("research_release/openplc-macyste-transfer-v2/requirements.txt", "requirements.txt"),
    ("research_release/openplc-macyste-transfer-v2/PREREGISTRATION.md", "PREREGISTRATION.md"),
    ("research_release/openplc-macyste-transfer-v2/preregistration.json", "preregistration.json"),
    ("research_release/openplc-macyste-transfer-v2/rc2-main-figure.svg", "results/rc2-main-figure.svg"),
    ("research_release/openplc-macyste-transfer-v2/THIRD-PARTY-NOTICE.md", "THIRD-PARTY-NOTICE.md"),
    ("research_release/openplc-macyste-transfer-v2/MACYSTE-PROVENANCE.json", "MACYSTE-PROVENANCE.json"),
    ("research_release/openplc-macyste-transfer-v2/SECURITY.md", "SECURITY.md"),
    ("research_release/openplc-macyste-transfer-v2/CONTRIBUTING.md", "CONTRIBUTING.md"),
    ("research_release/openplc-macyste-transfer-v2/.gitignore", ".gitignore"),
    ("LICENSE", "LICENSE"),
    ("pytest.ini", "pytest.ini"),
    ("ml/__init__.py", "ml/__init__.py"),
    ("ml/lodo_generalization.py", "ml/lodo_generalization.py"),
    ("ml/openplc_macyste_rc2.py", "ml/openplc_macyste_rc2.py"),
    ("tests/test_openplc_macyste_rc2_contract.py", "tests/test_openplc_macyste_rc2_contract.py"),
    ("tests/test_openplc_macyste_rc2_result.py", "tests/test_openplc_macyste_rc2_result.py"),
    ("scripts/reproduce_openplc_macyste_rc2.py", "scripts/reproduce_openplc_macyste_rc2.py"),
    ("scripts/verify_research_artifact.py", "scripts/verify_artifact.py"),
    ("scripts/build_openplc_macyste_rc2_artifact.py", "provenance/build_openplc_macyste_rc2_artifact.py"),
    (".codex-validation/ds02-independent-faults-20260815/dataset.csv", ".codex-validation/ds02-independent-faults-20260815/dataset.csv"),
    ("docs/sonuclar/OPENPLC-BAGIMSIZ-ARIZA-KOSULARI-2026-08-15.json", "docs/sonuclar/OPENPLC-BAGIMSIZ-ARIZA-KOSULARI-2026-08-15.json"),
    ("docs/sonuclar/OPENPLC-MACYSTE-RC2-ONKAYITLI-SONUC-2026-08-17.json", "docs/sonuclar/OPENPLC-MACYSTE-RC2-ONKAYITLI-SONUC-2026-08-17.json"),
    ("docs/sonuclar/OPENPLC-MACYSTE-RC2-ONKAYITLI-SONUC-2026-08-17.md", "docs/sonuclar/OPENPLC-MACYSTE-RC2-ONKAYITLI-SONUC-2026-08-17.md"),
    ("docs/sonuclar/OPENPLC-MACYSTE-RC2-ONKAYITLI-SONUC-2026-08-17.json", "results/rc2-results.json"),
    ("external/MaCySTe/LICENSE.txt", "THIRD_PARTY_LICENSES/MaCySTe-LICENSE.txt"),
    ("external/MaCySTe/README.md", "THIRD_PARTY_LICENSES/MaCySTe-README.md"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def safe_output(path: Path) -> Path:
    resolved = path.resolve()
    allowed = (ROOT / "output" / "research-artifact").resolve()
    if not _within(resolved, allowed) or resolved == allowed:
        raise ValueError(f"output must be a child of {allowed}")
    return resolved


def copy_file(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"required artifact input missing: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def iter_payload_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "artifact-manifest.json":
            yield path


def package_revision() -> str:
    """Return the newest commit touching an input that is copied into RC2."""
    tracked_inputs = [
        "research_release/openplc-macyste-transfer-v2",
        "ml/openplc_macyste_rc2.py",
        "ml/lodo_generalization.py",
        "scripts/build_openplc_macyste_rc2_artifact.py",
        "scripts/reproduce_openplc_macyste_rc2.py",
        "scripts/verify_research_artifact.py",
        "tests/test_openplc_macyste_rc2_contract.py",
        "tests/test_openplc_macyste_rc2_result.py",
        "docs/sonuclar/OPENPLC-MACYSTE-RC2-ONKAYITLI-SONUC-2026-08-17.json",
        "docs/sonuclar/OPENPLC-MACYSTE-RC2-ONKAYITLI-SONUC-2026-08-17.md",
    ]
    completed = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", *tracked_inputs],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    revision = completed.stdout.strip()
    if not revision:
        raise ValueError("could not resolve RC2 package source revision")
    return revision


def assert_boundary(output: Path, spec: dict[str, Any]) -> None:
    relative_paths = {
        path.relative_to(output).as_posix() for path in iter_payload_files(output)
    }
    for relative in sorted(relative_paths):
        if any(relative.startswith(prefix) for prefix in spec["forbidden_prefixes"]):
            raise ValueError(f"forbidden product prefix in artifact: {relative}")
        if relative in spec["forbidden_exact_paths"]:
            raise ValueError(f"forbidden product path in artifact: {relative}")
        if relative.lower().endswith((".key", ".pcap", ".pcapng", ".edge-update")):
            raise ValueError(f"forbidden sensitive extension in artifact: {relative}")


def build_manifest(output: Path, spec: dict[str, Any]) -> dict[str, Any]:
    records = []
    total_bytes = 0
    for path in iter_payload_files(output):
        size = path.stat().st_size
        total_bytes += size
        records.append(
            {
                "path": path.relative_to(output).as_posix(),
                "bytes": size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema": "maritime.research-artifact-manifest/v1",
        "artifact_id": spec["artifact_id"],
        "version": spec["version"],
        "status": spec["status"],
        "release_eligible": not spec["publication_blockers"],
        "source_revision": package_revision(),
        "reproducibility": spec["reproducibility"],
        "files": records,
        "file_count": len(records),
        "total_bytes": total_bytes,
        "publication_blockers": spec["publication_blockers"],
    }


def build(output: Path, *, make_zip: bool = True) -> tuple[Path, Path | None]:
    output = safe_output(output)
    spec = json.loads((TEMPLATE / "release-spec.json").read_text(encoding="utf-8"))
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    for source_name, target_name in STATIC_FILES:
        copy_file(ROOT / source_name, output / target_name)

    prereg_target = output / "research_release" / "openplc-macyste-transfer-v2"
    copy_file(TEMPLATE / "PREREGISTRATION.md", prereg_target / "PREREGISTRATION.md")
    copy_file(TEMPLATE / "preregistration.json", prereg_target / "preregistration.json")

    campaign_target = output / "captures" / "macyste" / CAMPAIGN_ID
    for name in ("manifest.json", "derived-manifest-v0.4.json", "ACCEPTED-LIMITATIONS.json"):
        copy_file(CAMPAIGN / name, campaign_target / name)
    for run in RUNS:
        copy_file(
            CAMPAIGN / run / "events-v0.4.csv",
            campaign_target / run / "events-v0.4.csv",
        )

    assert_boundary(output, spec)
    manifest = build_manifest(output, spec)
    manifest_path = output / "artifact-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    zip_path = None
    if make_zip:
        zip_path = output.parent / f"{ARCHIVE_ROOT}.zip"
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    archive.write(path, Path(ARCHIVE_ROOT) / path.relative_to(output))
    return manifest_path, zip_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-zip", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest, archive = build(args.output, make_zip=not args.no_zip)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"[FAILED] {exc}", file=sys.stderr)
        return 2
    print(f"[PASSED] manifest: {manifest}")
    if archive is not None:
        print(f"[PASSED] archive: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
