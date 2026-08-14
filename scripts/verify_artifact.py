"""Verify a built research artifact without running the ML analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifact-manifest.json"
SPEC = ROOT / "release-spec.json"
FORBIDDEN_CONTENT = {
    "host user path": re.compile(rb"(?i)[a-z]:[\\/]+users[\\/]+"),
    "placeholder identity": re.compile(
        rb"(?i)TODO-(?:kullanici|soyad|universite|repository)"
    ),
    "private development email": re.compile(rb"(?i)rftknl@outlook\.com"),
    "GitHub access token": re.compile(
        rb"(?:gh[opsu]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"
    ),
    "AWS access key": re.compile(rb"(?:AKIA|ASIA)[A-Z0-9]{16}"),
    "private key": re.compile(
        rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify(root: Path = ROOT) -> dict[str, int]:
    manifest = load_json(root / MANIFEST.name)
    spec = load_json(root / SPEC.name)
    if manifest.get("schema") != "maritime.research-artifact-manifest/v1":
        raise ValueError("unsupported artifact manifest schema")
    if manifest.get("artifact_id") != spec.get("artifact_id"):
        raise ValueError("artifact id does not match release spec")

    declared = {record["path"]: record for record in manifest["files"]}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name != MANIFEST.name
        and ".artifact-work" not in path.parts
        and "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts
        and ".git" not in path.parts
    }
    if set(declared) != actual:
        missing = sorted(set(declared) - actual)
        unexpected = sorted(actual - set(declared))
        raise ValueError(f"manifest file set mismatch: missing={missing}, unexpected={unexpected}")

    total_bytes = 0
    for relative, record in declared.items():
        path = root / relative
        size = path.stat().st_size
        total_bytes += size
        if size != record["bytes"]:
            raise ValueError(f"size mismatch: {relative}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"hash mismatch: {relative}")
        if any(relative.startswith(prefix) for prefix in spec["forbidden_prefixes"]):
            raise ValueError(f"forbidden product prefix: {relative}")
        if relative in spec["forbidden_exact_paths"]:
            raise ValueError(f"forbidden product path: {relative}")
        if relative.lower().endswith((".key", ".pcap", ".pcapng", ".edge-update")):
            raise ValueError(f"forbidden sensitive extension: {relative}")
        if path.is_symlink():
            raise ValueError(f"symbolic link not permitted in artifact: {relative}")
        content = path.read_bytes()
        for label, pattern in FORBIDDEN_CONTENT.items():
            if pattern.search(content):
                raise ValueError(f"forbidden {label} in artifact: {relative}")

    if len(declared) != manifest["file_count"]:
        raise ValueError("manifest file_count mismatch")
    if total_bytes != manifest["total_bytes"]:
        raise ValueError("manifest total_bytes mismatch")
    return {"files": len(declared), "bytes": total_bytes}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = verify(args.root.resolve())
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"[FAILED] {exc}", file=sys.stderr)
        return 2
    print(f"[PASSED] artifact integrity: {result['files']} files, {result['bytes']} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
