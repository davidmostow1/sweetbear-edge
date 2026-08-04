#!/usr/bin/env python3
"""Capture one MLB data snapshot: official stats, Statcast, slate context.

Designed to run unattended on a schedule. Writes a content-addressed,
timestamped directory and prints a short report.

Usage:
    python scripts/mlb_snapshot.py                    # today, UTC
    python scripts/mlb_snapshot.py --date 2026-08-04
    python scripts/mlb_snapshot.py --out data/mlb

Exit codes:
    0  every artifact captured
    1  some artifacts failed (details printed; partial data still written)
    2  the snapshot could not run at all
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sweetbear.mlbdata import snapshot  # noqa: E402


def archive(source_dir: Path, archive_dir: Path, keep_days: int) -> Path:
    """Compress a snapshot directory and prune archives past ``keep_days``.

    Raw snapshots run ~1.7 MB and compress to ~220 KB. At five captures a day
    that is roughly 1 MB daily, so archives are pruned while manifests are
    kept forever: the manifests are the provenance chain and are tiny, the
    payloads are bulk that can be re-fetched for any date Savant still serves.
    """
    archive_dir.mkdir(parents=True, exist_ok=True)
    target = archive_dir / f"{source_dir.name}.tgz"
    with tarfile.open(target, "w:gz") as tar:
        tar.add(source_dir, arcname=source_dir.name)

    cutoff = time.time() - (keep_days * 86400)
    for old in archive_dir.glob("*.tgz"):
        if old.stat().st_mtime < cutoff:
            old.unlink()

    return target


def append_manifest_index(manifest: dict, index_path: Path) -> None:
    """Append this capture's provenance to a permanent JSONL index.

    Append-only by construction. The digests recorded here are what let a
    later audit prove which bytes a prediction actually saw, so this file must
    never be rewritten -- only added to.
    """
    index_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "slate_date": manifest["slate_date"],
        "captured_at": manifest["captured_at"],
        "artifact_count": manifest["artifact_count"],
        "error_count": manifest["error_count"],
        "artifacts": {a["name"]: a["digest"] for a in manifest["artifacts"]},
        "errors": [e["name"] for e in manifest["errors"]],
    }
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="Slate date, YYYY-MM-DD (default: today UTC)",
    )
    parser.add_argument("--out", type=Path, default=Path("data/mlb"))
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season for Statcast leaderboards (default: year of --date)",
    )
    parser.add_argument(
        "--archive-to",
        type=Path,
        default=None,
        help="Compress the snapshot into this directory and prune old archives",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=None,
        help="Append-only JSONL manifest index for permanent provenance",
    )
    parser.add_argument(
        "--keep-days", type=int, default=30, help="Archive retention (default 30)"
    )
    parser.add_argument(
        "--prune-raw",
        action="store_true",
        help="Delete the uncompressed snapshot after archiving",
    )
    args = parser.parse_args()

    try:
        manifest = snapshot(args.date, args.out, year=args.season)
    except Exception as exc:  # noqa: BLE001
        print(f"SNAPSHOT FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print(f"# MLB snapshot — slate {manifest['slate_date']}")
    print(f"captured_at: {manifest['captured_at']}")
    print(f"artifacts:   {manifest['artifact_count']}  errors: {manifest['error_count']}\n")

    for artifact in manifest["artifacts"]:
        print(f"  {artifact['name']:<40} rows={artifact['rows']:<6} {artifact['digest'][:12]}")

    if args.index:
        append_manifest_index(manifest, args.index)
        print(f"\nprovenance appended to {args.index}")

    if args.archive_to:
        captured = sorted(args.out.iterdir())[-1]
        target = archive(captured, args.archive_to, args.keep_days)
        size_kb = target.stat().st_size / 1024
        print(f"archived to {target} ({size_kb:.0f} KB, keeping {args.keep_days}d)")
        if args.prune_raw:
            shutil.rmtree(captured)

    if manifest["errors"]:
        print("\n## Failures\n")
        for error in manifest["errors"]:
            print(f"  {error['name']}: {error['error']}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
