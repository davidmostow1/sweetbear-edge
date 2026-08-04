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
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sweetbear.mlbdata import snapshot  # noqa: E402


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

    if manifest["errors"]:
        print("\n## Failures\n")
        for error in manifest["errors"]:
            print(f"  {error['name']}: {error['error']}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
