#!/usr/bin/env python3
"""Ingest captured MLB archives into the queryable fact store, or replay them.

The capture step writes archives. This step turns them into a dataset that can
answer "what did we know at time T?" -- which is what a walk-forward evaluation
needs and what an archive alone cannot provide.

Safe to run repeatedly and safe to run on a schedule: ingestion is idempotent,
so a retry after a partial failure converges rather than duplicating.

Usage:
    python scripts/mlb_ingest.py                       # ingest new archives
    python scripts/mlb_ingest.py --rebuild             # replay everything
    python scripts/mlb_ingest.py --as-of 2026-08-05T12:00:00+00:00 --metric xera
    python scripts/mlb_ingest.py --revisions

Exit codes:
    0  success
    1  some archives failed (details printed; the rest still ingested)
    2  could not run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sweetbear.store import (  # noqa: E402
    as_of,
    connect,
    coverage,
    ingest_all,
    rebuild,
    revisions,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, default=Path("data/mlb_archives"))
    parser.add_argument("--db", type=Path, default=Path("data/mlb_facts.db"))
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop derived state and replay every archive from scratch",
    )
    parser.add_argument("--as-of", dest="as_of_time", default=None)
    parser.add_argument("--metric", default=None)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--revisions", action="store_true")
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args()

    if not args.archives.exists():
        print(f"ERROR: no archive directory at {args.archives}", file=sys.stderr)
        return 2

    if args.rebuild:
        result = rebuild(args.db, args.archives)
        print("# Replay (rebuilt from scratch)\n")
    else:
        conn = connect(args.db)
        result = ingest_all(conn, args.archives)
        conn.close()
        print("# Ingest\n")

    print(result.summary())

    conn = connect(args.db)
    try:
        if args.as_of_time:
            rows = as_of(
                conn, args.as_of_time, metric=args.metric, entity_id=args.entity
            )
            print(f"\n## Known as of {args.as_of_time} ({len(rows):,} facts)\n")
            for row in rows[: args.limit]:
                value = row["value_num"] if row["value_num"] is not None else row["value_text"]
                print(
                    f"  {row['entity_id']:>8}  {row['metric']:<28} {str(value):>12}"
                    f"   observed {row['observed_at'][:19]}"
                )

        if args.revisions:
            changes = revisions(conn, limit=args.limit)
            print(f"\n## Revisions ({len(changes)} shown)\n")
            if not changes:
                print("  none — no captured value has changed between snapshots")
            for change in changes:
                print(
                    f"  {change['entity_id']:>8}  {change['metric']:<28}"
                    f" {change['old_value']} -> {change['new_value']}"
                    f"   ({change['old_observed'][:16]} -> {change['new_observed'][:16]})"
                )

        report = coverage(conn)
        print("\n## Store coverage\n")
        for key in ("facts", "entities", "metrics", "captures", "artifacts_ingested"):
            print(f"  {key:<20} {report[key]:,}")
        if report["first_observed"]:
            print(f"  window               {report['first_observed'][:19]}"
                  f" -> {report['last_observed'][:19]}")
    finally:
        conn.close()

    return 1 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
