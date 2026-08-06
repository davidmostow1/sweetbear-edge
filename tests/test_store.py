"""Tests for the idempotent, replayable fact store.

Two properties define this module and everything else is secondary: ingesting
the same input twice must change nothing, and replaying from scratch must
produce the identical store. If either fails, the archive is not a dataset.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tarfile
from pathlib import Path

import pytest

from sweetbear.store import (
    as_of,
    connect,
    coverage,
    ingest_all,
    ingest_archive,
    rebuild,
    revisions,
)

PITCHER_BOARD = [
    {
        "last_name, first_name": "Alpha, A",
        "player_id": "111",
        "year": "2026",
        "era": "3.50",
        "xera": "3.10",
    },
    {
        "last_name, first_name": "Beta, B",
        "player_id": "222",
        "year": "2026",
        "era": "4.00",
        "xera": "4.40",
    },
]

ARSENAL_BOARD = [
    {
        "last_name, first_name": "Alpha, A",
        "player_id": "111",
        "year": "2026",
        "pitch_type": "FF",
        "pitch_name": "4-Seam Fastball",
        "whiff_percent": "25.0",
    },
    {
        "last_name, first_name": "Alpha, A",
        "player_id": "111",
        "year": "2026",
        "pitch_type": "SL",
        "pitch_name": "Slider",
        "whiff_percent": "38.5",
    },
]

PROBABLES = [
    {
        "game_pk": 900001,
        "game_date": "2026-08-05T17:05:00Z",
        "status": "Scheduled",
        "side": "home",
        "team_id": 143,
        "team_name": "Team",
        "opponent_id": 120,
        "pitcher_id": 111,
        "pitcher_name": "Alpha, A",
        "venue_id": 2681,
        "venue_name": "A Park",
    }
]


def _digest(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_archive(
    path: Path,
    captured_at: str,
    pitcher_board=None,
    arsenal=None,
    probables=None,
) -> Path:
    """Write a snapshot archive matching what mlbdata.snapshot produces."""
    pitcher_board = PITCHER_BOARD if pitcher_board is None else pitcher_board
    arsenal = ARSENAL_BOARD if arsenal is None else arsenal
    probables = PROBABLES if probables is None else probables

    stamp = captured_at.replace(":", "").replace("-", "")[:15] + "Z"
    staging = path.parent / f"_stage_{stamp}"
    inner = staging / stamp
    inner.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "savant_pitcher_expected_statistics": pitcher_board,
        "savant_pitcher_pitch_arsenal": arsenal,
        "probable_pitchers": probables,
    }
    manifest = {
        "slate_date": captured_at[:10],
        "captured_at": captured_at,
        "artifacts": [
            {"name": name, "digest": _digest(payload), "rows": len(payload)}
            for name, payload in artifacts.items()
        ],
        "errors": [],
    }
    for name, payload in artifacts.items():
        (inner / f"{name}.json").write_text(json.dumps(payload))
    (inner / "manifest.json").write_text(json.dumps(manifest))

    archive = path / f"{stamp}.tgz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(inner, arcname=stamp)
    return archive


@pytest.fixture
def archive_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "archives"
    directory.mkdir()
    build_archive(directory, "2026-08-05T05:00:00+00:00")
    return directory


# --- the two defining properties -------------------------------------------


def test_reingesting_the_same_archive_writes_nothing(archive_dir: Path, tmp_path: Path):
    """Idempotence. An unattended scheduler that retries must not duplicate."""
    conn = connect(tmp_path / "facts.db")

    first = ingest_all(conn, archive_dir)
    second = ingest_all(conn, archive_dir)

    assert first.facts_written > 0
    assert second.facts_written == 0
    assert second.artifacts_ingested == 0
    assert second.artifacts_skipped == first.artifacts_ingested


def test_replay_from_scratch_is_deterministic(archive_dir: Path, tmp_path: Path):
    """Replay. The store is a pure function of the archives."""

    def fingerprint(db: Path) -> str:
        conn = connect(db)
        rows = conn.execute(
            "SELECT entity_type, entity_id, metric, valid_date, observed_at,"
            " value_num FROM facts ORDER BY 1,2,3,4,5"
        ).fetchall()
        conn.close()
        return hashlib.sha256(repr([tuple(r) for r in rows]).encode()).hexdigest()

    first = rebuild(tmp_path / "a.db", archive_dir)
    second = rebuild(tmp_path / "b.db", archive_dir)

    assert first.facts_written == second.facts_written
    assert fingerprint(tmp_path / "a.db") == fingerprint(tmp_path / "b.db")


def test_rebuild_discards_prior_state(archive_dir: Path, tmp_path: Path):
    db = tmp_path / "facts.db"
    conn = connect(db)
    conn.execute(
        "INSERT INTO facts VALUES ('pitcher','999','junk','2026','2026-01-01',"
        "1.0,NULL,'fake','deadbeef')"
    )
    conn.commit()
    conn.close()

    rebuild(db, archive_dir)

    conn = connect(db)
    junk = conn.execute("SELECT COUNT(*) FROM facts WHERE metric='junk'").fetchone()[0]
    conn.close()
    assert junk == 0


# --- anti-lookahead --------------------------------------------------------


def test_as_of_cannot_see_a_later_observation(tmp_path: Path):
    """The core guarantee: a query before a revision returns the old value."""
    directory = tmp_path / "archives"
    directory.mkdir()
    build_archive(directory, "2026-08-05T05:00:00+00:00")

    revised = [dict(row) for row in PITCHER_BOARD]
    revised[0]["xera"] = "2.50"
    build_archive(directory, "2026-08-05T21:00:00+00:00", pitcher_board=revised)

    conn = connect(tmp_path / "facts.db")
    ingest_all(conn, directory)

    before = as_of(conn, "2026-08-05T12:00:00+00:00", entity_id="111", metric="xera")
    after = as_of(conn, "2026-08-06T00:00:00+00:00", entity_id="111", metric="xera")

    assert before[0]["value_num"] == pytest.approx(3.10)
    assert after[0]["value_num"] == pytest.approx(2.50)


def test_as_of_before_any_capture_returns_nothing(archive_dir: Path, tmp_path: Path):
    conn = connect(tmp_path / "facts.db")
    ingest_all(conn, archive_dir)
    assert as_of(conn, "2020-01-01T00:00:00+00:00") == []


def test_as_of_filters_compose(archive_dir: Path, tmp_path: Path):
    conn = connect(tmp_path / "facts.db")
    ingest_all(conn, archive_dir)

    rows = as_of(
        conn, "2026-12-31T00:00:00+00:00", entity_type="pitcher", metric="era"
    )
    assert {row["entity_id"] for row in rows} == {"111", "222"}
    assert all(row["metric"] == "era" for row in rows)


def test_as_of_rows_carry_their_provenance(archive_dir: Path, tmp_path: Path):
    conn = connect(tmp_path / "facts.db")
    ingest_all(conn, archive_dir)
    row = as_of(conn, "2026-12-31T00:00:00+00:00", entity_id="111", metric="era")[0]
    assert row["source"] == "savant_pitcher_expected_statistics"
    assert len(row["source_digest"]) == 64


# --- revisions -------------------------------------------------------------


def test_revisions_surface_changed_values(tmp_path: Path):
    directory = tmp_path / "archives"
    directory.mkdir()
    build_archive(directory, "2026-08-05T05:00:00+00:00")
    revised = [dict(row) for row in PITCHER_BOARD]
    revised[0]["xera"] = "2.50"
    build_archive(directory, "2026-08-05T21:00:00+00:00", pitcher_board=revised)

    conn = connect(tmp_path / "facts.db")
    ingest_all(conn, directory)

    changes = [r for r in revisions(conn) if r["metric"] == "xera"]
    assert len(changes) == 1
    assert changes[0]["old_value"] == pytest.approx(3.10)
    assert changes[0]["new_value"] == pytest.approx(2.50)


def test_unchanged_values_are_not_reported_as_revisions(tmp_path: Path):
    directory = tmp_path / "archives"
    directory.mkdir()
    build_archive(directory, "2026-08-05T05:00:00+00:00")
    build_archive(directory, "2026-08-05T21:00:00+00:00")  # identical values

    conn = connect(tmp_path / "facts.db")
    ingest_all(conn, directory)

    assert revisions(conn) == []


# --- parsing ---------------------------------------------------------------


def test_arsenal_metrics_are_namespaced_by_pitch_type(archive_dir: Path, tmp_path: Path):
    """Without a pitch prefix, two pitch types collide on the primary key and
    one silently overwrites the other."""
    conn = connect(tmp_path / "facts.db")
    ingest_all(conn, archive_dir)

    rows = as_of(conn, "2026-12-31T00:00:00+00:00", entity_id="111")
    metrics = {row["metric"]: row["value_num"] for row in rows}

    assert metrics["ff.whiff_percent"] == pytest.approx(25.0)
    assert metrics["sl.whiff_percent"] == pytest.approx(38.5)


def test_probable_pitcher_facts_use_game_date_not_season(archive_dir: Path, tmp_path: Path):
    """A start assignment describes one game, not a season."""
    conn = connect(tmp_path / "facts.db")
    ingest_all(conn, archive_dir)

    rows = as_of(conn, "2026-12-31T00:00:00+00:00", metric="start.opponent_id")
    assert rows[0]["valid_date"] == "2026-08-05"
    assert rows[0]["value_num"] == pytest.approx(120)


def test_text_values_are_preserved_separately(archive_dir: Path, tmp_path: Path):
    conn = connect(tmp_path / "facts.db")
    ingest_all(conn, archive_dir)
    rows = as_of(conn, "2026-12-31T00:00:00+00:00", metric="start.venue_name")
    assert rows[0]["value_text"] == "A Park"
    assert rows[0]["value_num"] is None


def test_identity_columns_do_not_become_metrics(archive_dir: Path, tmp_path: Path):
    conn = connect(tmp_path / "facts.db")
    ingest_all(conn, archive_dir)
    rows = as_of(conn, "2026-12-31T00:00:00+00:00")
    metrics = {row["metric"] for row in rows}
    assert "player_id" not in metrics
    assert "last_name, first_name" not in metrics
    assert "pitch_name" not in metrics


# --- robustness ------------------------------------------------------------


def test_unreadable_archive_is_recorded_not_raised(tmp_path: Path):
    directory = tmp_path / "archives"
    directory.mkdir()
    (directory / "broken.tgz").write_bytes(b"not a tarball")

    conn = connect(tmp_path / "facts.db")
    result = ingest_all(conn, directory)

    assert result.errors
    assert result.facts_written == 0


def test_archive_without_manifest_is_recorded_not_raised(tmp_path: Path):
    directory = tmp_path / "archives"
    directory.mkdir()
    stray = tmp_path / "stray"
    stray.mkdir()
    (stray / "savant_pitcher_expected_statistics.json").write_text("[]")
    with tarfile.open(directory / "nomanifest.tgz", "w:gz") as tar:
        tar.add(stray, arcname="nomanifest")

    conn = connect(tmp_path / "facts.db")
    result = ingest_all(conn, directory)

    assert any("manifest" in error for error in result.errors)


def test_coverage_reports_what_was_stored(archive_dir: Path, tmp_path: Path):
    conn = connect(tmp_path / "facts.db")
    ingest_all(conn, archive_dir)
    report = coverage(conn)

    assert report["facts"] > 0
    assert report["captures"] == 1
    assert report["artifacts_ingested"] == 3
    assert any(
        source["source"] == "savant_pitcher_expected_statistics"
        for source in report["by_source"]
    )
