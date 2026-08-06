"""Idempotent, replayable fact store over captured MLB archives.

The capture layer (:mod:`sweetbear.mlbdata`) writes compressed archives. That
is an archive, not a dataset: it cannot be queried, it cannot deduplicate, and
it cannot answer the only question that matters for anti-lookahead work --
*what did we know at time T?* Adding more archives on top of that grows storage
without adding capability.

This module is the bridge. It ingests archives into a **bitemporal fact store**
and can rebuild that store from scratch at any time.

Two time axes, and the distinction is the whole point:

* ``valid_date`` -- what period the value describes (a 2026 season xERA).
* ``observed_at`` -- when we captured it.

Keeping both is what makes lookahead detectable rather than merely forbidden.
A feature used by a prediction made at time T is legitimate only if it was
observed at or before T; :func:`as_of` enforces that by construction, so a
backtest cannot accidentally consume a value that did not exist yet. Keeping
every observation rather than overwriting also makes silent upstream revisions
visible: the same fact captured twice with different values is a revision, and
:func:`revisions` lists them.

**Idempotence.** Ingestion is keyed on
``(entity_type, entity_id, metric, valid_date, observed_at)`` with the source
digest carried alongside. Re-ingesting the same archive is a no-op. Running the
pipeline twice, or replaying a partially-completed run, converges to the same
state -- which is what lets the ingest be safely retried by an unattended
scheduler that may have failed halfway through.

**Replay.** :func:`rebuild` drops derived state and re-ingests every archive in
chronological order. The store is therefore disposable: it never needs to be
committed or backed up, because it is a pure function of the archives. If the
parsing logic changes, replay produces a new consistent state rather than a
mixture of old and new interpretations.
"""

from __future__ import annotations

import json
import sqlite3
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

__all__ = [
    "SCHEMA",
    "IngestResult",
    "connect",
    "ingest_archive",
    "ingest_all",
    "rebuild",
    "as_of",
    "revisions",
    "coverage",
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    entity_type   TEXT NOT NULL,
    entity_id     TEXT NOT NULL,
    metric        TEXT NOT NULL,
    valid_date    TEXT NOT NULL,
    observed_at   TEXT NOT NULL,
    value_num     REAL,
    value_text    TEXT,
    source        TEXT NOT NULL,
    source_digest TEXT NOT NULL,
    PRIMARY KEY (entity_type, entity_id, metric, valid_date, observed_at)
);

CREATE INDEX IF NOT EXISTS facts_asof
    ON facts (entity_type, metric, observed_at);
CREATE INDEX IF NOT EXISTS facts_entity
    ON facts (entity_id, metric, observed_at);

-- One row per artifact successfully ingested. Ingestion consults this first,
-- so re-processing an archive is skipped rather than merely deduplicated
-- row-by-row: cheaper, and it makes "already done" an explicit fact.
CREATE TABLE IF NOT EXISTS ingested_artifacts (
    archive       TEXT NOT NULL,
    artifact      TEXT NOT NULL,
    digest        TEXT NOT NULL,
    observed_at   TEXT NOT NULL,
    rows_written  INTEGER NOT NULL,
    PRIMARY KEY (archive, artifact, digest)
);
"""

#: Columns that identify rather than measure. Everything else in a Savant board
#: is treated as a metric, so new columns appear automatically rather than
#: needing a schema change -- the store should not silently drop a metric just
#: because nobody updated a whitelist.
_SAVANT_IDENTITY = {"last_name, first_name", "player_id", "year", "team_name_alt"}


@dataclass(frozen=True)
class IngestResult:
    archives_seen: int = 0
    archives_ingested: int = 0
    artifacts_ingested: int = 0
    artifacts_skipped: int = 0
    facts_written: int = 0
    errors: tuple[str, ...] = ()

    def merge(self, other: "IngestResult") -> "IngestResult":
        return IngestResult(
            archives_seen=self.archives_seen + other.archives_seen,
            archives_ingested=self.archives_ingested + other.archives_ingested,
            artifacts_ingested=self.artifacts_ingested + other.artifacts_ingested,
            artifacts_skipped=self.artifacts_skipped + other.artifacts_skipped,
            facts_written=self.facts_written + other.facts_written,
            errors=self.errors + other.errors,
        )

    def summary(self) -> str:
        lines = [
            f"archives seen       {self.archives_seen}",
            f"archives ingested   {self.archives_ingested}",
            f"artifacts ingested  {self.artifacts_ingested}",
            f"artifacts skipped   {self.artifacts_skipped} (already present)",
            f"facts written       {self.facts_written:,}",
        ]
        if self.errors:
            lines.append(f"errors              {len(self.errors)}")
            lines.extend(f"  - {e}" for e in self.errors[:10])
        return "\n".join(lines)


def connect(path: Path | str) -> sqlite3.Connection:
    """Open (creating if needed) a fact store."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def _as_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _savant_facts(
    rows: list[dict[str, Any]],
    entity_type: str,
    source: str,
    observed_at: str,
    digest: str,
) -> Iterator[tuple]:
    """One fact per (player, metric). ``valid_date`` is the season the board
    covers, which is what a season-to-date leaderboard actually describes."""
    for row in rows:
        player_id = str(row.get("player_id") or "").strip()
        if not player_id:
            continue
        valid_date = str(row.get("year") or "").strip() or "unknown"

        # pitch_arsenal has one row per pitch type per pitcher, so the metric
        # name must carry the pitch or rows collide on the primary key and
        # silently overwrite each other.
        prefix = ""
        pitch = row.get("pitch_type")
        if pitch:
            prefix = f"{str(pitch).strip().lower()}."

        for column, raw in row.items():
            if column in _SAVANT_IDENTITY or column == "pitch_type":
                continue
            if column == "pitch_name":
                continue
            number = _as_number(raw)
            yield (
                entity_type,
                player_id,
                f"{prefix}{column}",
                valid_date,
                observed_at,
                number,
                None if number is not None else (str(raw) if raw != "" else None),
                source,
                digest,
            )


def _probable_pitcher_facts(
    rows: list[dict[str, Any]], observed_at: str, digest: str
) -> Iterator[tuple]:
    """Slate context: who starts, against whom, where.

    ``valid_date`` here is the game date rather than the season, because a
    probable-starter assignment describes one specific game.
    """
    for row in rows:
        pitcher_id = str(row.get("pitcher_id") or "").strip()
        game_date = str(row.get("game_date") or "")[:10] or "unknown"
        if not pitcher_id:
            continue
        for metric, key in (
            ("start.game_pk", "game_pk"),
            ("start.team_id", "team_id"),
            ("start.opponent_id", "opponent_id"),
            ("start.venue_id", "venue_id"),
            ("start.side", "side"),
            ("start.status", "status"),
            ("start.venue_name", "venue_name"),
        ):
            raw = row.get(key)
            if raw is None:
                continue
            number = _as_number(raw)
            yield (
                "pitcher",
                pitcher_id,
                metric,
                game_date,
                observed_at,
                number,
                None if number is not None else str(raw),
                "probable_pitchers",
                digest,
            )


def _artifact_facts(
    name: str, payload: Any, observed_at: str, digest: str
) -> Iterator[tuple]:
    if name.startswith("savant_"):
        entity_type = "batter" if "_batter_" in name else "pitcher"
        if isinstance(payload, list):
            yield from _savant_facts(payload, entity_type, name, observed_at, digest)
    elif name == "probable_pitchers":
        if isinstance(payload, list):
            yield from _probable_pitcher_facts(payload, observed_at, digest)
    # schedule.json is retained in the archive but not shredded into facts:
    # its useful content is already extracted into probable_pitchers, and
    # storing the raw nested payload as facts would add rows without adding
    # anything queryable.


def ingest_archive(conn: sqlite3.Connection, archive: Path) -> IngestResult:
    """Ingest one ``.tgz`` snapshot. Safe to call repeatedly on the same file."""
    archive = Path(archive)
    written = 0
    ingested = 0
    skipped = 0
    errors: list[str] = []

    try:
        tar = tarfile.open(archive, "r:gz")
    except (tarfile.TarError, OSError) as exc:
        return IngestResult(
            archives_seen=1, errors=(f"{archive.name}: unreadable ({exc})",)
        )

    with tar:
        members = {
            Path(m.name).name: m
            for m in tar.getmembers()
            if m.isfile() and m.name.endswith(".json")
        }
        manifest_member = members.get("manifest.json")
        if manifest_member is None:
            return IngestResult(
                archives_seen=1, errors=(f"{archive.name}: no manifest.json",)
            )

        manifest = json.load(tar.extractfile(manifest_member))
        observed_at = manifest["captured_at"]
        digests = {a["name"]: a["digest"] for a in manifest.get("artifacts", [])}

        for artifact_name, digest in digests.items():
            member = members.get(f"{artifact_name}.json")
            if member is None:
                continue

            already = conn.execute(
                "SELECT 1 FROM ingested_artifacts WHERE archive=? AND artifact=? AND digest=?",
                (archive.name, artifact_name, digest),
            ).fetchone()
            if already:
                skipped += 1
                continue

            try:
                payload = json.load(tar.extractfile(member))
            except (json.JSONDecodeError, OSError) as exc:
                errors.append(f"{archive.name}/{artifact_name}: {exc}")
                continue

            facts = list(_artifact_facts(artifact_name, payload, observed_at, digest))
            if facts:
                conn.executemany(
                    "INSERT OR IGNORE INTO facts (entity_type, entity_id, metric,"
                    " valid_date, observed_at, value_num, value_text, source,"
                    " source_digest) VALUES (?,?,?,?,?,?,?,?,?)",
                    facts,
                )
            conn.execute(
                "INSERT OR REPLACE INTO ingested_artifacts"
                " (archive, artifact, digest, observed_at, rows_written)"
                " VALUES (?,?,?,?,?)",
                (archive.name, artifact_name, digest, observed_at, len(facts)),
            )
            written += len(facts)
            ingested += 1

    conn.commit()
    return IngestResult(
        archives_seen=1,
        archives_ingested=1 if ingested else 0,
        artifacts_ingested=ingested,
        artifacts_skipped=skipped,
        facts_written=written,
        errors=tuple(errors),
    )


def ingest_all(conn: sqlite3.Connection, archive_dir: Path) -> IngestResult:
    """Ingest every archive in chronological order.

    Order matters for readability of the resulting history, not for
    correctness -- facts carry their own ``observed_at``, so an out-of-order
    ingest converges to the same state.
    """
    total = IngestResult()
    for archive in sorted(Path(archive_dir).glob("*.tgz")):
        total = total.merge(ingest_archive(conn, archive))
    return total


def rebuild(db_path: Path | str, archive_dir: Path | str) -> IngestResult:
    """Drop all derived state and replay every archive from scratch.

    The store is a pure function of the archives, so this is always safe and
    never loses anything that was not already reproducible. Use it after
    changing parsing logic, so the result is one consistent interpretation
    rather than a mixture of old and new.
    """
    db_path = Path(db_path)
    if db_path.exists():
        db_path.unlink()
    conn = connect(db_path)
    try:
        return ingest_all(conn, Path(archive_dir))
    finally:
        conn.close()


# --- queries ---------------------------------------------------------------


def as_of(
    conn: sqlite3.Connection,
    when: str,
    entity_type: str | None = None,
    metric: str | None = None,
    entity_id: str | None = None,
) -> list[dict[str, Any]]:
    """The most recent value of each fact **known at or before** ``when``.

    This is the anti-lookahead primitive. A feature vector assembled from this
    function cannot contain a value that did not exist at ``when``, regardless
    of what the caller does afterward -- the guarantee lives in the query, not
    in the discipline of whoever writes the caller.
    """
    clauses = ["observed_at <= ?"]
    params: list[Any] = [when]
    if entity_type:
        clauses.append("entity_type = ?")
        params.append(entity_type)
    if metric:
        clauses.append("metric = ?")
        params.append(metric)
    if entity_id:
        clauses.append("entity_id = ?")
        params.append(str(entity_id))

    where = " AND ".join(clauses)
    rows = conn.execute(
        f"""
        SELECT entity_type, entity_id, metric, valid_date, value_num, value_text,
               observed_at, source, source_digest
        FROM facts
        WHERE {where}
          AND observed_at = (
              SELECT MAX(f2.observed_at) FROM facts f2
              WHERE f2.entity_type = facts.entity_type
                AND f2.entity_id   = facts.entity_id
                AND f2.metric      = facts.metric
                AND f2.valid_date  = facts.valid_date
                AND f2.observed_at <= ?
          )
        ORDER BY entity_id, metric
        """,
        (*params, when),
    ).fetchall()
    return [dict(row) for row in rows]


def revisions(conn: sqlite3.Connection, limit: int = 100) -> list[dict[str, Any]]:
    """Facts whose value changed across captures.

    A changed value for the same entity, metric, and period is an upstream
    revision. Ordinary in-season stats move constantly and that is expected;
    what matters is that the movement is *visible and attributable* rather
    than silently overwriting history.
    """
    rows = conn.execute(
        """
        SELECT a.entity_type, a.entity_id, a.metric, a.valid_date,
               a.value_num AS old_value, a.observed_at AS old_observed,
               b.value_num AS new_value, b.observed_at AS new_observed
        FROM facts a
        JOIN facts b
          ON  a.entity_type = b.entity_type
          AND a.entity_id   = b.entity_id
          AND a.metric      = b.metric
          AND a.valid_date  = b.valid_date
          AND a.observed_at < b.observed_at
        WHERE a.value_num IS NOT b.value_num
          AND NOT EXISTS (
              SELECT 1 FROM facts m
              WHERE m.entity_type = a.entity_type AND m.entity_id = a.entity_id
                AND m.metric = a.metric AND m.valid_date = a.valid_date
                AND m.observed_at > a.observed_at AND m.observed_at < b.observed_at
          )
        ORDER BY b.observed_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(row) for row in rows]


def coverage(conn: sqlite3.Connection) -> dict[str, Any]:
    """What the store actually holds, for sanity-checking a replay."""
    def scalar(sql: str) -> Any:
        return conn.execute(sql).fetchone()[0]

    by_source = conn.execute(
        "SELECT source, COUNT(*) AS facts, COUNT(DISTINCT entity_id) AS entities"
        " FROM facts GROUP BY source ORDER BY facts DESC"
    ).fetchall()

    return {
        "facts": scalar("SELECT COUNT(*) FROM facts"),
        "entities": scalar("SELECT COUNT(DISTINCT entity_id) FROM facts"),
        "metrics": scalar("SELECT COUNT(DISTINCT metric) FROM facts"),
        "captures": scalar("SELECT COUNT(DISTINCT observed_at) FROM facts"),
        "first_observed": scalar("SELECT MIN(observed_at) FROM facts"),
        "last_observed": scalar("SELECT MAX(observed_at) FROM facts"),
        "artifacts_ingested": scalar("SELECT COUNT(*) FROM ingested_artifacts"),
        "by_source": [dict(row) for row in by_source],
    }
