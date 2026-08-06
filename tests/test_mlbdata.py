"""Tests for MLB data ingestion.

Network calls are stubbed. These tests verify parsing, provenance, and failure
handling -- the parts that must be right regardless of what the upstream
sources happen to be serving today. A test that depends on live MLB data would
fail every offseason and tell us nothing about the code.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sweetbear import mlbdata
from sweetbear.mlbdata import (
    SAVANT_LEADERBOARDS,
    Capture,
    fetch_csv,
    fetch_json,
    fetch_probable_pitchers,
    fetch_savant_leaderboard,
    snapshot,
)

SCHEDULE_FIXTURE = {
    "dates": [
        {
            "games": [
                {
                    "gamePk": 823431,
                    "gameDate": "2026-08-04T17:05:00Z",
                    "status": {"detailedState": "Scheduled"},
                    "venue": {"id": 2681, "name": "Citizens Bank Park"},
                    "teams": {
                        "away": {
                            "team": {"id": 120, "name": "Washington Nationals"},
                            "probablePitcher": {"id": 1, "fullName": "A Pitcher"},
                        },
                        "home": {
                            "team": {"id": 143, "name": "Philadelphia Phillies"},
                            "probablePitcher": {"id": 2, "fullName": "B Pitcher"},
                        },
                    },
                },
                {
                    # Starters not yet announced -- a normal morning state.
                    "gamePk": 823432,
                    "gameDate": "2026-08-04T23:10:00Z",
                    "status": {"detailedState": "Scheduled"},
                    "venue": {"id": 15, "name": "Coors Field"},
                    "teams": {
                        "away": {"team": {"id": 137, "name": "San Francisco Giants"}},
                        "home": {"team": {"id": 115, "name": "Colorado Rockies"}},
                    },
                },
            ]
        }
    ]
}

SAVANT_CSV = (
    '﻿"last_name, first_name","player_id","era","xera"\n'
    '"Alcantara, Sandy","645261","3.50","3.10"\n'
    '"Nola, Aaron","656302","4.20","3.95"\n'
)


@pytest.fixture
def stub_request(monkeypatch):
    """Replace the network layer with a recorded-response map."""
    calls: list[str] = []
    responses: dict[str, bytes] = {}

    def fake_request(url, timeout, retries):
        calls.append(url)
        for fragment, payload in responses.items():
            if fragment in url:
                return payload
        raise RuntimeError(f"no stub for {url}")

    monkeypatch.setattr(mlbdata, "_request", fake_request)
    monkeypatch.setattr(mlbdata.time, "sleep", lambda _s: None)
    return calls, responses


# --- provenance ------------------------------------------------------------


def test_json_capture_records_digest_of_exact_bytes(stub_request):
    calls, responses = stub_request
    raw = json.dumps(SCHEDULE_FIXTURE).encode()
    responses["schedule"] = raw

    capture = fetch_json("https://statsapi.mlb.com/api/v1/schedule?x=1")

    assert capture.digest == hashlib.sha256(raw).hexdigest()
    assert capture.source == "statsapi"
    assert capture.content_type == "application/json"
    assert capture.captured_at.endswith("+00:00")


def test_csv_capture_strips_bom_and_counts_rows(stub_request):
    calls, responses = stub_request
    responses["leaderboard"] = SAVANT_CSV.encode("utf-8")

    capture = fetch_csv("https://baseballsavant.mlb.com/leaderboard/x?csv=true")

    assert capture.rows == 2
    # The BOM must not leak into the first column name, or every lookup of
    # that field silently misses.
    assert "last_name, first_name" in capture.data[0]
    assert capture.data[0]["xera"] == "3.10"


def test_identical_bytes_produce_identical_digests(stub_request):
    calls, responses = stub_request
    responses["leaderboard"] = SAVANT_CSV.encode("utf-8")
    first = fetch_csv("https://baseballsavant.mlb.com/leaderboard/x?csv=true")
    second = fetch_csv("https://baseballsavant.mlb.com/leaderboard/x?csv=true")
    assert first.digest == second.digest


def test_revised_source_data_changes_the_digest(stub_request):
    """The case timestamps alone cannot catch: a silently corrected stat line."""
    calls, responses = stub_request
    responses["leaderboard"] = SAVANT_CSV.encode("utf-8")
    before = fetch_csv("https://baseballsavant.mlb.com/leaderboard/x?csv=true")

    responses["leaderboard"] = SAVANT_CSV.replace("3.10", "3.08").encode("utf-8")
    after = fetch_csv("https://baseballsavant.mlb.com/leaderboard/x?csv=true")

    assert before.digest != after.digest


def test_to_metadata_omits_the_payload():
    capture = Capture(
        source="s", url="u", captured_at="t", digest="d",
        content_type="c", rows=1, data={"big": "payload"},
    )
    assert "data" not in capture.to_metadata()
    assert capture.to_metadata()["digest"] == "d"


# --- schedule parsing ------------------------------------------------------


def test_probable_pitchers_flattens_both_sides(stub_request):
    calls, responses = stub_request
    responses["schedule"] = json.dumps(SCHEDULE_FIXTURE).encode()

    rows = fetch_probable_pitchers("2026-08-04")

    assert len(rows) == 2  # only the game with announced starters
    names = {row["pitcher_name"] for row in rows}
    assert names == {"A Pitcher", "B Pitcher"}


def test_probable_pitchers_records_opponent_and_venue(stub_request):
    calls, responses = stub_request
    responses["schedule"] = json.dumps(SCHEDULE_FIXTURE).encode()

    away = next(r for r in fetch_probable_pitchers("2026-08-04") if r["side"] == "away")

    assert away["team_id"] == 120
    assert away["opponent_id"] == 143  # the other side, not its own team
    assert away["venue_name"] == "Citizens Bank Park"


def test_unannounced_starters_are_omitted_not_faked(stub_request):
    """A game with no probable pitcher must produce no row at all."""
    calls, responses = stub_request
    responses["schedule"] = json.dumps(SCHEDULE_FIXTURE).encode()

    rows = fetch_probable_pitchers("2026-08-04")

    assert all(row["game_pk"] != 823432 for row in rows)
    assert all(row["pitcher_id"] is not None for row in rows)


def test_probable_pitchers_carry_source_provenance(stub_request):
    calls, responses = stub_request
    responses["schedule"] = json.dumps(SCHEDULE_FIXTURE).encode()

    rows = fetch_probable_pitchers("2026-08-04")

    assert all(row["source_digest"] for row in rows)
    assert len({row["source_digest"] for row in rows}) == 1


# --- savant boards ---------------------------------------------------------


def test_savant_leaderboard_rejects_unknown_board():
    with pytest.raises(ValueError, match="unknown leaderboard"):
        fetch_savant_leaderboard("does_not_exist")


def test_savant_leaderboard_rejects_bad_player_type():
    with pytest.raises(ValueError, match="pitcher.*batter"):
        fetch_savant_leaderboard("expected_statistics", "umpire")


def test_savant_leaderboard_builds_expected_url(stub_request):
    calls, responses = stub_request
    responses["leaderboard"] = SAVANT_CSV.encode("utf-8")

    fetch_savant_leaderboard("expected_statistics", "batter", 2026)

    url = calls[-1]
    assert "type=batter" in url
    assert "year=2026" in url
    assert "csv=true" in url


def test_every_registered_board_has_a_path():
    for board, (path, extra) in SAVANT_LEADERBOARDS.items():
        assert path.startswith("/leaderboard/"), board
        assert isinstance(extra, dict), board


# --- snapshot orchestration ------------------------------------------------


def test_snapshot_writes_manifest_and_artifacts(stub_request, tmp_path: Path):
    calls, responses = stub_request
    responses["schedule"] = json.dumps(SCHEDULE_FIXTURE).encode()
    responses["leaderboard"] = SAVANT_CSV.encode("utf-8")

    manifest = snapshot("2026-08-04", tmp_path, polite_delay=0.0)

    assert manifest["error_count"] == 0
    assert manifest["artifact_count"] > 0

    written = list(tmp_path.iterdir())
    assert len(written) == 1  # one timestamped directory
    assert (written[0] / "manifest.json").exists()
    assert (written[0] / "schedule.json").exists()


def test_snapshot_records_failures_instead_of_hiding_them(stub_request, tmp_path: Path):
    """One slow board must not cost the whole slate, and the gap must show."""
    calls, responses = stub_request
    responses["schedule"] = json.dumps(SCHEDULE_FIXTURE).encode()
    # No stub for "leaderboard" -> every Savant fetch raises.

    manifest = snapshot("2026-08-04", tmp_path, polite_delay=0.0)

    assert manifest["error_count"] > 0
    assert any("savant" in error["name"] for error in manifest["errors"])
    # The schedule still made it through.
    assert any(a["name"] == "schedule" for a in manifest["artifacts"])


def test_snapshot_skips_batter_pitch_arsenal(stub_request, tmp_path: Path):
    """Savant returns an empty board rather than an error, so skip explicitly."""
    calls, responses = stub_request
    responses["schedule"] = json.dumps(SCHEDULE_FIXTURE).encode()
    responses["leaderboard"] = SAVANT_CSV.encode("utf-8")

    manifest = snapshot("2026-08-04", tmp_path, polite_delay=0.0)

    names = {a["name"] for a in manifest["artifacts"]}
    assert "savant_pitcher_pitch_arsenal" in names
    assert "savant_batter_pitch_arsenal" not in names


def test_snapshot_directories_are_timestamped_uniquely(stub_request, tmp_path: Path):
    calls, responses = stub_request
    responses["schedule"] = json.dumps(SCHEDULE_FIXTURE).encode()
    responses["leaderboard"] = SAVANT_CSV.encode("utf-8")

    manifest = snapshot("2026-08-04", tmp_path, polite_delay=0.0)
    directory = next(iter(tmp_path.iterdir()))

    # Directory name must be a UTC stamp, so snapshots sort chronologically
    # and never overwrite one another.
    assert directory.name.endswith("Z")
    assert manifest["slate_date"] == "2026-08-04"


# --- retry policy ----------------------------------------------------------


def test_client_errors_are_not_retried(monkeypatch):
    """A 404 is an answer. Retrying it wastes the source's capacity."""
    import urllib.error

    attempts = {"count": 0}

    def always_404(request, timeout):  # noqa: ARG001
        attempts["count"] += 1
        raise urllib.error.HTTPError("u", 404, "Not Found", {}, None)

    monkeypatch.setattr(mlbdata.urllib.request, "urlopen", always_404)
    monkeypatch.setattr(mlbdata.time, "sleep", lambda _s: None)

    with pytest.raises(urllib.error.HTTPError):
        mlbdata._request("https://example.test/missing", timeout=1, retries=3)

    assert attempts["count"] == 1


def test_server_errors_are_retried(monkeypatch):
    import urllib.error

    attempts = {"count": 0}

    def always_503(request, timeout):  # noqa: ARG001
        attempts["count"] += 1
        raise urllib.error.HTTPError("u", 503, "Unavailable", {}, None)

    monkeypatch.setattr(mlbdata.urllib.request, "urlopen", always_503)
    monkeypatch.setattr(mlbdata.time, "sleep", lambda _s: None)

    with pytest.raises(RuntimeError, match="failed to fetch"):
        mlbdata._request("https://example.test/flaky", timeout=1, retries=3)

    assert attempts["count"] == 3
