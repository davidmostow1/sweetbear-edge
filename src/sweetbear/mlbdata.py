"""MLB data ingestion: official stats, Statcast advanced metrics, game context.

Two sources, both free and public:

* **MLB StatsAPI** (``statsapi.mlb.com``) -- the official feed. Schedule,
  probable starters, rosters, lineups, per-player season and game-by-game
  splits, live game state.
* **Baseball Savant** (``baseballsavant.mlb.com``) -- Statcast. This is where
  the descriptive statistics stop and the predictive ones start: expected
  outcomes (xBA, xSLG, xwOBA, xERA) that strip batted-ball luck, contact
  quality (barrel rate, exit velocity, hard-hit rate), and per-pitch arsenal
  performance (whiff rate, put-away rate, run value by pitch type).

Every fetch is stamped and hashed. That is not bookkeeping for its own sake:
an anti-lookahead guarantee needs evidence that is immutable and
content-addressed, not merely timestamped, because a stat line can be revised
after the fact while keeping an honest capture time. A digest of exactly the
bytes that were used lets a later audit prove the feature values a prediction
saw are the values that existed when it was made.

Nothing here models anything. It fetches, stamps, and stores.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "STATSAPI_BASE",
    "SAVANT_BASE",
    "SAVANT_LEADERBOARDS",
    "Capture",
    "fetch_json",
    "fetch_csv",
    "fetch_schedule",
    "fetch_probable_pitchers",
    "fetch_player_stats",
    "fetch_game_lineups",
    "fetch_savant_leaderboard",
    "snapshot",
]

STATSAPI_BASE = "https://statsapi.mlb.com/api/v1"
SAVANT_BASE = "https://baseballsavant.mlb.com"

USER_AGENT = "sweetbear-edge/0.1 (research; contact via repository)"
DEFAULT_TIMEOUT = 60
DEFAULT_RETRIES = 3
POLITE_DELAY_SECONDS = 1.0

#: Savant leaderboards worth pulling, keyed by a stable short name. Each entry
#: is (path, extra query params). ``type`` is filled in per player type.
#:
#: These are chosen for predictive content rather than completeness. Expected
#: statistics separate skill from batted-ball luck; contact quality is the most
#: stable batter signal in small samples; arsenal stats are what actually drive
#: a strikeout projection, since whiff and put-away rates by pitch type
#: anticipate strikeouts far better than a pitcher's own past strikeout total.
SAVANT_LEADERBOARDS: dict[str, tuple[str, dict[str, str]]] = {
    "expected_statistics": (
        "/leaderboard/expected_statistics",
        {"position": "", "team": "", "filterType": "bip", "min": "q"},
    ),
    "contact_quality": (
        "/leaderboard/statcast",
        {"position": "", "team": "", "min": "q"},
    ),
    "pitch_arsenal": (
        "/leaderboard/pitch-arsenal-stats",
        {"pitchType": "", "team": "", "min": "10"},
    ),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Capture:
    """One fetched artifact, with the provenance needed to audit it later.

    ``digest`` is the SHA-256 of the exact bytes received. Two captures with
    the same digest are the same evidence; a changed digest for the same URL
    means the source revised its data, which is precisely the silent-revision
    case that timestamps alone cannot detect.
    """

    source: str
    url: str
    captured_at: str
    digest: str
    content_type: str
    rows: int
    data: Any = field(repr=False, default=None)

    def to_metadata(self) -> dict[str, Any]:
        """Provenance without the payload, for indexes and manifests."""
        return {
            "source": self.source,
            "url": self.url,
            "captured_at": self.captured_at,
            "digest": self.digest,
            "content_type": self.content_type,
            "rows": self.rows,
        }


def _request(url: str, timeout: int, retries: int) -> bytes:
    """GET with retry on transient failure. Raises on permanent failure.

    Retries only network-level errors and 5xx: a 404 is an answer, not a
    hiccup, and retrying it just wastes the source's capacity.
    """
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code < 500:
                raise
            last_error = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
        if attempt < retries - 1:
            time.sleep(2.0**attempt)
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def fetch_json(
    url: str,
    source: str = "statsapi",
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> Capture:
    raw = _request(url, timeout, retries)
    payload = json.loads(raw.decode("utf-8"))
    return Capture(
        source=source,
        url=url,
        captured_at=_utc_now(),
        digest=hashlib.sha256(raw).hexdigest(),
        content_type="application/json",
        rows=len(payload) if isinstance(payload, list) else 1,
        data=payload,
    )


def fetch_csv(
    url: str,
    source: str = "savant",
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> Capture:
    raw = _request(url, timeout, retries)
    text = raw.decode("utf-8-sig")  # Savant emits a BOM
    rows = list(csv.DictReader(io.StringIO(text)))
    return Capture(
        source=source,
        url=url,
        captured_at=_utc_now(),
        digest=hashlib.sha256(raw).hexdigest(),
        content_type="text/csv",
        rows=len(rows),
        data=rows,
    )


# --- MLB StatsAPI ----------------------------------------------------------


def fetch_schedule(date: str, sport_id: int = 1, **kwargs: Any) -> Capture:
    """Games for one date (``YYYY-MM-DD``), with probable pitchers and venue."""
    params = urllib.parse.urlencode(
        {
            "sportId": sport_id,
            "date": date,
            "hydrate": "probablePitcher,team,venue,linescore,weather",
        }
    )
    return fetch_json(f"{STATSAPI_BASE}/schedule?{params}", **kwargs)


def fetch_probable_pitchers(date: str, **kwargs: Any) -> list[dict[str, Any]]:
    """Flatten the schedule into one row per probable starter.

    Returns an empty list rather than raising when starters are not yet
    announced -- an unannounced starter is a normal state of the world in the
    morning, not an error, and the caller needs to distinguish "no starter yet"
    from "the fetch failed."
    """
    capture = fetch_schedule(date, **kwargs)
    out: list[dict[str, Any]] = []
    for day in capture.data.get("dates", []):
        for game in day.get("games", []):
            for side in ("away", "home"):
                team = game.get("teams", {}).get(side, {})
                pitcher = team.get("probablePitcher")
                if not pitcher:
                    continue
                out.append(
                    {
                        "game_pk": game.get("gamePk"),
                        "game_date": game.get("gameDate"),
                        "status": game.get("status", {}).get("detailedState"),
                        "side": side,
                        "team_id": team.get("team", {}).get("id"),
                        "team_name": team.get("team", {}).get("name"),
                        "opponent_id": (
                            game.get("teams", {})
                            .get("home" if side == "away" else "away", {})
                            .get("team", {})
                            .get("id")
                        ),
                        "pitcher_id": pitcher.get("id"),
                        "pitcher_name": pitcher.get("fullName"),
                        "venue_id": game.get("venue", {}).get("id"),
                        "venue_name": game.get("venue", {}).get("name"),
                        "captured_at": capture.captured_at,
                        "source_digest": capture.digest,
                    }
                )
    return out


def fetch_player_stats(
    player_id: int,
    group: str = "pitching",
    stats: str = "season",
    season: int | None = None,
    **kwargs: Any,
) -> Capture:
    """Season, career, or game-log splits for one player.

    ``stats='gameLog'`` is the one that matters for rolling windows, since a
    season aggregate cannot tell you whether a pitcher's strikeout rate is
    trending up or collapsing.
    """
    params: dict[str, Any] = {"stats": stats, "group": group}
    if season is not None:
        params["season"] = season
    query = urllib.parse.urlencode(params)
    return fetch_json(f"{STATSAPI_BASE}/people/{player_id}/stats?{query}", **kwargs)


def fetch_game_lineups(game_pk: int, **kwargs: Any) -> Capture:
    """Confirmed batting orders and the handedness of everyone involved.

    Lineup state is a gating fact, not a nice-to-have: a batter projection made
    before the card is posted is a projection about someone who may not play,
    and the capture time here is what proves which of those two situations a
    prediction was made in.
    """
    return fetch_json(
        f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live", **kwargs
    )


# --- Baseball Savant / Statcast -------------------------------------------


def fetch_savant_leaderboard(
    board: str,
    player_type: str = "pitcher",
    year: int | None = None,
    **kwargs: Any,
) -> Capture:
    """One Savant leaderboard as parsed CSV rows.

    ``board`` must be a key of :data:`SAVANT_LEADERBOARDS`. ``player_type`` is
    ``'pitcher'`` or ``'batter'``.
    """
    if board not in SAVANT_LEADERBOARDS:
        raise ValueError(
            f"unknown leaderboard {board!r}; available: {sorted(SAVANT_LEADERBOARDS)}"
        )
    if player_type not in ("pitcher", "batter"):
        raise ValueError(f"player_type must be 'pitcher' or 'batter': {player_type!r}")

    path, extra = SAVANT_LEADERBOARDS[board]
    params = {"type": player_type, "year": year or datetime.now().year, "csv": "true"}
    params.update(extra)
    query = urllib.parse.urlencode(params)
    return fetch_csv(f"{SAVANT_BASE}{path}?{query}", **kwargs)


# --- Full snapshot ---------------------------------------------------------


def snapshot(
    date: str,
    out_dir: Path,
    year: int | None = None,
    boards: Iterable[str] | None = None,
    polite_delay: float = POLITE_DELAY_SECONDS,
) -> dict[str, Any]:
    """Capture a full picture of one slate, and write it content-addressed.

    Layout:

        out_dir/
          <captured_at>/manifest.json       provenance for every artifact
          <captured_at>/<name>.json         the payloads themselves

    A failure on one artifact does not abandon the rest -- a Savant board being
    slow should not cost you the lineups. Failures are recorded in the manifest
    with their error, so a gap in the data is visible rather than silent, which
    matters because a quietly-missing feature is indistinguishable from a
    feature that was genuinely absent.
    """
    out_dir = Path(out_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = out_dir / stamp
    target.mkdir(parents=True, exist_ok=True)

    year = year or int(date[:4])
    boards = list(boards) if boards is not None else list(SAVANT_LEADERBOARDS)

    artifacts: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    def record(name: str, capture: Capture) -> None:
        (target / f"{name}.json").write_text(
            json.dumps(capture.data, indent=1), encoding="utf-8"
        )
        artifacts.append({"name": name, **capture.to_metadata()})

    def attempt(name: str, fn: Any) -> None:
        try:
            record(name, fn())
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            errors.append({"name": name, "error": f"{type(exc).__name__}: {exc}"})
        time.sleep(polite_delay)

    attempt("schedule", lambda: fetch_schedule(date))

    for player_type in ("pitcher", "batter"):
        for board in boards:
            # pitch_arsenal is pitcher-side only; Savant returns an empty board
            # for batters rather than erroring, so skip it explicitly.
            if board == "pitch_arsenal" and player_type == "batter":
                continue
            attempt(
                f"savant_{player_type}_{board}",
                lambda b=board, p=player_type: fetch_savant_leaderboard(b, p, year),
            )

    try:
        probables = fetch_probable_pitchers(date)
        (target / "probable_pitchers.json").write_text(
            json.dumps(probables, indent=1), encoding="utf-8"
        )
        artifacts.append(
            {
                "name": "probable_pitchers",
                "source": "statsapi",
                "url": "derived:schedule",
                "captured_at": _utc_now(),
                "digest": hashlib.sha256(
                    json.dumps(probables, sort_keys=True).encode()
                ).hexdigest(),
                "content_type": "application/json",
                "rows": len(probables),
            }
        )
    except Exception as exc:  # noqa: BLE001
        errors.append({"name": "probable_pitchers", "error": str(exc)})

    manifest = {
        "slate_date": date,
        "captured_at": _utc_now(),
        "season": year,
        "artifacts": artifacts,
        "errors": errors,
        "artifact_count": len(artifacts),
        "error_count": len(errors),
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
