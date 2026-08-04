#!/usr/bin/env python3
"""Automated state snapshot of davidmostow1/bear-edge-platform.

Runs the falsifiable checklist from the adversarial audit against whatever the
repository currently contains, and emits a machine-readable state file plus a
human-readable diff against the previous snapshot.

The point is to remove the human from the loop. Every check here answers a
question that was asked in the audit and given a specific, verifiable answer;
re-running it says whether that answer has changed. Nothing here interprets or
editorialises -- it reports what is in the tree at a named commit, so a later
session can trust it without re-deriving it.

Usage:
    python scripts/bear_edge_snapshot.py --repo /path/to/bear-edge-platform
    python scripts/bear_edge_snapshot.py --repo <path> --previous state.json

Exit codes:
    0  snapshot succeeded (whether or not anything changed)
    1  snapshot could not run (repo missing, git failure)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Every check below maps to a finding in the audit. Keep the keys stable --
# downstream diffing is by key, so renaming one silently drops its history.
CHECKS_DOC = {
    "branch_shas": "Which branches exist and where they point",
    "test_count": "npm test totals",
    "evaluation_verdicts": "Is LEAN wired into the application contract",
    "lean_in_app_code": "Any LEAN verdict reference outside the DB migration",
    "uncertainty_method": "Row-level vs event-cluster bootstrap",
    "cluster_bootstrap_present": "Does bootstrapClusterMeanInterval exist",
    "promotion_policy": "Promotion gate values",
    "model_evidence": "trainingCutoff / calibrationReportId / implementationDigest",
    "batter_features": "Batter model feature sets",
    "bet_placement_code": "Any order-submission code (must stay absent)",
}


def run(cmd: list[str], cwd: Path | None = None, timeout: int = 300) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"
    except FileNotFoundError as exc:
        return 127, str(exc)


def branch_shas(repo: Path) -> dict[str, str]:
    code, out = run(["git", "ls-remote", "--heads", "origin"], cwd=repo)
    if code != 0:
        return {"_error": out.strip()[:400]}
    shas: dict[str, str] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].startswith("refs/heads/"):
            shas[parts[1].removeprefix("refs/heads/")] = parts[0]
    return shas


def test_count(repo: Path) -> dict[str, Any]:
    """Run the suite. Absent node_modules is reported, not silently installed."""
    if not (repo / "node_modules").exists():
        code, out = run(["npm", "install", "--no-audit", "--no-fund"], cwd=repo, timeout=300)
        if code != 0:
            return {"ran": False, "reason": "npm install failed"}
    code, out = run(["npm", "test"], cwd=repo, timeout=600)
    result: dict[str, Any] = {"ran": True, "exit_code": code}
    for field in ("tests", "pass", "fail"):
        match = re.search(rf"^# {field} (\d+)$", out, re.MULTILINE)
        if match:
            result[field] = int(match.group(1))
    return result


def read_text(repo: Path, rel: str) -> str:
    path = repo / rel
    return path.read_text(encoding="utf-8") if path.exists() else ""


def evaluation_verdicts(repo: Path) -> list[str]:
    text = read_text(repo, "src/audit/record-contract.js")
    match = re.search(r"EVALUATION_VERDICTS\s*=\s*Object\.freeze\(\[(.*?)\]\)", text, re.S)
    if not match:
        return []
    return re.findall(r'"([A-Z]+)"', match.group(1))


def grep_count(repo: Path, pattern: str, path: str = "src") -> int:
    code, out = run(
        ["grep", "-rlE", pattern, "--include=*.js", path], cwd=repo, timeout=60
    )
    if code not in (0, 1):
        return -1
    return len([line for line in out.splitlines() if line.strip()])


def registry(repo: Path) -> dict[str, Any]:
    raw = read_text(repo, "models/registry.json")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def collect(repo: Path) -> dict[str, Any]:
    reg = registry(repo)
    models = reg.get("models", [])

    code, head = run(["git", "rev-parse", "HEAD"], cwd=repo)
    verdicts = evaluation_verdicts(repo)

    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "head": head.strip() if code == 0 else None,
        "branch_shas": branch_shas(repo),
        "test_count": test_count(repo),
        "evaluation_verdicts": verdicts,
        "lean_wired_into_app": "LEAN" in verdicts,
        # The DB migration widened a CHECK constraint without touching code;
        # this counts genuine verdict-level references, not the unrelated
        # `candidate.lean` over/under side field.
        "lean_verdict_refs_in_src": grep_count(repo, r'verdict.*"LEAN"|"LEAN".*verdict'),
        "uncertainty_method": reg.get("promotionPolicy", {}).get(
            "requiredUncertaintyMethod"
        ),
        "cluster_bootstrap_present": bool(
            re.search(
                r"function bootstrapClusterMeanInterval",
                read_text(repo, "src/calibration/metrics.js"),
            )
        ),
        "promotion_policy": reg.get("promotionPolicy", {}),
        "model_evidence": [
            {
                "modelId": m.get("modelId"),
                "marketFamily": m.get("marketFamily"),
                "modelStatus": m.get("modelStatus"),
                "trainingCutoff": m.get("trainingCutoff"),
                "calibrationReportId": m.get("calibrationReportId"),
                "implementationDigest": (
                    m.get("calculationImplementation", {}).get("implementationDigest")
                ),
                "featureCount": len(m.get("featureSet", [])),
            }
            for m in models
        ],
        "batter_features": {
            m.get("marketFamily"): m.get("featureSet", [])
            for m in models
            if "batter" in str(m.get("marketFamily", ""))
        },
        # Must stay zero. A nonzero value is the single most urgent alert this
        # script can raise: it means executable betting code has appeared.
        "bet_placement_files": grep_count(
            repo, r"placeBet|submitBet|placeWager|executeBet|submitOrder"
        ),
    }


def summarise(state: dict[str, Any]) -> list[str]:
    """The standing findings, restated against current data."""
    lines: list[str] = []
    models = state.get("model_evidence", [])
    if models:
        no_cutoff = sum(1 for m in models if m.get("trainingCutoff") is None)
        no_calib = sum(1 for m in models if m.get("calibrationReportId") is None)
        no_digest = sum(1 for m in models if m.get("implementationDigest") is None)
        not_research = [
            m for m in models if m.get("modelStatus") != "research_only"
        ]
        lines.append(f"- models registered: {len(models)}")
        lines.append(f"- trainingCutoff unpopulated: {no_cutoff}/{len(models)}")
        lines.append(f"- calibrationReportId unpopulated: {no_calib}/{len(models)}")
        lines.append(f"- implementationDigest unpopulated: {no_digest}/{len(models)}")
        if not_research:
            lines.append(
                f"- !! MODEL STATUS CHANGED: {[m['modelId'] for m in not_research]}"
            )
        else:
            lines.append("- all models still research_only")

    lines.append(
        f"- uncertainty method: {state.get('uncertainty_method')}"
        f" (cluster bootstrap present: {state.get('cluster_bootstrap_present')})"
    )
    lines.append(
        f"- evaluation verdicts: {state.get('evaluation_verdicts')}"
        f" | LEAN wired: {state.get('lean_wired_into_app')}"
    )
    tests = state.get("test_count", {})
    if tests.get("ran"):
        lines.append(
            f"- tests: {tests.get('pass')} pass / {tests.get('fail')} fail"
            f" of {tests.get('tests')}"
        )
    placement = state.get("bet_placement_files", 0)
    if placement and placement > 0:
        lines.append(f"- !! BET PLACEMENT CODE DETECTED in {placement} file(s)")
    else:
        lines.append("- no bet-placement code present (expected)")
    return lines


MATERIAL_KEYS = (
    "evaluation_verdicts",
    "lean_wired_into_app",
    "uncertainty_method",
    "cluster_bootstrap_present",
    "promotion_policy",
    "model_evidence",
    "batter_features",
    "bet_placement_files",
)


def diff(previous: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Only material changes. Timestamps and SHAs alone are not news."""
    changes: list[str] = []
    for key in MATERIAL_KEYS:
        before, after = previous.get(key), current.get(key)
        if before != after:
            changes.append(f"### {key}\n- before: `{before}`\n- after:  `{after}`")

    old_branches = previous.get("branch_shas", {})
    new_branches = current.get("branch_shas", {})
    added = set(new_branches) - set(old_branches)
    moved = {
        b for b in set(new_branches) & set(old_branches)
        if new_branches[b] != old_branches[b]
    }
    if added:
        changes.append(f"### new branches\n- {sorted(added)}")
    if moved:
        changes.append(
            "### branches moved\n"
            + "\n".join(
                f"- `{b}`: {old_branches[b][:8]} -> {new_branches[b][:8]}"
                for b in sorted(moved)
            )
        )
    return changes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--previous", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not (args.repo / ".git").exists():
        print(f"ERROR: {args.repo} is not a git repository", file=sys.stderr)
        return 1

    run(["git", "fetch", "origin"], cwd=args.repo)
    state = collect(args.repo)

    print(f"# Bear Edge snapshot — {state['captured_at']}")
    print(f"\nHEAD: `{state['head']}`\n")
    print("## Standing findings, current values\n")
    print("\n".join(summarise(state)))

    if args.previous and args.previous.exists():
        try:
            prev = json.loads(args.previous.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prev = {}
        changes = diff(prev, state)
        print("\n## Change since previous snapshot\n")
        if changes:
            print("\n\n".join(changes))
        else:
            print("No material change.")
    else:
        print("\n## Change since previous snapshot\n\nNo previous snapshot supplied.")

    if args.out:
        args.out.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(f"\nState written to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
