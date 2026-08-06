"""Price hits, total bases, and home runs for one batter-game, all from one
underlying plate-appearance distribution.

Run: python examples/batter_props.py
"""

from __future__ import annotations

from sweetbear.batters import (
    LEAGUE_PA_OUTCOME_RATES,
    BatterProfile,
    PitcherAllowedProfile,
    build_batter_distribution,
)

batter = BatterProfile(
    name="Sample Slugger",
    rates={
        "strikeout": 0.16,
        "walk": 0.10,
        "hbp": 0.01,
        "single": 0.16,
        "double": 0.06,
        "triple": 0.005,
        "home_run": 0.045,
        "out_in_play": 0.46,
    },
    expected_plate_appearances=4.3,
)
# A slightly tougher-than-average pitcher: more strikeouts, fewer home runs
# allowed, rest of the mass redistributed to keep the eight rates at 1.
opponent = PitcherAllowedProfile(
    name="Sample Ace",
    rates={
        "strikeout": 0.27,
        "walk": 0.075,
        "hbp": 0.009,
        "single": 0.13,
        "double": 0.040,
        "triple": 0.003,
        "home_run": 0.023,
        "out_in_play": 0.450,
    },
)

dist = build_batter_distribution(
    batter, opponent, correlation_group="2026-07-31-GAME1-Slugger"
)

print(dist.summary(stats=("hits", "total_bases", "home_runs", "walks", "strikeouts")))

market = {
    0.5: (1.35, 3.15),
    1.5: (2.30, 1.62),
}

print("\n\nHits, priced against a sample book (positive EV marked with *):\n")
print("  line  side     model    fair     book   book fair      EV")
print("  " + "-" * 58)
for q in sorted(dist.price_against_market("hits", market), key=lambda x: (x.line, x.side)):
    star = "*" if q.has_edge else " "
    print(
        f"{star} {q.line:<5.1f} {q.side:<6} {q.conditional_probability:7.2%} "
        f"{q.fair_decimal_odds:7.2f} {q.market_decimal_odds:7.2f} "
        f"{q.market_fair_probability:9.2%} {q.expected_value:+7.2%}"
    )

print(
    "\nHits, total bases, and home runs above all share the correlation group "
    f"{dist.correlation_group!r}.\nBacktested together, they count as ONE "
    "observation -- not three -- because they are three views\nof the same "
    "plate appearances, not three independent events.\n\n"
    "Runs and RBIs are deliberately not offered: they depend on baserunners "
    "and the rest\nof the lineup, which this engine does not model. Calling "
    "dist.pmf_for('runs') raises,\nrather than silently faking a number from "
    "isolated batter rates."
)
