"""Price a full strikeout ladder for one start, and check it against a book.

Run: python examples/strikeout_ladder.py
"""

from __future__ import annotations

from sweetbear.strikeouts import (
    OpponentProfile,
    PitcherProfile,
    build_strikeout_distribution,
)

# A strikeout pitcher (30% of batters faced) with a typical starter's workload,
# against a lineup that whiffs a bit more than league average.
pitcher = PitcherProfile(
    name="Sample Ace",
    strikeout_rate=0.30,
    expected_batters_faced=24.0,
    batters_faced_sd=3.5,
)
opponent = OpponentProfile(name="Sample Lineup", strikeout_rate=0.245)

dist = build_strikeout_distribution(
    pitcher, opponent, league_strikeout_rate=0.22, correlation_group="2026-07-30-GAME1"
)

print(dist.summary())

# A book's two-sided prices for part of the ladder. Both sides are required:
# de-vigging one side alone is impossible.
market = {
    4.5: (1.35, 3.15),
    5.5: (1.80, 2.00),
    6.5: (2.65, 1.48),
    7.5: (4.40, 1.20),
}

print("\n\nPriced against the book (positive EV marked with *):\n")
print("  line  side     model    fair     book   book fair      EV")
print("  " + "-" * 58)

quotes = dist.price_against_market(market)
for q in sorted(quotes, key=lambda x: (x.line, x.side)):
    star = "*" if q.has_edge else " "
    print(
        f"{star} {q.line:<5.1f} {q.side:<6} {q.conditional_probability:7.2%} "
        f"{q.fair_decimal_odds:7.2f} {q.market_decimal_odds:7.2f} "
        f"{q.market_fair_probability:9.2%} {q.expected_value:+7.2%}"
    )

edges = [q for q in quotes if q.has_edge]
print(f"\n{len(edges)} of {len(quotes)} quotes show positive expected value.")
print(
    "\nREAD THIS BEFORE GETTING EXCITED. The book prices above are invented for\n"
    "the demo, and the resulting edges are absurd -- a real edge is 1-4%, not\n"
    "140%. An edge that large never means free money. It means the model and\n"
    "the market disagree wildly, and the market is a live auction with millions\n"
    "of dollars behind it while the model is a few lines of arithmetic. When\n"
    "you see a number like this against real prices, the correct conclusion is\n"
    "that the model is broken -- wrong pitcher, stale workload estimate, or a\n"
    "line that has already moved."
)
print(
    "\nEvery quote above shares the correlation group "
    f"{dist.correlation_group!r}.\nBacktested, this entire ladder counts as ONE "
    "observation, not eight.\nThat is the difference between a real edge and a "
    "sample-size illusion."
)
