"""End-to-end walkthrough on synthetic data.

Run with ``python examples/end_to_end.py``.

Two scenarios are simulated against the same machinery. In one the model is
noisier than the book and has no real edge; in the other it sees something the
book does not. Nothing in the pipeline is told which is which. The point of the
exercise is that the output separates them *before* the profit does -- closing
line value turns negative or positive hundreds of bets before the bankroll
curve is statistically distinguishable from noise.
"""

from __future__ import annotations

import numpy as np

from sweetbear import (
    Bet,
    IsotonicCalibrator,
    brier_score,
    devig,
    expected_calibration_error,
    kelly_stake,
    log_loss,
    odds,
    run_backtest,
    walk_forward_calibrate,
)

N = 6000
BOOK_MARGIN = 1.0476  # a standard -110 / -110 two-way book


def simulate(
    open_noise: float, close_noise: float, model_noise: float, seed: int
) -> list[Bet]:
    """Build a settled bet history from a known ground truth.

    All three parameters are standard deviations in log-odds space around the
    true probability. You bet into the *opening* line and are graded against
    the *closing* line, which is sharper because money has arrived in between.
    Beating the close therefore requires being sharper than the opening price
    -- which is exactly the real-world condition, and the reason CLV is worth
    measuring at all.
    """
    rng = np.random.default_rng(seed)
    logit_true = rng.normal(0.0, 0.9, size=N)
    true_p = 1.0 / (1.0 + np.exp(-logit_true))

    def priced(noise: float) -> np.ndarray:
        p = 1.0 / (1.0 + np.exp(-(logit_true + rng.normal(0.0, noise, size=N))))
        return np.clip(p, 0.05, 0.95)

    open_p = priced(open_noise)
    offered = 1.0 / (np.stack([open_p, 1.0 - open_p], axis=1) * BOOK_MARGIN)

    close_p = priced(close_noise)
    closing = 1.0 / (np.stack([close_p, 1.0 - close_p], axis=1) * BOOK_MARGIN)
    closing_fair = devig(closing, method="shin")[:, 0]

    model_p = 1.0 / (1.0 + np.exp(-(logit_true + rng.normal(0.0, model_noise, size=N))))
    outcomes = (rng.uniform(size=N) < true_p).astype(int)

    return [
        Bet(
            timestamp=i,
            probability=float(model_p[i]),
            decimal_odds=float(offered[i, 0]),
            outcome=int(outcomes[i]),
            closing_fair_probability=float(closing_fair[i]),
        )
        for i in range(N)
    ]


def probability_quality(tag: str, bets: list[Bet]) -> None:
    y = np.array([b.outcome for b in bets], dtype=float)
    p = np.array([b.probability for b in bets])
    print(
        f"  {tag:<14} brier={brier_score(y, p):.4f}  "
        f"logloss={log_loss(y, p):.4f}  ece={expected_calibration_error(y, p):.4f}"
    )


def run_scenario(
    title: str, open_noise: float, close_noise: float, model_noise: float, seed: int
) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")
    sharper = "model" if model_noise < open_noise else "book"
    print(
        f"opening line noise={open_noise}  closing={close_noise}  "
        f"model={model_noise}  ({sharper} is sharper than the open)\n"
    )

    bets = simulate(open_noise, close_noise, model_noise, seed)
    calibrated = walk_forward_calibrate(
        bets, IsotonicCalibrator, min_train=1000, refit_every=250
    )

    print("probability quality")
    probability_quality("raw", bets)
    probability_quality("recalibrated", calibrated)

    print("\nbacktest (recalibrated, quarter Kelly, 1% minimum edge)")
    result = run_backtest(
        calibrated, staking=kelly_stake(0.25, cap=0.05), min_edge=0.01
    )
    print("  " + result.summary().replace("\n", "\n  "))

    if result.mean_clv is not None:
        print(
            f"\n  CLV said {result.mean_clv:+.2%}; the bankroll delivered "
            f"{result.yield_:+.2%}."
        )


def main() -> None:
    market = odds.american_to_decimal([-110, -110])
    print(f"reference book: overround={odds.overround(market):.4f}  "
          f"vig={odds.vig(market):.2%}")

    run_scenario(
        "SCENARIO 1 -- sharp book, mediocre model (no real edge)",
        open_noise=0.06,
        close_noise=0.04,
        model_noise=0.35,
        seed=20260730,
    )
    run_scenario(
        "SCENARIO 2 -- informed model, beatable opening line (real edge)",
        open_noise=0.20,
        close_noise=0.14,
        model_noise=0.10,
        seed=20260730,
    )


if __name__ == "__main__":
    main()
