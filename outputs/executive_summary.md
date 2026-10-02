# Executive Summary

## Objective

Build a reproducible Black-Litterman public-equity optimizer that combines benchmark-implied returns, transparent analyst views, and long-only max-Sharpe constraints.

## Run Setup

- Price window: 2021-01-04 to 2026-06-02
- Price source mode: cached_adjusted_prices
- Saved price SHA-256: 230c017bd49c03453e33ce5fa33bd571e5fe9f92fe7b58a4f3e6d7b688fa4f87
- Asset count: 10
- Daily observations: 1359
- Risk aversion estimate: 5.6620

## Key Results

- Benchmark model Sharpe ratio: 0.5016
- Optimized model Sharpe ratio: 0.5105
- Model Sharpe improvement: 0.0089
- Top optimized weight: AMZN at 18.55%
- Top risk contributor: AMZN at 28.84% of portfolio variance
- Largest active sector exposure: Information Technology at -9.35%

## Portfolio Interpretation

The optimized portfolio is positioned from Black-Litterman posterior returns rather than raw historical returns. The model compares benchmark-implied priors against transparent view assumptions, then solves a constrained long-only max-Sharpe allocation.

## Outputs To Review

- `outputs/optimized_weights.csv`
- `outputs/portfolio_metrics.csv`
- `outputs/sector_exposures.csv`
- `outputs/risk_contributions.csv`
- `outputs/efficient_frontier.png`
- `outputs/sector_exposures.png`
- `outputs/risk_contributions.png`

## Limitations

Historical metrics and cumulative growth are in-sample constant-weight diagnostics: the full price window determines covariance, risk aversion and final weights, which are then applied to that same window. They are not a chronological out-of-sample backtest or evidence that this allocation could have been traded at the start. Analyst views are undated demonstration assumptions.

This is a portfolio analytics demonstration, not investment advice. Views, benchmark weights, risk-free rate, tau, and constraints are demo assumptions. The workflow does not include transaction costs, taxes, liquidity, factor risk, or live production controls.
