# Executive Summary

## Objective

Build a reproducible Black-Litterman public-equity optimizer that combines benchmark-implied returns, transparent analyst views, and long-only max-Sharpe constraints.

## Run Setup

- Fitting price window: 2021-01-04 to 2024-12-31
- Price source mode: cached_adjusted_prices
- Saved price SHA-256: 230c017bd49c03453e33ce5fa33bd571e5fe9f92fe7b58a4f3e6d7b688fa4f87
- Asset count: 10
- Daily observations: 1005
- Risk aversion estimate: 5.6001

## Chronological Scenario Holdout

- Training cutoff (exclusive): 2025-01-01
- Entry at first holdout close: 2025-01-02
- Held-out return dates: 2025-01-03 to 2026-06-02
- Held-out return observations: 353
- Acquisition cost assumption: 10 basis points per acquired dollar for each portfolio
- Benchmark net total return: 30.77%
- Optimized net total return: 25.19%
- Optimized minus benchmark net total return: -5.58 percentage points

Training-only fitted weights enter at the first holdout close. Each portfolio starts with $1 cash, pays its acquisition fee, and holds fixed adjusted-price units with drifting weights. Terminal wealth is marked without an exit liquidation; no later turnover is assumed.

Inspect holdout/manifest.json, daily_wealth.csv, entry_trades.csv, weights.csv, realized_metrics.csv and cumulative_wealth.png. Configuration, source CSVs, training prices and outputs have SHA-256 records.

This is a chronological holdout conditional on frozen, undated scenario views. It does not establish these views were available historically. The selected current universe is not a point-in-time constituent/delisting dataset, and the adjusted-price snapshot is a revised vintage. One caller-specified cutoff is used without searching for a winning split.

## Training Model Results

- Benchmark model Sharpe ratio: 0.5022
- Optimized model Sharpe ratio: 0.5079
- Model Sharpe improvement: 0.0057
- Top optimized weight: AMZN at 17.53%
- Top risk contributor: AMZN at 27.77% of portfolio variance
- Largest active sector exposure: Consumer Staples at 9.27%

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

Historical metrics in portfolio_metrics.csv and cumulative_returns.png are in-sample constant-weight diagnostics for the fitting window. They are not a chronological out-of-sample backtest or evidence that this allocation could have been traded at the start. Analyst views are undated demonstration assumptions.

This is a portfolio analytics demonstration, not investment advice. Views, benchmark weights, risk-free rate, tau, and constraints are demo assumptions. The holdout uses an explicit acquisition-fee scenario when enabled; it does not model taxes, variable spreads, liquidity or production execution.
