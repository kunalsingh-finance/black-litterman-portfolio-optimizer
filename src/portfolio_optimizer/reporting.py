from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_weight_chart(weights: pd.Series, benchmark_weights: pd.Series, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(
        {
            "Benchmark": benchmark_weights.loc[weights.index],
            "Optimized": weights,
        }
    )
    axis = comparison.plot(kind="bar", figsize=(11, 6), width=0.78)
    axis.set_title("Benchmark vs Black-Litterman Optimized Weights")
    axis.set_ylabel("Portfolio weight")
    axis.set_xlabel("Asset")
    axis.legend(loc="best")
    axis.figure.tight_layout()
    axis.figure.savefig(output_path, dpi=160)
    plt.close(axis.figure)
    return output_path


def save_return_chart(returns: pd.DataFrame, weights: pd.Series, benchmark_weights: pd.Series, output_path: Path, *, training_only: bool = False) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    optimized_returns = returns.loc[:, weights.index] @ weights
    benchmark_returns = returns.loc[:, benchmark_weights.index] @ benchmark_weights
    cumulative = pd.DataFrame(
        {
            "Benchmark": (1.0 + benchmark_returns).cumprod(),
            "Optimized": (1.0 + optimized_returns).cumprod(),
        }
    )
    axis = cumulative.plot(figsize=(11, 6))
    label = "Training In-sample" if training_only else "In-sample"
    axis.set_title(f"{label} Constant-weight Return Diagnostic\nWeights estimated using the same fitting window")
    axis.set_ylabel("Growth of $1")
    axis.set_xlabel("Date")
    axis.legend(loc="best")
    axis.figure.tight_layout()
    axis.figure.savefig(output_path, dpi=160)
    plt.close(axis.figure)
    return output_path


def save_holdout_chart(daily: pd.DataFrame, output_path: Path, transaction_cost_bps: float) -> Path:
    figure, axis = plt.subplots(figsize=(11, 6))
    for name, color in (("benchmark", "#1f77b4"), ("optimized", "#d62728")):
        axis.plot(daily.index, daily[f"{name}_gross_wealth"], color=color, linestyle="--", alpha=0.55, label=f"{name.title()} gross")
        axis.plot(daily.index, daily[f"{name}_net_wealth"], color=color, label=f"{name.title()} net entry fee")
    axis.set_title(f"Chronological Holdout: Frozen Scenario Allocation\nFirst-close entry, fixed adjusted-price units; {transaction_cost_bps:g} bps acquisition fee")
    axis.set_ylabel("Marked wealth per $1 initial cash (no exit liquidation)")
    axis.set_xlabel("Holdout date")
    axis.legend(loc="best")
    figure.text(0.5, 0.015, "Undated scenario views | selected current universe | revised price vintage | no historical execution claim", ha="center", fontsize=9)
    figure.tight_layout(rect=[0, 0.045, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def save_risk_contribution_chart(risk_contributions: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_contributions = risk_contributions.sort_values("percent_risk_contribution", ascending=False)
    axis = sorted_contributions["percent_risk_contribution"].plot(kind="bar", figsize=(11, 6), color="#2a6f97")
    axis.set_title("Optimized Portfolio Risk Contribution")
    axis.set_ylabel("Percent of portfolio variance")
    axis.set_xlabel("Asset")
    axis.figure.tight_layout()
    axis.figure.savefig(output_path, dpi=160)
    plt.close(axis.figure)
    return output_path


def save_sector_exposure_chart(sector_exposures: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exposure_columns = ["benchmark_weight", "optimized_weight"]
    axis = sector_exposures.loc[:, exposure_columns].plot(kind="bar", figsize=(11, 6), width=0.76)
    axis.set_title("Benchmark vs Optimized Sector Exposure")
    axis.set_ylabel("Portfolio weight")
    axis.set_xlabel("Sector")
    axis.legend(["Benchmark", "Optimized"], loc="best")
    axis.figure.tight_layout()
    axis.figure.savefig(output_path, dpi=160)
    plt.close(axis.figure)
    return output_path


def save_executive_summary(
    summary: dict[str, str | float | int],
    sector_exposures: pd.DataFrame,
    risk_contributions: pd.DataFrame,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    largest_active_sector = str(sector_exposures["active_weight"].abs().idxmax())
    largest_active_value = float(sector_exposures.loc[largest_active_sector, "active_weight"])
    top_risk_asset = str(risk_contributions["percent_risk_contribution"].idxmax())
    top_risk_value = float(risk_contributions.loc[top_risk_asset, "percent_risk_contribution"])
    optimized_sharpe = float(summary["optimized_model_sharpe_ratio"])
    benchmark_sharpe = float(summary["benchmark_model_sharpe_ratio"])
    sharpe_change = optimized_sharpe - benchmark_sharpe
    has_holdout = "holdout_manifest" in summary

    lines = [
        "# Executive Summary",
        "",
        "## Objective",
        "",
        "Build a reproducible Black-Litterman public-equity optimizer that combines benchmark-implied returns, transparent analyst views, and long-only max-Sharpe constraints.",
        "",
        "## Run Setup",
        "",
        f"- Fitting price window: {summary['start_date']} to {summary['end_date']}",
        f"- Price source mode: {summary['price_source_mode']}",
        f"- Saved price SHA-256: {summary['price_file_sha256']}",
        f"- Asset count: {summary['asset_count']}",
        f"- Daily observations: {summary['observation_count']}",
        f"- Risk aversion estimate: {float(summary['risk_aversion']):.4f}",
        "",
        "## Training Model Results" if has_holdout else "## Key Results",
        "",
        f"- Benchmark model Sharpe ratio: {benchmark_sharpe:.4f}",
        f"- Optimized model Sharpe ratio: {optimized_sharpe:.4f}",
        f"- Model Sharpe improvement: {sharpe_change:.4f}",
        f"- Top optimized weight: {summary['top_weight']} at {float(summary['top_weight_value']):.2%}",
        f"- Top risk contributor: {top_risk_asset} at {top_risk_value:.2%} of portfolio variance",
        f"- Largest active sector exposure: {largest_active_sector} at {largest_active_value:.2%}",
        "",
        "## Portfolio Interpretation",
        "",
        "The optimized portfolio is positioned from Black-Litterman posterior returns rather than raw historical returns. The model compares benchmark-implied priors against transparent view assumptions, then solves a constrained long-only max-Sharpe allocation.",
        "",
        "## Outputs To Review",
        "",
        "- `outputs/optimized_weights.csv`",
        "- `outputs/portfolio_metrics.csv`",
        "- `outputs/sector_exposures.csv`",
        "- `outputs/risk_contributions.csv`",
        "- `outputs/efficient_frontier.png`",
        "- `outputs/sector_exposures.png`",
        "- `outputs/risk_contributions.png`",
        "",
        "## Limitations",
        "",
        "Historical metrics in portfolio_metrics.csv and cumulative_returns.png are in-sample constant-weight diagnostics for the fitting window. They are not a chronological out-of-sample backtest or evidence that this allocation could have been traded at the start. Analyst views are undated demonstration assumptions.",
        "",
        "This is a portfolio analytics demonstration, not investment advice. Views, benchmark weights, risk-free rate, tau, and constraints are demo assumptions. The holdout uses an explicit acquisition-fee scenario when enabled; it does not model taxes, variable spreads, liquidity or production execution.",
        "",
    ]
    if has_holdout:
        insert_at = lines.index("## Training Model Results")
        lines[insert_at:insert_at] = [
            "## Chronological Scenario Holdout", "",
            f"- Training cutoff (exclusive): {summary['training_cutoff_exclusive']}",
            f"- Entry at first holdout close: {summary['holdout_entry_close_date']}",
            f"- Held-out return dates: {summary['holdout_first_return_date']} to {summary['holdout_last_price_date']}",
            f"- Held-out return observations: {summary['holdout_return_observations']}",
            f"- Acquisition cost assumption: {float(summary['holdout_transaction_cost_bps']):g} basis points per acquired dollar for each portfolio",
            f"- Benchmark net total return: {float(summary['holdout_benchmark_net_total_return']):.2%}",
            f"- Optimized net total return: {float(summary['holdout_optimized_net_total_return']):.2%}",
            f"- Optimized minus benchmark net total return: {100 * float(summary['holdout_net_return_difference']):.2f} percentage points",
            "", "Training-only fitted weights enter at the first holdout close. Each portfolio starts with $1 cash, pays its acquisition fee, and holds fixed adjusted-price units with drifting weights. Terminal wealth is marked without an exit liquidation; no later turnover is assumed.",
            "", "Inspect holdout/manifest.json, daily_wealth.csv, entry_trades.csv, weights.csv, realized_metrics.csv and cumulative_wealth.png. Configuration, source CSVs, training prices and outputs have SHA-256 records.",
            "", "This is a chronological holdout conditional on frozen, undated scenario views. It does not establish these views were available historically. The selected current universe is not a point-in-time constituent/delisting dataset, and the adjusted-price snapshot is a revised vintage. One caller-specified cutoff is used without searching for a winning split.", "",
        ]
    output_path.write_text("\n".join(lines), encoding="ascii")
    return output_path


def save_efficient_frontier_chart(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    benchmark_weights: pd.Series,
    optimized_weights: pd.Series,
    risk_free_rate: float,
    max_weight: float,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    assets = list(expected_returns.index)
    rng = np.random.default_rng(42)
    sampled_returns: list[float] = []
    sampled_volatilities: list[float] = []
    sampled_sharpes: list[float] = []
    covariance_values = covariance.loc[assets, assets].to_numpy(dtype=float)
    expected_values = expected_returns.loc[assets].to_numpy(dtype=float)

    attempts = 0
    while len(sampled_returns) < 3000 and attempts < 20000:
        attempts += 1
        weights = rng.dirichlet(np.ones(len(assets)))
        if float(weights.max()) > max_weight:
            continue
        portfolio_return = float(weights @ expected_values)
        portfolio_volatility = float(np.sqrt(weights @ covariance_values @ weights))
        sampled_returns.append(portfolio_return)
        sampled_volatilities.append(portfolio_volatility)
        sampled_sharpes.append((portfolio_return - risk_free_rate) / portfolio_volatility)

    if not sampled_returns:
        raise ValueError("Could not sample feasible portfolios for the efficient frontier chart.")

    benchmark = benchmark_weights.loc[assets].to_numpy(dtype=float)
    optimized = optimized_weights.loc[assets].to_numpy(dtype=float)
    benchmark_return = float(benchmark @ expected_values)
    benchmark_volatility = float(np.sqrt(benchmark @ covariance_values @ benchmark))
    optimized_return = float(optimized @ expected_values)
    optimized_volatility = float(np.sqrt(optimized @ covariance_values @ optimized))

    figure, axis = plt.subplots(figsize=(11, 6))
    scatter = axis.scatter(
        sampled_volatilities,
        sampled_returns,
        c=sampled_sharpes,
        cmap="viridis",
        alpha=0.45,
        s=16,
    )
    axis.scatter([benchmark_volatility], [benchmark_return], color="#1f77b4", marker="D", s=80, label="Benchmark")
    axis.scatter([optimized_volatility], [optimized_return], color="#d62728", marker="*", s=150, label="Optimized")
    axis.set_title("Feasible Portfolio Set: Black-Litterman Expected Return vs Risk")
    axis.set_xlabel("Annualized volatility")
    axis.set_ylabel("Black-Litterman expected return")
    axis.legend(loc="best")
    figure.colorbar(scatter, ax=axis, label="Sharpe ratio")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path
