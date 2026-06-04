from __future__ import annotations

from pathlib import Path

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


def save_return_chart(returns: pd.DataFrame, weights: pd.Series, benchmark_weights: pd.Series, output_path: Path) -> Path:
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
    axis.set_title("Historical Cumulative Return Backtest")
    axis.set_ylabel("Growth of $1")
    axis.set_xlabel("Date")
    axis.legend(loc="best")
    axis.figure.tight_layout()
    axis.figure.savefig(output_path, dpi=160)
    plt.close(axis.figure)
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
