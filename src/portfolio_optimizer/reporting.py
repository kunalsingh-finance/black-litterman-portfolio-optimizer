from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
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
