from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.portfolio_optimizer.data import (
    download_price_history,
    load_benchmark_weights,
    load_price_history,
    load_sector_map,
    load_views,
)
from src.portfolio_optimizer.optimizer import (
    black_litterman_posterior_returns,
    compute_annualized_covariance,
    compute_annualized_returns,
    compute_benchmark_returns,
    compute_equilibrium_returns,
    compute_portfolio_metrics,
    compute_risk_contributions,
    compute_sector_exposures,
    compute_simple_returns,
    compute_view_uncertainty,
    estimate_risk_aversion,
    optimize_max_sharpe_portfolio,
    optimize_long_only_portfolio,
    save_matrix,
    save_series,
)
from src.portfolio_optimizer.reporting import (
    save_efficient_frontier_chart,
    save_executive_summary,
    save_return_chart,
    save_risk_contribution_chart,
    save_sector_exposure_chart,
    save_weight_chart,
)


def _parse_symbols(symbols_text: str) -> list[str]:
    symbols = [symbol.strip().upper() for symbol in symbols_text.split(",")]
    if not symbols or any(not symbol for symbol in symbols):
        raise ValueError("Symbols must be provided as a comma-separated list.")
    if len(set(symbols)) != len(symbols):
        raise ValueError("Symbols list contains duplicates.")
    return symbols


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Black-Litterman portfolio optimizer demo.")
    parser.add_argument("--symbols", required=True, help="Comma-separated ticker universe.")
    parser.add_argument("--start-date", required=True, help="Price history start date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", required=True, help="Price history end date in YYYY-MM-DD format.")
    parser.add_argument("--benchmark-weights", required=True, type=Path, help="Benchmark weights CSV path.")
    parser.add_argument("--sector-map", required=True, type=Path, help="Asset sector map CSV path.")
    parser.add_argument("--views", required=True, type=Path, help="Black-Litterman views CSV path.")
    parser.add_argument("--prices-output", required=True, type=Path, help="Downloaded price CSV output path.")
    parser.add_argument("--outputs-dir", required=True, type=Path, help="Directory for model outputs.")
    parser.add_argument("--risk-free-rate", required=True, type=float, help="Annual risk-free rate assumption.")
    parser.add_argument("--tau", required=True, type=float, help="Black-Litterman tau scalar.")
    parser.add_argument("--max-weight", required=True, type=float, help="Maximum long-only asset weight.")
    parser.add_argument("--iterations", required=True, type=int, help="Projected-gradient optimizer iterations.")
    return parser


def run_demo(args: argparse.Namespace) -> dict[str, str | float | int]:
    symbols = _parse_symbols(str(args.symbols))
    prices = download_price_history(symbols, str(args.start_date), str(args.end_date), args.prices_output)
    loaded_prices = load_price_history(args.prices_output)
    returns = compute_simple_returns(loaded_prices)
    annualized_covariance = compute_annualized_covariance(returns, 252)
    historical_returns = compute_annualized_returns(returns, 252)
    benchmark_weights = load_benchmark_weights(args.benchmark_weights, symbols)
    sector_map = load_sector_map(args.sector_map, symbols)
    benchmark_returns = compute_benchmark_returns(returns, benchmark_weights)
    risk_aversion = estimate_risk_aversion(benchmark_returns, float(args.risk_free_rate), 252)
    equilibrium_returns = compute_equilibrium_returns(annualized_covariance, benchmark_weights, risk_aversion)
    views = load_views(args.views, symbols)
    omega = compute_view_uncertainty(
        annualized_covariance,
        views["pick_matrix"],
        views["confidences"],
        float(args.tau),
    )
    posterior_returns = black_litterman_posterior_returns(
        annualized_covariance,
        equilibrium_returns,
        views["pick_matrix"],
        views["view_returns"],
        omega,
        float(args.tau),
    )
    utility_weights = optimize_long_only_portfolio(
        posterior_returns,
        annualized_covariance,
        risk_aversion,
        float(args.max_weight),
        int(args.iterations),
    )
    optimized_weights = optimize_max_sharpe_portfolio(
        posterior_returns,
        annualized_covariance,
        float(args.risk_free_rate),
        float(args.max_weight),
    )
    benchmark_historical_metrics = compute_portfolio_metrics(
        benchmark_weights,
        historical_returns,
        annualized_covariance,
        float(args.risk_free_rate),
    )
    optimized_historical_metrics = compute_portfolio_metrics(
        optimized_weights,
        historical_returns,
        annualized_covariance,
        float(args.risk_free_rate),
    )
    benchmark_model_metrics = compute_portfolio_metrics(
        benchmark_weights,
        posterior_returns,
        annualized_covariance,
        float(args.risk_free_rate),
    )
    optimized_model_metrics = compute_portfolio_metrics(
        optimized_weights,
        posterior_returns,
        annualized_covariance,
        float(args.risk_free_rate),
    )

    outputs_dir = Path(args.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    save_series(equilibrium_returns, outputs_dir / "equilibrium_returns.csv", "equilibrium_return")
    save_series(posterior_returns, outputs_dir / "posterior_returns.csv", "posterior_return")
    save_series(optimized_weights, outputs_dir / "optimized_weights.csv", "optimized_weight")
    save_series(utility_weights, outputs_dir / "utility_weights.csv", "utility_weight")
    save_matrix(annualized_covariance, outputs_dir / "annualized_covariance.csv")

    metrics = pd.DataFrame(
        {
            "benchmark_historical": benchmark_historical_metrics,
            "optimized_historical": optimized_historical_metrics,
            "benchmark_model": benchmark_model_metrics,
            "optimized_model": optimized_model_metrics,
        }
    )
    metrics.to_csv(outputs_dir / "portfolio_metrics.csv", index_label="metric", float_format="%.8f")
    risk_contributions = compute_risk_contributions(optimized_weights, annualized_covariance)
    risk_contributions.to_csv(outputs_dir / "risk_contributions.csv", index_label="asset", float_format="%.8f")
    sector_exposures = compute_sector_exposures(optimized_weights, benchmark_weights, sector_map)
    sector_exposures.to_csv(outputs_dir / "sector_exposures.csv", index_label="sector", float_format="%.8f")
    views["pick_matrix"].to_csv(outputs_dir / "view_matrix.csv", index_label="view_name", float_format="%.8f")
    omega.to_csv(outputs_dir / "view_uncertainty.csv", index_label="view_name", float_format="%.8f")
    save_weight_chart(optimized_weights, benchmark_weights, outputs_dir / "weights.png")
    save_return_chart(returns, optimized_weights, benchmark_weights, outputs_dir / "cumulative_returns.png")
    save_risk_contribution_chart(risk_contributions, outputs_dir / "risk_contributions.png")
    save_sector_exposure_chart(sector_exposures, outputs_dir / "sector_exposures.png")
    save_efficient_frontier_chart(
        posterior_returns,
        annualized_covariance,
        benchmark_weights,
        optimized_weights,
        float(args.risk_free_rate),
        float(args.max_weight),
        outputs_dir / "efficient_frontier.png",
    )

    summary: dict[str, str | float | int] = {
        "start_date": str(prices.index.min().date()),
        "end_date": str(prices.index.max().date()),
        "asset_count": len(symbols),
        "observation_count": int(prices.shape[0]),
        "risk_aversion": float(risk_aversion),
        "benchmark_historical_return": float(benchmark_historical_metrics.loc["expected_return"]),
        "optimized_historical_return": float(optimized_historical_metrics.loc["expected_return"]),
        "benchmark_model_return": float(benchmark_model_metrics.loc["expected_return"]),
        "optimized_model_return": float(optimized_model_metrics.loc["expected_return"]),
        "benchmark_model_volatility": float(benchmark_model_metrics.loc["volatility"]),
        "optimized_model_volatility": float(optimized_model_metrics.loc["volatility"]),
        "benchmark_model_sharpe_ratio": float(benchmark_model_metrics.loc["sharpe_ratio"]),
        "optimized_model_sharpe_ratio": float(optimized_model_metrics.loc["sharpe_ratio"]),
        "top_weight": str(optimized_weights.idxmax()),
        "top_weight_value": float(optimized_weights.max()),
        "top_risk_contributor": str(risk_contributions["percent_risk_contribution"].idxmax()),
        "top_risk_contribution": float(risk_contributions["percent_risk_contribution"].max()),
        "largest_active_sector": str(sector_exposures["active_weight"].abs().idxmax()),
        "largest_active_sector_weight": float(
            sector_exposures.loc[sector_exposures["active_weight"].abs().idxmax(), "active_weight"]
        ),
    }
    (outputs_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="ascii")
    save_executive_summary(summary, sector_exposures, risk_contributions, outputs_dir / "executive_summary.md")
    return summary


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    summary = run_demo(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
