from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change().dropna(how="any")
    if returns.empty:
        raise ValueError("Price history does not contain enough rows to compute returns.")
    return returns


def compute_annualized_covariance(returns: pd.DataFrame, trading_days: int) -> pd.DataFrame:
    covariance = returns.cov() * float(trading_days)
    if covariance.isna().any().any():
        raise ValueError("Annualized covariance contains missing values.")
    return covariance


def compute_annualized_returns(returns: pd.DataFrame, trading_days: int) -> pd.Series:
    annualized_returns = returns.mean() * float(trading_days)
    if annualized_returns.isna().any():
        raise ValueError("Annualized return vector contains missing values.")
    return annualized_returns


def compute_benchmark_returns(returns: pd.DataFrame, benchmark_weights: pd.Series) -> pd.Series:
    aligned_weights = benchmark_weights.loc[list(returns.columns)]
    return returns @ aligned_weights


def estimate_risk_aversion(benchmark_returns: pd.Series, risk_free_rate: float, trading_days: int) -> float:
    annualized_excess_return = float(benchmark_returns.mean() * float(trading_days) - risk_free_rate)
    annualized_variance = float(benchmark_returns.var() * float(trading_days))
    if annualized_variance <= 0.0:
        raise ValueError("Benchmark variance must be positive.")
    risk_aversion = annualized_excess_return / annualized_variance
    return max(risk_aversion, 1.0)


def compute_equilibrium_returns(
    covariance: pd.DataFrame,
    benchmark_weights: pd.Series,
    risk_aversion: float,
) -> pd.Series:
    aligned_weights = benchmark_weights.loc[list(covariance.columns)]
    implied_returns = float(risk_aversion) * covariance.to_numpy(dtype=float) @ aligned_weights.to_numpy(dtype=float)
    return pd.Series(implied_returns, index=covariance.columns, name="equilibrium_return")


def compute_view_uncertainty(
    covariance: pd.DataFrame,
    pick_matrix: pd.DataFrame,
    confidences: pd.Series,
    tau: float,
) -> pd.DataFrame:
    covariance_array = covariance.loc[pick_matrix.columns, pick_matrix.columns].to_numpy(dtype=float)
    pick_array = pick_matrix.to_numpy(dtype=float)
    view_variances = np.diag(pick_array @ (float(tau) * covariance_array) @ pick_array.T)
    confidence_values = confidences.loc[pick_matrix.index].to_numpy(dtype=float)
    uncertainty = view_variances * (1.0 - confidence_values) / confidence_values
    uncertainty = np.maximum(uncertainty, 1.0e-8)
    return pd.DataFrame(np.diag(uncertainty), index=pick_matrix.index, columns=pick_matrix.index)


def black_litterman_posterior_returns(
    covariance: pd.DataFrame,
    equilibrium_returns: pd.Series,
    pick_matrix: pd.DataFrame,
    view_returns: pd.Series,
    omega: pd.DataFrame,
    tau: float,
) -> pd.Series:
    assets = list(covariance.columns)
    sigma = covariance.loc[assets, assets].to_numpy(dtype=float)
    pi = equilibrium_returns.loc[assets].to_numpy(dtype=float)
    p_matrix = pick_matrix.loc[:, assets].to_numpy(dtype=float)
    q_vector = view_returns.loc[pick_matrix.index].to_numpy(dtype=float)
    omega_matrix = omega.loc[pick_matrix.index, pick_matrix.index].to_numpy(dtype=float)

    tau_sigma_inverse = np.linalg.inv(float(tau) * sigma)
    omega_inverse = np.linalg.inv(omega_matrix)
    left_side = tau_sigma_inverse + p_matrix.T @ omega_inverse @ p_matrix
    right_side = tau_sigma_inverse @ pi + p_matrix.T @ omega_inverse @ q_vector
    posterior = np.linalg.solve(left_side, right_side)
    return pd.Series(posterior, index=assets, name="posterior_return")


def _project_to_capped_simplex(values: np.ndarray, max_weight: float) -> np.ndarray:
    if max_weight <= 0.0:
        raise ValueError("max_weight must be positive.")
    if max_weight * float(len(values)) < 1.0:
        raise ValueError("max_weight is too low for the number of assets.")

    lower_bound = float(values.min() - max_weight)
    upper_bound = float(values.max())
    for _ in range(120):
        midpoint = (lower_bound + upper_bound) / 2.0
        projected = np.clip(values - midpoint, 0.0, max_weight)
        if float(projected.sum()) > 1.0:
            lower_bound = midpoint
        else:
            upper_bound = midpoint
    result = np.clip(values - upper_bound, 0.0, max_weight)
    return result / float(result.sum())


def optimize_long_only_portfolio(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_aversion: float,
    max_weight: float,
    iterations: int,
) -> pd.Series:
    assets = list(expected_returns.index)
    sigma = covariance.loc[assets, assets].to_numpy(dtype=float)
    mu = expected_returns.to_numpy(dtype=float)
    eigenvalues = np.linalg.eigvalsh(sigma)
    largest_eigenvalue = max(float(eigenvalues.max()), 1.0e-8)
    step_size = 0.5 / (float(risk_aversion) * largest_eigenvalue)
    weights = np.full(len(assets), 1.0 / float(len(assets)), dtype=float)

    for _ in range(iterations):
        gradient = mu - float(risk_aversion) * sigma @ weights
        weights = _project_to_capped_simplex(weights + step_size * gradient, max_weight)

    return pd.Series(weights, index=assets, name="optimized_weight")


def optimize_max_sharpe_portfolio(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float,
    max_weight: float,
) -> pd.Series:
    assets = list(expected_returns.index)
    if max_weight * float(len(assets)) < 1.0:
        raise ValueError("max_weight is too low for the number of assets.")

    mu = expected_returns.to_numpy(dtype=float)
    sigma = covariance.loc[assets, assets].to_numpy(dtype=float)
    initial_weights = np.full(len(assets), 1.0 / float(len(assets)), dtype=float)
    bounds = [(0.0, float(max_weight)) for _ in assets]
    constraints = [{"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)}]

    def negative_sharpe_ratio(weights: np.ndarray) -> float:
        expected_return = float(weights @ mu)
        volatility = float(np.sqrt(weights @ sigma @ weights))
        if volatility <= 0.0:
            raise ValueError("Portfolio volatility must be positive during optimization.")
        return -((expected_return - risk_free_rate) / volatility)

    result = minimize(
        negative_sharpe_ratio,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1.0e-12, "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"Max-Sharpe optimization failed: {result.message}.")

    optimized_weights = np.asarray(result.x, dtype=float)
    optimized_weights = optimized_weights / float(optimized_weights.sum())
    return pd.Series(optimized_weights, index=assets, name="optimized_weight")


def compute_portfolio_metrics(
    weights: pd.Series,
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float,
) -> pd.Series:
    assets = list(weights.index)
    weight_values = weights.to_numpy(dtype=float)
    expected_return = float(weight_values @ expected_returns.loc[assets].to_numpy(dtype=float))
    volatility = float(np.sqrt(weight_values @ covariance.loc[assets, assets].to_numpy(dtype=float) @ weight_values))
    sharpe_ratio = (expected_return - risk_free_rate) / volatility
    return pd.Series(
        {
            "expected_return": expected_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
        },
        name="portfolio_metrics",
    )


def compute_risk_contributions(weights: pd.Series, covariance: pd.DataFrame) -> pd.DataFrame:
    assets = list(weights.index)
    weight_values = weights.to_numpy(dtype=float)
    covariance_values = covariance.loc[assets, assets].to_numpy(dtype=float)
    portfolio_variance = float(weight_values @ covariance_values @ weight_values)
    if portfolio_variance <= 0.0:
        raise ValueError("Portfolio variance must be positive.")

    marginal_contribution = covariance_values @ weight_values
    component_variance = weight_values * marginal_contribution
    percent_contribution = component_variance / portfolio_variance
    return pd.DataFrame(
        {
            "weight": weight_values,
            "component_variance": component_variance,
            "percent_risk_contribution": percent_contribution,
        },
        index=assets,
    )


def compute_sector_exposures(
    optimized_weights: pd.Series,
    benchmark_weights: pd.Series,
    sector_map: pd.Series,
) -> pd.DataFrame:
    assets = list(optimized_weights.index)
    ordered_sector_map = sector_map.loc[assets]
    optimized_frame = pd.DataFrame(
        {
            "sector": ordered_sector_map,
            "optimized_weight": optimized_weights.loc[assets],
            "benchmark_weight": benchmark_weights.loc[assets],
        }
    )
    sector_exposures = optimized_frame.groupby("sector", sort=True).sum()
    sector_exposures["active_weight"] = (
        sector_exposures["optimized_weight"] - sector_exposures["benchmark_weight"]
    )
    return sector_exposures.loc[:, ["benchmark_weight", "optimized_weight", "active_weight"]]


def save_series(series: pd.Series, output_path: Path, value_name: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = series.rename(value_name).reset_index()
    frame.columns = ["asset", value_name]
    frame.to_csv(output_path, index=False, float_format="%.8f")
    return output_path


def save_matrix(matrix: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_path, index_label="asset", float_format="%.8f")
    return output_path
