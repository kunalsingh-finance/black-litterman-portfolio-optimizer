from __future__ import annotations

import numpy as np
import pandas as pd

from src.portfolio_optimizer.optimizer import (
    black_litterman_posterior_returns,
    compute_equilibrium_returns,
    compute_view_uncertainty,
    optimize_long_only_portfolio,
    optimize_max_sharpe_portfolio,
)


def test_optimize_long_only_portfolio_respects_constraints() -> None:
    assets = ["A", "B", "C", "D"]
    expected_returns = pd.Series([0.10, 0.08, 0.06, 0.04], index=assets)
    covariance = pd.DataFrame(
        np.eye(4) * 0.04,
        index=assets,
        columns=assets,
    )

    weights = optimize_long_only_portfolio(expected_returns, covariance, 3.0, 0.40, 250)

    assert np.isclose(float(weights.sum()), 1.0)
    assert (weights >= 0.0).all()
    assert (weights <= 0.4000001).all()


def test_optimize_max_sharpe_portfolio_respects_constraints() -> None:
    assets = ["A", "B", "C", "D"]
    expected_returns = pd.Series([0.12, 0.09, 0.06, 0.03], index=assets)
    covariance = pd.DataFrame(
        np.eye(4) * 0.05,
        index=assets,
        columns=assets,
    )

    weights = optimize_max_sharpe_portfolio(expected_returns, covariance, 0.02, 0.45)

    assert np.isclose(float(weights.sum()), 1.0)
    assert (weights >= 0.0).all()
    assert (weights <= 0.4500001).all()


def test_black_litterman_relative_view_moves_posterior_returns() -> None:
    assets = ["A", "B"]
    covariance = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.03]],
        index=assets,
        columns=assets,
    )
    benchmark_weights = pd.Series([0.5, 0.5], index=assets)
    equilibrium_returns = compute_equilibrium_returns(covariance, benchmark_weights, 2.5)
    pick_matrix = pd.DataFrame([[1.0, -1.0]], index=["A over B"], columns=assets)
    view_returns = pd.Series([0.03], index=["A over B"])
    confidences = pd.Series([0.75], index=["A over B"])
    omega = compute_view_uncertainty(covariance, pick_matrix, confidences, 0.05)

    posterior = black_litterman_posterior_returns(
        covariance,
        equilibrium_returns,
        pick_matrix,
        view_returns,
        omega,
        0.05,
    )

    assert posterior.loc["A"] - posterior.loc["B"] > equilibrium_returns.loc["A"] - equilibrium_returns.loc["B"]
