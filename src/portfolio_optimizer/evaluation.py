"""Chronological scenario evaluation using fixed adjusted-price units."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def validate_training_cutoff(train_end: str) -> pd.Timestamp:
    cutoff = pd.Timestamp(train_end)
    if pd.isna(cutoff) or cutoff.tzinfo is not None or cutoff != cutoff.normalize():
        raise ValueError("Training cutoff must be a valid calendar date.")
    return cutoff


def validate_transaction_cost_bps(transaction_cost_bps: float) -> float:
    cost_rate = float(transaction_cost_bps) / 10_000.0
    if not np.isfinite(cost_rate) or cost_rate < 0:
        raise ValueError("Transaction cost basis points must be finite and non-negative.")
    return cost_rate


def split_training_holdout(
    prices: pd.DataFrame, train_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The cutoff is exclusive for estimation and inclusive for holdout prices."""
    cutoff = validate_training_cutoff(train_end)
    training = prices.loc[prices.index < cutoff].copy()
    holdout = prices.loc[prices.index >= cutoff].copy()
    if len(training) < 252:
        raise ValueError("Training needs at least 252 price rows before the exclusive cutoff.")
    if len(holdout) < 21:
        raise ValueError("Holdout needs at least 20 returns after its entry-close price (21 price rows).")
    return training, holdout


def buy_and_hold_path(
    prices: pd.DataFrame, weights: pd.Series, transaction_cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fund once from $1 cash, charge acquisition fees, and never rebalance.

    These units use vendor adjusted prices, including embedded distribution
    adjustments; they are not broker share holdings or a separate cash ledger.
    Terminal wealth is marked, without an assumed liquidation or exit fee.
    """
    cost_rate = validate_transaction_cost_bps(transaction_cost_bps)
    if len(prices) < 2 or prices.index.hasnans or prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("Holdout prices need at least two unique increasing observations.")
    if prices.columns.has_duplicates or weights.index.has_duplicates or set(prices.columns) != set(weights.index):
        raise ValueError("Holdout prices and unique portfolio weights must have exactly the same assets.")
    ordered = weights.reindex(prices.columns).astype(float)
    if not np.isfinite(ordered.to_numpy()).all() or (ordered < 0).any() or not np.isclose(ordered.sum(), 1.0, atol=1e-8, rtol=0):
        raise ValueError("Holdout weights must be finite, non-negative and sum to one.")
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("Holdout prices must be finite and positive.")

    # Cash budget: acquired notional + fee_rate * acquired notional = $1.
    acquired_notional = 1.0 / (1.0 + cost_rate)
    fee = cost_rate * acquired_notional
    price_ratios = prices.div(prices.iloc[0])
    asset_values = price_ratios.mul(ordered)
    gross_wealth = asset_values.sum(axis=1)
    net_wealth = gross_wealth * acquired_notional
    drift_weights = asset_values.div(gross_wealth, axis=0)
    daily = pd.DataFrame({"gross_wealth": gross_wealth, "net_wealth": net_wealth,
                          "market_return": gross_wealth.pct_change(fill_method=None),
                          "traded_notional": 0.0, "transaction_fee": 0.0}, index=prices.index)
    # Entry is at this close; no return from the training close is captured.
    daily.iloc[0, daily.columns.get_loc("traded_notional")] = acquired_notional
    daily.iloc[0, daily.columns.get_loc("transaction_fee")] = fee
    trades = pd.DataFrame({"asset": prices.columns, "date": str(prices.index[0].date()),
                           "target_weight": ordered.to_numpy(), "adjusted_entry_price": prices.iloc[0].to_numpy(),
                           "fixed_adjusted_units": (acquired_notional * ordered / prices.iloc[0]).to_numpy(),
                           "traded_notional": (acquired_notional * ordered).to_numpy(),
                           "transaction_fee": (fee * ordered).to_numpy()})
    return daily, drift_weights, trades


def realized_metrics(daily: pd.DataFrame, risk_free_rate: float) -> dict[str, Any]:
    """Fees affect total return/CAGR/drawdown, not the market-return Sharpe."""
    if not np.isfinite(risk_free_rate):
        raise ValueError("Risk-free rate assumption must be finite.")
    market_returns = daily["market_return"].iloc[1:]
    count = len(market_returns)
    volatility = float(market_returns.std(ddof=1) * np.sqrt(252)) if count > 1 else 0.0
    gross_end, net_end = float(daily["gross_wealth"].iloc[-1]), float(daily["net_wealth"].iloc[-1])
    gross = pd.Series([1.0, *daily["gross_wealth"].tolist()])
    net = pd.Series([1.0, *daily["net_wealth"].tolist()])
    return {"return_observations": count,
            "gross_total_return": gross_end - 1.0, "net_total_return": net_end - 1.0,
            "gross_cagr_252": gross_end ** (252.0 / count) - 1.0,
            "net_cagr_252": net_end ** (252.0 / count) - 1.0,
            "annualized_market_volatility": volatility,
            "gross_market_sharpe_excluding_entry_fee": (float(market_returns.mean()) * 252 - risk_free_rate) / volatility if volatility > 0 else None,
            "gross_max_drawdown": float((gross / gross.cummax() - 1.0).min()),
            "net_max_drawdown_from_initial_cash": float((net / net.cummax() - 1.0).min()),
            "entry_target_allocation": 1.0,
            "actual_traded_notional_over_initial_cash": float(daily["traded_notional"].sum()),
            "entry_transaction_fee_over_initial_cash": float(daily["transaction_fee"].sum()),
            "subsequent_traded_notional": float(daily["traded_notional"].iloc[1:].sum())}


def export_holdout(
    output_dir: Path, training_prices: pd.DataFrame, holdout_prices: pd.DataFrame,
    optimized_weights: pd.Series, benchmark_weights: pd.Series,
    transaction_cost_bps: float, risk_free_rate: float, configuration: dict[str, Any],
    source_hashes: dict[str, str], price_source_mode: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    daily_frames, trade_frames, weight_frames, metrics = [], [], [], {}
    for name, weights in (("benchmark", benchmark_weights), ("optimized", optimized_weights)):
        daily, drift, trades = buy_and_hold_path(holdout_prices, weights, transaction_cost_bps)
        daily_frames.append(daily.add_prefix(f"{name}_"))
        trade_frames.append(trades.assign(portfolio=name))
        weight_frames.append(pd.DataFrame({f"{name}_entry_weight": weights,
                                          f"{name}_end_weight": drift.iloc[-1]}))
        metrics[name] = realized_metrics(daily, risk_free_rate)
    combined = pd.concat(daily_frames, axis=1)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_dir / "daily_wealth.csv", index_label="date", float_format="%.12f")
    pd.concat(trade_frames, ignore_index=True).to_csv(output_dir / "entry_trades.csv", index=False, float_format="%.12f")
    pd.concat(weight_frames, axis=1).to_csv(output_dir / "weights.csv", index_label="asset", float_format="%.12f")
    pd.DataFrame.from_dict(metrics, orient="index").to_csv(output_dir / "realized_metrics.csv", index_label="portfolio", float_format="%.12f")
    config_bytes = json.dumps(configuration, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    manifest = {"evaluation": "chronological_buy_and_hold_scenario_holdout",
                "configuration": configuration, "configuration_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "source_csv_sha256": source_hashes, "price_source_mode": price_source_mode,
                "training_prices_sha256": hashlib.sha256(training_prices.to_csv(index_label="date", float_format="%.12f").encode("utf-8")).hexdigest(),
                "training_first_price_date": str(training_prices.index[0].date()),
                "training_last_price_date": str(training_prices.index[-1].date()),
                "training_price_rows": len(training_prices),
                "entry_close_date": str(holdout_prices.index[0].date()),
                "first_heldout_return_date": str(holdout_prices.index[1].date()),
                "last_heldout_price_date": str(holdout_prices.index[-1].date()),
                "heldout_return_observations": len(holdout_prices) - 1,
                "timing": "Fit only before the exclusive cutoff; enter at first holdout close; returns begin at next observed close",
                "holdings": "Fixed adjusted-price units, drifting weights; distributions embedded in vendor adjusted-price convention",
                "cost_model": "Same $1 initial cash per portfolio; entry notional=1/(1+cost_rate); fees charged on acquired notional; no later trades or exit liquidation",
                "turnover_definition": "Sum of absolute acquired notional divided by initial cash; entry-only purchases, not annualized half-turnover",
                "views": "Undated scenario assumptions frozen for this run; no evidence these views were available at the historical cutoff",
                "universe": "Current selected universe; not a point-in-time constituent or delisting dataset",
                "source_vintage": "Revised adjusted-price snapshot, not an archived data vintage available at the training cutoff",
                "split_selection": "One caller-specified cutoff; no automatic split search or winner selection",
                "risk_free_rate": "Constant annual assumption; market Sharpe uses arithmetic annual excess return and excludes entry fee",
                "annualization": "252 trading observations; CAGR includes entry fee over observed holding intervals",
                "metrics": metrics,
                "net_terminal_return_difference_optimized_minus_benchmark": metrics["optimized"]["net_total_return"] - metrics["benchmark"]["net_total_return"]}
    manifest["output_sha256"] = {name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
                                 for name in ("daily_wealth.csv", "entry_trades.csv", "weights.csv", "realized_metrics.csv")}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return manifest, combined


def remove_owned_holdout_pack(outputs_dir: Path) -> None:
    """A successful legacy-mode run must not present a prior holdout as current."""
    output_root = outputs_dir.resolve()
    folder = outputs_dir / "holdout"
    if not folder.exists():
        return
    if folder.resolve().parent != output_root:
        raise ValueError("Holdout folder must remain inside the requested output directory.")
    manifest_path = folder / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("evaluation") != "chronological_buy_and_hold_scenario_holdout":
        return
    # Only these generated filenames belong to this workflow. Preserve other files.
    owned = [folder / name for name in ("daily_wealth.csv", "entry_trades.csv", "weights.csv",
                                       "realized_metrics.csv", "cumulative_wealth.png", "manifest.json")]
    if any(path.exists() and path.resolve().parent != folder.resolve() for path in owned):
        raise ValueError("Owned holdout files must remain inside their output pack.")
    for path in owned:
        path.unlink(missing_ok=True)
