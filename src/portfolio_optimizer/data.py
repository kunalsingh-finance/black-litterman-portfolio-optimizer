from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import pandas as pd
import yfinance as yf


class ViewInputs(TypedDict):
    pick_matrix: pd.DataFrame
    view_returns: pd.Series
    confidences: pd.Series


def download_price_history(
    symbols: list[str],
    start_date: str,
    end_date: str,
    output_path: Path,
) -> pd.DataFrame:
    if not symbols:
        raise ValueError("At least one symbol is required.")
    if len(set(symbols)) != len(symbols):
        raise ValueError("Symbol list contains duplicates.")

    raw_prices = yf.download(
        tickers=symbols,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=False,
    )
    if raw_prices.empty:
        raise ValueError("Yahoo Finance returned no price rows.")

    if isinstance(raw_prices.columns, pd.MultiIndex):
        if "Close" not in raw_prices.columns.get_level_values(0):
            raise ValueError("Yahoo Finance response does not include adjusted close prices.")
        prices = raw_prices["Close"].copy()
    else:
        if "Close" not in raw_prices.columns:
            raise ValueError("Yahoo Finance response does not include adjusted close prices.")
        prices = raw_prices[["Close"]].copy()
        prices.columns = symbols

    prices = prices.loc[:, symbols].dropna(how="any")
    if prices.empty:
        raise ValueError("Downloaded price history has no overlapping dates.")
    if prices.shape[0] < 252:
        raise ValueError("Downloaded price history needs at least 252 overlapping rows.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(output_path, index_label="date", float_format="%.6f")
    return prices


def load_price_history(price_path: Path) -> pd.DataFrame:
    prices = pd.read_csv(price_path, parse_dates=["date"], index_col="date")
    if prices.empty:
        raise ValueError(f"Price file is empty: {price_path}.")
    if prices.isna().any().any():
        raise ValueError(f"Price file contains missing values: {price_path}.")
    return prices.astype(float)


def load_benchmark_weights(weight_path: Path, assets: list[str]) -> pd.Series:
    weights_frame = pd.read_csv(weight_path)
    required_columns = {"asset", "benchmark_weight"}
    missing_columns = required_columns.difference(set(weights_frame.columns))
    if missing_columns:
        joined_columns = ", ".join(sorted(missing_columns))
        raise ValueError(f"Benchmark weight file is missing columns: {joined_columns}.")

    weights = weights_frame.set_index("asset")["benchmark_weight"].astype(float)
    missing_assets = [asset for asset in assets if asset not in weights.index]
    if missing_assets:
        joined_assets = ", ".join(missing_assets)
        raise ValueError(f"Benchmark weights are missing assets: {joined_assets}.")

    ordered_weights = weights.loc[assets]
    if (ordered_weights < 0.0).any():
        raise ValueError("Benchmark weights must be non-negative.")

    total_weight = float(ordered_weights.sum())
    if total_weight <= 0.0:
        raise ValueError("Benchmark weights must have a positive total.")
    return ordered_weights / total_weight


def load_sector_map(sector_map_path: Path, assets: list[str]) -> pd.Series:
    sector_frame = pd.read_csv(sector_map_path)
    required_columns = {"asset", "sector"}
    missing_columns = required_columns.difference(set(sector_frame.columns))
    if missing_columns:
        joined_columns = ", ".join(sorted(missing_columns))
        raise ValueError(f"Sector map file is missing columns: {joined_columns}.")

    sector_map = sector_frame.set_index("asset")["sector"].astype(str)
    missing_assets = [asset for asset in assets if asset not in sector_map.index]
    if missing_assets:
        joined_assets = ", ".join(missing_assets)
        raise ValueError(f"Sector map is missing assets: {joined_assets}.")

    ordered_sector_map = sector_map.loc[assets]
    if ordered_sector_map.str.strip().eq("").any():
        raise ValueError("Sector map contains blank sector labels.")
    return ordered_sector_map


def load_views(view_path: Path, assets: list[str]) -> ViewInputs:
    views_frame = pd.read_csv(view_path)
    required_columns = {"view_name", "asset", "coefficient", "view_return", "confidence"}
    missing_columns = required_columns.difference(set(views_frame.columns))
    if missing_columns:
        joined_columns = ", ".join(sorted(missing_columns))
        raise ValueError(f"Views file is missing columns: {joined_columns}.")

    unknown_assets = sorted(set(views_frame["asset"]).difference(set(assets)))
    if unknown_assets:
        joined_assets = ", ".join(unknown_assets)
        raise ValueError(f"Views file includes assets outside the universe: {joined_assets}.")

    view_names = list(dict.fromkeys(views_frame["view_name"].astype(str)))
    pick_matrix = pd.DataFrame(0.0, index=view_names, columns=assets)
    view_returns = pd.Series(index=view_names, dtype=float)
    confidences = pd.Series(index=view_names, dtype=float)

    for view_name in view_names:
        view_rows = views_frame.loc[views_frame["view_name"] == view_name]
        unique_returns = view_rows["view_return"].astype(float).unique()
        unique_confidences = view_rows["confidence"].astype(float).unique()
        if len(unique_returns) != 1:
            raise ValueError(f"View {view_name} has more than one view_return.")
        if len(unique_confidences) != 1:
            raise ValueError(f"View {view_name} has more than one confidence.")

        view_returns.loc[view_name] = float(unique_returns[0])
        confidences.loc[view_name] = float(unique_confidences[0])
        for row in view_rows.itertuples(index=False):
            pick_matrix.loc[view_name, str(row.asset)] = float(row.coefficient)

    if (confidences <= 0.0).any() or (confidences > 1.0).any():
        raise ValueError("View confidence values must be in the interval (0, 1].")
    if (pick_matrix.abs().sum(axis=1) == 0.0).any():
        raise ValueError("Each view must contain at least one non-zero coefficient.")

    return {
        "pick_matrix": pick_matrix,
        "view_returns": view_returns,
        "confidences": confidences,
    }
