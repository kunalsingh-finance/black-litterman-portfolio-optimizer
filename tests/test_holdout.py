"""Verify chronology and cash accounting rather than an optimizer winner."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.portfolio_optimizer.evaluation import buy_and_hold_path, realized_metrics, split_training_holdout
from src.portfolio_optimizer.run_demo import _build_parser, run_demo

ROOT = Path(__file__).resolve().parents[1]


def command_arguments(output, prices=None):
    return ["--symbols", "AAPL,MSFT,NVDA,AMZN,GOOGL,JPM,XOM,UNH,PG,JNJ",
            "--start-date", "2021-01-01", "--end-date", "2026-06-03",
            "--benchmark-weights", str(ROOT / "data/benchmark_weights.csv"),
            "--sector-map", str(ROOT / "data/sector_map.csv"), "--views", str(ROOT / "data/views.csv"),
            "--prices-input", str(prices or ROOT / "data/prices.csv"), "--outputs-dir", str(output),
            "--risk-free-rate", "0.04", "--tau", "0.05", "--max-weight", "0.25", "--iterations", "750",
            "--train-end", "2025-01-01", "--transaction-cost-bps", "10"]


def stub_charts(monkeypatch):
    def chart_stub(*args, **kwargs):
        path = next(value for value in args if isinstance(value, Path))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test chart stub")
        return path
    for name in ("save_weight_chart", "save_return_chart", "save_risk_contribution_chart",
                 "save_sector_exposure_chart", "save_efficient_frontier_chart", "save_holdout_chart"):
        monkeypatch.setattr(f"src.portfolio_optimizer.run_demo.{name}", chart_stub)


def test_buy_and_hold_drifts_and_accounts_for_entry_cost_exactly():
    prices = pd.DataFrame({"A": [100.0, 110.0, 120.0], "B": [100.0, 90.0, 100.0]},
                          index=pd.date_range("2025-01-01", periods=3))
    daily, drift, trades = buy_and_hold_path(prices, pd.Series({"A": 0.5, "B": 0.5}), 100.0)
    assert daily["gross_wealth"].tolist() == pytest.approx([1.0, 1.0, 1.1])
    assert daily["net_wealth"].tolist() == pytest.approx(np.array([1.0, 1.0, 1.1]) / 1.01)
    assert drift.iloc[1]["A"] == pytest.approx(0.55)
    assert drift.iloc[-1]["A"] == pytest.approx(0.6 / 1.1)
    assert trades["traded_notional"].sum() + trades["transaction_fee"].sum() == pytest.approx(1.0)
    assert trades["transaction_fee"].sum() == pytest.approx(0.01 / 1.01)
    assert pd.isna(daily["market_return"].iloc[0])
    assert daily["traded_notional"].iloc[1:].sum() == 0
    assert daily["transaction_fee"].iloc[1:].sum() == 0
    units = trades.set_index("asset")["fixed_adjusted_units"]
    np.testing.assert_allclose(prices.mul(units).sum(axis=1), daily["net_wealth"])
    metrics = realized_metrics(daily, 0.04)
    assert metrics["return_observations"] == 2
    assert metrics["net_total_return"] == pytest.approx(1.1 / 1.01 - 1)
    assert metrics["net_cagr_252"] == pytest.approx((1.1 / 1.01) ** 126 - 1)


def test_zero_cost_matches_gross_and_fees_do_not_improve_wealth():
    prices = pd.DataFrame({"A": [100.0, 95.0, 90.0]}, index=pd.date_range("2025-01-01", periods=3))
    free, _, _ = buy_and_hold_path(prices, pd.Series({"A": 1.0}), 0)
    paid, _, _ = buy_and_hold_path(prices, pd.Series({"A": 1.0}), 10)
    pd.testing.assert_series_equal(free["gross_wealth"], free["net_wealth"], check_names=False)
    assert (paid["net_wealth"] <= paid["gross_wealth"]).all()
    assert realized_metrics(paid, 0.04)["net_max_drawdown_from_initial_cash"] == pytest.approx(0.9 / 1.001 - 1)


@pytest.mark.parametrize("cost", [-1.0, float("nan"), float("inf")])
def test_invalid_costs_are_rejected(cost):
    prices = pd.DataFrame({"A": [100.0, 101.0]}, index=pd.date_range("2025-01-01", periods=2))
    with pytest.raises(ValueError, match="basis points"):
        buy_and_hold_path(prices, pd.Series({"A": 1.0}), cost)


@pytest.mark.parametrize("training_rows,holdout_rows,message", [(251, 30, "252 price"), (252, 20, "20 returns")])
def test_insufficient_training_or_holdout_is_rejected(training_rows, holdout_rows, message):
    index = pd.bdate_range("2023-01-01", periods=training_rows + holdout_rows)
    prices = pd.DataFrame({"A": np.arange(len(index)) + 100.0}, index=index)
    with pytest.raises(ValueError, match=message):
        split_training_holdout(prices, str(index[training_rows].date()))


def test_cutoff_is_exclusive_for_fit_and_entry_precedes_first_return():
    index = pd.bdate_range("2023-01-01", periods=273)
    prices = pd.DataFrame({"A": np.arange(len(index)) + 100.0}, index=index)
    training, holdout = split_training_holdout(prices, str(index[252].date()))
    assert len(training) == 252 and len(holdout) == 21
    assert training.index[-1] < holdout.index[0]
    daily, _, _ = buy_and_hold_path(holdout, pd.Series({"A": 1.0}), 0)
    assert pd.isna(daily["market_return"].iloc[0])
    assert daily["market_return"].iloc[1] == pytest.approx(prices.iloc[253, 0] / prices.iloc[252, 0] - 1)


def test_future_price_changes_cannot_change_fitted_inputs_or_weights(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("Cached evaluation tried to download")

    monkeypatch.setattr("src.portfolio_optimizer.run_demo.download_price_history", forbidden)
    stub_charts(monkeypatch)
    prices = pd.read_csv(ROOT / "data/prices.csv", parse_dates=["date"], index_col="date")
    changed = prices.copy()
    future = changed.index >= "2025-01-01"
    steps = np.arange(1, future.sum() + 1)[:, None]
    changed.loc[future] *= 1.0 + steps * np.linspace(0.001, 0.003, len(prices.columns))
    original_path, changed_path = tmp_path / "original.csv", tmp_path / "changed.csv"
    prices.to_csv(original_path)
    changed.to_csv(changed_path)
    outputs = [tmp_path / "first", tmp_path / "second"]
    for output, source in zip(outputs, (original_path, changed_path)):
        run_demo(_build_parser().parse_args(command_arguments(output, source)))
    for filename in ("optimized_weights.csv", "utility_weights.csv", "annualized_covariance.csv",
                     "equilibrium_returns.csv", "posterior_returns.csv", "view_uncertainty.csv", "portfolio_metrics.csv"):
        assert (outputs[0] / filename).read_bytes() == (outputs[1] / filename).read_bytes()
    manifests = [json.loads((output / "holdout/manifest.json").read_text()) for output in outputs]
    assert manifests[0]["training_prices_sha256"] == manifests[1]["training_prices_sha256"]
    assert manifests[0]["configuration_sha256"] == manifests[1]["configuration_sha256"]
    assert manifests[0]["source_csv_sha256"]["prices"] != manifests[1]["source_csv_sha256"]["prices"]
    assert manifests[0]["metrics"]["optimized"]["net_total_return"] != manifests[1]["metrics"]["optimized"]["net_total_return"]


@pytest.mark.parametrize("train_end,cost", [("NaT", 10), ("2025-01-01T12:00", 10),
                                            ("2025-01-01", -1), ("2025-01-01", float("nan"))])
def test_invalid_holdout_options_do_not_mutate_prior_outputs_or_download(monkeypatch, tmp_path, train_end, cost):
    output = tmp_path / "output"
    output.mkdir()
    prior = output / "summary.json"
    prior.write_text('{"prior": true}')
    args = _build_parser().parse_args(command_arguments(output))
    args.train_end, args.transaction_cost_bps = train_end, cost
    args.prices_input = None
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid options reached the downloader")
    monkeypatch.setattr("src.portfolio_optimizer.run_demo.download_price_history", forbidden)
    with pytest.raises(ValueError):
        run_demo(args)
    assert prior.read_text() == '{"prior": true}'
    assert list(output.iterdir()) == [prior]


def test_switch_to_legacy_mode_removes_only_owned_stale_holdout_artifacts(monkeypatch, tmp_path):
    output = tmp_path / "output"
    pack = output / "holdout"
    pack.mkdir(parents=True)
    (pack / "manifest.json").write_text(json.dumps({"evaluation": "chronological_buy_and_hold_scenario_holdout"}))
    for name in ("daily_wealth.csv", "entry_trades.csv", "weights.csv", "realized_metrics.csv", "cumulative_wealth.png"):
        (pack / name).write_text("prior generated artifact")
    note = pack / "personal_notes.txt"
    note.write_text("preserve user material")
    args = _build_parser().parse_args(command_arguments(output))
    args.train_end = None
    stub_charts(monkeypatch)
    summary = run_demo(args)
    assert summary["active_holdout_manifest"] is None
    assert summary["historical_evaluation"] == "in_sample_constant_weight_diagnostic"
    assert list(pack.iterdir()) == [note]
    assert note.read_text() == "preserve user material"
    assert "Chronological Scenario Holdout" not in (output / "executive_summary.md").read_text()


def test_documented_offline_holdout_cli_artifacts(tmp_path):
    output = tmp_path / "output"
    subprocess.run([sys.executable, "-m", "src.portfolio_optimizer.run_demo", *command_arguments(output)],
                   cwd=ROOT, check=True, capture_output=True, text=True)
    manifest = json.loads((output / "holdout/manifest.json").read_text())
    assert manifest["training_last_price_date"] == "2024-12-31"
    assert manifest["entry_close_date"] == "2025-01-02"
    assert manifest["first_heldout_return_date"] == "2025-01-03"
    assert manifest["last_heldout_price_date"] == "2026-06-02"
    assert manifest["heldout_return_observations"] >= 20
    for name, value in manifest["output_sha256"].items():
        assert hashlib.sha256((output / "holdout" / name).read_bytes()).hexdigest() == value
    for key, path in {"prices": "prices.csv", "views": "views.csv", "benchmark_weights": "benchmark_weights.csv", "sector_map": "sector_map.csv"}.items():
        assert manifest["source_csv_sha256"][key] == hashlib.sha256((ROOT / "data" / path).read_bytes()).hexdigest()
    assert "Undated scenario" in manifest["views"]
    assert "revised" in manifest["source_vintage"].lower()
    trades = pd.read_csv(output / "holdout/entry_trades.csv")
    for name in ("benchmark", "optimized"):
        portfolio = trades.loc[trades["portfolio"] == name]
        assert portfolio["traded_notional"].sum() + portfolio["transaction_fee"].sum() == pytest.approx(1.0)
    daily = pd.read_csv(output / "holdout/daily_wealth.csv")
    assert pd.isna(daily.loc[0, "optimized_market_return"])
    assert daily.loc[1:, "optimized_traded_notional"].sum() == 0
    weights = pd.read_csv(output / "optimized_weights.csv")
    assert weights["optimized_weight"].sum() == pytest.approx(1.0)
    assert weights["optimized_weight"].max() <= 0.25000001
    summary = (output / "executive_summary.md").read_text()
    assert "Chronological Scenario Holdout" in summary and "revised vintage" in summary
