import hashlib
from pathlib import Path

import pandas as pd
import pytest

from src.portfolio_optimizer.data import load_price_history
from src.portfolio_optimizer.run_demo import _build_parser, run_demo

ROOT = Path(__file__).resolve().parents[1]


def test_cached_demo_does_not_download_and_labels_in_sample(monkeypatch, tmp_path):
    def forbid_download(*args, **kwargs):
        raise AssertionError("Offline mode made a market-data request")
    monkeypatch.setattr("src.portfolio_optimizer.run_demo.download_price_history", forbid_download)
    args = _build_parser().parse_args([
        "--symbols", "AAPL,MSFT,NVDA,AMZN,GOOGL,JPM,XOM,UNH,PG,JNJ",
        "--start-date", "2021-01-01", "--end-date", "2026-06-03",
        "--benchmark-weights", str(ROOT / "data/benchmark_weights.csv"),
        "--sector-map", str(ROOT / "data/sector_map.csv"),
        "--views", str(ROOT / "data/views.csv"),
        "--prices-input", str(ROOT / "data/prices.csv"),
        "--outputs-dir", str(tmp_path), "--risk-free-rate", "0.04",
        "--tau", "0.05", "--max-weight", "0.25", "--iterations", "750",
    ])
    summary = run_demo(args)
    assert summary["historical_evaluation"] == "in_sample_constant_weight_diagnostic"
    assert summary["price_source_mode"] == "cached_adjusted_prices"
    assert summary["price_file_sha256"] == hashlib.sha256(args.prices_input.read_bytes()).hexdigest()
    assert "not a chronological out-of-sample backtest" in (tmp_path / "executive_summary.md").read_text()
    assert "optimized_in_sample" in pd.read_csv(tmp_path / "portfolio_metrics.csv").columns


@pytest.mark.parametrize("dates,values", [(["2024-01-01", "2024-01-01"], [100, 101]),
                                          (["2024-01-01", "2024-01-02"], [100, 0])])
def test_invalid_saved_prices_are_rejected(tmp_path, dates, values):
    path = tmp_path / "prices.csv"
    pd.DataFrame({"date": dates, "A": values}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_price_history(path)
