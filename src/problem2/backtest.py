"""Backtest: validate the dual-state model against historical data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .model import (
    DEVICE_ORDER,
    estimate_params,
    extract_maintenance_schedule,
    load_daily_permeability,
    load_seasonality_summary,
    load_warmup_history,
)
from .simulate import generate_maintenance_dates, seasonal_effect


def seasonal_effect_for_series(doys: np.ndarray, params) -> np.ndarray:
    """Vectorized seasonal effect for an array of day-of-year values."""
    amp = params.seasonal_amplitude
    peak = params.seasonal_peak_day
    phase = 2 * np.pi * (doys.astype(float) - peak) / 365.25
    return amp * np.cos(phase)


def backtest_one_device(
    device: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    params,
    schedule,
    daily: pd.DataFrame,
    maintenance: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Simulate from start_date to end_date and compare with actual data.

    Returns DataFrame with columns: date, actual_P, predicted_P, C, F, S, maintenance_type
    """
    from datetime import date, timedelta

    _epoch = date(1899, 12, 30)

    def _ordinal_to_doy(ordinal: int) -> int:
        d = _epoch + timedelta(days=ordinal)
        return d.timetuple().tm_yday

    def _date_to_ordinal(d: pd.Timestamp) -> int:
        return (d.date() - _epoch).days

    start_ord = _date_to_ordinal(start_date)
    end_ord = _date_to_ordinal(end_date)

    # Get device data
    dev_data = daily[
        (daily["device"] == device)
        & (daily["date"] >= start_date - pd.Timedelta(days=90))
        & (daily["date"] <= end_date)
    ].sort_values("date")

    # Get maintenance records up to start_date
    dev_maint = maintenance[
        (maintenance["device"] == device) & (maintenance["date"] <= start_date)
    ].sort_values("date")

    # Compute days since last maintenance at start_date
    if len(dev_maint) > 0:
        last_maint_date = dev_maint["date"].max()
        days_since = (start_date - last_maint_date).days
    else:
        days_since = 30  # fallback

    # Initialize state from actual data near start_date
    # Use season-adjusted permeability to estimate C and F robustly
    pre_start = dev_data[dev_data["date"] < start_date]

    if len(pre_start) >= 20:
        # Compute approximate season-adjusted values
        pre_dates = pre_start["date"]
        pre_values = pre_start["permeability_median"].dropna()

        if len(pre_values) >= 10:
            # Estimate C as the 90th percentile of (P - Ŝ) in pre-period
            # (capacity = near-peak season-adjusted permeability)
            doys = pre_dates.dt.dayofyear.to_numpy()
            # Approximate seasonal effect using simple harmonic
            S_approx = seasonal_effect_for_series(doys, params)
            P_sa = pre_values.to_numpy() - S_approx[:len(pre_values)]

            # C ≈ upper envelope of season-adjusted P
            C0 = float(np.percentile(P_sa, 85))
            # F ≈ C - P_sa at the most recent point
            recent_sa = P_sa[-1]
            F0 = max(C0 - recent_sa, 0.0)
            recent_P = pre_values.iloc[-1]
        else:
            recent_P = params.recent_level
            doy0 = _ordinal_to_doy(start_ord)
            S0 = seasonal_effect(doy0, params)
            F0 = params.beta * min(days_since, 90)
            C0 = recent_P + F0 - S0
    else:
        recent_P = params.recent_level
        doy0 = _ordinal_to_doy(start_ord)
        S0 = seasonal_effect(doy0, params)
        F0 = params.beta * min(days_since, 90)
        C0 = recent_P + F0 - S0

    # Generate maintenance schedule
    # For backtest: use ACTUAL historical maintenance dates, not random ones
    future_maint = maintenance[
        (maintenance["device"] == device)
        & (maintenance["date"] > start_date)
        & (maintenance["date"] <= end_date)
    ]
    # Build event dict with actual dates
    event_dict = {}
    for _, row in future_maint.iterrows():
        ord_val = _date_to_ordinal(row["date"])
        event_dict[ord_val] = row["maintenance_type"]

    # Simulate
    C, F = C0, F0
    day = start_ord
    records = []

    while day <= end_ord:
        doy = _ordinal_to_doy(day)
        S = seasonal_effect(doy, params)

        # Evolution
        aging_noise = rng.normal(0, max(params.alpha, 0.001) * 0.5)
        fouling_noise = rng.normal(0, params.beta * 0.3)
        obs_noise = rng.normal(0, params.sigma)

        C = C - params.alpha + aging_noise
        F = max(F + params.beta + fouling_noise, 0.0)
        P = C - F + S + obs_noise

        # Maintenance
        mtype = event_dict.get(day, None)
        if mtype:
            recovery = (
                params.medium_recovery
                if mtype == "中维护"
                else (params.major_recovery or params.medium_recovery * 1.2)
            )
            F = max(F - recovery, 0.0)

        current_date = _epoch + timedelta(days=day)
        records.append(
            {
                "date": current_date,
                "predicted_P": P,
                "C": C,
                "F": F,
                "S": S,
                "maintenance": mtype,
            }
        )
        day += 1

    sim_df = pd.DataFrame(records)
    sim_df["date"] = pd.to_datetime(sim_df["date"])

    # Merge with actual data
    actual = dev_data[["date", "permeability_median"]].copy()
    actual = actual.rename(columns={"permeability_median": "actual_P"})

    merged = sim_df.merge(actual, on="date", how="left")
    merged["device"] = device
    return merged


def run_backtest(
    tables_dir: Path,
    processed_dir: Path,
    backtest_start: str = "2025-04-10",
    backtest_end: str = "2026-04-10",
    n_runs: int = 100,
    seed: int = 2026,
) -> pd.DataFrame:
    """Run backtest for all devices, averaging over multiple MC runs."""
    seasonality = load_seasonality_summary(tables_dir)
    params = estimate_params(tables_dir, processed_dir, seasonality)
    schedules = extract_maintenance_schedule(processed_dir)
    daily = load_daily_permeability(processed_dir)
    maintenance = pd.read_csv(
        processed_dir / "maintenance_records.csv", parse_dates=["date"]
    )

    start_date = pd.Timestamp(backtest_start)
    end_date = pd.Timestamp(backtest_end)

    all_runs = []
    for device in DEVICE_ORDER:
        print(f"  Backtesting {device}...")
        p = params[device]
        s = schedules[device]

        device_runs = []
        for i in range(n_runs):
            run_rng = np.random.default_rng(seed + i * 1000)
            result = backtest_one_device(
                device, start_date, end_date, p, s, daily, maintenance, run_rng
            )
            result["run"] = i
            device_runs.append(result)

        # Average across runs
        all_device = pd.concat(device_runs, ignore_index=True)
        avg = (
            all_device.groupby(["device", "date"])
            .agg(
                predicted_P_mean=("predicted_P", "mean"),
                predicted_P_std=("predicted_P", "std"),
                predicted_P_q05=("predicted_P", lambda x: x.quantile(0.05)),
                predicted_P_q95=("predicted_P", lambda x: x.quantile(0.95)),
                actual_P=("actual_P", "first"),
                C_mean=("C", "mean"),
                F_mean=("F", "mean"),
                S_mean=("S", "mean"),
            )
            .reset_index()
        )
        all_runs.append(avg)

    return pd.concat(all_runs, ignore_index=True)


def compute_backtest_metrics(results: pd.DataFrame) -> pd.DataFrame:
    """Compute per-device error metrics."""
    records = []
    for device, group in results.groupby("device"):
        valid = group.dropna(subset=["actual_P", "predicted_P_mean"])
        if len(valid) < 10:
            continue
        errors = valid["actual_P"] - valid["predicted_P_mean"]
        mae = float(errors.abs().mean())
        rmse = float(np.sqrt((errors**2).mean()))
        # Coverage: fraction of actual values within 90% prediction interval
        in_interval = (valid["actual_P"] >= valid["predicted_P_q05"]) & (
            valid["actual_P"] <= valid["predicted_P_q95"]
        )
        coverage = float(in_interval.mean())
        # Bias
        bias = float(errors.mean())
        records.append(
            {
                "device": device,
                "n_days": len(valid),
                "MAE": mae,
                "RMSE": rmse,
                "bias": bias,
                "coverage_90pct": coverage,
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Backtest Problem 2 model")
    parser.add_argument("--tables-dir", type=Path, default=Path("outputs/problem1/tables"))
    parser.add_argument("--processed-dir", type=Path, default=Path("outputs/problem1/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/problem2"))
    parser.add_argument("--start", default="2025-04-10")
    parser.add_argument("--end", default="2026-04-10")
    parser.add_argument("--n-runs", type=int, default=100)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Backtest: {args.start} → {args.end} ({args.n_runs} MC runs)")
    results = run_backtest(
        args.tables_dir,
        args.processed_dir,
        args.start,
        args.end,
        args.n_runs,
    )

    metrics = compute_backtest_metrics(results)
    print("\n=== Backtest Metrics ===")
    print(
        f"{'Device':<8} {'n_days':>6} {'MAE':>8} {'RMSE':>8} {'Bias':>8} {'Coverage':>10}"
    )
    print("-" * 55)
    for _, row in metrics.iterrows():
        print(
            f"{row['device']:<8} {row['n_days']:>6.0f} "
            f"{row['MAE']:>8.2f} {row['RMSE']:>8.2f} "
            f"{row['bias']:>8.2f} {row['coverage_90pct']:>10.2f}"
        )

    # Save
    metrics.to_csv(
        args.output_dir / "tables" / "backtest_metrics.csv", index=False, encoding="utf-8-sig"
    )
    results.to_csv(
        args.output_dir / "tables" / "backtest_trajectories.csv", index=False, encoding="utf-8-sig"
    )
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
