"""Leakage-free temporal backtest for the Problem 2 state model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.problem1.analysis import (
    add_maintenance_clock,
    envelope_decay,
    estimate_cycle_slopes,
    event_study,
    fit_seasonality,
)

from .model import (
    DEVICE_ORDER,
    FilterParams,
    load_daily_permeability,
    load_maintenance_records,
)
from .simulate import seasonal_effect


def _training_params(
    daily: pd.DataFrame,
    maintenance: pd.DataFrame,
    cutoff: pd.Timestamp,
    damage_share: float = 0.20,
) -> dict[str, FilterParams]:
    """Estimate all parameters using observations strictly before ``cutoff``."""
    train_daily = daily.loc[daily["date"] < cutoff].copy()
    train_maintenance = maintenance.loc[maintenance["date"] < cutoff].copy()
    clocked = add_maintenance_clock(train_daily, train_maintenance)
    season_frame, _, details = fit_seasonality(clocked)
    coefficient_rows = details.loc[details["table"] == "coefficients"]
    coefficients = coefficient_rows.set_index("term")["coefficient"].to_dict()
    slopes = estimate_cycle_slopes(season_frame)
    _, event_metrics, _ = event_study(season_frame, train_maintenance)
    _, envelopes = envelope_decay(season_frame, train_maintenance)

    positive_ed = envelopes.loc[
        envelopes["envelope_decline_per_year"] > 0,
        "envelope_decline_per_year",
    ]
    population_ed = float(positive_ed.median()) if len(positive_ed) else 10.0
    population_medium = float(
        event_metrics.loc[
            event_metrics["maintenance_type"] == "中维护",
            "counterfactual_gain_3d",
        ].median()
    )
    population_major = float(
        event_metrics.loc[
            event_metrics["maintenance_type"] == "大维护",
            "counterfactual_gain_3d",
        ].median()
    )
    population_decline = float(slopes["decline_rate_per_day"].median())
    params: dict[str, FilterParams] = {}

    for device in DEVICE_ORDER:
        dev_daily = season_frame.loc[season_frame["device"] == device].sort_values(
            "date"
        )
        dev_maint = train_maintenance.loc[
            train_maintenance["device"] == device
        ].sort_values("date")
        dev_slope = slopes.loc[
            slopes["device"] == device, "decline_rate_per_day"
        ]
        decline = (
            float(dev_slope.median()) if len(dev_slope) else population_decline
        )
        dev_envelope = envelopes.loc[
            envelopes["device"] == device, "envelope_decline_per_year"
        ]
        ed = float(dev_envelope.iloc[0]) if len(dev_envelope) else population_ed
        if not np.isfinite(ed) or ed <= 0:
            ed = population_ed
        ed = float(np.clip(ed, 2.0, 3.0 * population_ed))
        span_years = max(
            (dev_daily["date"].max() - dev_daily["date"].min()).days / 365.25,
            0.5,
        )
        medium_per_year = (
            (dev_maint["maintenance_type"] == "中维护").sum() / span_years
        )
        major_per_year = (
            (dev_maint["maintenance_type"] == "大维护").sum() / span_years
        )
        weight = medium_per_year + 3.0 * major_per_year
        medium_damage = ed * damage_share / weight if weight else 0.0
        alpha = ed * (1.0 - damage_share) / 365.25

        dev_events = event_metrics.loc[event_metrics["device"] == device]
        medium_values = dev_events.loc[
            dev_events["maintenance_type"] == "中维护",
            "counterfactual_gain_3d",
        ]
        major_values = dev_events.loc[
            dev_events["maintenance_type"] == "大维护",
            "counterfactual_gain_3d",
        ]
        medium_gain = (
            float(medium_values.median())
            if medium_values.notna().any()
            else population_medium
        )
        major_gain = (
            float(major_values.median())
            if major_values.notna().any()
            else population_major
        )
        sigma = float(dev_daily["model_residual"].std())
        recent_level = float(
            dev_daily.tail(30)["permeability_median"].median()
        )
        last_event = dev_maint.iloc[-1] if len(dev_maint) else None
        days_since = (
            float((cutoff - last_event["date"]).days)
            if last_event is not None
            else 30.0
        )
        mediums_since_major = 0
        for mtype in reversed(dev_maint["maintenance_type"].tolist()):
            if mtype == "大维护":
                break
            mediums_since_major += int(mtype == "中维护")
        params[device] = FilterParams(
            device=device,
            alpha=alpha,
            beta=max(decline - alpha, 0.01),
            seasonal_sin_1y=float(coefficients["sin_1y"]),
            seasonal_cos_1y=float(coefficients["cos_1y"]),
            seasonal_sin_2y=float(coefficients["sin_2y"]),
            seasonal_cos_2y=float(coefficients["cos_2y"]),
            medium_recovery=max(medium_gain, 1.0),
            major_recovery=max(major_gain, 1.0),
            medium_damage=max(medium_damage, 0.0),
            major_damage=max(3.0 * medium_damage, 0.0),
            sigma=max(sigma, 0.1),
            recent_level=recent_level,
            days_since_maintenance=days_since,
            mediums_since_major=mediums_since_major,
            last_maintenance_type=(
                str(last_event["maintenance_type"])
                if last_event is not None
                else "观测起点"
            ),
            historical_irreversible_decline_per_year=ed,
            damage_share_assumption=damage_share,
        )
    return params


def backtest_one_device(
    device: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    params: FilterParams,
    daily: pd.DataFrame,
    maintenance: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Simulate with actual future maintenance dates and training-only params."""
    epoch = pd.Timestamp("1899-12-30")
    start_ord = int((start_date - epoch).days)
    end_ord = int((end_date - epoch).days)
    pre = daily.loc[
        (daily["device"] == device)
        & (daily["date"] < start_date)
        & (daily["date"] >= start_date - pd.Timedelta(days=int(90))),
        ["date", "permeability_median"],
    ].dropna()
    if len(pre) >= 10:
        doys = pre["date"].dt.dayofyear.to_numpy(dtype=int)
        seasonal = np.array([seasonal_effect(day, params) for day in doys])
        adjusted = pre["permeability_median"].to_numpy(dtype=float) - seasonal
        C = float(np.percentile(adjusted, 85))
        F = max(C - adjusted[-1], 0.0)
    else:
        start_season = seasonal_effect(start_date.dayofyear, params)
        F = max(params.beta * params.days_since_maintenance, 0.0)
        C = params.recent_level + F - start_season

    events = maintenance.loc[
        (maintenance["device"] == device)
        & (maintenance["date"] >= start_date)
        & (maintenance["date"] <= end_date)
    ]
    event_dict = {
        int((row["date"] - epoch).days): row["maintenance_type"]
        for _, row in events.iterrows()
    }
    records = []
    for ordinal in range(start_ord, end_ord + 1):
        current_date = epoch + pd.Timedelta(days=int(ordinal))
        C = C - params.alpha + rng.normal(0, max(params.alpha, 0.001) * 0.5)
        F = max(F + params.beta + rng.normal(0, params.beta * 0.3), 0.0)
        predicted = (
            C
            - F
            + seasonal_effect(current_date.dayofyear, params)
            + rng.normal(0, params.sigma)
        )
        event_type = event_dict.get(ordinal)
        if event_type == "中维护":
            C -= params.medium_damage
            F = max(F - params.medium_recovery, 0.0)
        elif event_type == "大维护":
            C -= params.major_damage
            F = max(F - params.major_recovery, 0.0)
        records.append(
            {
                "device": device,
                "date": current_date,
                "predicted_P": predicted,
                "C": C,
                "F": F,
                "maintenance": event_type,
            }
        )
    result = pd.DataFrame(records)
    actual = daily.loc[
        daily["device"] == device, ["date", "permeability_median"]
    ].rename(columns={"permeability_median": "actual_P"})
    return result.merge(actual, on="date", how="left")


def run_backtest(
    tables_dir: Path,
    processed_dir: Path,
    backtest_start: str = "2025-04-10",
    backtest_end: str = "2026-04-10",
    n_runs: int = 100,
    seed: int = 2026,
) -> pd.DataFrame:
    del tables_dir  # retained for CLI compatibility
    daily = load_daily_permeability(processed_dir)
    maintenance = load_maintenance_records(processed_dir)
    start_date = pd.Timestamp(backtest_start)
    end_date = pd.Timestamp(backtest_end)
    params = _training_params(daily, maintenance, start_date)
    all_devices = []
    for device in DEVICE_ORDER:
        runs = []
        for index in range(n_runs):
            run = backtest_one_device(
                device,
                start_date,
                end_date,
                params[device],
                daily,
                maintenance,
                np.random.default_rng(seed + index * 1000 + int(device[1:])),
            )
            run["run"] = index
            runs.append(run)
        combined = pd.concat(runs, ignore_index=True)
        average = (
            combined.groupby(["device", "date"])
            .agg(
                predicted_P_mean=("predicted_P", "mean"),
                predicted_P_std=("predicted_P", "std"),
                predicted_P_q05=("predicted_P", lambda x: x.quantile(0.05)),
                predicted_P_q95=("predicted_P", lambda x: x.quantile(0.95)),
                actual_P=("actual_P", "first"),
                C_mean=("C", "mean"),
                F_mean=("F", "mean"),
            )
            .reset_index()
        )
        all_devices.append(average)
    return pd.concat(all_devices, ignore_index=True)


def compute_backtest_metrics(results: pd.DataFrame) -> pd.DataFrame:
    records = []
    for device, group in results.groupby("device"):
        valid = group.dropna(subset=["actual_P", "predicted_P_mean"])
        errors = valid["actual_P"] - valid["predicted_P_mean"]
        in_interval = (valid["actual_P"] >= valid["predicted_P_q05"]) & (
            valid["actual_P"] <= valid["predicted_P_q95"]
        )
        records.append(
            {
                "device": device,
                "n_days": len(valid),
                "MAE": float(errors.abs().mean()),
                "RMSE": float(np.sqrt(np.mean(errors**2))),
                "bias": float(errors.mean()),
                "coverage_90pct": float(in_interval.mean()),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Leakage-free Problem 2 backtest")
    parser.add_argument("--tables-dir", type=Path, default=Path("outputs/problem1/tables"))
    parser.add_argument("--processed-dir", type=Path, default=Path("outputs/problem1/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/problem2"))
    parser.add_argument("--start", default="2025-04-10")
    parser.add_argument("--end", default="2026-04-10")
    parser.add_argument("--n-runs", type=int, default=100)
    args = parser.parse_args()
    tables = args.output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    results = run_backtest(
        args.tables_dir,
        args.processed_dir,
        args.start,
        args.end,
        args.n_runs,
    )
    metrics = compute_backtest_metrics(results)
    metrics.to_csv(tables / "backtest_metrics.csv", index=False, encoding="utf-8-sig")
    results.to_csv(
        tables / "backtest_trajectories.csv", index=False, encoding="utf-8-sig"
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
