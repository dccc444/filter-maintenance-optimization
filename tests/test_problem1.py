from __future__ import annotations

import numpy as np
import pandas as pd

from src.problem1.analysis import (
    _event_effects_for_one,
    add_maintenance_clock,
    aggregate_daily,
    build_maintenance_slope_change,
    estimate_effect_duration,
    normalize_device,
)


def test_normalize_device() -> None:
    assert normalize_device("A_1") == "A1"
    assert normalize_device("a10") == "A10"


def test_daily_aggregation_uses_median_and_coverage() -> None:
    hourly = pd.DataFrame(
        {
            "device": ["A1"] * 4,
            "time": pd.to_datetime(
                [
                    "2025-01-01 00:00",
                    "2025-01-01 01:00",
                    "2025-01-01 02:00",
                    "2025-01-01 03:00",
                ]
            ),
            "per": [10.0, 11.0, 12.0, 1000.0],
            "per_clean": [10.0, 11.0, 12.0, np.nan],
        }
    )
    daily = aggregate_daily(hourly)
    assert daily.loc[0, "permeability_median"] == 11.0
    assert daily.loc[0, "valid_hours"] == 3
    assert daily.loc[0, "hour_coverage"] == 3 / 24


def test_maintenance_clock_resets_on_event_date() -> None:
    daily = pd.DataFrame(
        {
            "device": ["A1"] * 3,
            "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
            "permeability_median": [10.0, 11.0, 10.0],
        }
    )
    maintenance = pd.DataFrame(
        {
            "device": ["A1"],
            "date": pd.to_datetime(["2025-01-02"]),
            "maintenance_type": ["中维护"],
        }
    )
    result = add_maintenance_clock(daily, maintenance)
    assert result["cycle_id"].tolist() == [0, 1, 1]
    assert result["days_since_maintenance"].tolist() == [0.0, 0.0, 1.0]


def test_counterfactual_event_gain_on_synthetic_data() -> None:
    dates = pd.date_range("2025-01-01", periods=70)
    event_date = pd.Timestamp("2025-02-01")
    relative = (dates - event_date).days.to_numpy()
    baseline = 100 - 0.2 * relative
    effect = np.where(relative >= 1, 15.0, 0.0)
    frame = pd.DataFrame(
        {
            "date": dates,
            "permeability_season_adjusted": baseline + effect,
        }
    )
    result = _event_effects_for_one(frame, event_date)
    assert result is not None
    _, metrics = result
    assert abs(metrics["counterfactual_gain_3d"] - 15.0) < 1e-8
    assert abs(metrics["effect_day30"] - 15.0) < 1e-8


def test_effect_duration_requires_persistent_crossing() -> None:
    curve = pd.DataFrame(
        {
            "relative_day": np.arange(1, 16),
            "effect": [10, 10, 9, 8, 1, 8, 7, 6, 1, 1, 1, 0.5, 0.5, 0.5, 0.5],
        }
    )
    result = estimate_effect_duration(
        curve, initial_gain=10.0, threshold_ratio=0.2, smooth_window=1
    )
    assert result["duration_status"] == "threshold_crossed"
    assert result["effective_duration_days"] == 9
    assert not result["duration_censored"]


def test_effect_duration_reports_right_censoring() -> None:
    curve = pd.DataFrame(
        {"relative_day": np.arange(1, 11), "effect": np.full(10, 8.0)}
    )
    result = estimate_effect_duration(curve, initial_gain=10.0)
    assert result["duration_status"] == "right_censored"
    assert result["effective_duration_days"] == 10
    assert result["duration_censored"]


def test_matched_slope_change_uses_equal_windows() -> None:
    dates = pd.date_range("2025-01-01", periods=80)
    event_date = dates[39]
    relative = (dates - event_date).days.to_numpy()
    values = np.where(relative < 0, 100 - 0.2 * relative, 115 - 0.1 * relative)
    season_frame = pd.DataFrame(
        {
            "device": "A1",
            "date": dates,
            "permeability_season_adjusted": values,
        }
    )
    maintenance = pd.DataFrame(
        {
            "device": ["A1"],
            "date": [event_date],
            "maintenance_type": ["中维护"],
        }
    )
    result = build_maintenance_slope_change(season_frame, maintenance)
    assert abs(result.loc[0, "pre_decline_rate"] - 0.2) < 1e-8
    assert abs(result.loc[0, "post_decline_rate"] - 0.1) < 1e-8
    assert abs(result.loc[0, "decline_rate_change"] + 0.1) < 1e-8
