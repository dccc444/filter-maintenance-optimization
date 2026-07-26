from __future__ import annotations

import numpy as np

from src.problem2.model import FilterParams, MaintenanceSchedule
from src.problem2.simulate import (
    expected_post_major_mean,
    generate_maintenance_dates,
    seasonal_effect,
    simulate_one_device,
)


def _params(**overrides) -> FilterParams:
    values = {
        "device": "A1",
        "alpha": 0.02,
        "beta": 0.2,
        "seasonal_sin_1y": 1.0,
        "seasonal_cos_1y": 2.0,
        "seasonal_sin_2y": 3.0,
        "seasonal_cos_2y": 4.0,
        "medium_recovery": 15.0,
        "major_recovery": 25.0,
        "medium_damage": 0.2,
        "major_damage": 0.6,
        "sigma": 1.0,
        "recent_level": 80.0,
        "days_since_maintenance": 0.0,
        "mediums_since_major": 0,
        "last_maintenance_type": "中维护",
        "historical_irreversible_decline_per_year": 10.0,
        "damage_share_assumption": 0.2,
    }
    values.update(overrides)
    return FilterParams(**values)


def _schedule(**overrides) -> MaintenanceSchedule:
    values = {
        "device": "A1",
        "medium_gap_mean": 10.0,
        "medium_gap_std": 0.0,
        "major_gap_mean": 10.0,
        "major_gap_std": 0.0,
        "medium_between_major": 2,
    }
    values.update(overrides)
    return MaintenanceSchedule(**values)


def test_seasonal_effect_uses_all_four_coefficients() -> None:
    params = _params()
    day = 100
    expected = (
        np.sin(2 * np.pi * day / 365.25)
        + 2 * np.cos(2 * np.pi * day / 365.25)
        + 3 * np.sin(4 * np.pi * day / 365.25)
        + 4 * np.cos(4 * np.pi * day / 365.25)
    )
    assert abs(seasonal_effect(day, params) - expected) < 1e-12


def test_schedule_has_no_post_major_off_by_one() -> None:
    events = generate_maintenance_dates(
        1000, _schedule(), np.random.default_rng(1), max_years=1
    )
    assert [event_type for _, event_type in events[:6]] == [
        "中维护",
        "中维护",
        "大维护",
        "中维护",
        "中维护",
        "大维护",
    ]


def test_first_gap_continues_current_cycle() -> None:
    events = generate_maintenance_dates(
        1000,
        _schedule(medium_gap_mean=20.0),
        np.random.default_rng(1),
        max_years=1,
        days_since_last=7,
    )
    assert events[0][0] == 1013


def test_post_major_condition_is_a_30_day_mean() -> None:
    params = _params(
        alpha=0.0,
        beta=1.0,
        seasonal_sin_1y=0.0,
        seasonal_cos_1y=0.0,
        seasonal_sin_2y=0.0,
        seasonal_cos_2y=0.0,
        major_damage=0.0,
        major_recovery=10.0,
    )
    value = expected_post_major_mean(50.0, 20.0, 1000, params, window_days=30)
    expected = np.mean([50.0 - (10.0 + day) for day in range(1, 31)])
    assert abs(value - expected) < 1e-12
    already_applied = expected_post_major_mean(
        50.0,
        10.0,
        1000,
        params,
        window_days=30,
        maintenance_already_applied=True,
    )
    assert abs(already_applied - expected) < 1e-12


def test_unfinished_simulation_is_right_censored() -> None:
    params = _params(
        alpha=0.0,
        beta=0.0,
        sigma=0.1,
        recent_level=100.0,
        medium_damage=0.0,
        major_damage=0.0,
        seasonal_sin_1y=0.0,
        seasonal_cos_1y=0.0,
        seasonal_sin_2y=0.0,
        seasonal_cos_2y=0.0,
    )
    result = simulate_one_device(
        params,
        _schedule(medium_gap_mean=1000.0, major_gap_mean=1000.0),
        np.random.default_rng(1),
        warmup_history=[100.0] * 365,
        max_years=1,
    )
    assert result["right_censored"]
    assert not result["event_observed"]
