"""Monte Carlo simulator for filter lifetime and maintenance policy analysis."""

from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta

import numpy as np

from .model import FilterParams, MaintenanceSchedule

EXCEL_EPOCH = date(1899, 12, 30)


def _ordinal_to_doy(ordinal: int) -> int:
    return (EXCEL_EPOCH + timedelta(days=int(ordinal))).timetuple().tm_yday


def seasonal_effect(day_of_year: int, params: FilterParams) -> float:
    """Evaluate the exact four-coefficient harmonic model from Problem 1."""
    day = float(day_of_year)
    return float(
        params.seasonal_sin_1y * np.sin(2 * np.pi * day / 365.25)
        + params.seasonal_cos_1y * np.cos(2 * np.pi * day / 365.25)
        + params.seasonal_sin_2y * np.sin(4 * np.pi * day / 365.25)
        + params.seasonal_cos_2y * np.cos(4 * np.pi * day / 365.25)
    )


def generate_maintenance_dates(
    start_date: int,
    schedule: MaintenanceSchedule,
    rng: np.random.Generator,
    max_years: int = 25,
    days_since_last: float = 0.0,
    mediums_since_major: int = 0,
) -> list[tuple[int, str]]:
    """Continue the observed fixed schedule from the current maintenance state."""
    events: list[tuple[int, str]] = []
    current_day = int(start_date)
    end_day = int(start_date + max_years * 365.25)
    medium_count = int(max(mediums_since_major, 0))
    first = True
    while current_day < end_day:
        is_major = medium_count >= schedule.medium_between_major
        if is_major:
            full_gap = max(
                14.0,
                float(rng.normal(schedule.major_gap_mean, schedule.major_gap_std)),
            )
            event_type = "大维护"
        else:
            full_gap = max(
                14.0,
                float(rng.normal(schedule.medium_gap_mean, schedule.medium_gap_std)),
            )
            event_type = "中维护"
        remaining = full_gap - days_since_last if first else full_gap
        gap = max(int(round(remaining)), 1)
        current_day += gap
        if current_day > end_day:
            break
        events.append((current_day, event_type))
        if event_type == "大维护":
            medium_count = 0
        else:
            medium_count += 1
        first = False
    return events


def expected_post_major_mean(
    C: float,
    F: float,
    day: int,
    params: FilterParams,
    window_days: int = 30,
    maintenance_already_applied: bool = False,
) -> float:
    """Expected mean permeability after a hypothetical major maintenance."""
    projected_C = C
    projected_F = F
    if not maintenance_already_applied:
        projected_C -= params.major_damage
        projected_F = max(projected_F - params.major_recovery, 0.0)
    values = []
    for offset in range(1, window_days + 1):
        projected_C -= params.alpha
        projected_F = max(projected_F + params.beta, 0.0)
        values.append(
            projected_C
            - projected_F
            + seasonal_effect(_ordinal_to_doy(day + offset), params)
        )
    return float(np.mean(values))


def simulate_one_device(
    params: FilterParams,
    schedule: MaintenanceSchedule,
    rng: np.random.Generator,
    prediction_start_ordinal: int = 46122,
    warmup_history: list[float] | None = None,
    return_trajectory: bool = False,
    threshold: float = 37.0,
    recovery_window_days: int = 30,
    max_years: int = 25,
    purchase_cost: float = 300.0,
    medium_cost: float = 3.0,
    major_cost: float = 12.0,
) -> dict:
    """Simulate one device until the two-condition end-of-life rule or censoring."""
    start_day = int(prediction_start_ordinal)
    start_doy = _ordinal_to_doy(start_day)
    S0 = seasonal_effect(start_doy, params)
    if warmup_history and len(warmup_history) >= 30:
        recent = np.asarray(warmup_history[-90:], dtype=float)
        n = len(recent)
        doys = [
            _ordinal_to_doy(start_day - n + index + 1) for index in range(n)
        ]
        season = np.array([seasonal_effect(day, params) for day in doys])
        adjusted = recent - season
        C = float(np.percentile(adjusted, 85))
        F = max(C - adjusted[-1], 0.0)
    else:
        F = max(params.beta * params.days_since_maintenance, 0.0)
        C = params.recent_level + F - S0

    events = generate_maintenance_dates(
        start_day,
        schedule,
        rng,
        max_years=max_years,
        days_since_last=params.days_since_maintenance,
        mediums_since_major=params.mediums_since_major,
    )
    event_idx = 0
    window_365 = list(warmup_history[-365:]) if warmup_history else []
    daily_p: list[float] = []
    daily_day: list[int] = []
    medium_count = 0
    major_count = 0
    terminated = False
    final_post_major_mean = np.nan
    day = start_day
    max_days = int(round(max_years * 365.25))

    for _ in range(max_days):
        major_today = False
        C = C - params.alpha + rng.normal(0, max(params.alpha, 0.001) * 0.5)
        F = max(F + params.beta + rng.normal(0, params.beta * 0.3), 0.0)
        P = (
            C
            - F
            + seasonal_effect(_ordinal_to_doy(day), params)
            + rng.normal(0, params.sigma)
        )
        daily_p.append(float(P))
        daily_day.append(day)
        window_365.append(float(P))
        if len(window_365) > 365:
            window_365.pop(0)

        while event_idx < len(events) and day >= events[event_idx][0]:
            event_type = events[event_idx][1]
            event_idx += 1
            if event_type == "中维护":
                medium_count += 1
                C -= params.medium_damage
                F = max(F - params.medium_recovery, 0.0)
            else:
                major_count += 1
                C -= params.major_damage
                F = max(F - params.major_recovery, 0.0)
                major_today = True

        if len(window_365) == 365 and float(np.mean(window_365)) < threshold:
            final_post_major_mean = expected_post_major_mean(
                C,
                F,
                day,
                params,
                window_days=recovery_window_days,
                maintenance_already_applied=major_today,
            )
            if final_post_major_mean < threshold:
                terminated = True
                break
        day += 1

    observed_days = int(day - start_day)
    if not terminated:
        observed_days = max_days
        day = start_day + max_days
    total_cost = purchase_cost + medium_count * medium_cost + major_count * major_cost
    years = max(observed_days / 365.25, 1 / 365.25)
    result = {
        "device": params.device,
        "total_lifetime_days": observed_days,
        "event_observed": terminated,
        "right_censored": not terminated,
        "start_date": (EXCEL_EPOCH + timedelta(days=start_day)).isoformat(),
        "end_date": (EXCEL_EPOCH + timedelta(days=day)).isoformat(),
        "medium_maintenance_count": medium_count,
        "major_maintenance_count": major_count,
        "final_annual_mean": (
            float(np.mean(window_365)) if len(window_365) == 365 else np.nan
        ),
        "post_major_30d_mean": final_post_major_mean,
        "mean_simulated_permeability": (
            float(np.mean(daily_p)) if daily_p else np.nan
        ),
        "final_C": float(C),
        "final_F": float(F),
        "total_cost": float(total_cost),
        "annualized_cost": float(total_cost / years),
        "threshold": threshold,
        "terminated_by_two_condition_rule": terminated,
    }
    if return_trajectory:
        result["daily_P"] = daily_p
        result["daily_day"] = daily_day
    return result


def monte_carlo_simulate(
    params: FilterParams,
    schedule: MaintenanceSchedule,
    n_runs: int = 1000,
    seed: int = 2026,
    warmup_history: list[float] | None = None,
    save_trajectories: int = 0,
    **simulation_options,
) -> tuple[list[dict], list[dict] | None]:
    results: list[dict] = []
    trajectories = [] if save_trajectories else None
    for index in range(n_runs):
        run_rng = np.random.default_rng(
            seed + index * 1000 + int(params.device[1:])
        )
        perturbed = deepcopy(params)
        shared_irreversible = np.exp(run_rng.normal(0, 0.20))
        perturbed.alpha = max(params.alpha * shared_irreversible, 0.0005)
        perturbed.beta = max(params.beta * np.exp(run_rng.normal(0, 0.15)), 0.001)
        perturbed.medium_recovery = max(
            params.medium_recovery * np.exp(run_rng.normal(0, 0.20)), 1.0
        )
        perturbed.major_recovery = max(
            params.major_recovery * np.exp(run_rng.normal(0, 0.20)), 1.0
        )
        perturbed.medium_damage = max(
            params.medium_damage * shared_irreversible * np.exp(run_rng.normal(0, 0.25)),
            0.0,
        )
        perturbed.major_damage = max(
            params.major_damage * shared_irreversible * np.exp(run_rng.normal(0, 0.25)),
            0.0,
        )
        result = simulate_one_device(
            perturbed,
            schedule,
            run_rng,
            warmup_history=warmup_history,
            return_trajectory=bool(save_trajectories and index < save_trajectories),
            **simulation_options,
        )
        results.append(result)
        if trajectories is not None and index < save_trajectories:
            trajectories.append(result)
    return results, trajectories


def _km_quantile(times: np.ndarray, observed: np.ndarray, quantile: float) -> float:
    """Kaplan--Meier lifetime quantile; NaN when follow-up is insufficient."""
    survival = 1.0
    target = 1.0 - quantile
    for time in sorted(np.unique(times)):
        at_risk = int(np.sum(times >= time))
        events = int(np.sum((times == time) & observed))
        if events:
            survival *= 1.0 - events / at_risk
        if survival <= target:
            return float(time)
    return np.nan


def summarise_results(results: list[dict], device: str) -> dict:
    times = np.array([r["total_lifetime_days"] for r in results], dtype=float)
    observed = np.array([r["event_observed"] for r in results], dtype=bool)
    medium_counts = np.array([r["medium_maintenance_count"] for r in results])
    major_counts = np.array([r["major_maintenance_count"] for r in results])
    median = _km_quantile(times, observed, 0.5)
    low = _km_quantile(times, observed, 0.025)
    high = _km_quantile(times, observed, 0.975)
    return {
        "device": device,
        "n_runs": len(results),
        "observed_eol_runs": int(observed.sum()),
        "right_censored_runs": int((~observed).sum()),
        "right_censored_share": float((~observed).mean()),
        "median_lifetime_days": median,
        "ci95_low_days": low,
        "ci95_high_days": high,
        "max_followup_days": float(times.max()),
        "median_medium_count": float(np.median(medium_counts)),
        "median_major_count": float(np.median(major_counts)),
        "median_annualized_cost": float(
            np.median([r["annualized_cost"] for r in results])
        ),
        "start_date": results[0]["start_date"],
        "median_end_date": (
            _ordinal_to_date_str(
                _date_to_ordinal(results[0]["start_date"]) + int(median)
            )
            if np.isfinite(median)
            else None
        ),
    }


def _date_to_ordinal(date_str: str) -> int:
    return (date.fromisoformat(date_str) - EXCEL_EPOCH).days


def _ordinal_to_date_str(ordinal: int) -> str:
    return (EXCEL_EPOCH + timedelta(days=int(ordinal))).isoformat()
