""""Monte Carlo simulation engine for filter lifetime prediction."""

from __future__ import annotations

import numpy as np

from .model import FilterParams, MaintenanceSchedule


def seasonal_effect(day_of_year: int, params: FilterParams) -> float:
    """Compute seasonal component S(t) for given day of year (1-365).

    Two-harmonic model matching Problem 1: annual + semiannual.
    """
    day = float(day_of_year)
    peak = params.seasonal_peak_day
    # Annual harmonic
    phase1 = 2 * np.pi * (day - peak) / 365.25
    annual = params.seasonal_amplitude * np.cos(phase1)
    # Semiannual harmonic
    phase2 = 4 * np.pi * (day - peak * 0.5) / 365.25
    semi = params.semiannual_amplitude * np.cos(phase2)
    return annual + semi


def generate_maintenance_dates(
    start_date: int,  # ordinal day
    schedule: MaintenanceSchedule,
    rng: np.random.Generator,
    max_years: int = 10,
) -> list[tuple[int, str]]:
    """Generate maintenance event dates following the fixed schedule pattern.

    Returns list of (day_ordinal, '中维护'/'大维护').
    """
    events: list[tuple[int, str]] = []
    current_day = start_date
    end_day = start_date + int(max_years * 365.25)

    medium_count = 0
    while current_day < end_day:
        # Determine if next maintenance should be medium or major
        if medium_count >= schedule.medium_between_major_mean:
            # Major maintenance
            interval = max(
                30,
                rng.normal(schedule.major_interval_mean, schedule.major_interval_std),
            )
            current_day += int(round(interval))
            events.append((current_day, "大维护"))
            medium_count = 0
        else:
            # Medium maintenance
            interval = max(
                14,
                rng.normal(
                    schedule.medium_interval_mean, schedule.medium_interval_std
                ),
            )
            current_day += int(round(interval))
            events.append((current_day, "中维护"))
        medium_count += 1

    return events


def simulate_one_device(
    params: FilterParams,
    schedule: MaintenanceSchedule,
    rng: np.random.Generator,
    prediction_start_ordinal: int = 46122,  # 2026-04-10 (Excel epoch)
    warmup_history: list[float] | None = None,
    return_trajectory: bool = False,
    debug: bool = False,
) -> dict:
    """Simulate one filter device from prediction start to end-of-life.

    warmup_history: last 365 days of actual daily permeability to
    pre-populate the sliding window (prevents cold-start bias).
    """
    from datetime import date, timedelta

    _epoch = date(1899, 12, 30)

    def _ordinal_to_doy(ordinal: int) -> int:
        """Convert Excel ordinal to day-of-year (1-366)."""
        d = _epoch + timedelta(days=ordinal)
        return d.timetuple().tm_yday

    # Initialize state at prediction start
    # Use 85th percentile of season-adjusted P from warmup history (robust to seasonal troughs)
    # This matches the backtest initialization method.
    day_of_year = _ordinal_to_doy(prediction_start_ordinal)
    S0 = seasonal_effect(day_of_year, params)

    if warmup_history and len(warmup_history) >= 30:
        # Estimate C as upper envelope of recent season-adjusted permeability
        recent = np.array(warmup_history[-90:])  # last ~90 days
        # Approximate seasonal effects for each day (simplified: use average doy offset)
        n = len(recent)
        doys = [(day_of_year - n + i - 1) % 365 + 1 for i in range(n)]
        S_arr = np.array([seasonal_effect(int(d), params) for d in doys])
        P_sa = recent - S_arr
        C = float(np.percentile(P_sa, 85))
        F = max(C - (recent[-1] - S_arr[-1]), 0.0)
    else:
        # Fallback: use recent_level parameter
        current_fouling = params.beta * params.days_since_maintenance
        C = params.recent_level + current_fouling - S0
        F = current_fouling

    # Generate maintenance schedule from prediction start
    maint_events = generate_maintenance_dates(
        prediction_start_ordinal, schedule, rng, max_years=15
    )

    # Tracking
    daily_P: list[float] = []
    daily_day: list[int] = []
    medium_count = 0
    major_count = 0
    maintenance_dates: list[int] = []

    # Note: daily C -= alpha already accounts for the full annual envelope decline ED.
    # No additional per-event damage is applied — alpha rate is the TOTAL irreversible loss.

    # Event pointer
    event_idx = 0
    event_dates = [e[0] for e in maint_events]
    event_types = [e[1] for e in maint_events]

    # Simulation loop
    day = prediction_start_ordinal
    max_days = 365 * 12  # 12 years max

    # Pre-populate 365-day window with historical data (cold-start fix)
    window_365: list[float] = []
    if warmup_history:
        window_365 = list(warmup_history[-365:])
    else:
        window_365 = []

    last_major_day = prediction_start_ordinal
    major_attempted_recently = False

    for step in range(max_days):
        doy = _ordinal_to_doy(day)
        S = seasonal_effect(doy, params)

        # State evolution — noise decomposition:
        #   aging_noise: proportional to aging rate (tiny, smooth decline)
        #   fouling_noise: proportional to fouling rate (moderate daily variation)
        #   obs_noise: the bulk of residual volatility (iid measurement/process noise)
        sigma_total = params.sigma
        aging_noise = rng.normal(0, max(params.alpha, 0.001) * 0.5)
        fouling_noise = rng.normal(0, params.beta * 0.3)
        obs_noise = rng.normal(0, sigma_total)

        # Irreversible aging: slow, small daily decline
        C = C - params.alpha + aging_noise

        # Reversible fouling: grows daily, can fluctuate (negative = brief cleaning)
        F = max(F + params.beta + fouling_noise, 0.0)

        # Observed permeability
        P = C - F + S + obs_noise

        daily_P.append(P)
        daily_day.append(day)

        # 365-day sliding window
        window_365.append(P)
        while len(window_365) > 365:
            window_365.pop(0)

        # Check if maintenance event occurs today
        if event_idx < len(event_dates) and day >= event_dates[event_idx]:
            mtype = event_types[event_idx]
            event_idx += 1
            maintenance_dates.append(day)

            if mtype == "中维护":
                medium_count += 1
                recovery = params.medium_recovery
                F = max(F - recovery * params.medium_retention, 0.0)
            else:  # 大维护
                major_count += 1
                last_major_day = day
                major_attempted_recently = True
                recovery = params.major_recovery or params.medium_recovery * 1.2
                F = max(F - recovery * (params.major_retention or 1.0), 0.0)

        # Lifetime check (after at least 365 days of data)
        if len(window_365) >= 365:
            annual_mean = sum(window_365) / len(window_365)

            # Condition 1: annual mean < 37
            if annual_mean < 37:
                # Condition 2: even major maintenance wouldn't recover
                # Simulate: if we did major maintenance now
                simulated_recovery = params.major_recovery or params.medium_recovery * 1.2
                simulated_F = max(F - simulated_recovery, 0.0)
                # Post-major P estimate (30-day window after repair)
                post_major_estimate = C - simulated_F + S
                if post_major_estimate < 37:
                    break

        day += 1

    # Compute results
    total_days = day - prediction_start_ordinal
    lifetime_ordinal = day

    # Convert ordinal to date (Excel epoch: 1899-12-30)
    epoch = date(1899, 12, 30)
    start_date_obj = epoch + timedelta(days=prediction_start_ordinal)
    end_date_obj = epoch + timedelta(days=lifetime_ordinal)

    result = {
        "device": params.device,
        "total_lifetime_days": total_days,
        "start_date": start_date_obj.isoformat(),
        "end_date": end_date_obj.isoformat(),
        "remaining_days": total_days,
        "medium_maintenance_count": medium_count,
        "major_maintenance_count": major_count,
        "final_annual_mean": float(sum(window_365[-365:]) / 365) if len(window_365) >= 365 else None,
        "final_C": float(C),
        "final_F": float(F),
    }
    if return_trajectory:
        result["daily_P"] = daily_P
        result["daily_day"] = daily_day
    return result


def monte_carlo_simulate(
    params: FilterParams,
    schedule: MaintenanceSchedule,
    n_runs: int = 1000,
    seed: int = 2026,
    warmup_history: list[float] | None = None,
    save_trajectories: int = 0,
) -> tuple[list[dict], list[dict] | None]:
    """Run Monte Carlo simulation for one device.

    If save_trajectories > 0, also save that many sample trajectories.
    Returns (results, trajectories_or_None).
    """
    rng = np.random.default_rng(seed)
    results = []
    trajectories = [] if save_trajectories > 0 else None

    for i in range(n_runs):
        run_seed = int(seed + i * 1000 + int(params.device[1:]))
        run_rng = np.random.default_rng(run_seed)

        # Perturb parameters for this run (parameter uncertainty)
        from copy import deepcopy

        perturbed = deepcopy(params)
        # Alpha: multiplicative noise (log-normal, tighter for long-horizon stability)
        perturbed.alpha = max(
            params.alpha * np.exp(run_rng.normal(0, 0.2)), 0.001
        )
        # Beta: multiplicative noise (log-normal, same paradigm as alpha)
        perturbed.beta = max(
            params.beta * np.exp(run_rng.normal(0, 0.15)), 0.001
        )
        # Medium recovery: additive noise
        perturbed.medium_recovery = max(
            params.medium_recovery + run_rng.normal(0, params.medium_recovery * 0.2),
            1.0,
        )
        # Major recovery
        if perturbed.major_recovery is not None:
            perturbed.major_recovery = max(
                perturbed.major_recovery
                + run_rng.normal(0, perturbed.major_recovery * 0.2),
                1.0,
            )

        result = simulate_one_device(
            perturbed,
            schedule,
            run_rng,
            warmup_history=warmup_history,
            return_trajectory=(save_trajectories > 0 and i < save_trajectories),
        )
        results.append(result)

        if save_trajectories > 0 and i < save_trajectories:
            trajectories.append(result)

    return results, trajectories


def summarise_results(
    results: list[dict], device: str
) -> dict:
    """Compute summary statistics from Monte Carlo runs."""
    lifetimes = np.array([r["total_lifetime_days"] for r in results])
    medium_counts = np.array([r["medium_maintenance_count"] for r in results])
    major_counts = np.array([r["major_maintenance_count"] for r in results])

    return {
        "device": device,
        "n_runs": len(results),
        "median_lifetime_days": float(np.median(lifetimes)),
        "mean_lifetime_days": float(np.mean(lifetimes)),
        "std_lifetime_days": float(np.std(lifetimes, ddof=1)),
        "ci95_low_days": float(np.quantile(lifetimes, 0.025)),
        "ci95_high_days": float(np.quantile(lifetimes, 0.975)),
        "ci50_low_days": float(np.quantile(lifetimes, 0.25)),
        "ci50_high_days": float(np.quantile(lifetimes, 0.75)),
        "median_medium_count": float(np.median(medium_counts)),
        "median_major_count": float(np.median(major_counts)),
        "start_date": results[0]["start_date"],
        "median_end_date": _ordinal_to_date_str(
            _date_to_ordinal(results[0]["start_date"])
            + int(np.median(lifetimes))
        ),
    }


def _date_to_ordinal(date_str: str) -> int:
    """Convert ISO date string to Excel ordinal."""
    from datetime import date

    d = date.fromisoformat(date_str)
    epoch = date(1899, 12, 30)
    return (d - epoch).days


def _ordinal_to_date_str(ordinal: int) -> str:
    """Convert Excel ordinal to ISO date string."""
    from datetime import date, timedelta

    epoch = date(1899, 12, 30)
    return (epoch + timedelta(days=ordinal)).isoformat()
