""""Problem 2: Dual-state filter degradation model and lifetime prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEVICE_ORDER = [f"A{i}" for i in range(1, 11)]
RANDOM_SEED = 2026


@dataclass
class FilterParams:
    """Estimated parameters for one filter device."""

    device: str
    # Irreversible aging rate (per day)
    alpha: float
    # Reversible fouling growth rate (per day)
    beta: float
    # Seasonal amplitude (half peak-to-trough)
    seasonal_amplitude: float
    # Seasonal peak day-of-year
    seasonal_peak_day: int
    # Semiannual harmonic amplitude (from Problem 1 seasonality_summary)
    semiannual_amplitude: float
    # Semiannual phase offset (days relative to annual peak)
    semiannual_phase_days: float
    # Medium maintenance: fraction of fouling cleared (0-1)
    medium_recovery: float
    # Medium maintenance: retention factor at 30 days
    medium_retention: float
    # Major maintenance: fraction of fouling cleared
    major_recovery: float | None
    # Major maintenance: retention factor at 30 days
    major_retention: float | None
    # Process noise std (residual volatility)
    sigma: float
    # Recent permeability level (last 30 days median)
    recent_level: float
    # Current cycle days since maintenance (at prediction start)
    days_since_maintenance: float
    # Last maintenance type before prediction start
    last_maintenance_type: str


@dataclass
class MaintenanceSchedule:
    """Fixed maintenance schedule for one device."""

    device: str
    # Medium maintenance: typical interval (days) and its std
    medium_interval_mean: float
    medium_interval_std: float
    # Major maintenance: typical interval (days) and its std
    major_interval_mean: float
    major_interval_std: float
    # Number of medium between majors
    medium_between_major_mean: float
    # Seasonal preference (month→relative frequency, optional)
    seasonal_profile: dict[int, float] | None = None


def load_device_indicators(tables_dir: Path) -> pd.DataFrame:
    return pd.read_csv(tables_dir / "device_indicators.csv").set_index("device")


def load_seasonality_summary(tables_dir: Path) -> dict:
    import json

    return json.loads((tables_dir / "seasonality_summary.json").read_text())


def load_maintenance_records(processed_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(processed_dir / "maintenance_records.csv", parse_dates=["date"])
    return df


def load_daily_permeability(processed_dir: Path) -> pd.DataFrame:
    return pd.read_csv(
        processed_dir / "daily_permeability.csv", parse_dates=["date"]
    )


def load_warmup_history(
    processed_dir: Path,
) -> dict[str, list[float]]:
    """Load last 365 days of actual daily permeability for each device.

    Used to pre-populate the sliding window in simulations.
    """
    daily = load_daily_permeability(processed_dir)
    warmup: dict[str, list[float]] = {}
    for device in DEVICE_ORDER:
        dev_data = daily[
            (daily["device"] == device) & daily["permeability_median"].notna()
        ].sort_values("date")
        # Take last 365 days of usable observations
        recent = dev_data.tail(365)["permeability_median"].tolist()
        warmup[device] = recent
    return warmup


def estimate_params(
    tables_dir: Path,
    processed_dir: Path,
    seasonality: dict,
) -> dict[str, FilterParams]:
    """Estimate FilterParams for all 10 devices from Problem 1 outputs."""
    indicators = load_device_indicators(tables_dir)
    daily = load_daily_permeability(processed_dir)
    maintenance = load_maintenance_records(processed_dir)

    seasonal_peak = seasonality["seasonal_peak_day"]
    seasonal_amp = seasonality["seasonal_amplitude"]

    params: dict[str, FilterParams] = {}
    for device in DEVICE_ORDER:
        row = indicators.loc[device]

        # Irreversible aging from envelope decline (per year → per day)
        ed = row.get("envelope_decline_per_year", np.nan)
        # Compute median envelope decline for regularization
        all_ed = indicators["envelope_decline_per_year"].dropna()
        med_ed = float(all_ed.median())
        if np.isnan(ed) or ed < 0:
            # Negative or missing: use regularized population median
            ed = max(med_ed, 2.0)  # floor at 2%/year
        # Cap extreme values at 3x population median
        ed = min(ed, med_ed * 3)
        alpha = ed / 365.25  # daily rate

        # Net decline rate = alpha + beta → beta = DR - alpha
        dr = row["decline_rate_median"]
        beta = max(dr - alpha, 0.01)  # floor at small positive

        # Medium maintenance recovery
        medium_gain = row.get("medium_gain_3d_median", np.nan)
        if np.isnan(medium_gain):
            medium_gain = indicators["medium_gain_3d_median"].median()
        # Recovery fraction: gain relative to typical pre-maintenance fouling level
        # Fouling accumulated over ~interval days: F ≈ beta * interval
        medium_retention = row.get("medium_retention_day30_median", 1.0)
        if np.isnan(medium_retention):
            medium_retention = 1.0

        # Major maintenance recovery
        major_gain = row.get("major_gain_3d_median", np.nan)
        major_retention = row.get("major_retention_day30_median", np.nan)
        if np.isnan(major_gain):
            # A4, A8: share from other devices
            major_gain = indicators["major_gain_3d_median"].median()
            major_retention = indicators["major_retention_day30_median"].median()

        # Process noise
        sigma = row.get("residual_volatility", 8.0)
        if np.isnan(sigma):
            sigma = 8.0

        # Current state (at prediction start: 2026-04-10)
        recent_level = row.get("recent_30_observation_median", 70.0)
        if np.isnan(recent_level):
            recent_level = 70.0

        # Days since last maintenance
        dev_daily = daily[daily["device"] == device].sort_values("date")
        dev_maint = maintenance[maintenance["device"] == device]
        last_date = dev_daily["date"].max()
        days_since = 0.0
        last_type = "观测起点"
        if len(dev_maint) > 0:
            last_maint_date = dev_maint["date"].max()
            days_since = float((last_date - last_maint_date).days)
            last_type = dev_maint.loc[
                dev_maint["date"] == last_maint_date, "maintenance_type"
            ].iloc[0]

        params[device] = FilterParams(
            device=device,
            alpha=alpha,
            beta=beta,
            seasonal_amplitude=seasonal_amp,
            seasonal_peak_day=seasonal_peak,
            semiannual_amplitude=seasonality.get("semiannual_amplitude", 2.84),
            semiannual_phase_days=seasonality.get("seasonal_peak_day", 234) * 0.5,
            medium_recovery=medium_gain,
            medium_retention=medium_retention,
            major_recovery=float(major_gain) if major_gain is not None else None,
            major_retention=float(major_retention) if major_retention is not None else None,
            sigma=sigma,
            recent_level=recent_level,
            days_since_maintenance=days_since,
            last_maintenance_type=last_type,
        )
    return params


def extract_maintenance_schedule(
    processed_dir: Path,
) -> dict[str, MaintenanceSchedule]:
    """Extract fixed maintenance patterns from historical records."""
    maintenance = load_maintenance_records(processed_dir)

    schedules: dict[str, MaintenanceSchedule] = {}
    for device in DEVICE_ORDER:
        dev_maint = maintenance[maintenance["device"] == device].sort_values(
            "date"
        )
        if len(dev_maint) < 2:
            # Fallback: use population average
            continue

        dates = dev_maint["date"].to_numpy()
        types = dev_maint["maintenance_type"].to_numpy()
        intervals = np.diff(dates).astype("timedelta64[D]").astype(float)

        # Separate medium and major intervals
        medium_intervals = []
        major_intervals = []
        medium_between_major = []
        count_since_major = 0

        for i in range(len(types)):
            if types[i] == "大维护":
                if count_since_major > 0:
                    medium_between_major.append(count_since_major)
                count_since_major = 0
            elif types[i] == "中维护":
                count_since_major += 1

        for i in range(len(intervals)):
            if types[i + 1] == "中维护" and types[i] == "中维护":
                medium_intervals.append(intervals[i])
            elif types[i + 1] == "大维护":
                if types[i] == "中维护":
                    major_intervals.append(intervals[i])
                else:
                    major_intervals.append(intervals[i])

        # If no major maintenance interval data, use scaled medium interval
        if not major_intervals:
            major_intervals = [
                np.mean(medium_intervals) * 4 if medium_intervals else 180
            ]
        if not medium_intervals:
            medium_intervals = [np.mean(major_intervals) / 4 if major_intervals else 45]

        med_arr = np.array(medium_intervals) if medium_intervals else np.array([60.0])
        maj_arr = np.array(major_intervals) if major_intervals else np.array([med_arr.mean() * 4])

        # For devices with no major records, use population median for med_between
        has_majors = len(major_intervals) > 0 and any(
            t == "大维护" for t in types
        )
        if has_majors and medium_between_major:
            med_between = np.array(medium_between_major)
        else:
            med_between = np.array([4.0])  # typical: 4 medium per major

        # Monthly maintenance frequency (seasonal preference)
        dev_maint_copy = dev_maint.copy()
        dev_maint_copy["month"] = dev_maint_copy["date"].dt.month
        month_counts = dev_maint_copy.groupby("month").size()
        seasonal_profile = {}
        for m in range(1, 13):
            if m in month_counts.index:
                seasonal_profile[m] = float(month_counts[m] / len(dev_maint_copy))

        schedules[device] = MaintenanceSchedule(
            device=device,
            medium_interval_mean=float(np.mean(med_arr)),
            medium_interval_std=float(np.std(med_arr, ddof=1)) if len(med_arr) > 1 else 5.0,
            major_interval_mean=float(np.mean(maj_arr)),
            major_interval_std=float(np.std(maj_arr, ddof=1)) if len(maj_arr) > 1 else 20.0,
            medium_between_major_mean=float(np.mean(med_between)),
            seasonal_profile=seasonal_profile if len(seasonal_profile) > 1 else None,
        )

    # Fill missing devices with population average
    for device in DEVICE_ORDER:
        if device not in schedules:
            all_medium = [s.medium_interval_mean for s in schedules.values()]
            all_major = [s.major_interval_mean for s in schedules.values()]
            all_between = [s.medium_between_major_mean for s in schedules.values()]
            schedules[device] = MaintenanceSchedule(
                device=device,
                medium_interval_mean=float(np.mean(all_medium)),
                medium_interval_std=float(np.mean([s.medium_interval_std for s in schedules.values()])),
                major_interval_mean=float(np.mean(all_major)),
                major_interval_std=float(np.mean([s.major_interval_std for s in schedules.values()])),
                medium_between_major_mean=float(np.mean(all_between)),
            )

    return schedules
