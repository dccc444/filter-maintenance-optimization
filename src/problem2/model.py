"""Problem 2 data interfaces and parameter estimation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEVICE_ORDER = [f"A{i}" for i in range(1, 11)]
RANDOM_SEED = 2026


@dataclass
class FilterParams:
    """Parameters and current state for one filter."""

    device: str
    alpha: float
    beta: float
    seasonal_sin_1y: float
    seasonal_cos_1y: float
    seasonal_sin_2y: float
    seasonal_cos_2y: float
    medium_recovery: float
    major_recovery: float
    medium_damage: float
    major_damage: float
    sigma: float
    recent_level: float
    days_since_maintenance: float
    mediums_since_major: int
    last_maintenance_type: str
    historical_irreversible_decline_per_year: float
    damage_share_assumption: float


@dataclass
class MaintenanceSchedule:
    """Observed fixed-maintenance policy.

    ``*_gap`` is the interval from the previous recorded event to an event of
    that type. A major event replaces, rather than accompanies, the next medium
    event.
    """

    device: str
    medium_gap_mean: float
    medium_gap_std: float
    major_gap_mean: float
    major_gap_std: float
    medium_between_major: int
    seasonal_profile: dict[int, float] | None = None

    # Compatibility aliases used by reports created before the terminology fix.
    @property
    def medium_interval_mean(self) -> float:
        return self.medium_gap_mean

    @property
    def medium_interval_std(self) -> float:
        return self.medium_gap_std

    @property
    def major_interval_mean(self) -> float:
        return self.major_gap_mean

    @property
    def major_interval_std(self) -> float:
        return self.major_gap_std

    @property
    def medium_between_major_mean(self) -> float:
        return float(self.medium_between_major)


def load_device_indicators(tables_dir: Path) -> pd.DataFrame:
    return pd.read_csv(tables_dir / "device_indicators.csv").set_index("device")


def load_seasonality_summary(tables_dir: Path) -> dict:
    import json

    return json.loads((tables_dir / "seasonality_summary.json").read_text(encoding="utf-8"))


def load_seasonality_coefficients(tables_dir: Path) -> dict[str, float]:
    details = pd.read_csv(tables_dir / "seasonality_details.csv")
    coefficients = details.loc[details["table"] == "coefficients"].copy()
    coefficients["coefficient"] = pd.to_numeric(
        coefficients["coefficient"], errors="coerce"
    )
    mapping = coefficients.set_index("term")["coefficient"].to_dict()
    required = ["sin_1y", "cos_1y", "sin_2y", "cos_2y"]
    missing = [term for term in required if term not in mapping]
    if missing:
        raise ValueError(f"第一问季节系数缺失: {missing}")
    return {term: float(mapping[term]) for term in required}


def load_maintenance_records(processed_dir: Path) -> pd.DataFrame:
    return pd.read_csv(
        processed_dir / "maintenance_records.csv", parse_dates=["date"]
    )


def load_daily_permeability(processed_dir: Path) -> pd.DataFrame:
    return pd.read_csv(
        processed_dir / "daily_permeability.csv", parse_dates=["date"]
    )


def load_warmup_history(processed_dir: Path) -> dict[str, list[float]]:
    """Return exactly the last 365 calendar days for each device.

    Short gaps are interpolated only for initialization; any remaining gaps are
    filled with the device median so that the simulation's 365-day window
    represents calendar days rather than 365 scattered observations.
    """
    daily = load_daily_permeability(processed_dir)
    warmup: dict[str, list[float]] = {}
    for device in DEVICE_ORDER:
        dev = daily.loc[daily["device"] == device].sort_values("date")
        end = dev["date"].max()
        index = pd.date_range(end=end, periods=365, freq="D")
        series = (
            dev.set_index("date")["permeability_median"]
            .reindex(index)
            .interpolate(limit=7, limit_direction="both")
        )
        series = series.fillna(series.median())
        warmup[device] = series.astype(float).tolist()
    return warmup


def _regularized_envelope_decline(indicators: pd.DataFrame, device: str) -> float:
    positive = indicators["envelope_decline_per_year"].dropna()
    positive = positive.loc[positive > 0]
    population = float(positive.median()) if len(positive) else 10.0
    value = float(indicators.loc[device, "envelope_decline_per_year"])
    if not np.isfinite(value) or value <= 0:
        value = population
    return float(np.clip(value, 2.0, 3.0 * population))


def estimate_params(
    tables_dir: Path,
    processed_dir: Path,
    seasonality: dict | None = None,
    damage_share: float = 0.20,
) -> dict[str, FilterParams]:
    """Estimate device parameters from the completed Problem 1 interface.

    Irreversible envelope decline cannot identify natural aging and
    maintenance damage separately. For policy simulation we therefore allocate
    a documented fraction of historical irreversible decline to event damage.
    The default 20% is a baseline scenario and is exposed for sensitivity
    analysis in Problems 3 and 4.
    """
    indicators = load_device_indicators(tables_dir)
    daily = load_daily_permeability(processed_dir)
    maintenance = load_maintenance_records(processed_dir)
    coefficients = load_seasonality_coefficients(tables_dir)

    params: dict[str, FilterParams] = {}
    for device in DEVICE_ORDER:
        row = indicators.loc[device]
        dev_daily = daily.loc[daily["device"] == device].sort_values("date")
        dev_maint = maintenance.loc[maintenance["device"] == device].sort_values(
            "date"
        )
        ed = _regularized_envelope_decline(indicators, device)
        span_years = max(
            (dev_daily["date"].max() - dev_daily["date"].min()).days / 365.25,
            1.0,
        )
        medium_per_year = (
            (dev_maint["maintenance_type"] == "中维护").sum() / span_years
        )
        major_per_year = (
            (dev_maint["maintenance_type"] == "大维护").sum() / span_years
        )
        weighted_events = medium_per_year + 3.0 * major_per_year
        medium_damage = (
            ed * damage_share / weighted_events if weighted_events > 0 else 0.0
        )
        major_damage = 3.0 * medium_damage
        alpha = ed * (1.0 - damage_share) / 365.25

        decline_rate = float(row["decline_rate_median"])
        beta = max(decline_rate - alpha, 0.01)

        medium_gain = float(row.get("medium_gain_3d_median", np.nan))
        if not np.isfinite(medium_gain):
            medium_gain = float(indicators["medium_gain_3d_median"].median())
        major_gain = float(row.get("major_gain_3d_median", np.nan))
        if not np.isfinite(major_gain):
            major_gain = float(indicators["major_gain_3d_median"].median())

        sigma = float(row.get("residual_volatility", 8.0))
        if not np.isfinite(sigma):
            sigma = 8.0
        recent_level = float(row.get("recent_30_observation_median", 70.0))
        if not np.isfinite(recent_level):
            recent_level = 70.0

        last_date = dev_daily["date"].max()
        days_since = 0.0
        last_type = "观测起点"
        mediums_since_major = 0
        if len(dev_maint):
            past = dev_maint.loc[dev_maint["date"] <= last_date]
            last_event = past.iloc[-1]
            days_since = float((last_date - last_event["date"]).days)
            last_type = str(last_event["maintenance_type"])
            for mtype in reversed(past["maintenance_type"].tolist()):
                if mtype == "大维护":
                    break
                if mtype == "中维护":
                    mediums_since_major += 1
            if not (past["maintenance_type"] == "大维护").any():
                # A4/A8 have no recorded major event. Continue the imputed
                # population four-medium cycle without treating the full
                # two-year history as one overdue cycle.
                mediums_since_major %= 4

        params[device] = FilterParams(
            device=device,
            alpha=float(alpha),
            beta=float(beta),
            seasonal_sin_1y=coefficients["sin_1y"],
            seasonal_cos_1y=coefficients["cos_1y"],
            seasonal_sin_2y=coefficients["sin_2y"],
            seasonal_cos_2y=coefficients["cos_2y"],
            medium_recovery=float(max(medium_gain, 1.0)),
            major_recovery=float(max(major_gain, 1.0)),
            medium_damage=float(max(medium_damage, 0.0)),
            major_damage=float(max(major_damage, 0.0)),
            sigma=float(max(sigma, 0.1)),
            recent_level=recent_level,
            days_since_maintenance=days_since,
            mediums_since_major=mediums_since_major,
            last_maintenance_type=last_type,
            historical_irreversible_decline_per_year=ed,
            damage_share_assumption=damage_share,
        )
    return params


def extract_maintenance_schedule(
    processed_dir: Path,
) -> dict[str, MaintenanceSchedule]:
    """Extract event-to-event gaps and medium/major sequencing."""
    maintenance = load_maintenance_records(processed_dir)
    raw: dict[str, dict] = {}
    all_major_gaps: list[float] = []
    all_between: list[int] = []

    for device in DEVICE_ORDER:
        dev = maintenance.loc[maintenance["device"] == device].sort_values("date")
        dates = dev["date"].to_numpy()
        types = dev["maintenance_type"].tolist()
        gaps = (
            np.diff(dates).astype("timedelta64[D]").astype(float)
            if len(dates) >= 2
            else np.array([], dtype=float)
        )
        medium_gaps = [
            float(gaps[i]) for i in range(len(gaps)) if types[i + 1] == "中维护"
        ]
        major_gaps = [
            float(gaps[i]) for i in range(len(gaps)) if types[i + 1] == "大维护"
        ]
        all_major_gaps.extend(major_gaps)

        between: list[int] = []
        count = 0
        seen_major = False
        for mtype in types:
            if mtype == "大维护":
                if seen_major or count:
                    between.append(count)
                count = 0
                seen_major = True
            else:
                count += 1
        all_between.extend([value for value in between if value > 0])
        month_counts = dev["date"].dt.month.value_counts(normalize=True)
        raw[device] = {
            "medium_gaps": medium_gaps,
            "major_gaps": major_gaps,
            "between": [value for value in between if value > 0],
            "seasonal_profile": {
                int(month): float(value) for month, value in month_counts.items()
            },
        }

    population_major_gap = float(np.median(all_major_gaps)) if all_major_gaps else 60.0
    population_between = int(round(np.median(all_between))) if all_between else 4
    schedules: dict[str, MaintenanceSchedule] = {}
    for device in DEVICE_ORDER:
        item = raw[device]
        medium = np.array(item["medium_gaps"] or [57.0], dtype=float)
        major = np.array(item["major_gaps"] or [population_major_gap], dtype=float)
        between = item["between"]
        target = int(round(np.median(between))) if between else population_between
        schedules[device] = MaintenanceSchedule(
            device=device,
            medium_gap_mean=float(medium.mean()),
            medium_gap_std=float(medium.std(ddof=1)) if len(medium) > 1 else 7.0,
            major_gap_mean=float(major.mean()),
            major_gap_std=float(major.std(ddof=1)) if len(major) > 1 else 10.0,
            medium_between_major=max(target, 1),
            seasonal_profile=item["seasonal_profile"] or None,
        )
    return schedules
