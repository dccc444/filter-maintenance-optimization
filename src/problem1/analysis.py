from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


DEVICE_ORDER = [f"A{i}" for i in range(1, 11)]
MAINTENANCE_ORDER = ["中维护", "大维护"]
RANDOM_SEED = 2026


def normalize_device(value: object) -> str:
    text = str(value).strip().upper().replace("_", "")
    if not text.startswith("A"):
        raise ValueError(f"无法识别设备编号: {value!r}")
    number = int(text[1:])
    if number not in range(1, 11):
        raise ValueError(f"设备编号超出 A1-A10: {value!r}")
    return f"A{number}"


def _find_attachment(data_dir: Path, suffix_number: int) -> Path:
    candidates = sorted(data_dir.glob(f"*{suffix_number}.xlsx"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"在 {data_dir} 中应恰好找到一个 *{suffix_number}.xlsx，实际为 {candidates}"
        )
    return candidates[0]


def load_inputs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    permeability_path = _find_attachment(data_dir, 1)
    maintenance_path = _find_attachment(data_dir, 2)

    pieces: list[pd.DataFrame] = []
    workbook = pd.ExcelFile(permeability_path)
    for sheet in workbook.sheet_names:
        frame = pd.read_excel(permeability_path, sheet_name=sheet)
        required = {"time", "per"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{sheet} 缺少字段 {required - set(frame.columns)}")
        frame = frame.loc[:, ["time", "per"]].copy()
        frame["device"] = normalize_device(sheet)
        frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
        frame["per"] = pd.to_numeric(frame["per"], errors="coerce")
        pieces.append(frame)

    hourly = pd.concat(pieces, ignore_index=True)
    hourly = hourly.loc[hourly["time"].notna()].copy()
    hourly = hourly.sort_values(["device", "time"]).reset_index(drop=True)

    maintenance = pd.read_excel(maintenance_path)
    if maintenance.shape[1] < 3:
        raise ValueError("维护记录至少需要三列：编号、日期、维护类型")
    maintenance = maintenance.iloc[:, :3].copy()
    maintenance.columns = ["device", "date", "maintenance_type"]
    maintenance["device"] = maintenance["device"].map(normalize_device)
    maintenance["date"] = pd.to_datetime(maintenance["date"], errors="coerce").dt.normalize()
    maintenance["maintenance_type"] = maintenance["maintenance_type"].astype(str).str.strip()
    maintenance = maintenance.dropna(subset=["date"])
    unknown = set(maintenance["maintenance_type"]) - set(MAINTENANCE_ORDER)
    if unknown:
        raise ValueError(f"发现未知维护类型: {unknown}")
    maintenance = maintenance.sort_values(["device", "date"]).reset_index(drop=True)
    return hourly, maintenance


def flag_hourly_outliers(
    hourly: pd.DataFrame,
    maintenance: pd.DataFrame,
    window: int = 25,
    threshold: float = 6.0,
) -> pd.DataFrame:
    """Flag isolated sensor spikes without treating maintenance jumps as outliers."""
    result = hourly.copy()
    result["near_maintenance"] = False
    result["is_outlier"] = False

    for device, idx in result.groupby("device", sort=False).groups.items():
        loc = np.asarray(list(idx))
        values = result.loc[loc, "per"]
        rolling_median = values.rolling(window, center=True, min_periods=8).median()
        abs_dev = (values - rolling_median).abs()
        rolling_mad = abs_dev.rolling(window, center=True, min_periods=8).median()
        robust_scale = 1.4826 * rolling_mad
        candidate = (abs_dev > threshold * robust_scale) & (robust_scale > 0.05)

        device_dates = maintenance.loc[maintenance["device"] == device, "date"]
        near = pd.Series(False, index=values.index)
        observed_dates = result.loc[loc, "time"].dt.normalize()
        for date in device_dates:
            near |= observed_dates.between(date - pd.Timedelta(days=2), date + pd.Timedelta(days=2))
        result.loc[loc, "near_maintenance"] = near.to_numpy()
        result.loc[loc, "is_outlier"] = (candidate & ~near).fillna(False).to_numpy()

    result["per_clean"] = result["per"].mask(result["is_outlier"])
    return result


def build_quality_table(hourly: pd.DataFrame) -> pd.DataFrame:
    records = []
    for device, group in hourly.groupby("device", sort=False):
        start = group["time"].min()
        end = group["time"].max()
        expected_hours = int(np.floor((end - start).total_seconds() / 3600)) + 1
        calendar_days = (end.normalize() - start.normalize()).days + 1
        valid = int(group["per"].notna().sum())
        valid_by_day = (
            group.assign(date=group["time"].dt.normalize())
            .groupby("date")["per"]
            .count()
        )
        usable_days = int((valid_by_day >= 3).sum())
        records.append(
            {
                "device": device,
                "start_time": start,
                "end_time": end,
                "rows": len(group),
                "valid_values": valid,
                "missing_values": int(group["per"].isna().sum()),
                "missing_rate": float(group["per"].isna().mean()),
                "duplicate_timestamps": int(group["time"].duplicated().sum()),
                "expected_calendar_hours": expected_hours,
                "calendar_hour_coverage": valid / expected_hours,
                "calendar_days": calendar_days,
                "usable_days": usable_days,
                "calendar_day_coverage": usable_days / calendar_days,
                "outliers_flagged": int(group.get("is_outlier", False).sum()),
                "minimum": group["per"].min(),
                "maximum": group["per"].max(),
            }
        )
    return pd.DataFrame(records).sort_values(
        "device", key=lambda s: s.map(DEVICE_ORDER.index)
    )


def aggregate_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    data = hourly.copy()
    data["date"] = data["time"].dt.normalize()
    value_column = "per_clean" if "per_clean" in data.columns else "per"
    daily = (
        data.groupby(["device", "date"], observed=True)
        .agg(
            observed_rows=("time", "size"),
            valid_hours=(value_column, "count"),
            permeability_median=(value_column, "median"),
            permeability_mean=(value_column, "mean"),
            permeability_std=(value_column, "std"),
            permeability_min=(value_column, "min"),
            permeability_max=(value_column, "max"),
        )
        .reset_index()
    )
    daily["hour_coverage"] = (daily["valid_hours"] / 24.0).clip(upper=1.0)
    daily["usable"] = daily["valid_hours"] >= 3
    daily.loc[~daily["usable"], [
        "permeability_median",
        "permeability_mean",
        "permeability_std",
        "permeability_min",
        "permeability_max",
    ]] = np.nan
    return daily.sort_values(["device", "date"]).reset_index(drop=True)


def add_maintenance_clock(
    daily: pd.DataFrame, maintenance: pd.DataFrame
) -> pd.DataFrame:
    result = daily.copy()
    result["days_since_maintenance"] = np.nan
    result["cycle_id"] = 0
    result["previous_maintenance_type"] = "观测起点"
    for device, idx in result.groupby("device", sort=False).groups.items():
        loc = np.asarray(list(idx))
        dates = result.loc[loc, "date"].to_numpy(dtype="datetime64[D]")
        events = maintenance.loc[maintenance["device"] == device].copy()
        event_dates = events["date"].to_numpy(dtype="datetime64[D]")
        positions = np.searchsorted(event_dates, dates, side="right")
        result.loc[loc, "cycle_id"] = positions
        if len(event_dates):
            has_previous = positions > 0
            days = np.full(len(loc), np.nan)
            days[has_previous] = (
                dates[has_previous] - event_dates[positions[has_previous] - 1]
            ).astype("timedelta64[D]").astype(float)
            types = np.full(len(loc), "观测起点", dtype=object)
            event_types = events["maintenance_type"].to_numpy()
            types[has_previous] = event_types[positions[has_previous] - 1]
            result.loc[loc, "days_since_maintenance"] = days
            result.loc[loc, "previous_maintenance_type"] = types
    result["days_since_maintenance"] = result["days_since_maintenance"].fillna(0.0)
    return result


def _harmonic_terms(dates: pd.Series) -> np.ndarray:
    day = dates.dt.dayofyear.to_numpy(dtype=float)
    return np.column_stack(
        [
            np.sin(2 * np.pi * day / 365.25),
            np.cos(2 * np.pi * day / 365.25),
            np.sin(4 * np.pi * day / 365.25),
            np.cos(4 * np.pi * day / 365.25),
        ]
    )


def _base_design(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    devices = pd.Categorical(frame["device"], categories=DEVICE_ORDER, ordered=True)
    device_codes = devices.codes
    n = len(frame)
    elapsed_years = (
        (frame["date"] - frame["date"].min()).dt.days.to_numpy(dtype=float) / 365.25
    )
    columns = [np.ones(n)]
    names = ["intercept"]
    for i, device in enumerate(DEVICE_ORDER[1:], start=1):
        columns.append((device_codes == i).astype(float))
        names.append(f"device_{device}")
    for i, device in enumerate(DEVICE_ORDER):
        columns.append(elapsed_years * (device_codes == i))
        names.append(f"trend_{device}")
    dsm = frame["days_since_maintenance"].to_numpy(dtype=float)
    columns.extend([dsm / 30.0, (dsm / 30.0) ** 2])
    names.extend(["days_since_maintenance_30d", "days_since_maintenance_sq"])
    return np.column_stack(columns), names


def fit_seasonality(daily: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    frame = daily.loc[daily["permeability_median"].notna()].copy()
    base_x, base_names = _base_design(frame)
    harmonic_x = _harmonic_terms(frame["date"])
    full_x = np.column_stack([base_x, harmonic_x])
    full_names = base_names + ["sin_1y", "cos_1y", "sin_2y", "cos_2y"]
    y = frame["permeability_median"].to_numpy(dtype=float)

    beta_base, *_ = np.linalg.lstsq(base_x, y, rcond=None)
    beta_full, *_ = np.linalg.lstsq(full_x, y, rcond=None)
    residual_base = y - base_x @ beta_base
    residual_full = y - full_x @ beta_full
    ssr_base = float(residual_base @ residual_base)
    ssr_full = float(residual_full @ residual_full)
    q = harmonic_x.shape[1]
    df_resid = len(y) - full_x.shape[1]
    f_stat = ((ssr_base - ssr_full) / q) / (ssr_full / df_resid)
    p_value = float(stats.f.sf(f_stat, q, df_resid))

    grid_dates = pd.Series(pd.date_range("2025-01-01", periods=365, freq="D"))
    grid_h = _harmonic_terms(grid_dates)
    harmonic_beta = beta_full[-4:]
    seasonal_grid = grid_h @ harmonic_beta
    amplitude = float((seasonal_grid.max() - seasonal_grid.min()) / 2)
    peak_i = int(np.argmax(seasonal_grid))
    trough_i = int(np.argmin(seasonal_grid))

    frame["model_fitted"] = full_x @ beta_full
    frame["detrended_residual"] = residual_base
    frame["model_residual"] = residual_full
    frame["seasonal_component"] = harmonic_x @ harmonic_beta
    frame["permeability_season_adjusted"] = (
        frame["permeability_median"] - frame["seasonal_component"]
    )

    summary = {
        "n_observations": len(y),
        "base_parameters": base_x.shape[1],
        "full_parameters": full_x.shape[1],
        "ssr_without_season": ssr_base,
        "ssr_with_season": ssr_full,
        "relative_ssr_reduction": (ssr_base - ssr_full) / ssr_base,
        "f_statistic": float(f_stat),
        "df_numerator": q,
        "df_denominator": df_resid,
        "p_value": p_value,
        "seasonal_amplitude": amplitude,
        "seasonal_peak_day": peak_i + 1,
        "seasonal_trough_day": trough_i + 1,
        "annual_amplitude": float(np.hypot(harmonic_beta[0], harmonic_beta[1])),
        "semiannual_amplitude": float(np.hypot(harmonic_beta[2], harmonic_beta[3])),
    }
    coefficient_table = pd.DataFrame(
        {"term": full_names, "coefficient": beta_full}
    )
    seasonal_curve = pd.DataFrame(
        {
            "day_of_year": np.arange(1, 366),
            "date_reference": grid_dates,
            "seasonal_effect": seasonal_grid,
        }
    )
    return frame, summary, pd.concat(
        [
            coefficient_table.assign(table="coefficients"),
            seasonal_curve.rename(
                columns={
                    "day_of_year": "term",
                    "seasonal_effect": "coefficient",
                }
            )
            .loc[:, ["term", "coefficient"]]
            .assign(table="seasonal_curve"),
        ],
        ignore_index=True,
    )


def dominant_periods(season_frame: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    aggregate = (
        season_frame.groupby("date", as_index=False)["detrended_residual"].median()
        .set_index("date")
        .asfreq("D")
    )
    series = aggregate["detrended_residual"].interpolate(limit=7, limit_direction="both")
    series = series - series.mean()
    n = len(series)
    spectrum = np.abs(np.fft.rfft(series.to_numpy())) ** 2
    frequencies = np.fft.rfftfreq(n, d=1.0)
    periods = np.full_like(frequencies, np.inf, dtype=float)
    nonzero = frequencies > 0
    periods[nonzero] = 1 / frequencies[nonzero]
    valid = nonzero & (periods >= 7) & (periods <= 500)
    table = pd.DataFrame(
        {
            "period_days": periods[valid],
            "power": spectrum[valid],
        }
    )
    table["relative_power"] = table["power"] / table["power"].sum()
    return table.nlargest(top_n, "power").reset_index(drop=True)


def estimate_cycle_slopes(
    season_frame: pd.DataFrame,
) -> pd.DataFrame:
    records = []
    for (device, cycle_id), group in season_frame.groupby(["device", "cycle_id"]):
        group = group.dropna(subset=["permeability_season_adjusted"]).sort_values("date")
        if len(group) < 14:
            continue
        elapsed = (group["date"] - group["date"].min()).dt.days.to_numpy(dtype=float)
        span = float(elapsed.max() - elapsed.min())
        if span < 14:
            continue
        slope, intercept, low, high = stats.theilslopes(
            group["permeability_season_adjusted"].to_numpy(dtype=float), elapsed
        )
        records.append(
            {
                "device": device,
                "cycle_id": int(cycle_id),
                "cycle_start": group["date"].min(),
                "cycle_end": group["date"].max(),
                "previous_maintenance_type": group["previous_maintenance_type"].iloc[0],
                "n_observed_days": len(group),
                "span_days": span,
                "slope_per_day": slope,
                "decline_rate_per_day": -slope,
                "slope_ci_low": low,
                "slope_ci_high": high,
                "intercept": intercept,
            }
        )
    return pd.DataFrame(records)


def _event_effects_for_one(
    device_daily: pd.DataFrame,
    event_date: pd.Timestamp,
    max_day: int = 30,
) -> tuple[pd.DataFrame, dict] | None:
    data = device_daily.set_index("date").sort_index()
    event_day = np.datetime64(pd.Timestamp(event_date).date(), "D")
    relative_all = (
        data.index.to_numpy(dtype="datetime64[D]") - event_day
    ).astype("timedelta64[D]").astype(int)
    pre = data.loc[
        (relative_all >= -21) & (relative_all <= -1),
        "permeability_season_adjusted",
    ].dropna()
    if len(pre) < 10:
        return None
    x_pre = (pre.index - event_date).days.to_numpy(dtype=float)
    slope, intercept, *_ = stats.theilslopes(pre.to_numpy(dtype=float), x_pre)

    window = data.loc[
        (relative_all >= -14) & (relative_all <= max_day)
    ].copy()
    relative_day = (window.index - event_date).days.to_numpy(dtype=int)
    predicted = intercept + slope * relative_day
    effect = window["permeability_season_adjusted"].to_numpy(dtype=float) - predicted
    curve = pd.DataFrame(
        {
            "relative_day": relative_day,
            "actual_adjusted": window["permeability_season_adjusted"].to_numpy(),
            "counterfactual": predicted,
            "effect": effect,
        }
    )

    def mean_effect(start: int, end: int) -> float:
        values = curve.loc[curve["relative_day"].between(start, end), "effect"]
        return float(values.mean()) if values.notna().sum() else np.nan

    pre_naive = data.loc[
        (relative_all >= -3) & (relative_all <= -1),
        "permeability_season_adjusted",
    ].mean()
    post_naive = data.loc[
        (relative_all >= 1) & (relative_all <= 3),
        "permeability_season_adjusted",
    ].mean()
    metrics = {
        "pre_observations": len(pre),
        "pre_slope_per_day": slope,
        "naive_gain_3d": float(post_naive - pre_naive),
        "counterfactual_gain_3d": mean_effect(1, 3),
        "effect_day7": mean_effect(5, 9),
        "effect_day14": mean_effect(12, 16),
        "effect_day30": mean_effect(27, 30),
    }
    gain = metrics["counterfactual_gain_3d"]
    metrics["retention_day30"] = (
        metrics["effect_day30"] / gain if np.isfinite(gain) and abs(gain) > 1e-9 else np.nan
    )
    return curve, metrics


def event_study(
    season_frame: pd.DataFrame, maintenance: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    curves = []
    metrics = []
    for event_id, event in maintenance.reset_index(drop=True).iterrows():
        device_daily = season_frame.loc[season_frame["device"] == event["device"]]
        result = _event_effects_for_one(device_daily, event["date"])
        if result is None:
            continue
        curve, record = result
        curve["event_id"] = event_id
        curve["device"] = event["device"]
        curve["event_date"] = event["date"]
        curve["maintenance_type"] = event["maintenance_type"]
        curves.append(curve)
        record.update(
            {
                "event_id": event_id,
                "device": event["device"],
                "event_date": event["date"],
                "maintenance_type": event["maintenance_type"],
            }
        )
        metrics.append(record)
    curves_df = pd.concat(curves, ignore_index=True)
    metrics_df = pd.DataFrame(metrics)

    rng = np.random.default_rng(RANDOM_SEED)
    summaries = []
    for maintenance_type in MAINTENANCE_ORDER:
        subset = metrics_df.loc[metrics_df["maintenance_type"] == maintenance_type]
        for metric in [
            "naive_gain_3d",
            "counterfactual_gain_3d",
            "effect_day7",
            "effect_day14",
            "effect_day30",
            "retention_day30",
        ]:
            values = subset[metric].dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            boot = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
            summaries.append(
                {
                    "maintenance_type": maintenance_type,
                    "metric": metric,
                    "n_events": len(values),
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    "ci95_low": float(np.quantile(boot, 0.025)),
                    "ci95_high": float(np.quantile(boot, 0.975)),
                    "positive_share": float((values > 0).mean()),
                }
            )
    summary_df = pd.DataFrame(summaries)
    return curves_df, metrics_df, summary_df


def event_curve_summary(curves: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    records = []
    for (maintenance_type, relative_day), group in curves.groupby(
        ["maintenance_type", "relative_day"]
    ):
        event_values = group.groupby("event_id")["effect"].mean().dropna().to_numpy()
        if not len(event_values):
            continue
        boot = rng.choice(
            event_values, size=(1000, len(event_values)), replace=True
        ).mean(axis=1)
        records.append(
            {
                "maintenance_type": maintenance_type,
                "relative_day": relative_day,
                "n_events": len(event_values),
                "mean_effect": float(event_values.mean()),
                "median_effect": float(np.median(event_values)),
                "ci95_low": float(np.quantile(boot, 0.025)),
                "ci95_high": float(np.quantile(boot, 0.975)),
            }
        )
    return pd.DataFrame(records)


def compare_maintenance_types(metrics: pd.DataFrame) -> pd.DataFrame:
    records = []
    for metric in [
        "counterfactual_gain_3d",
        "effect_day7",
        "effect_day14",
        "effect_day30",
        "retention_day30",
    ]:
        medium = metrics.loc[metrics["maintenance_type"] == "中维护", metric].dropna()
        major = metrics.loc[metrics["maintenance_type"] == "大维护", metric].dropna()
        if len(medium) and len(major):
            statistic, p_value = stats.mannwhitneyu(
                medium, major, alternative="two-sided"
            )
            records.append(
                {
                    "metric": metric,
                    "medium_n": len(medium),
                    "major_n": len(major),
                    "medium_median": medium.median(),
                    "major_median": major.median(),
                    "mann_whitney_u": statistic,
                    "p_value": p_value,
                }
            )
    return pd.DataFrame(records)


def envelope_decay(
    season_frame: pd.DataFrame, maintenance: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    points = []
    for event_id, event in maintenance.reset_index(drop=True).iterrows():
        subset = season_frame.loc[
            (season_frame["device"] == event["device"])
            & season_frame["date"].between(
                event["date"] + pd.Timedelta(days=1),
                event["date"] + pd.Timedelta(days=7),
            ),
            "permeability_season_adjusted",
        ].dropna()
        if len(subset) >= 3:
            points.append(
                {
                    "event_id": event_id,
                    "device": event["device"],
                    "event_date": event["date"],
                    "maintenance_type": event["maintenance_type"],
                    "post_7d_level": float(subset.median()),
                    "n_days": len(subset),
                }
            )
    points_df = pd.DataFrame(points)
    records = []
    for device, group in points_df.groupby("device"):
        if len(group) < 4:
            continue
        elapsed = (
            (group["event_date"] - group["event_date"].min()).dt.days.to_numpy(dtype=float)
            / 365.25
        )
        slope, intercept, low, high = stats.theilslopes(
            group["post_7d_level"].to_numpy(), elapsed
        )
        records.append(
            {
                "device": device,
                "n_events": len(group),
                "envelope_slope_per_year": slope,
                "envelope_decline_per_year": -slope,
                "slope_ci_low": low,
                "slope_ci_high": high,
                "intercept": intercept,
            }
        )
    return points_df, pd.DataFrame(records)


def device_seasonal_amplitudes(daily: pd.DataFrame) -> pd.DataFrame:
    records = []
    for device, group in daily.groupby("device"):
        group = group.dropna(subset=["permeability_median"]).copy()
        if len(group) < 180:
            continue
        elapsed = (group["date"] - group["date"].min()).dt.days.to_numpy() / 365.25
        dsm = group["days_since_maintenance"].to_numpy() / 30.0
        h = _harmonic_terms(group["date"])
        x = np.column_stack([np.ones(len(group)), elapsed, dsm, dsm**2, h])
        beta, *_ = np.linalg.lstsq(
            x, group["permeability_median"].to_numpy(dtype=float), rcond=None
        )
        grid = _harmonic_terms(pd.Series(pd.date_range("2025-01-01", periods=365)))
        component = grid @ beta[-4:]
        records.append(
            {
                "device": device,
                "seasonal_amplitude": (component.max() - component.min()) / 2,
                "annual_amplitude": np.hypot(beta[-4], beta[-3]),
                "semiannual_amplitude": np.hypot(beta[-2], beta[-1]),
            }
        )
    return pd.DataFrame(records)


def build_indicator_table(
    quality: pd.DataFrame,
    season_frame: pd.DataFrame,
    cycle_slopes: pd.DataFrame,
    event_metrics: pd.DataFrame,
    envelope_summary: pd.DataFrame,
    device_seasonality: pd.DataFrame,
) -> pd.DataFrame:
    indicators = quality.loc[
        :, ["device", "calendar_hour_coverage", "missing_rate", "outliers_flagged"]
    ].copy()
    slope_summary = (
        cycle_slopes.groupby("device")["decline_rate_per_day"]
        .agg(
            decline_rate_median="median",
            decline_rate_mean="mean",
            decline_rate_q25=lambda x: x.quantile(0.25),
            decline_rate_q75=lambda x: x.quantile(0.75),
            valid_cycles="count",
        )
        .reset_index()
    )
    indicators = indicators.merge(slope_summary, on="device", how="left")
    indicators = indicators.merge(device_seasonality, on="device", how="left")

    for maintenance_type, prefix in [("中维护", "medium"), ("大维护", "major")]:
        event_summary = (
            event_metrics.loc[event_metrics["maintenance_type"] == maintenance_type]
            .groupby("device")
            .agg(
                **{
                    f"{prefix}_events": ("event_id", "count"),
                    f"{prefix}_gain_3d_median": (
                        "counterfactual_gain_3d",
                        "median",
                    ),
                    f"{prefix}_effect_day30_median": ("effect_day30", "median"),
                    f"{prefix}_retention_day30_median": (
                        "retention_day30",
                        "median",
                    ),
                }
            )
            .reset_index()
        )
        indicators = indicators.merge(event_summary, on="device", how="left")

    indicators = indicators.merge(
        envelope_summary.loc[:, ["device", "envelope_decline_per_year"]],
        on="device",
        how="left",
    )
    residual = (
        season_frame.groupby("device")["model_residual"]
        .std()
        .rename("residual_volatility")
        .reset_index()
    )
    recent = (
        season_frame.sort_values("date")
        .groupby("device")
        .tail(30)
        .groupby("device")["permeability_median"]
        .median()
        .rename("recent_30_observation_median")
        .reset_index()
    )
    indicators = indicators.merge(residual, on="device", how="left")
    indicators = indicators.merge(recent, on="device", how="left")
    for col in ["medium_events", "major_events"]:
        if col in indicators:
            indicators[col] = indicators[col].fillna(0).astype(int)
    return indicators.sort_values(
        "device", key=lambda s: s.map(DEVICE_ORDER.index)
    ).reset_index(drop=True)


def monthly_coverage(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    frame["month"] = frame["date"].dt.to_period("M").astype(str)
    return (
        frame.groupby(["device", "month"], as_index=False)["hour_coverage"]
        .mean()
        .sort_values(["device", "month"])
    )


@dataclass
class AnalysisResults:
    hourly: pd.DataFrame
    maintenance: pd.DataFrame
    quality: pd.DataFrame
    daily: pd.DataFrame
    season_frame: pd.DataFrame
    seasonality_summary: dict
    seasonality_details: pd.DataFrame
    periods: pd.DataFrame
    cycle_slopes: pd.DataFrame
    event_curves: pd.DataFrame
    event_metrics: pd.DataFrame
    event_summary: pd.DataFrame
    event_curve_summary: pd.DataFrame
    maintenance_comparison: pd.DataFrame
    envelope_points: pd.DataFrame
    envelope_summary: pd.DataFrame
    indicators: pd.DataFrame
    monthly_coverage: pd.DataFrame


def run_analysis(data_dir: Path) -> AnalysisResults:
    hourly_raw, maintenance = load_inputs(data_dir)
    hourly = flag_hourly_outliers(hourly_raw, maintenance)
    quality = build_quality_table(hourly)
    daily = aggregate_daily(hourly)
    daily = add_maintenance_clock(daily, maintenance)
    season_frame, seasonality_summary, seasonality_details = fit_seasonality(daily)
    periods = dominant_periods(season_frame)
    cycle_slopes = estimate_cycle_slopes(season_frame)
    event_curves, event_metrics, event_summary = event_study(
        season_frame, maintenance
    )
    curve_summary = event_curve_summary(event_curves)
    maintenance_comparison = compare_maintenance_types(event_metrics)
    envelope_points, envelope_summary = envelope_decay(season_frame, maintenance)
    per_device_season = device_seasonal_amplitudes(daily)
    indicators = build_indicator_table(
        quality,
        season_frame,
        cycle_slopes,
        event_metrics,
        envelope_summary,
        per_device_season,
    )
    return AnalysisResults(
        hourly=hourly,
        maintenance=maintenance,
        quality=quality,
        daily=daily,
        season_frame=season_frame,
        seasonality_summary=seasonality_summary,
        seasonality_details=seasonality_details,
        periods=periods,
        cycle_slopes=cycle_slopes,
        event_curves=event_curves,
        event_metrics=event_metrics,
        event_summary=event_summary,
        event_curve_summary=curve_summary,
        maintenance_comparison=maintenance_comparison,
        envelope_points=envelope_points,
        envelope_summary=envelope_summary,
        indicators=indicators,
        monthly_coverage=monthly_coverage(daily),
    )
