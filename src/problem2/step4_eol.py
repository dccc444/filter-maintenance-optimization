"""Step 4: Define and validate end-of-life criteria."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .model import DEVICE_ORDER, load_daily_permeability, load_maintenance_records


def compute_rolling_annual_mean(
    daily: pd.DataFrame,
) -> pd.DataFrame:
    """Compute 365-day rolling mean permeability for each device."""
    records = []
    for device in DEVICE_ORDER:
        dev = daily[
            (daily["device"] == device) & daily["permeability_median"].notna()
        ].sort_values("date")
        if len(dev) < 365:
            continue

        dev_copy = dev.set_index("date")
        rolling = dev_copy["permeability_median"].rolling("365D", min_periods=180).mean()

        for i, (date, val) in enumerate(rolling.items()):
            if pd.notna(val):
                records.append(
                    {
                        "device": device,
                        "date": date,
                        "annual_mean_P": val,
                        "below_37": val < 37,
                    }
                )
    return pd.DataFrame(records)


def compute_post_maintenance_recovery(
    daily: pd.DataFrame,
    maintenance: pd.DataFrame,
    params: dict,
    window_days: int = 30,
) -> pd.DataFrame:
    """For each maintenance event, compute the average P in the post-maintenance window."""
    records = []
    for _, event in maintenance.iterrows():
        device = event["device"]
        event_date = event["date"]
        mtype = event["maintenance_type"]

        post_window = daily[
            (daily["device"] == device)
            & (daily["date"] > event_date)
            & (daily["date"] <= event_date + pd.Timedelta(days=window_days))
            & daily["permeability_median"].notna()
        ]

        if len(post_window) < 5:
            continue

        post_mean = float(post_window["permeability_median"].mean())
        records.append(
            {
                "device": device,
                "event_date": event_date,
                "maintenance_type": mtype,
                f"post_{window_days}d_mean": post_mean,
                "below_37": post_mean < 37,
            }
        )
    return pd.DataFrame(records)


def evaluate_end_of_life_status(
    processed_dir: Path,
    tables_dir: Path,
) -> dict:
    """Evaluate how close each device is to end-of-life.

    Uses historical data to compute:
    1. Current 365-day trailing average
    2. Recovery after most recent maintenance
    3. Projected time until threshold
    """
    daily = load_daily_permeability(processed_dir)
    maintenance = load_maintenance_records(processed_dir)

    # Load device indicators for alpha/beta
    indicators = pd.read_csv(tables_dir / "device_indicators.csv").set_index("device")

    # Rolling annual mean
    annual = compute_rolling_annual_mean(daily)

    # Latest annual mean per device
    latest_annual = (
        annual.sort_values("date")
        .groupby("device")
        .tail(1)
        .set_index("device")
    )

    # Post-maintenance recovery for major maintenance events
    post_major = compute_post_maintenance_recovery(
        daily,
        maintenance[maintenance["maintenance_type"] == "大维护"],
        {},
        window_days=30,
    )

    # Most recent major maintenance recovery per device
    latest_major = (
        post_major.sort_values("event_date")
        .groupby("device")
        .tail(1)
        .set_index("device")
    )

    # Build status table
    status = []
    for device in DEVICE_ORDER:
        row = {"device": device}

        # Current annual mean
        if device in latest_annual.index:
            row["current_annual_mean"] = round(
                latest_annual.loc[device, "annual_mean_P"], 1
            )
            row["currently_below_37"] = bool(
                latest_annual.loc[device, "below_37"]
            )
        else:
            row["current_annual_mean"] = None
            row["currently_below_37"] = None

        # Latest major maintenance recovery
        if device in latest_major.index:
            row["last_major_30d_mean"] = round(
                latest_major.loc[device, "post_30d_mean"], 1
            )
            row["major_recovery_below_37"] = bool(
                latest_major.loc[device, "below_37"]
            )
        else:
            row["last_major_30d_mean"] = None
            row["major_recovery_below_37"] = None

        # Projection: rough estimate of time to annual_mean < 37
        # Annual mean decline ≈ alpha*365 + beta*average_fouling
        if device in indicators.index:
            alpha = float(indicators.loc[device, "envelope_decline_per_year"])
            if alpha < 0:
                alpha = float(
                    indicators["envelope_decline_per_year"]
                    .clip(lower=0)
                    .median()
                )
            current = row["current_annual_mean"]
            if current and current > 37:
                years = (current - 37) / max(alpha, 1.0)
                row["projected_years_to_37"] = round(years, 2)
            else:
                row["projected_years_to_37"] = 0.0 if current else None
        else:
            row["projected_years_to_37"] = None

        # End-of-life status
        annual_below = row.get("currently_below_37", False)
        major_below = row.get("major_recovery_below_37", False)
        if annual_below and major_below:
            row["eol_status"] = "已终止"
        elif annual_below:
            row["eol_status"] = "年平均 < 37 (大维护待确认)"
        elif major_below:
            row["eol_status"] = "大维护恢复不足 (待观察)"
        else:
            row["eol_status"] = "正常运行"

        status.append(row)

    df_status = pd.DataFrame(status)

    # Also compute historical statistics
    n_ever_below = int(annual.groupby("device")["below_37"].any().sum())

    return {
        "device_status": df_status,
        "n_devices_ever_below_annual": n_ever_below,
        "annual_mean_stats": latest_annual.reset_index(),
        "post_major_stats": post_major,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Step 4: End-of-life criteria")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("outputs/problem1/processed"),
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("outputs/problem1/tables"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/problem2"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "tables").mkdir(parents=True, exist_ok=True)

    results = evaluate_end_of_life_status(args.processed_dir, args.tables_dir)

    # Print
    print("=== 第四步：寿命终止标准与当前状态 ===\n")
    print("终止标准（数学化）：")
    print("  条件一: P̄₃₆₅(t) < 37  (365日滑动平均透水率低于37)")
    print("  条件二: E[P̄₃₀^大维护后(t)] < 37  (大维护后30日平均仍不能恢复到37)")
    print("  两者同时满足 → 寿命终止\n")

    print("当前各设备状态:")
    print(
        f"{'设备':<6} {'年平均P':>8} {'<37?':>6} {'大维护后30d':>12} {'<37?':>6} "
        f"{'预估到37(年)':>12} {'状态':<18}"
    )
    print("-" * 80)
    for _, r in results["device_status"].iterrows():
        ann = f"{r['current_annual_mean']:.1f}" if pd.notna(r["current_annual_mean"]) else "N/A"
        maj = f"{r['last_major_30d_mean']:.1f}" if pd.notna(r["last_major_30d_mean"]) else "无记录"
        proj = f"{r['projected_years_to_37']:.1f}" if pd.notna(r["projected_years_to_37"]) else "N/A"
        a_below = "是" if r.get("currently_below_37") else "否"
        m_below = "是" if r.get("major_recovery_below_37") else ("?" if maj == "无记录" else "否")
        print(
            f"{r['device']:<6} {ann:>8} {a_below:>6} {maj:>12} {m_below:>6} "
            f"{proj:>12} {r['eol_status']:<18}"
        )

    # Save
    results["device_status"].to_csv(
        args.output_dir / "tables" / "eol_status.csv", index=False, encoding="utf-8-sig"
    )
    print(f"\n输出: {args.output_dir / 'eol_status.csv'}")


if __name__ == "__main__":
    main()
