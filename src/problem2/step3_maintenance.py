"""Step 3: Analyse and export current fixed maintenance patterns."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .model import DEVICE_ORDER, extract_maintenance_schedule, load_maintenance_records


def analyse_maintenance_patterns(processed_dir: Path) -> dict:
    """Deep-dive into maintenance patterns from historical records.

    Returns a dict with per-device and population-level statistics.
    """
    maintenance = load_maintenance_records(processed_dir)
    schedules = extract_maintenance_schedule(processed_dir)

    # Per-device analysis
    device_stats = []
    for device in DEVICE_ORDER:
        dev = maintenance[maintenance["device"] == device].sort_values("date")
        s = schedules[device]

        medium = dev[dev["maintenance_type"] == "中维护"]
        major = dev[dev["maintenance_type"] == "大维护"]

        # Intervals between consecutive maintenances
        dates = dev["date"].to_numpy()
        if len(dates) >= 2:
            intervals = np.diff(dates).astype("timedelta64[D]").astype(float)
        else:
            intervals = np.array([])

        # Monthly distribution
        dev_copy = dev.copy()
        dev_copy["month"] = dev_copy["date"].dt.month
        month_counts = dev_copy.groupby("month").size()

        # Compute seasonal concentration: chi-squared test against uniform
        total = len(dev)
        expected = total / 12
        observed = np.array([month_counts.get(m, 0) for m in range(1, 13)])
        if total >= 12:
            chi2 = float(np.sum((observed - expected) ** 2 / expected))
        else:
            chi2 = float("nan")

        device_stats.append(
            {
                "device": device,
                "total_events": len(dev),
                "medium_count": len(medium),
                "major_count": len(major),
                "first_date": str(dev["date"].min().date()) if len(dev) else None,
                "last_date": str(dev["date"].max().date()) if len(dev) else None,
                "interval_mean_days": float(np.mean(intervals)) if len(intervals) else None,
                "interval_std_days": float(np.std(intervals, ddof=1)) if len(intervals) > 1 else None,
                "interval_min_days": float(np.min(intervals)) if len(intervals) else None,
                "interval_max_days": float(np.max(intervals)) if len(intervals) else None,
                "medium_interval_mean": s.medium_interval_mean,
                "medium_interval_std": s.medium_interval_std,
                "major_interval_mean": s.major_interval_mean,
                "major_interval_std": s.major_interval_std,
                "medium_between_major": s.medium_between_major_mean,
                "chi2_seasonality": round(chi2, 2) if not np.isnan(chi2) else None,
                "peak_month": int(np.argmax(observed) + 1) if total > 0 else None,
                "trough_month": int(np.argmin(observed) + 1) if total > 0 else None,
            }
        )

    df_devices = pd.DataFrame(device_stats)

    # Population-level summary
    med_intervals = df_devices["medium_interval_mean"].dropna()
    maj_intervals = df_devices["major_interval_mean"].dropna()
    med_between = df_devices["medium_between_major"].dropna()

    population = {
        "n_devices": 10,
        "total_maintenance_events": int(df_devices["total_events"].sum()),
        "total_medium": int(df_devices["medium_count"].sum()),
        "total_major": int(df_devices["major_count"].sum()),
        "medium_interval": {
            "mean": float(med_intervals.mean()),
            "std": float(med_intervals.std(ddof=1)),
            "q25": float(med_intervals.quantile(0.25)),
            "q75": float(med_intervals.quantile(0.75)),
        },
        "major_interval": {
            "mean": float(maj_intervals.mean()),
            "std": float(maj_intervals.std(ddof=1)),
            "q25": float(maj_intervals.quantile(0.25)),
            "q75": float(maj_intervals.quantile(0.75)),
        },
        "medium_between_major": {
            "mean": float(med_between.mean()),
            "std": float(med_between.std(ddof=1)),
        },
    }

    # Seasonal pattern: aggregate all devices by month
    all_maint = maintenance.copy()
    all_maint["month"] = all_maint["date"].dt.month
    month_dist = all_maint.groupby("month").size().to_dict()
    seasonal = {str(m): month_dist.get(m, 0) for m in range(1, 13)}

    return {
        "devices": df_devices,
        "population": population,
        "monthly_distribution": seasonal,
        "schedules": schedules,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Step 3: Maintenance pattern analysis")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("outputs/problem1/processed"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/problem2"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = analyse_maintenance_patterns(args.processed_dir)

    # Save
    results["devices"].to_csv(
        args.output_dir / "tables" / "maintenance_patterns.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with open(args.output_dir / "tables" / "maintenance_patterns.json", "w") as f:
        json.dump(
            {
                "population": results["population"],
                "monthly_distribution": results["monthly_distribution"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Print summary
    print("=== 第三步：当前固定维护规律分析 ===\n")

    pop = results["population"]
    print(f"总计维护事件: {pop['total_maintenance_events']}")
    print(f"  中维护: {pop['total_medium']} 次, 大维护: {pop['total_major']} 次")
    print()

    print("设备级维护间隔:")
    print(
        f"{'设备':<6} {'总次数':>5} {'中维护':>5} {'大维护':>5} "
        f"{'中间隔':>8} {'中间隔std':>9} {'大间隔':>8} {'中/大':>6} {'峰值月':>6}"
    )
    print("-" * 75)
    for _, r in results["devices"].iterrows():
        peak = int(r["peak_month"]) if pd.notna(r["peak_month"]) else "-"
        print(
            f"{r['device']:<6} {r['total_events']:>5.0f} {r['medium_count']:>5.0f} "
            f"{r['major_count']:>5.0f} "
            f"{r['medium_interval_mean']:>8.1f} {r['medium_interval_std']:>9.1f} "
            f"{r['major_interval_mean']:>8.1f} {r['medium_between_major']:>6.1f} "
            f"{peak:>6}"
        )

    print("\n月度维护分布 (全部设备合计):")
    monthly = results["monthly_distribution"]
    for m in range(1, 13):
        count = monthly.get(str(m), 0)
        bar = "█" * count
        print(f"  {m:>2}月: {count:>3}次  {bar}")

    print(f"\n人口级统计:")
    mi = pop["medium_interval"]
    print(f"  中维护间隔: {mi['mean']:.1f} ± {mi['std']:.1f} 天 (IQR: {mi['q25']:.0f}-{mi['q75']:.0f})")
    ma = pop["major_interval"]
    print(f"  大维护间隔: {ma['mean']:.1f} ± {ma['std']:.1f} 天 (IQR: {ma['q25']:.0f}-{ma['q75']:.0f})")
    mb = pop["medium_between_major"]
    print(f"  大维护间中维护次数: {mb['mean']:.1f} ± {mb['std']:.1f}")

    print(f"\n输出: {args.output_dir / 'maintenance_patterns.csv'}")
    print(f"      {args.output_dir / 'maintenance_patterns.json'}")


if __name__ == "__main__":
    main()
