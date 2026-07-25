"""Run Problem 2: estimate parameters and predict filter lifetimes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .model import (
    DEVICE_ORDER,
    estimate_params,
    extract_maintenance_schedule,
    load_seasonality_summary,
    load_warmup_history,
)
from .simulate import monte_carlo_simulate, summarise_results


def main() -> None:
    parser = argparse.ArgumentParser(description="B题第二问：过滤器寿命预测")
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("outputs/problem1/tables"),
    )
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
    parser.add_argument(
        "--n-runs",
        type=int,
        default=1000,
        help="Monte Carlo simulation runs per device",
    )
    parser.add_argument("--device", type=str, default=None, help="Single device to run")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load Problem 1 outputs
    print("Loading Problem 1 outputs...")
    seasonality = load_seasonality_summary(args.tables_dir)
    params = estimate_params(args.tables_dir, args.processed_dir, seasonality)
    schedules = extract_maintenance_schedule(args.processed_dir)

    # Print parameter summary
    print("\n=== Parameter Estimates ===")
    print(
        f"{'Device':<8} {'alpha/day':>10} {'beta/day':>10} "
        f"{'Med.Rec.':>10} {'Maj.Rec.':>10} {'sigma':>8} {'Recent':>8}"
    )
    print("-" * 70)
    for device in DEVICE_ORDER:
        p = params[device]
        major_rec = f"{p.major_recovery:.1f}" if p.major_recovery else "N/A"
        print(
            f"{device:<8} {p.alpha:>10.4f} {p.beta:>10.4f} "
            f"{p.medium_recovery:>10.1f} {major_rec:>10} "
            f"{p.sigma:>8.1f} {p.recent_level:>8.1f}"
        )

    print("\n=== Maintenance Schedules ===")
    print(f"{'Device':<8} {'Med.Interval':>14} {'Maj.Interval':>14} {'Med/Maj':>10}")
    print("-" * 50)
    for device in DEVICE_ORDER:
        s = schedules[device]
        print(
            f"{device:<8} {s.medium_interval_mean:>8.1f}±{s.medium_interval_std:<4.1f}"
            f" {s.major_interval_mean:>8.1f}±{s.major_interval_std:<4.1f}"
            f" {s.medium_between_major_mean:>10.1f}"
        )

    # Load warmup history
    warmup = load_warmup_history(args.processed_dir)

    # Run simulation
    devices_to_run = [args.device] if args.device else DEVICE_ORDER
    print(f"\n=== Monte Carlo Simulation ({args.n_runs} runs) ===")

    all_summaries = []
    for device in devices_to_run:
        print(f"\nSimulating {device}...")
        results, trajectories = monte_carlo_simulate(
            params[device],
            schedules[device],
            n_runs=args.n_runs,
            warmup_history=warmup.get(device),
            save_trajectories=5,
        )
        summary = summarise_results(results, device)
        all_summaries.append(summary)

        print(
            f"  Median lifetime: {summary['median_lifetime_days']:.0f} days "
            f"({summary['median_lifetime_days']/365.25:.1f} years)"
        )
        print(
            f"  95% CI: [{summary['ci95_low_days']:.0f}, "
            f"{summary['ci95_high_days']:.0f}] days"
        )
        print(f"  Median end date: {summary['median_end_date']}")
        print(
            f"  Median maintenance: "
            f"{summary['median_medium_count']:.0f} medium, "
            f"{summary['median_major_count']:.0f} major"
        )

    # Save results
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(
        args.output_dir / "tables" / "lifetime_predictions.csv", index=False, encoding="utf-8-sig"
    )

    # Save parameters
    param_records = []
    for device in DEVICE_ORDER:
        p = params[device]
        s = schedules[device]
        param_records.append(
            {
                "device": device,
                "alpha_per_day": p.alpha,
                "beta_per_day": p.beta,
                "seasonal_amplitude": p.seasonal_amplitude,
                "seasonal_peak_day": p.seasonal_peak_day,
                "medium_recovery": p.medium_recovery,
                "medium_retention": p.medium_retention,
                "major_recovery": p.major_recovery,
                "major_retention": p.major_retention,
                "sigma": p.sigma,
                "recent_level": p.recent_level,
                "days_since_maintenance": p.days_since_maintenance,
                "last_maintenance_type": p.last_maintenance_type,
                "medium_interval_mean": s.medium_interval_mean,
                "medium_interval_std": s.medium_interval_std,
                "major_interval_mean": s.major_interval_mean,
                "major_interval_std": s.major_interval_std,
                "medium_between_major_mean": s.medium_between_major_mean,
            }
        )
    pd.DataFrame(param_records).to_csv(
        args.output_dir / "tables" / "model_parameters.csv", index=False, encoding="utf-8-sig"
    )

    print(f"\nResults saved to {args.output_dir.resolve()}")
    print("Files: lifetime_predictions.csv, model_parameters.csv")


if __name__ == "__main__":
    main()
