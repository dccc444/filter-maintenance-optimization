"""Run Problem 2: estimate parameters and predict filter lifetimes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib").resolve()))

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
    parser.add_argument(
        "--max-years",
        type=int,
        default=25,
        help="Maximum follow-up; unfinished runs are reported as right-censored",
    )
    parser.add_argument(
        "--damage-share",
        type=float,
        default=0.20,
        help="Share of historical irreversible decline allocated to maintenance damage",
    )
    parser.add_argument(
        "--backtest-runs",
        type=int,
        default=100,
        help="Training-only temporal backtest simulations per device",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(parents=True, exist_ok=True)

    # Load Problem 1 outputs
    print("Loading Problem 1 outputs...")
    seasonality = load_seasonality_summary(args.tables_dir)
    params = estimate_params(
        args.tables_dir,
        args.processed_dir,
        seasonality,
        damage_share=args.damage_share,
    )
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
        major_rec = f"{p.major_recovery:.1f}"
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
            max_years=args.max_years,
        )
        summary = summarise_results(results, device)
        all_summaries.append(summary)

        print(
            f"  KM median lifetime: {summary['median_lifetime_days']:.0f} days "
            f"({summary['median_lifetime_days']/365.25:.1f} years)"
        )
        upper = (
            f"{summary['ci95_high_days']:.0f}"
            if pd.notna(summary["ci95_high_days"])
            else f">{summary['max_followup_days']:.0f}"
        )
        print(f"  95% PI: [{summary['ci95_low_days']:.0f}, {upper}] days")
        print(
            f"  Right-censored: {summary['right_censored_runs']}/"
            f"{summary['n_runs']}"
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
                "seasonal_sin_1y": p.seasonal_sin_1y,
                "seasonal_cos_1y": p.seasonal_cos_1y,
                "seasonal_sin_2y": p.seasonal_sin_2y,
                "seasonal_cos_2y": p.seasonal_cos_2y,
                "medium_recovery": p.medium_recovery,
                "major_recovery": p.major_recovery,
                "medium_damage": p.medium_damage,
                "major_damage": p.major_damage,
                "sigma": p.sigma,
                "recent_level": p.recent_level,
                "days_since_maintenance": p.days_since_maintenance,
                "last_maintenance_type": p.last_maintenance_type,
                "mediums_since_major": p.mediums_since_major,
                "historical_irreversible_decline_per_year":
                    p.historical_irreversible_decline_per_year,
                "damage_share_assumption": p.damage_share_assumption,
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
    interface = {
        "version": 1,
        "purpose": "第三问维护策略优化与第四问长期风险评估的统一输入接口",
        "simulation_entrypoint": "src.problem2.simulate.simulate_one_device",
        "prediction_start": "2026-04-10",
        "end_of_life_rule": {
            "annual_window_days": 365,
            "permeability_threshold": 37.0,
            "post_major_window_days": 30,
            "logic": "annual_mean_below_threshold AND post_major_mean_below_threshold",
        },
        "baseline": {
            "maintenance_damage_share": args.damage_share,
            "maximum_followup_years": args.max_years,
            "monte_carlo_runs_per_device": args.n_runs,
            "costs": {
                "purchase": 300.0,
                "medium_maintenance": 3.0,
                "major_maintenance": 12.0,
            },
        },
        "required_sensitivity_scenarios": {
            "maintenance_damage_share": [0.0, 0.2, 0.4],
            "devices_without_observed_major_maintenance": ["A4", "A8"],
            "right_censoring": "use event_observed and Kaplan-Meier summaries",
        },
        "policy_variables": [
            "medium_gap_mean",
            "major_gap_mean",
            "medium_between_major",
        ],
        "outputs": [
            "total_lifetime_days",
            "event_observed",
            "right_censored",
            "medium_maintenance_count",
            "major_maintenance_count",
            "mean_simulated_permeability",
            "final_annual_mean",
            "post_major_30d_mean",
            "total_cost",
            "annualized_cost",
        ],
        "device_parameter_table": "tables/model_parameters.csv",
        "baseline_lifetime_table": "tables/lifetime_predictions.csv",
        "validation_table": "tables/backtest_metrics.csv",
    }
    (args.output_dir / "problem3_4_interface.json").write_text(
        json.dumps(interface, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not args.device:
        from .backtest import compute_backtest_metrics, run_backtest
        from .plots import (
            fig01_parameter_heatmap,
            fig02_lifetime_predictions,
            fig03_maintenance_intervals,
            fig04_eol_status,
            fig05_monthly_maintenance,
        )
        from .step3_maintenance import analyse_maintenance_patterns
        from .step4_eol import evaluate_end_of_life_status

        patterns = analyse_maintenance_patterns(args.processed_dir)
        patterns["devices"].to_csv(
            args.output_dir / "tables" / "maintenance_patterns.csv",
            index=False,
            encoding="utf-8-sig",
        )
        (args.output_dir / "tables" / "maintenance_patterns.json").write_text(
            json.dumps(
                {
                    "population": patterns["population"],
                    "monthly_distribution": patterns["monthly_distribution"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        eol = evaluate_end_of_life_status(args.processed_dir, args.tables_dir)
        eol["device_status"].to_csv(
            args.output_dir / "tables" / "eol_status.csv",
            index=False,
            encoding="utf-8-sig",
        )
        backtest = run_backtest(
            args.tables_dir,
            args.processed_dir,
            n_runs=args.backtest_runs,
        )
        compute_backtest_metrics(backtest).to_csv(
            args.output_dir / "tables" / "backtest_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        backtest.to_csv(
            args.output_dir / "tables" / "backtest_trajectories.csv",
            index=False,
            encoding="utf-8-sig",
        )
        figure_dir = args.output_dir / "figures"
        fig01_parameter_heatmap(figure_dir)
        fig02_lifetime_predictions(figure_dir)
        fig03_maintenance_intervals(figure_dir)
        fig04_eol_status(figure_dir)
        fig05_monthly_maintenance(figure_dir)

    print(f"\nResults saved to {args.output_dir.resolve()}")
    print("Problem 2 tables, validation outputs and figures are complete.")


if __name__ == "__main__":
    main()
