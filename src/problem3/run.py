"""Run Problem 3 maintenance-policy optimisation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib").resolve()))

from src.problem2.model import (
    DEVICE_ORDER,
    estimate_params,
    extract_maintenance_schedule,
    load_seasonality_summary,
    load_warmup_history,
)

from .optimize import choose_best_specs, evaluate_policy
from .plots import make_problem3_plots
from .policies import PolicySpec, generate_candidate_specs


STRATEGY_BY_FAMILY = {
    "固定周期": "优化固定周期",
    "透水率触发": "透水率触发",
    "状态触发": "状态触发",
}


def _portfolio_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    baseline_cost = comparison.loc[
        comparison["strategy"] == "当前固定方案", "renewal_annual_cost"
    ].sum()
    records = []
    for strategy, group in comparison.groupby("strategy", sort=False):
        total_cost = float(group["renewal_annual_cost"].sum())
        records.append(
            {
                "strategy": strategy,
                "factory_annual_cost": total_cost,
                "average_cost_per_filter": total_cost / len(group),
                "cost_saving_rate": (baseline_cost - total_cost) / baseline_cost,
                "average_median_lifetime_years": float(
                    group["median_lifetime_days"].mean() / 365.25
                ),
                "total_medium_maintenances_per_year": float(
                    group["medium_maintenances_per_year"].sum()
                ),
                "total_major_maintenances_per_year": float(
                    group["major_maintenances_per_year"].sum()
                ),
                "mean_eol_within_2y_probability": float(
                    group["eol_within_2y_probability"].mean()
                ),
                "maximum_right_censored_share": float(
                    group["right_censored_share"].max()
                ),
            }
        )
    return pd.DataFrame(records)


def _spec_record(device: str, strategy: str, spec: PolicySpec) -> dict:
    return {"device": device, "strategy": strategy, **spec.to_record()}


def main() -> None:
    parser = argparse.ArgumentParser(description="B题第三问：最优维护方案")
    parser.add_argument("--tables-dir", type=Path, default=Path("outputs/problem1/tables"))
    parser.add_argument(
        "--processed-dir", type=Path, default=Path("outputs/problem1/processed")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/problem3"))
    parser.add_argument("--screen-runs", type=int, default=40)
    parser.add_argument("--final-runs", type=int, default=300)
    parser.add_argument("--scenario-runs", type=int, default=100)
    parser.add_argument("--screen-years", type=int, default=15)
    parser.add_argument("--max-years", type=int, default=25)
    args = parser.parse_args()

    tables_dir = args.output_dir / "tables"
    figures_dir = args.output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    seasonality = load_seasonality_summary(args.tables_dir)
    params = estimate_params(
        args.tables_dir, args.processed_dir, seasonality, damage_share=0.20
    )
    schedules = extract_maintenance_schedule(args.processed_dir)
    warmup = load_warmup_history(args.processed_dir)
    specs = generate_candidate_specs()

    print(
        f"Screening {len(specs)} candidates × {len(DEVICE_ORDER)} devices × "
        f"{args.screen_runs} Monte Carlo runs..."
    )
    screening_records = []
    for device in DEVICE_ORDER:
        for spec in specs:
            screening_records.append(
                evaluate_policy(
                    params[device],
                    schedules[device],
                    warmup[device],
                    strategy=STRATEGY_BY_FAMILY[spec.family],
                    spec=spec,
                    n_runs=args.screen_runs,
                    max_years=args.screen_years,
                )
            )
        print(f"  {device} complete")
    screening = pd.DataFrame(screening_records)
    screening.to_csv(
        tables_dir / "candidate_screening.csv", index=False, encoding="utf-8-sig"
    )

    selected = choose_best_specs(screening_records, specs)
    selected_records = []
    for device in DEVICE_ORDER:
        for family, strategy in STRATEGY_BY_FAMILY.items():
            selected_records.append(
                _spec_record(device, strategy, selected[(device, family)])
            )

    print("Re-evaluating current and selected policies at full Monte Carlo size...")
    comparison_records = []
    for device in DEVICE_ORDER:
        comparison_records.append(
            evaluate_policy(
                params[device],
                schedules[device],
                warmup[device],
                strategy="当前固定方案",
                spec=None,
                n_runs=args.final_runs,
                max_years=args.max_years,
            )
        )
        for family, strategy in STRATEGY_BY_FAMILY.items():
            comparison_records.append(
                evaluate_policy(
                    params[device],
                    schedules[device],
                    warmup[device],
                    strategy=strategy,
                    spec=selected[(device, family)],
                    n_runs=args.final_runs,
                    max_years=args.max_years,
                )
            )
        print(f"  {device} complete")
    comparison = pd.DataFrame(comparison_records)
    recommended_rows = []
    for device in DEVICE_ORDER:
        candidates = comparison.loc[
            (comparison["device"] == device)
            & (comparison["strategy"] != "当前固定方案")
        ]
        feasible = candidates.loc[candidates["right_censored_share"] <= 0.20]
        winner = (feasible if len(feasible) else candidates).sort_values(
            ["renewal_annual_cost", "median_lifetime_days"],
            ascending=[True, False],
        ).iloc[0]
        recommended_spec = selected[(device, str(winner["family"]))]
        selected[(device, "推荐混合方案")] = recommended_spec
        selected_records.append(
            _spec_record(device, "推荐混合方案", recommended_spec)
        )
        recommended = winner.copy()
        recommended["strategy"] = "推荐混合方案"
        recommended_rows.append(recommended)
    comparison = pd.concat(
        [comparison, pd.DataFrame(recommended_rows)],
        ignore_index=True,
        sort=False,
    )
    pd.DataFrame(selected_records).to_csv(
        tables_dir / "selected_policies.csv", index=False, encoding="utf-8-sig"
    )
    comparison.to_csv(
        tables_dir / "strategy_comparison.csv", index=False, encoding="utf-8-sig"
    )
    portfolio = _portfolio_summary(comparison)
    portfolio.to_csv(
        tables_dir / "portfolio_summary.csv", index=False, encoding="utf-8-sig"
    )

    print("Checking baseline and recommended policies under damage scenarios...")
    robustness_records = []
    for damage_share in (0.0, 0.4):
        scenario_params = estimate_params(
            args.tables_dir,
            args.processed_dir,
            seasonality,
            damage_share=damage_share,
        )
        for device in DEVICE_ORDER:
            for strategy, spec in (
                ("当前固定方案", None),
                ("推荐混合方案", selected[(device, "推荐混合方案")]),
            ):
                record = evaluate_policy(
                    scenario_params[device],
                    schedules[device],
                    warmup[device],
                    strategy=strategy,
                    spec=spec,
                    n_runs=args.scenario_runs,
                    max_years=args.max_years,
                )
                record["damage_share"] = damage_share
                robustness_records.append(record)
    baseline_scenario = comparison.loc[
        comparison["strategy"].isin(["当前固定方案", "推荐混合方案"])
    ].copy()
    baseline_scenario["damage_share"] = 0.2
    robustness = pd.concat(
        [baseline_scenario, pd.DataFrame(robustness_records)],
        ignore_index=True,
        sort=False,
    )
    robustness.to_csv(
        tables_dir / "damage_scenario_robustness.csv",
        index=False,
        encoding="utf-8-sig",
    )

    machine_policy = {
        device: selected[(device, "推荐混合方案")].to_record()
        for device in DEVICE_ORDER
    }
    (args.output_dir / "recommended_policy.json").write_text(
        json.dumps(
            {
                "version": 1,
                "objective": "minimize renewal-equivalent annual cost",
                "cost_unit": "万元",
                "purchase_cost": 300,
                "medium_maintenance_cost": 3,
                "major_maintenance_cost": 12,
                "damage_share_baseline": 0.20,
                "policies": machine_policy,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    make_problem3_plots(screening, comparison, portfolio, figures_dir)
    print(portfolio.to_string(index=False))
    print(f"Problem 3 outputs saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
