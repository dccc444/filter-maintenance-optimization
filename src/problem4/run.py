"""Run Problem 4 price sensitivity and policy robustness analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.problem2.model import (
    DEVICE_ORDER,
    estimate_params,
    extract_maintenance_schedule,
    load_seasonality_summary,
    load_warmup_history,
)
from src.problem3.optimize import evaluate_policy
from src.problem3.policies import generate_candidate_specs

from .plots import make_problem4_plots
from .sensitivity import (
    coarse_price_scenarios,
    full_price_scenarios,
    one_factor_applicability,
    optimise_scenarios,
    select_top_pairs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="B题第四问：价格敏感性分析")
    parser.add_argument("--tables-dir", type=Path, default=Path("outputs/problem1/tables"))
    parser.add_argument(
        "--processed-dir", type=Path, default=Path("outputs/problem1/processed")
    )
    parser.add_argument(
        "--problem3-dir", type=Path, default=Path("outputs/problem3")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/problem4"))
    parser.add_argument("--refine-runs", type=int, default=200)
    parser.add_argument("--max-years", type=int, default=25)
    parser.add_argument("--top-n", type=int, default=3)
    args = parser.parse_args()

    output_tables = args.output_dir / "tables"
    figures_dir = args.output_dir / "figures"
    output_tables.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    screening = pd.read_csv(
        args.problem3_dir / "tables" / "candidate_screening.csv"
    )
    selected = pd.read_csv(
        args.problem3_dir / "tables" / "selected_policies.csv"
    )
    original_rows = selected.loc[
        selected["strategy"] == "推荐混合方案"
    ]
    original_policy = dict(
        zip(original_rows["device"], original_rows["candidate_id"])
    )
    candidate_specs = {
        spec.candidate_id: spec for spec in generate_candidate_specs()
    }

    pairs = select_top_pairs(
        screening,
        coarse_price_scenarios(),
        top_n=args.top_n,
    )
    pairs.update(original_policy.items())
    print(
        f"Refining {len(pairs)} device-candidate pairs × "
        f"{args.refine_runs} Monte Carlo runs..."
    )

    seasonality = load_seasonality_summary(args.tables_dir)
    params = estimate_params(
        args.tables_dir, args.processed_dir, seasonality, damage_share=0.20
    )
    schedules = extract_maintenance_schedule(args.processed_dir)
    warmup = load_warmup_history(args.processed_dir)
    refined_records = []
    for device in DEVICE_ORDER:
        device_pairs = sorted(candidate for dev, candidate in pairs if dev == device)
        for candidate_id in device_pairs:
            refined_records.append(
                evaluate_policy(
                    params[device],
                    schedules[device],
                    warmup[device],
                    strategy="第四问候选复核",
                    spec=candidate_specs[candidate_id],
                    n_runs=args.refine_runs,
                    max_years=args.max_years,
                )
            )
        print(f"  {device}: {len(device_pairs)} candidates complete")
    refined = pd.DataFrame(refined_records)
    refined.to_csv(
        output_tables / "refined_candidate_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    scenarios = full_price_scenarios()
    device_optima, summary = optimise_scenarios(
        refined,
        scenarios,
        original_policy,
    )
    scenarios.to_csv(
        output_tables / "price_scenarios.csv", index=False, encoding="utf-8-sig"
    )
    device_optima.to_csv(
        output_tables / "scenario_device_optima.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        output_tables / "sensitivity_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    applicability = one_factor_applicability(summary)
    applicability.to_csv(
        output_tables / "original_policy_applicability.csv",
        index=False,
        encoding="utf-8-sig",
    )

    base = summary.loc[
        (summary["scenario_type"] == "购置价单因素")
        & (summary["purchase_factor"] == 1.0)
    ].iloc[0]
    decision = {
        "version": 1,
        "tested_price_factor_range": [0.5, 1.5],
        "near_optimal_definition": "relative regret <= 5%",
        "base_price_result": {
            "reoptimized_factory_annual_cost": float(
                base["optimal_factory_annual_cost"]
            ),
            "problem3_plan_annual_cost": float(base["original_plan_annual_cost"]),
            "relative_regret": float(base["relative_regret"]),
        },
        "one_factor_applicability": applicability.to_dict(orient="records"),
    }
    (args.output_dir / "policy_robustness.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    make_problem4_plots(summary, figures_dir)
    print(applicability.to_string(index=False))
    print(f"Problem 4 outputs saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
