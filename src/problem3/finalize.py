"""Resume Problem 3 after candidate search and final policy evaluation."""

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

from .optimize import evaluate_policy
from .plots import make_problem3_plots
from .policies import PolicySpec
from .run import _portfolio_summary


def _optional(record: pd.Series, name: str, caster):
    value = record.get(name)
    return None if pd.isna(value) else caster(value)


def _load_spec(record: pd.Series) -> PolicySpec:
    return PolicySpec(
        candidate_id=str(record["candidate_id"]),
        family=str(record["family"]),
        label=str(record["label"]),
        medium_gap=_optional(record, "medium_gap", float),
        major_gap=_optional(record, "major_gap", float),
        medium_between_major=int(record["medium_between_major"]),
        medium_trigger=_optional(record, "medium_trigger", float),
        major_trigger=_optional(record, "major_trigger", float),
        fouling_trigger=_optional(record, "fouling_trigger", float),
        reserve_threshold=_optional(record, "reserve_threshold", float),
        min_gap=int(record["min_gap"]),
        max_gap=int(record["max_gap"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="继续第三问损伤情景和图表生成")
    parser.add_argument("--tables-dir", type=Path, default=Path("outputs/problem1/tables"))
    parser.add_argument(
        "--processed-dir", type=Path, default=Path("outputs/problem1/processed")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/problem3"))
    parser.add_argument("--scenario-runs", type=int, default=100)
    parser.add_argument("--max-years", type=int, default=25)
    args = parser.parse_args()

    output_tables = args.output_dir / "tables"
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    screening = pd.read_csv(output_tables / "candidate_screening.csv")
    comparison = pd.read_csv(output_tables / "strategy_comparison.csv")
    selected_table = pd.read_csv(output_tables / "selected_policies.csv")
    selected_rows = selected_table.loc[
        selected_table["strategy"] == "推荐混合方案"
    ].set_index("device")
    selected = {
        device: _load_spec(selected_rows.loc[device]) for device in DEVICE_ORDER
    }

    seasonality = load_seasonality_summary(args.tables_dir)
    schedules = extract_maintenance_schedule(args.processed_dir)
    warmup = load_warmup_history(args.processed_dir)
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
                ("推荐混合方案", selected[device]),
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
        print(f"damage_share={damage_share:.1f} complete")

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
        output_tables / "damage_scenario_robustness.csv",
        index=False,
        encoding="utf-8-sig",
    )

    portfolio = _portfolio_summary(comparison)
    portfolio.to_csv(
        output_tables / "portfolio_summary.csv", index=False, encoding="utf-8-sig"
    )
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
                "policies": {
                    device: selected[device].to_record() for device in DEVICE_ORDER
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    make_problem3_plots(screening, comparison, portfolio, figures_dir)
    print(portfolio.to_string(index=False))


if __name__ == "__main__":
    main()
