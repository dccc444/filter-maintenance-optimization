from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib").resolve()))

from .analysis import AnalysisResults, run_analysis
from .plots import make_all_plots
from .report import build_report


def save_outputs(results: AnalysisResults, output_dir: Path) -> None:
    processed = output_dir / "processed"
    tables = output_dir / "tables"
    figures = output_dir / "figures"
    processed.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    results.hourly.to_csv(
        processed / "hourly_permeability_cleaned.csv", index=False, encoding="utf-8-sig"
    )
    results.daily.to_csv(
        processed / "daily_permeability.csv", index=False, encoding="utf-8-sig"
    )
    results.maintenance.to_csv(
        processed / "maintenance_records.csv", index=False, encoding="utf-8-sig"
    )

    table_map: dict[str, pd.DataFrame] = {
        "data_quality.csv": results.quality,
        "seasonality_details.csv": results.seasonality_details,
        "dominant_periods.csv": results.periods,
        "cycle_slopes.csv": results.cycle_slopes,
        "maintenance_event_metrics.csv": results.event_metrics,
        "maintenance_effect_summary.csv": results.event_summary,
        "maintenance_event_curve.csv": results.event_curve_summary,
        "maintenance_type_comparison.csv": results.maintenance_comparison,
        "envelope_points.csv": results.envelope_points,
        "envelope_decay.csv": results.envelope_summary,
        "device_indicators.csv": results.indicators,
        "monthly_coverage.csv": results.monthly_coverage,
    }
    for name, frame in table_map.items():
        frame.to_csv(tables / name, index=False, encoding="utf-8-sig")
    (tables / "seasonality_summary.json").write_text(
        json.dumps(results.seasonality_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    make_all_plots(results, figures)
    build_report(results, output_dir / "第一问分析报告.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 B 题第一问完整分析")
    parser.add_argument("--data-dir", type=Path, default=Path("B题附件"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/problem1")
    )
    args = parser.parse_args()
    results = run_analysis(args.data_dir)
    save_outputs(results, args.output_dir)
    print(f"第一问分析完成：{args.output_dir.resolve()}")
    print(
        f"小时记录 {len(results.hourly):,}，日记录 {len(results.daily):,}，"
        f"维护事件 {len(results.maintenance)}，有效维护周期 {len(results.cycle_slopes)}"
    )


if __name__ == "__main__":
    main()
