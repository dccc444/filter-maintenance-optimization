"""Figures for Problem 3."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib").resolve()))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    if font_path.exists():
        import matplotlib.font_manager as fm

        fm.fontManager.addfont(str(font_path))
        matplotlib.rcParams["font.family"] = fm.FontProperties(
            fname=str(font_path)
        ).get_name()
    matplotlib.rcParams["axes.unicode_minus"] = False


def make_problem3_plots(
    screening: pd.DataFrame,
    comparison: pd.DataFrame,
    portfolio: pd.DataFrame,
    output_dir: Path,
) -> None:
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)

    order = ["当前固定方案", "优化固定周期", "透水率触发", "状态触发", "推荐混合方案"]
    palette = ["#7f8c8d", "#3498db", "#f39c12", "#9b59b6", "#27ae60"]
    view = portfolio.set_index("strategy").reindex(order).dropna().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(view["strategy"], view["factory_annual_cost"], color=palette[: len(view)])
    ax.bar_label(bars, fmt="%.1f", padding=4)
    ax.set_ylabel("全厂长期年均成本（万元/年）")
    ax.set_title("不同维护策略的全厂长期年均成本")
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    fig.savefig(output_dir / "01_策略年均成本比较.png", dpi=180)
    plt.close(fig)

    family_colors = {"固定周期": "#3498db", "透水率触发": "#f39c12", "状态触发": "#9b59b6"}
    fig, ax = plt.subplots(figsize=(10, 6))
    aggregate = (
        screening.groupby(["candidate_id", "family"], as_index=False)
        .agg(
            annual_cost=("renewal_annual_cost", "sum"),
            median_lifetime=("median_lifetime_days", "mean"),
        )
    )
    for family, group in aggregate.groupby("family"):
        ax.scatter(
            group["median_lifetime"] / 365.25,
            group["annual_cost"],
            label=family,
            alpha=0.75,
            s=55,
            color=family_colors[family],
        )
    ax.set_xlabel("10台设备平均剩余寿命中位数（年）")
    ax.set_ylabel("全厂长期年均成本（万元/年）")
    ax.set_title("候选维护参数的成本—寿命权衡")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "02_候选策略成本寿命权衡.png", dpi=180)
    plt.close(fig)

    pivot = comparison.pivot(
        index="device", columns="strategy", values="renewal_annual_cost"
    ).reindex([f"A{i}" for i in range(1, 11)])
    savings = 100 * (
        pivot["当前固定方案"] - pivot["推荐混合方案"]
    ) / pivot["当前固定方案"]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = np.where(savings >= 0, "#27ae60", "#e74c3c")
    bars = ax.bar(savings.index, savings.values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_ylabel("年均成本下降率")
    ax.set_title("推荐方案相对当前方案的设备级节约比例")
    fig.tight_layout()
    fig.savefig(output_dir / "03_设备级成本节约.png", dpi=180)
    plt.close(fig)

    rates = (
        comparison.groupby("strategy", as_index=False)[
            ["medium_maintenances_per_year", "major_maintenances_per_year"]
        ]
        .sum()
        .set_index("strategy")
        .reindex(order)
        .dropna()
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        rates.index,
        rates["medium_maintenances_per_year"],
        label="中维护",
        color="#3498db",
    )
    ax.bar(
        rates.index,
        rates["major_maintenances_per_year"],
        bottom=rates["medium_maintenances_per_year"],
        label="大维护",
        color="#e67e22",
    )
    ax.set_ylabel("10台设备合计维护次数/年")
    ax.set_title("不同策略的维护强度")
    ax.tick_params(axis="x", rotation=12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "04_策略维护强度比较.png", dpi=180)
    plt.close(fig)
