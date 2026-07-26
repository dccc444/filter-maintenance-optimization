"""Figures for Problem 4 price sensitivity."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib").resolve()))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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


def make_problem4_plots(summary: pd.DataFrame, output_dir: Path) -> None:
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    mappings = [
        ("购置价单因素", "purchase_factor", "过滤器购置价格"),
        ("中维护价单因素", "medium_factor", "中维护价格"),
        ("大维护价单因素", "major_factor", "大维护价格"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    for ax, (scenario_type, factor, title) in zip(axes, mappings):
        view = summary.loc[summary["scenario_type"] == scenario_type].sort_values(
            factor
        )
        ax.plot(
            view[factor],
            view["optimal_factory_annual_cost"],
            marker="o",
            color="#2c7fb8",
            label="重新优化成本",
        )
        ax.plot(
            view[factor],
            view["original_plan_annual_cost"],
            linestyle="--",
            color="#7f8c8d",
            label="第三问方案成本",
        )
        ax2 = ax.twinx()
        ax2.plot(
            view[factor],
            100 * view["relative_regret"],
            color="#d7301f",
            marker="s",
            label="后悔值",
        )
        ax2.axhline(5, color="#d7301f", linestyle=":", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("相对基准价格倍数")
        ax.set_ylabel("全厂年均成本（万元/年）")
        ax2.set_ylabel("第三问方案相对后悔值（%）")
    axes[0].legend(loc="upper left")
    fig.suptitle("单因素价格敏感性与第三问方案后悔值", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_dir / "01_单因素价格敏感性.png", dpi=180)
    plt.close(fig)

    for scenario_type, x, y, filename, title, x_label, y_label in [
        (
            "购置价×维护价",
            "maintenance_factor",
            "purchase_factor",
            "02_购置维护价格双因素.png",
            "购置价格与维护价格共同变化时的后悔值",
            "中、大维护价格倍数",
            "过滤器购置价格倍数",
        ),
        (
            "中维护价×大维护价",
            "major_factor",
            "medium_factor",
            "03_中大维护价格双因素.png",
            "中维护与大维护价格共同变化时的后悔值",
            "大维护价格倍数",
            "中维护价格倍数",
        ),
    ]:
        view = summary.loc[summary["scenario_type"] == scenario_type].copy()
        if scenario_type == "购置价×维护价":
            view["maintenance_factor"] = view["medium_factor"]
        pivot = view.pivot(index=y, columns=x, values="relative_regret") * 100
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            pivot.sort_index(ascending=False),
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            vmin=0,
            cbar_kws={"label": "第三问方案相对后悔值（%）"},
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    view = summary.loc[summary["scenario_type"] == "购置价单因素"].sort_values(
        "purchase_factor"
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        view["purchase_factor"],
        view["replacement_rate_per_year"],
        marker="o",
        label="更换次数/年",
        linewidth=2,
    )
    ax.plot(
        view["purchase_factor"],
        view["medium_maintenances_per_year"] / 10,
        marker="s",
        label="中维护次数/年（除以10）",
    )
    ax.plot(
        view["purchase_factor"],
        view["major_maintenances_per_year"],
        marker="^",
        label="大维护次数/年",
    )
    ax.set_xlabel("过滤器购置价格倍数")
    ax.set_ylabel("全厂决策强度")
    ax.set_title("购置价格变化对维护与更换决策的影响")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "04_购置价格与决策强度.png", dpi=180)
    plt.close(fig)
