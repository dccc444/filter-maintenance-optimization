from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import seaborn as sns

from .analysis import DEVICE_ORDER, MAINTENANCE_ORDER, AnalysisResults


COLORS = {"中维护": "#E69F00", "大维护": "#D55E00"}


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        chinese_font = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        chinese_font = "DejaVu Sans"
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = [chinese_font, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_daily_series(results: AnalysisResults, output_dir: Path) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(16, 20), sharex=True)
    for ax, device in zip(axes.flat, DEVICE_ORDER):
        data = results.daily.loc[results.daily["device"] == device]
        ax.plot(
            data["date"],
            data["permeability_median"],
            color="#0072B2",
            linewidth=0.8,
            label="日中位数",
        )
        events = results.maintenance.loc[results.maintenance["device"] == device]
        for maintenance_type in MAINTENANCE_ORDER:
            subset = events.loc[events["maintenance_type"] == maintenance_type]
            for date in subset["date"]:
                ax.axvline(
                    date,
                    color=COLORS[maintenance_type],
                    linewidth=0.8 if maintenance_type == "中维护" else 1.4,
                    alpha=0.65,
                )
        ax.axhline(37, color="#555555", linestyle="--", linewidth=0.8)
        ax.set_title(device)
        ax.set_ylabel("透水率")
    axes[-1, 0].set_xlabel("日期")
    axes[-1, 1].set_xlabel("日期")
    fig.suptitle("10 台过滤器日透水率与维护事件", fontsize=18, y=1.01)
    fig.text(
        0.5,
        0.002,
        "橙色：中维护；红色：大维护；虚线：阈值 37",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout()
    _save(fig, output_dir / "01_日透水率与维护事件.png")


def plot_monthly_coverage(results: AnalysisResults, output_dir: Path) -> None:
    pivot = results.monthly_coverage.pivot(
        index="device", columns="month", values="hour_coverage"
    ).reindex(DEVICE_ORDER)
    fig, ax = plt.subplots(figsize=(18, 5))
    sns.heatmap(
        pivot,
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "平均小时覆盖率"},
        ax=ax,
    )
    ax.set_title("数据完整性：设备—月份小时覆盖率")
    ax.set_xlabel("月份")
    ax.set_ylabel("设备")
    plt.xticks(rotation=60, ha="right")
    _save(fig, output_dir / "02_数据缺失热力图.png")


def plot_seasonality(results: AnalysisResults, output_dir: Path) -> None:
    curve = results.seasonality_details.loc[
        results.seasonality_details["table"] == "seasonal_curve"
    ].copy()
    curve["term"] = pd.to_numeric(curve["term"])
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(curve["term"], curve["coefficient"], color="#009E73", linewidth=2)
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xlabel("年内日序")
    ax.set_ylabel("季节效应（透水率）")
    summary = results.seasonality_summary
    ax.set_title(
        f"年/半年谐波季节效应（振幅={summary['seasonal_amplitude']:.2f}, "
        f"F={summary['f_statistic']:.1f}, p={summary['p_value']:.2g}）"
    )
    _save(fig, output_dir / "03_季节效应曲线.png")


def plot_cycle_slopes(results: AnalysisResults, output_dir: Path) -> None:
    frame = results.cycle_slopes.copy()
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.boxplot(
        data=frame,
        x="device",
        y="decline_rate_per_day",
        order=DEVICE_ORDER,
        color="#56B4E9",
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=frame,
        x="device",
        y="decline_rate_per_day",
        order=DEVICE_ORDER,
        color="#333333",
        alpha=0.55,
        size=3,
        ax=ax,
    )
    ax.axhline(0, color="#555555", linestyle="--", linewidth=0.8)
    ax.set_xlabel("设备")
    ax.set_ylabel("自然下降率（透水率/日）")
    ax.set_title("维护周期内季节校正后的自然下降率")
    _save(fig, output_dir / "04_维护周期下降率.png")


def plot_event_curve(results: AnalysisResults, output_dir: Path) -> None:
    frame = results.event_curve_summary
    fig, ax = plt.subplots(figsize=(12, 6))
    for maintenance_type in MAINTENANCE_ORDER:
        subset = frame.loc[frame["maintenance_type"] == maintenance_type].sort_values(
            "relative_day"
        )
        x = subset["relative_day"].to_numpy(dtype=float)
        mean = subset["mean_effect"].to_numpy(dtype=float)
        low = subset["ci95_low"].to_numpy(dtype=float)
        high = subset["ci95_high"].to_numpy(dtype=float)
        ax.plot(
            x,
            mean,
            label=maintenance_type,
            color=COLORS[maintenance_type],
            linewidth=2,
        )
        ax.fill_between(x, low, high, color=COLORS[maintenance_type], alpha=0.18)
    ax.axvline(0, color="#333333", linestyle="--", linewidth=1)
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set_xlabel("相对维护日")
    ax.set_ylabel("相对反事实轨迹的透水率增量")
    ax.set_title("中维护与大维护事件研究（阴影为事件 Bootstrap 95% 区间）")
    ax.legend()
    _save(fig, output_dir / "05_维护事件研究.png")


def plot_event_gains(results: AnalysisResults, output_dir: Path) -> None:
    frame = results.event_metrics.copy()
    long = frame.melt(
        id_vars=["maintenance_type", "event_id"],
        value_vars=[
            "counterfactual_gain_3d",
            "effect_day7",
            "effect_day14",
            "effect_day30",
        ],
        var_name="horizon",
        value_name="effect",
    )
    labels = {
        "counterfactual_gain_3d": "1–3 日",
        "effect_day7": "第 7 日",
        "effect_day14": "第 14 日",
        "effect_day30": "第 30 日",
    }
    long["horizon"] = long["horizon"].map(labels)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(
        data=long,
        x="horizon",
        y="effect",
        hue="maintenance_type",
        hue_order=MAINTENANCE_ORDER,
        palette=COLORS,
        showfliers=False,
        ax=ax,
    )
    ax.axhline(0, color="#555555", linestyle="--", linewidth=0.8)
    ax.set_xlabel("维护后时点")
    ax.set_ylabel("反事实维护效应")
    ax.set_title("维护恢复量及其保持情况")
    _save(fig, output_dir / "06_维护效果分布.png")


def plot_envelope(results: AnalysisResults, output_dir: Path) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(15, 18), sharex=True)
    for ax, device in zip(axes.flat, DEVICE_ORDER):
        points = results.envelope_points.loc[
            results.envelope_points["device"] == device
        ]
        for maintenance_type in MAINTENANCE_ORDER:
            subset = points.loc[points["maintenance_type"] == maintenance_type]
            ax.scatter(
                subset["event_date"],
                subset["post_7d_level"],
                label=maintenance_type,
                color=COLORS[maintenance_type],
                s=18,
                alpha=0.8,
            )
        summary = results.envelope_summary.loc[
            results.envelope_summary["device"] == device
        ]
        if len(summary) and len(points):
            row = summary.iloc[0]
            elapsed = (
                (points["event_date"] - points["event_date"].min()).dt.days / 365.25
            )
            fitted = row["intercept"] + row["envelope_slope_per_year"] * elapsed
            order = np.argsort(points["event_date"].to_numpy())
            ax.plot(
                points["event_date"].iloc[order],
                fitted.iloc[order],
                color="#0072B2",
                linewidth=1.2,
            )
            ax.set_title(
                f"{device}：上包络年变化 {row['envelope_slope_per_year']:.2f}"
            )
        else:
            ax.set_title(device)
        ax.set_ylabel("维护后 7 日水平")
    axes[-1, 0].set_xlabel("日期")
    axes[-1, 1].set_xlabel("日期")
    fig.suptitle("维护后上包络线：不可逆性能变化的代理指标", fontsize=17, y=1.01)
    fig.tight_layout()
    _save(fig, output_dir / "07_维护后上包络线.png")


def plot_indicator_heatmap(results: AnalysisResults, output_dir: Path) -> None:
    selected = results.indicators.set_index("device")[
        [
            "decline_rate_median",
            "seasonal_amplitude",
            "medium_gain_3d_median",
            "major_gain_3d_median",
            "envelope_decline_per_year",
            "residual_volatility",
        ]
    ].rename(
        columns={
            "decline_rate_median": "下降率 DR",
            "seasonal_amplitude": "季节振幅 SA",
            "medium_gain_3d_median": "中维护恢复 MG",
            "major_gain_3d_median": "大维护恢复 MG",
            "envelope_decline_per_year": "上包络衰减 ED",
            "residual_volatility": "波动 VI",
        }
    )
    standardized = (selected - selected.mean()) / selected.std(ddof=0)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(
        standardized,
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".1f",
        cbar_kws={"label": "列内标准分"},
        ax=ax,
    )
    ax.set_title("设备级透水率变化指标（标准化，仅用于横向比较）")
    ax.set_xlabel("指标")
    ax.set_ylabel("设备")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, output_dir / "08_设备指标热力图.png")


def make_all_plots(results: AnalysisResults, output_dir: Path) -> None:
    configure_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_daily_series(results, output_dir)
    plot_monthly_coverage(results, output_dir)
    plot_seasonality(results, output_dir)
    plot_cycle_slopes(results, output_dir)
    plot_event_curve(results, output_dir)
    plot_event_gains(results, output_dir)
    plot_envelope(results, output_dir)
    plot_indicator_heatmap(results, output_dir)
