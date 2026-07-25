"""Generate figures for Problem 2 report and paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# Style: match Problem 1 aesthetic
# Use CJK-capable font
import matplotlib.font_manager as fm
cjk_fonts = [f.name for f in fm.fontManager.ttflist if 'CJK' in f.name or 'WenQuanYi' in f.name]
if cjk_fonts:
    plt.rcParams["font.family"] = cjk_fonts[0]
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def fig01_parameter_heatmap(output_dir: Path) -> None:
    """Device-level parameter comparison heatmap."""
    params = pd.read_csv("outputs/problem2/tables/model_parameters.csv").set_index("device")

    columns = {
        "alpha_per_day": "α (老化率)",
        "beta_per_day": "β (堵塞率)",
        "medium_recovery": "中维护恢复",
        "major_recovery": "大维护恢复",
        "sigma": "σ (噪声)",
    }

    data = params[list(columns.keys())].copy()
    data.columns = list(columns.values())

    # Normalize each column to [0,1] for heatmap
    normed = (data - data.min()) / (data.max() - data.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(normed.T, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)

    # Annotate with original values
    for i in range(len(data)):
        for j, col in enumerate(data.columns):
            val = data.iloc[i, j]
            if pd.notna(val):
                ax.text(i, j, f"{val:.2f}", ha="center", va="center", fontsize=7)

    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(data.index)
    ax.set_yticks(range(len(data.columns)))
    ax.set_yticklabels(data.columns)
    ax.set_title("各设备退化模型参数对比（归一化热力图）", fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(output_dir / "01_参数对比热力图.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig02_lifetime_predictions(output_dir: Path) -> None:
    """Lifetime prediction bar chart with CI."""
    preds = pd.read_csv("outputs/problem2/tables/lifetime_predictions.csv")

    devices = preds["device"].tolist()
    median = preds["median_lifetime_days"].values / 365.25
    ci_low = preds["ci95_low_days"].values / 365.25
    ci_high = preds["ci95_high_days"].values / 365.25

    yerr_low = median - ci_low
    yerr_high = ci_high - median

    colors = []
    for m in median:
        if m > 3:
            colors.append("#2ecc71")
        elif m > 1.5:
            colors.append("#f39c12")
        else:
            colors.append("#e74c3c")

    fig, ax = plt.subplots(figsize=(8, 4))

    x = range(len(devices))
    bars = ax.bar(x, median, yerr=[yerr_low, yerr_high], capsize=4,
                  color=colors, edgecolor="white", linewidth=0.5, alpha=0.85)

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(devices)
    ax.set_ylabel("剩余寿命（年）")
    ax.set_title("10台过滤器剩余寿命预测（95% 预测区间）", fontweight="bold", pad=12)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="长寿命 (>3年)"),
        Patch(facecolor="#f39c12", label="中等 (1.5-3年)"),
        Patch(facecolor="#e74c3c", label="短寿命 (<1.5年)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.8)

    plt.tight_layout()
    fig.savefig(output_dir / "02_寿命预测条形图.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig03_maintenance_intervals(output_dir: Path) -> None:
    """Maintenance interval distribution per device."""
    patterns = pd.read_csv("outputs/problem2/tables/maintenance_patterns.csv")
    devices = patterns["device"].tolist()

    med_mean = patterns["medium_interval_mean"].values
    med_std = patterns["medium_interval_std"].values
    maj_mean = patterns["major_interval_mean"].values
    maj_std = patterns["major_interval_std"].values

    fig, ax = plt.subplots(figsize=(8, 4))

    x = np.arange(len(devices))
    width = 0.35

    bars1 = ax.bar(x - width/2, med_mean, width, yerr=med_std, capsize=3,
                   label="中维护间隔", color="#3498db", edgecolor="white")
    bars2 = ax.bar(x + width/2, maj_mean, width, yerr=maj_std, capsize=3,
                   label="大维护间隔", color="#e74c3c", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(devices)
    ax.set_ylabel("间隔（天）")
    ax.set_title("各设备维护间隔分布", fontweight="bold", pad=12)
    ax.legend(framealpha=0.8)

    # Annotate A4/A8 as "no major"
    for i, dev in enumerate(devices):
        if patterns.iloc[i]["major_count"] == 0:
            ax.annotate("无大维护", (x[i] + width/2, maj_mean[i]),
                       textcoords="offset points", xytext=(0, 10),
                       fontsize=6, ha="center", color="#e74c3c")

    plt.tight_layout()
    fig.savefig(output_dir / "03_维护间隔分布.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig04_eol_status(output_dir: Path) -> None:
    """End-of-life status: distance to threshold."""
    eol = pd.read_csv("outputs/problem2/tables/eol_status.csv")
    devices = eol["device"].tolist()
    annual = eol["current_annual_mean"].values
    threshold = 37

    fig, ax = plt.subplots(figsize=(8, 3.5))

    colors = ["#e74c3c" if a < 70 else "#f39c12" if a < 85 else "#2ecc71"
              for a in annual]

    x = range(len(devices))
    bars = ax.bar(x, annual, color=colors, edgecolor="white", linewidth=0.5)

    ax.axhline(y=threshold, color="red", linestyle="--", linewidth=1.5,
               label=f"寿命终止阈值 (37)")
    ax.fill_between([-1, len(devices)], 0, threshold, alpha=0.05, color="red")

    ax.set_xticks(x)
    ax.set_xticklabels(devices)
    ax.set_ylabel("365日滑动平均透水率")
    ax.set_title("当前各设备距寿命终止阈值的距离", fontweight="bold", pad=12)
    ax.legend(framealpha=0.8)

    # Annotate values
    for i, (dev, val) in enumerate(zip(devices, annual)):
        ax.text(i, val + 2, f"{val:.0f}", ha="center", fontsize=7)

    ax.set_ylim(0, max(annual) * 1.15)
    plt.tight_layout()
    fig.savefig(output_dir / "04_距阈值距离.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig05_monthly_maintenance(output_dir: Path) -> None:
    """Monthly maintenance distribution."""
    import json
    with open("outputs/problem2/tables/maintenance_patterns.json") as f:
        data = json.load(f)

    monthly = data["monthly_distribution"]
    months = range(1, 13)
    counts = [monthly.get(str(m), 0) for m in months]

    fig, ax = plt.subplots(figsize=(7, 3))

    colors = ["#3498db" if c > 12 else "#95a5a6" for c in counts]
    ax.bar(months, counts, color=colors, edgecolor="white")

    ax.set_xticks(months)
    ax.set_xlabel("月份")
    ax.set_ylabel("维护次数")
    ax.set_title("全部设备月度维护事件分布", fontweight="bold", pad=12)
    ax.axhline(y=np.mean(counts), color="red", linestyle="--", alpha=0.5,
               label=f"月均 ({np.mean(counts):.0f}次)")
    ax.legend(framealpha=0.8)

    plt.tight_layout()
    fig.savefig(output_dir / "05_月度维护分布.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    output_dir = Path("outputs/problem2/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Problem 2 figures...")
    fig01_parameter_heatmap(output_dir)
    print("  01_参数对比热力图.png")
    fig02_lifetime_predictions(output_dir)
    print("  02_寿命预测条形图.png")
    fig03_maintenance_intervals(output_dir)
    print("  03_维护间隔分布.png")
    fig04_eol_status(output_dir)
    print("  04_距阈值距离.png")
    fig05_monthly_maintenance(output_dir)
    print("  05_月度维护分布.png")
    print(f"\nFigures saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
