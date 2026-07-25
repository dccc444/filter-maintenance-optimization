from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import DEVICE_ORDER, AnalysisResults


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def _p_fmt(value: float) -> str:
    if value == 0:
        return "<1e-300"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def _markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 3) -> str:
    shown = frame.loc[:, columns].copy()
    for col in shown.select_dtypes(include=[np.number]).columns:
        shown[col] = shown[col].map(lambda x: _fmt(x, digits))
    return shown.to_markdown(index=False)


def build_report(results: AnalysisResults, path: Path) -> None:
    quality = results.quality
    season = results.seasonality_summary
    cycle = results.cycle_slopes
    event_summary = results.event_summary
    indicators = results.indicators

    medium_gain = event_summary.loc[
        (event_summary["maintenance_type"] == "中维护")
        & (event_summary["metric"] == "counterfactual_gain_3d")
    ].iloc[0]
    major_gain = event_summary.loc[
        (event_summary["maintenance_type"] == "大维护")
        & (event_summary["metric"] == "counterfactual_gain_3d")
    ].iloc[0]

    decline_by_device = (
        cycle.groupby("device")["decline_rate_per_day"]
        .median()
        .rename("中位下降率")
        .reset_index()
    )
    decline_by_device["_order"] = decline_by_device["device"].map(DEVICE_ORDER.index)
    decline_by_device = decline_by_device.sort_values("_order").drop(columns="_order")
    type_comparison = results.maintenance_comparison.loc[
        results.maintenance_comparison["metric"] == "counterfactual_gain_3d"
    ].iloc[0]

    lines = [
        "# 第一问：数据处理、透水率变化规律与维护影响",
        "",
        "## 1. 分析口径",
        "",
        "本问使用附件 1 的小时级透水率和附件 2 的中、大维护日期。维护导致的跃升是研究对象，不能作为异常值删除。程序仅将远离维护日前后 2 天、相对局部滚动中位数偏离超过 6 倍 MAD 的孤立点标记为传感器异常；趋势和维护效果分析使用每日不少于 3 个有效小时观测的日中位数。",
        "",
        "缺失小时不做全局线性填补。短缺口插值只用于频谱辅助分析，维护效果估计始终使用实际观测。季节性通过设备固定差异、设备长期趋势和距上次维护时间控制后的年/半年谐波项检验；下降率在维护周期内用 Theil–Sen 稳健斜率估计；维护效果用维护前 21 天趋势外推形成反事实轨迹。",
        "",
        "## 2. 数据质量",
        "",
        f"10 台设备共有 {int(quality['rows'].sum()):,} 条时间记录，其中有效透水率 {int(quality['valid_values'].sum()):,} 条。各设备按完整日历小时计算的覆盖率在 {_fmt(quality['calendar_hour_coverage'].min()*100,1)}%—{_fmt(quality['calendar_hour_coverage'].max()*100,1)}% 之间；每日不少于 3 个有效小时的日尺度覆盖率为 {_fmt(quality['calendar_day_coverage'].min()*100,1)}%—{_fmt(quality['calendar_day_coverage'].max()*100,1)}%，因此主分析采用日中位数。",
        "",
        _markdown_table(
            quality.assign(
                calendar_hour_coverage=quality["calendar_hour_coverage"] * 100,
                calendar_day_coverage=quality["calendar_day_coverage"] * 100,
                missing_rate=quality["missing_rate"] * 100,
            ),
            [
                "device",
                "rows",
                "valid_values",
                "missing_rate",
                "calendar_hour_coverage",
                "calendar_day_coverage",
                "outliers_flagged",
                "minimum",
                "maximum",
            ],
            2,
        ),
        "",
        "维护记录共 "
        f"{len(results.maintenance)} 次，其中中维护 "
        f"{int((results.maintenance['maintenance_type']=='中维护').sum())} 次、大维护 "
        f"{int((results.maintenance['maintenance_type']=='大维护').sum())} 次。A4、A8 在附件 2 中没有大维护记录，这与题面所述“一般每年 1—4 次”不一致，后续设备级大维护指标保持缺失，不进行人为填补。",
        "",
        "![日透水率与维护事件](figures/01_日透水率与维护事件.png)",
        "",
        "![数据缺失热力图](figures/02_数据缺失热力图.png)",
        "",
        "## 3. 周期性",
        "",
        f"加入年周期与半年周期谐波项后，残差平方和相对下降 {_fmt(season['relative_ssr_reduction']*100,2)}%；嵌套模型 F={_fmt(season['f_statistic'],2)}，p{_p_fmt(season['p_value']) if season['p_value'] == 0 else '=' + _p_fmt(season['p_value'])}。估计季节效应半振幅为 {_fmt(season['seasonal_amplitude'],2)}，年周期基波振幅为 {_fmt(season['annual_amplitude'],2)}，半年周期振幅为 {_fmt(season['semiannual_amplitude'],2)}。",
        "",
        f"季节曲线峰值约出现在年内第 {season['seasonal_peak_day']} 天，谷值约出现在第 {season['seasonal_trough_day']} 天。统计上季节项显著，但数据仅覆盖约两年，只有两个完整年循环，因此应将其解释为“存在显著季节性证据”，不能宣称已经稳定识别长期气候周期。",
        "",
        "频谱功率最大的候选周期如下（频谱只作辅助证据）：",
        "",
        _markdown_table(results.periods, ["period_days", "relative_power"], 3),
        "",
        "![季节效应曲线](figures/03_季节效应曲线.png)",
        "",
        "## 4. 下降趋势与规律",
        "",
        f"剔除季节项后，共获得 {len(cycle)} 个满足至少 14 个观测日且跨度不少于 14 天的维护周期。所有周期的稳健斜率均表现为下降，下降率中位数为 {_fmt(cycle['decline_rate_per_day'].median(),3)} 透水率/日，四分位区间为 {_fmt(cycle['decline_rate_per_day'].quantile(.25),3)}—{_fmt(cycle['decline_rate_per_day'].quantile(.75),3)}，完整范围为 {_fmt(cycle['decline_rate_per_day'].min(),3)}—{_fmt(cycle['decline_rate_per_day'].max(),3)}。这说明维护间隔内透水率持续下降的规律具有稳定性，但下降速度在设备和周期之间差异明显，寿命模型应使用分层参数和过程噪声，而不能给每台设备只拟合一条直线。",
        "",
        _markdown_table(decline_by_device, ["device", "中位下降率"], 3),
        "",
        "![维护周期下降率](figures/04_维护周期下降率.png)",
        "",
        "维护后 1—7 日季节校正水平的稳健回归斜率被定义为“上包络线变化率”，用于表征不可逆性能上限。正的年衰减值表示即便维护后，设备可恢复到的水平仍在下降；该指标将在第二问中作为固有性能老化速度的初值。",
        "",
        "![维护后上包络线](figures/07_维护后上包络线.png)",
        "",
        "## 5. 中维护和大维护的影响",
        "",
        f"中维护的 1—3 日反事实恢复量均值为 {_fmt(medium_gain['mean'],2)}，95% Bootstrap 区间为 [{_fmt(medium_gain['ci95_low'],2)}, {_fmt(medium_gain['ci95_high'],2)}]，有效事件数 {int(medium_gain['n_events'])}；大维护对应均值为 {_fmt(major_gain['mean'],2)}，95% 区间为 [{_fmt(major_gain['ci95_low'],2)}, {_fmt(major_gain['ci95_high'],2)}]，有效事件数 {int(major_gain['n_events'])}。",
        "",
        f"两类维护 1—3 日效果的 Mann–Whitney 检验 p={_p_fmt(type_comparison['p_value'])}。由于大维护仅有 {int(type_comparison['major_n'])} 个有效事件，且通常在状态更差时被选择，不能仅根据样本均值断言哪类维护的物理清洗能力更强；事件研究已控制维护前局部趋势和共同季节项，但仍可能存在按设备状态选择维护类型的剩余混杂。",
        "",
        "![维护事件研究](figures/05_维护事件研究.png)",
        "",
        "![维护效果分布](figures/06_维护效果分布.png)",
        "",
        "## 6. 透水率变化指标体系",
        "",
        "为后续寿命预测提供可解释输入，定义以下指标：",
        "",
        "1. **数据覆盖率 CR**：有效小时数/完整日历小时数；",
        "2. **自然下降率 DR**：维护周期内季节校正透水率 Theil–Sen 斜率的相反数；",
        "3. **季节振幅 SA**：设备级年/半年谐波曲线的半极差；",
        "4. **瞬时恢复量 MG**：维护后 1—3 日实际值相对维护前趋势反事实值的增量；",
        "5. **30 日保持率 MP30**：第 27—30 日维护效应/1—3 日恢复量；",
        "6. **上包络衰减 ED**：维护后 1—7 日水平随年份变化斜率的相反数；",
        "7. **波动指数 VI**：控制设备、长期趋势、维护时钟和季节项后的残差标准差。",
        "",
        "这些指标保留物理含义，不强行压缩为单一综合分数。第二问可将 DR、SA、MG、MP30、ED 和 VI 分别映射到堵塞增长、季节函数、维护恢复、维护保持、不可逆老化和过程噪声参数。",
        "",
        _markdown_table(
            indicators,
            [
                "device",
                "decline_rate_median",
                "seasonal_amplitude",
                "medium_gain_3d_median",
                "major_gain_3d_median",
                "medium_retention_day30_median",
                "major_retention_day30_median",
                "envelope_decline_per_year",
                "residual_volatility",
            ],
            3,
        ),
        "",
        "![设备指标热力图](figures/08_设备指标热力图.png)",
        "",
        "## 7. 第一问结论与第二问接口",
        "",
        "1. 小时级数据存在较大间断，但日尺度覆盖较好；以日中位数为主、小时级作质量检查是更稳健的口径。",
        "2. 控制设备差异、长期趋势和维护时钟后，年/半年季节项显著，但受限于只有约两年数据，外推时必须保留不确定性。",
        "3. 透水率呈“维护后跳升—周期内下降”的锯齿形；周期下降率存在明显设备差异与周期差异。",
        "4. 中维护和大维护总体均能带来短期恢复，但大维护样本少且存在状态选择偏差，第二问应采用跨设备共享信息的分层估计。",
        "5. 维护后上包络线并非恒定，说明只建模可逆堵塞不足，必须同时引入不可逆老化/维护损伤状态。",
        "6. `tables/device_indicators.csv`、`tables/cycle_slopes.csv`、`tables/maintenance_event_metrics.csv` 和 `processed/daily_permeability.csv` 构成第二问的直接数据接口。",
        "",
        "## 8. 局限性",
        "",
        "- 附件没有小维护记录，未记录小维护的作用会进入周期斜率和随机误差；",
        "- 维护类型不是随机分配，事件研究不能完全消除状态选择偏差；",
        "- 两年数据不足以高置信度区分稳定年周期与特定年份冲击；",
        "- 维护损伤不能仅凭单次前后跳升识别，需要第二问的双状态模型联合估计。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
