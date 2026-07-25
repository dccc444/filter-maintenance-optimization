# filter-maintenance-optimization

数学建模 B 题"过滤设备监测"的可复现分析项目。当前已实现第一问和第二问：

## 第一问：数据分析与指标构造

- 10 台过滤器小时级数据质量检查与异常标记
- 鲁棒日尺度聚合
- 年/半年谐波季节性检验与频谱辅助判断
- 按维护周期估计自然下降率
- 中维护、大维护的反事实事件研究
- 长期维护后上包络线衰减
- 设备级透水率变化指标体系（7 项指标）
- 自动生成论文可用表格、图片和 Markdown 报告

运行：

```powershell
.\.venv\Scripts\python -m src.problem1.run
```

## 第二问：寿命预测模型

- 双状态退化模型（不可逆性能上限 $C(t)$ + 可逆堵塞 $F(t)$ + 季节效应 $S(t)$）
- 从第一问指标估计 $\alpha/\beta/MG/\sigma$ 参数
- 提取当前固定维护规律（中维护约 57 天，大维护约每 4 次中维护后）
- 双条件寿命终止准则（年均 < 37 且大维护后不可恢复）
- 1000 次蒙特卡洛仿真，含参数不确定性扰动
- 回测验证（A2/A3/A8/A9 的 MAE < 5）
- 10 台设备寿命预测 + 95% 置信区间

运行：

```powershell
.\.venv\Scripts\python -m src.problem2.run --n-runs 1000
```

主要输出：

```text
outputs/problem2/
├─ figures/              # 5 张论文可用图片
├─ tables/               # 寿命预测表、参数表、回测指标等
└─ 第二问分析报告.md
```

论文正文初稿位于 `paper/第一问论文正文.md`，第二问论文正文位于 `filter-reading/08-第二问论文正文.md`。

## 数据

原始题目附件位于 `B题附件/`：

- `附件1.xlsx`：10 个工作表，记录小时级透水率
- `附件2.xlsx`：中维护和大维护记录

程序不会改写原始附件。第一问清洗和派生数据写入 `outputs/problem1/`，第二问结果写入 `outputs/problem2/`。

## 环境与运行

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m src.problem1.run    # 第一问
.\.venv\Scripts\python -m src.problem2.run    # 第二问
```

## 分析原则

1. 维护引起的透水率跳升不是异常值
2. 小时缺失不做全局线性填补，趋势分析使用每日中位数
3. 周期识别可使用短缺口插值，但维护效果估计只使用实际观测
4. 维护效果以维护前趋势外推为反事实，不直接用简单前后均值下结论
5. 只有约两年数据，年周期结论同时报告显著性、振幅和频谱证据
6. A4、A8 无大维护记录，设备级大维护指标保持缺失，不人为补造
7. 上包络衰减负值以总体中位数正则化，A4/A8 大维护参数跨设备共享估计
8. 寿命预测含参数不确定性（MC 中扰动 $\alpha/\beta/MG$），不以点估计代替区间
