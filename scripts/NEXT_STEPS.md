# EATA 论文实验指导书

- 每个实验任务包含：研究问题对应、目标、输入、统一协议、步骤、输出产物、验收标准。
- 与论文中的表/图（含占位表/图）严格对齐，便于后续替换占位符与补全结果。

---

## 0. 总体说明

### 0.1 研究问题（与论文 RQ 对齐）

- **RQ1（目标函数）**：Wasserstein/分布目标是否优于 MSE/点预测目标？
- **RQ2（搜索/引导）**：神经引导 + 语法增强是否显著提升搜索效率与策略质量？
- **RQ3（性能）**：可解释的符号策略能否在风险调整收益上接近/超过黑盒模型？

### 0.2 优先级定义

- **P0（必须）**：没有 P0 就无法投稿。
- **P1（重要）**：显著提升说服力与顶刊可接受性。
- **P2（可选）**：锦上添花。

### 0.3 输出目录规范（强制）

- **逐股票指标表**：`/Users/yin/Desktop/doing/eata/tables/`
- **论文图（PDF）**：`/Users/yin/Desktop/doing/eata/paper/figures/`
- **论文表（如需导出 tex/csv）**：`/Users/yin/Desktop/doing/eata/tables/`
- **原始运行日志/中间产物**：建议新增 `outputs/`（或复用现有结果目录），保证可追溯。

### 0.4 图-代码-研究问题（RQ）对应表（必须维护）

维护目标：论文中出现的图/表（尤其是占位项）都必须在此表中有：
- 论文 `label`
- 输出文件名（或表格文件名）
- 对应 RQ
- 输入数据文件（含 schema）
- 生成该图/表的最小可运行代码（见 0.5）

| 论文 label | 类型 | 论文引用文件/表 | 对应 RQ | 输入数据（建议路径） | 生成脚本（可一键运行） | 产物验收标准 |
|---|---|---|---|---|---|---|
| `tab:sp500_full_placeholder` | Table | `tables/sp500_full_table.tex`（用于替换 `\fake{TBD}`） | RQ3 | `tables/sp500_full_metrics.csv` | `analysis/build_sp500_full_table.py` | 表格行=方法（`EATA (Ours)`, `LightGBM`, `Buy & Hold`, `NEMoTS`），列=AR/SR/MDD/WinRate/Calmar；标注有效 `N` |
| `fig:sp500_full_placeholder` | Figure | `paper/figures/sp500_full_sharpe_boxplot.pdf` | RQ3 | `tables/sp500_full_metrics.csv` | `analysis/plot_sp500_full_sharpe_boxplot.py` | 箱线/小提琴图；显示 median/IQR/outliers；标题含 `N`；图内红色英文注明文件名/坐标/RQ/脚本名 |
| `fig:efficiency` | Figure | `paper/figures/fig4_search_efficiency.pdf` | RQ2 | `tables/search_efficiency_runs.csv` | `analysis/plot_search_efficiency.py` | 曲线=best-so-far；展示均值±标准差；缺真实数据时图内红色英文注明文件名/坐标/RQ/脚本名 |
| `fig:pareto` | Figure | `paper/figures/fig2_pareto_frontier.pdf` | RQ2/RQ3 | `tables/pareto_frontier_points.csv` | `analysis/plot_pareto_frontier.py` | x=complexity(AST nodes)，y=Sharpe；散点+frontier；缺真实数据时图内红色英文注明文件名/坐标/RQ/脚本名 |

---

## 0.5 作图与制表示例代码（可直接运行）

### 0.5.0 一键生成所有占位图/表（推荐）

```bash
python analysis/build_all_placeholders.py --base_dir /Users/yin/Desktop/doing/eata --force_fake
```

所有示例默认：
- 项目根目录：`/Users/yin/Desktop/doing/eata`
- 输入数据在 `tables/`
- 输出图在 `paper/figures/`

### 0.5.1 生成 S\&P 500 全量结果表（替换 `tab:sp500_full_placeholder`）

输入：`tables/sp500_full_metrics.csv`

必需字段（Schema）：
- `ticker`, `method`, `ar`, `sr`, `mdd`, `win_rate`, `calmar`

输出：`tables/sp500_full_table.tex`

```python
from pathlib import Path

import pandas as pd


BASE = Path('/Users/yin/Desktop/doing/eata')
CSV_PATH = BASE / 'tables' / 'sp500_full_metrics.csv'
OUT_TEX = BASE / 'tables' / 'sp500_full_table.tex'


df = pd.read_csv(CSV_PATH)
needed = {'ticker', 'method', 'ar', 'sr', 'mdd', 'win_rate', 'calmar'}
missing = needed - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

n_tickers = df['ticker'].nunique()

summary = (
    df.groupby('method')[['ar', 'sr', 'mdd', 'win_rate', 'calmar']]
    .mean(numeric_only=True)
    .reset_index()
)

def fmt_row(r) -> str:
    return (
        f"{r['method']} & {r['ar']:.2f} & {r['sr']:.2f} & {r['mdd']:.2f} & {r['win_rate']:.3f} & {r['calmar']:.2f} \\\\"
    )

rows = [fmt_row(r) for _, r in summary.iterrows()]

tex = "\n".join([
    "% Auto-generated table for tab:sp500_full_placeholder",
    f"% N_tickers = {n_tickers}",
    "\\resizebox{0.9\\columnwidth}{!}{%",
    "\\begin{tabular}{lccccc}",
    "\\toprule",
    "\\textbf{Method} & \\textbf{AR (\\%)} & \\textbf{SR} & \\textbf{MDD (\\%)} & \\textbf{Win Rate} & \\textbf{Calmar} \\\\",
    "\\midrule",
    *rows,
    "\\bottomrule",
    "\\end{tabular}%",
    "}",
])

OUT_TEX.write_text(tex)
print(f"Wrote: {OUT_TEX}")
```

### 0.5.2 生成 S\&P 500 Sharpe 分布图（替换 `fig:sp500_full_placeholder`）

输入：`tables/sp500_full_metrics.csv`

输出：`paper/figures/sp500_full_sharpe_boxplot.pdf`

```python
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


BASE = Path('/Users/yin/Desktop/doing/eata')
CSV_PATH = BASE / 'tables' / 'sp500_full_metrics.csv'
OUT_FIG = BASE / 'paper' / 'figures' / 'sp500_full_sharpe_boxplot.pdf'

df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=['sr'])

n_tickers = df['ticker'].nunique()

sns.set_style('whitegrid')
plt.figure(figsize=(8.6, 3.2))

ax = sns.boxplot(
    data=df,
    x='method',
    y='sr',
    showfliers=True,
)

ax.set_title(f"Per-stock Sharpe distribution over S&P 500 (N={n_tickers})")
ax.set_xlabel('Method')
ax.set_ylabel('Sharpe Ratio')
plt.xticks(rotation=25, ha='right')

plt.tight_layout()
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_FIG)
print(f"Wrote: {OUT_FIG}")
```

### 0.5.3 生成 Search Efficiency 图（对应 `fig:efficiency`）

建议输入：`tables/search_efficiency_runs.csv`

建议字段（Schema）：
- `run_id`：整数
- `method`：`eata` / `gp` / `random`
- `step`：迭代步（或 episode）
- `best_reward`：截至该步 best-so-far 指标（reward 或 Sharpe）

输出：`paper/figures/fig4_search_efficiency.pdf`

```python
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


BASE = Path('/Users/yin/Desktop/doing/eata')
CSV_PATH = BASE / 'tables' / 'search_efficiency_runs.csv'
OUT_FIG = BASE / 'paper' / 'figures' / 'fig4_search_efficiency.pdf'

df = pd.read_csv(CSV_PATH)
needed = {'run_id', 'method', 'step', 'best_reward'}
missing = needed - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

plt.figure(figsize=(6.8, 3.2))

for method, g in df.groupby('method'):
    stats = g.groupby('step')['best_reward'].agg(['mean', 'std']).reset_index()
    plt.plot(stats['step'], stats['mean'], label=method)
    plt.fill_between(
        stats['step'],
        stats['mean'] - stats['std'],
        stats['mean'] + stats['std'],
        alpha=0.15,
    )

plt.xlabel('Search step')
plt.ylabel('Best-so-far')
plt.legend(frameon=False)
plt.tight_layout()
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_FIG)
print(f"Wrote: {OUT_FIG}")
```

### 0.5.4 生成 Pareto Frontier 图（对应 `fig:pareto`）

建议输入：`tables/pareto_frontier_points.csv`

建议字段（Schema）：
- `method`
- `complexity`：AST node count
- `sr`：Sharpe（或你选择的性能指标）

输出：`paper/figures/fig2_pareto_frontier.pdf`

```python
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


BASE = Path('/Users/yin/Desktop/doing/eata')
CSV_PATH = BASE / 'tables' / 'pareto_frontier_points.csv'
OUT_FIG = BASE / 'paper' / 'figures' / 'fig2_pareto_frontier.pdf'

df = pd.read_csv(CSV_PATH)
needed = {'method', 'complexity', 'sr'}
missing = needed - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

sns.set_style('whitegrid')
plt.figure(figsize=(6.8, 3.2))
ax = sns.scatterplot(data=df, x='complexity', y='sr', hue='method', alpha=0.7, s=22)

ax.set_xlabel('Complexity (AST node count)')
ax.set_ylabel('Sharpe Ratio')
plt.tight_layout()
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_FIG)
print(f"Wrote: {OUT_FIG}")
```

---

## 1. P0：统一数据与回测协议（必须先做）

### 1.1 数据准备与清洗（P0）

- **目标**：保证所有方法在同一数据、同一清洗规则下比较。
- **输入**：
  - OHLCV（日频）
  - 股票列表（S\&P 500 constituents；或其可交易子集）
  - 时间区间：2020-01-01 至 2024-12-31
- **必须记录的清洗规则**：
  - 停牌/缺失交易日处理；
  - 拆股/分红/复权（如使用复权价，需明确）；
  - ticker 变更映射；
  - 异常值（极端跳变）处理策略。
- **输出产物**：
  - `tables/data_manifest.csv`：每个 ticker 数据覆盖率、缺失率、起止日期
  - `tables/data_cleaning_rules.md`：清洗规则文本
  - `tables/ticker_universe.csv`：最终纳入的 ticker 列表（含剔除原因列）
- **验收标准**：
  - 任意方法可复用同一份数据产物；
  - 数据覆盖率与剔除规则可复现；
  - 关键统计（样本数、交易日数）一致。

#### 1.1.1 输出文件字段约定（Schema）

- `tables/data_manifest.csv`：
  - `ticker`
  - `n_days`
  - `n_missing`
  - `missing_ratio`
  - `first_date`
  - `last_date`
- `tables/ticker_universe.csv`：
  - `ticker`
  - `source`（例如 sp500_constituents）
  - `start_date`, `end_date`
  - `coverage_ratio`
  - `excluded`（0/1）
  - `exclude_reason`

#### 1.1.2 数据导入：当前脚本状态与推荐做法（重要）

- `scripts/import_data.py` **当前为 Windows 路径硬编码版本**（`SOURCE_DATA_PATH` / `DB_PATH` 指向 `C:\\...`），在 mac 上直接运行会报错。
- 推荐做法（二选一）：
  - **方案 A（最小改动）**：直接在 `scripts/import_data.py` 顶部把 `SOURCE_DATA_PATH` 与 `DB_PATH` 改成你本机真实路径，然后运行 `python scripts/import_data.py`。
  - **方案 B（工程化）**：将 `scripts/import_data.py` 改为命令行参数版本（例如 `--source_dir`、`--db_path`、`--table_name`），并在 README/本指南中固定调用方式。

```bash
# 方案 A：修改脚本中的 SOURCE_DATA_PATH/DB_PATH 后执行
python scripts/import_data.py
```

### 1.2 统一回测协议（P0；必须写进论文 Experiments）

- **目标**：避免“基线被弱化/协议不一致”的审稿人质疑。
- **必须明确的协议项**：
  - **交易方向**：是否允许做空（若允许，是否借券成本=0？）；
  - **执行时点**：close-to-close / open-to-close（含 1-bar 延迟与否）；
  - **持仓规模**：单资产独立回测 / 多资产等权组合 / 市值加权；
  - **交易成本**：单边 bps（佣金+滑点+点差），与 Table~\ref{tab:costs} 一致；
  - **杠杆与资金**：默认 1x，无杠杆（除非明确）；
  - **指标定义**：AR/SR/MDD/Calmar/WinRate/Trades/Turnover 的公式。
- **输出产物**：
  - `tables/backtest_protocol.md`：协议全文（直接可复制到论文）
- **验收标准**：
  - 任意两次运行给定同一 seed，结果可复现（允许数值微小浮动需说明）。

#### 1.2.1 建议的“回测协议文本模板”（可直接复制到论文）

你可以在 `tables/backtest_protocol.md` 中采用如下结构（建议逐条写清）：

- Data: daily OHLCV; corporate actions handled by ...
- Split: rolling-window training (first 70\%) and out-of-sample testing (last 30\%)
- Signal execution: ...
- Position: ...
- Transaction cost: single-side bps ∈ {0,5,10,20}; applied on position changes
- Metrics: AR, SR, MDD, Calmar, WinRate, Trades, Turnover (definitions)

#### 1.2.2 建议的命令模板（占位，按你的代码实际参数改）

> 说明：本仓库当前可用的统一实验入口是 `run_experiments.py`，其参数以 `--mode {single,sweep,summary}` 为主（见 `python run_experiments.py --help`）。
> 注意 `run_experiments.py` 默认 `base_dir` 不是当前仓库路径，建议显式传入 `--base_dir /Users/yin/Desktop/doing/eata`。

```bash
# 单次实验：单个 ticker + 指定策略列表 + 多次 runs
python run_experiments.py \
  --mode single \
  --lookback 50 --lookahead 10 --stride 1 --depth 300 \
  --tickers AAPL \
  --strategies eata buy_and_hold macd arima gp lightgbm lstm transformer finrl_ppo \
  --runs 5 \
  --base_dir /Users/yin/Desktop/doing/eata

# 扫描实验：对多个 ticker 和参数网格做 sweep
python run_experiments.py \
  --mode sweep \
  --tickers AAPL MSFT GOOG AMZN \
  --strategies eata buy_and_hold macd lightgbm finrl_ppo \
  --runs 3 \
  --base_dir /Users/yin/Desktop/doing/eata

# 汇总实验：从 experiment_results 目录汇总生成 summary（具体汇总逻辑依脚本实现）
python run_experiments.py \
  --mode summary \
  --base_dir /Users/yin/Desktop/doing/eata
```

#### 1.2.3 重要说明：策略名称与脚本一致性

- `comparison_experiments/algorithms/baseline.py` 中的策略 key 包含：
  - `eata`, `buy_and_hold`, `macd`, `arima`, `gp`, `lightgbm`, `lstm`, `transformer`
  - FinRL 系列为 `finrl_ppo`（而不是 `ppo`）。
- 请以该文件中的 key 为准，否则会出现“未知策略/导入失败”。

#### 1.2.4 输出位置（本仓库真实行为）

- `run_experiments.py` 会将每次运行结果保存到：`<base_dir>/experiment_results/`
- 文件名形如：
  - `experiment_results_lookback{lb}_lookahead{la}_stride{s}_depth{d}_{ticker}_{timestamp}.csv`

---

## 2. P0：主实验（规模与泛化）

### 2.1 S\&P 500 全量实验（计划工作；论文已放占位表/图）（P0）

- **论文对齐**：
  - 占位表：`tab:sp500_full_placeholder`（`paper/4.experiments.tex`）
  - 占位图：`fig:sp500_full_placeholder`（`paper/4.experiments.tex`；Sharpe 分布图）
- **目标**：支撑 “S\&P 500 全量/大规模泛化” 的论断（并产出可替换占位符的真实结果）。
- **输入**：
  - S\&P 500 constituents 列表（可交易筛选规则需在 `data_cleaning_rules.md` 中记录）
  - 数据：2020-2024 OHLCV
- **方法**：对每个 ticker 在同一协议下运行以下方法：
  - Buy\&Hold、MACD、ARIMA、GP(Operon)、LightGBM、LSTM、Transformer、PPO(FinRL)、NEMoTS、EATA
- **输出产物（必须）**：
  - `tables/sp500_full_metrics.csv`：逐股票 AR/SR/MDD/WinRate/Calmar/Trades/Turnover
  - `tables/sp500_full_summary.json`：mean/std/median/IQR/N
  - `paper/figures/sp500_full_sharpe_boxplot.pdf`：Sharpe 分布（箱线/小提琴）

#### 2.1.0 执行可行性说明（当前仓库约束）

- `run_experiments.py` 的 `--tickers` 当前是“命令行枚举”，对 S\&P 500 全量（500 个 ticker）不够友好。
- 建议在后续工程化中给 `run_experiments.py` 增加 `--tickers_file tables/ticker_universe.csv` 之类的参数，避免命令行长度限制。

#### 2.1.1 输出文件字段约定（Schema）

- `tables/sp500_full_metrics.csv`（逐股票）：
  - `ticker`
  - `method`
  - `ar`（年化收益，% 或小数，必须统一）
  - `sr`
  - `mdd`
  - `win_rate`
  - `calmar`
  - `n_trades`
  - `turnover`
  - `start_date`, `end_date`
  - `seed`
- `tables/sp500_full_summary.json`（全局汇总）：
  - `n_tickers`
  - `metrics`: 每个 metric 的 mean/std/median/iqr

#### 2.1.2 绘图规范（替换占位图）

- 输出文件：`paper/figures/sp500_full_sharpe_boxplot.pdf`
- 图类型：boxplot/violin（需含 median/IQR/outliers）
- 图中必须标注：`N=<有效ticker数>`
- **验收标准**：
  - 全量 ticker 的有效 N（剔除规则明确）；
  - `tab:sp500_full_placeholder` 可被真实均值/方差替换；
  - `fig:sp500_full_placeholder` 的占位框可替换为真实 PDF。

### 2.2 样本外验证（时间外推）（P0）

- **目标**：避免“训练期拟合、测试期失效”的质疑。
- **设计**：
  - 训练：2020-2023；测试：2024
  - 可选：测试再拆分 2024H1/2024H2
- **输出产物**：
  - `tables/oos_validation.csv`：训练/测试 AR/SR/MDD + 表达式稳定性

#### 2.2.1 输出字段约定（Schema）

- `tables/oos_validation.csv`：
  - `ticker`, `method`, `period`（train/test/test_h1/test_h2）
  - `ar`, `sr`, `mdd`, `win_rate`, `calmar`
  - `expression_stability`（如果适用）
- **验收标准**：
  - 测试期性能不崩塌；
  - 明确表达式更新频率（滑动窗口的重训练策略）。

---

## 3. P0：统计显著性与数据窥探防护

### 3.1 显著性检验（P0）

- **目标**：让 Table~\ref{tab:main_results} 的显著性标注具有学术可信度。
- **建议方法（最少实现一项）**：
  - paired t-test（逐股票配对）
  - SPA test（Hansen, 2005）
  - Reality Check（White, 2000）
  - Deflated Sharpe（Bailey \& López de Prado, 2014）
- **输出产物**：
  - `tables/significance_tests.csv`：方法对比、p 值、校正方式
- **验收标准**：
  - 明确 $H_0$；
  - 明确样本单位（ticker-level）；
  - 多重比较校正策略可复现。

#### 3.1.1 输出字段约定（Schema）

- `tables/significance_tests.csv`：
  - `baseline`
  - `test`（paired_t / spa / reality_check / deflated_sharpe）
  - `metric`（sr/ar/calmar 等）
  - `n_samples`（ticker 数）
  - `statistic`
  - `p_value`
  - `p_adjustment`（none/bonferroni/fdr）
  - `alpha`
  - `conclusion`（reject/fail_to_reject）

---

## 4. P0/P1：消融实验（机制闭环）

### 4.1 Profit Head 消融（NoRL / NEMoTS-style）（P0）

- **目标**：直接验证 Profit Head 的必要性（对应 RQ1/RQ2）。
- **输出产物**：
  - `tables/ablation_profit_head.csv`
- **验收标准**：
  - SR/MDD 有显著差异；
  - 与论文 Table~\ref{tab:ablation_loss} 可对齐更新。

### 4.2 Grammar Augmentation 消融（P1）

- **输出产物**：`tables/ablation_grammar_aug.csv`

### 4.3 Neural Guidance 消融（\alpha=0）（P1）

- **输出产物**：`tables/ablation_neural_guidance.csv`

### 4.4 金融算子消融（仅基础算子）（P1）

- **输出产物**：`tables/ablation_fin_ops.csv`

---

## 5. P1：敏感性分析

### 5.1 信号阈值敏感性（P1）

- **目标**：证明 $Q_{25}/Q_{75}$ 并非拍脑袋。
- **输出产物**：`tables/sensitivity_signal_threshold.csv` + 对应图

### 5.2 关键超参数敏感性（P1）

- **参数**：$T^{\text{in}}, H, K, \omega, \alpha, L^{\max}$ 等
- **输出产物**：`tables/sensitivity_hparams.csv` + 对应曲线图

#### 5.2.1 参数调优脚本状态（重要）

- `scripts/parameter_tuning.py` 当前在你的环境下会报 `ModuleNotFoundError: No module named 'agent'`，原因是其 `sys.path` 添加方式指向 `scripts/` 目录而非项目根目录。
- 建议修复方向：
  - 将 `sys.path.append(os.path.dirname(os.path.abspath(__file__)))` 改为项目根目录；或
  - 在运行时设置 `PYTHONPATH` 指向项目根；并确保 `agent.py` 可被导入。

---

## 6. P1：必须补齐的图表（论文可视化证据）

- **累计收益曲线**：代表性 ticker + 全量分布统计
- **表达式演化过程**：best reward vs episode、表达式长度/复杂度 vs episode
- **失败案例分析**：挑选 EATA 明显弱于 Buy\&Hold 的 ticker，解释原因并给出表达式
- **计算效率**：每 ticker wall-clock time、CPU/GPU 占用，与基线对比

---

## 7. 交付检查表（提交前）

- [ ] `tables/` 中所有 CSV/JSON 可重现生成（不依赖手工编辑）
- [ ] `paper/figures/` 中所有图为 PDF，且与正文引用一致
- [ ] 论文中不再出现 `\fake{}`（除明确标注为计划占位的 S\&P500 全量图表）
