"""
生成对比实验的所有图表 - 基于62支股票的真实数据
生成4个图表：
1. real_cumulative_returns.pdf - 累积收益对比
2. real_correlation.pdf - 策略相关性矩阵（暂时基于汇总指标）
3. real_return_distribution.pdf - EATA收益分布（N=62）
4. real_risk_return.pdf - 风险-收益散点图（N=62）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Nature期刊标准字体和样式设置
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "axes.unicode_minus": False,
    "svg.fonttype": "none",      # SVG可编辑文本
    "pdf.fonttype": 42,          # TrueType文本
    "font.size": 9,              # Nature标准字号（密集图表7-9pt）
    "axes.spines.right": False,  # 只保留左下边框
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,     # 无边框图例
})

def load_data():
    """加载所有需要的数据"""
    # 1. 12个策略的汇总数据
    summary_df = pd.read_csv('../results/comparison_study/all_12_strategies_62stocks_final.csv')
    
    # 2. EATA的62支股票详细指标
    eata_metrics = pd.read_csv('../results/comparison_study/eata_62stocks_final_metrics.csv')
    
    # 3. 62支股票列表
    with open('../results/comparison_study/common_stocks_for_finrl.txt') as f:
        stocks = [line.strip() for line in f if line.strip()]
    
    return summary_df, eata_metrics, stocks


def plot_cumulative_returns(summary_df, output_dir):
    """绘制累积收益对比图 - 垂直柱状图"""
    print("生成累积收益对比图...")
    
    # 按Sharpe Ratio降序排序
    df_sorted = summary_df.sort_values('SR', ascending=False)
    
    strategies = df_sorted['Strategy'].values
    sharpe_ratios = df_sorted['SR'].values
    
    # 设置颜色：第一名蓝色，第二名橙色，其他灰色
    colors = []
    for i, s in enumerate(strategies):
        if i == 0:  # 第一名
            colors.append('#4472C4')  # 蓝色
        elif i == 1:  # 第二名
            colors.append('#FFA500')  # 橙色
        else:
            colors.append('#A6A6A6')  # 灰色
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x_pos = np.arange(len(strategies))
    bars = ax.bar(x_pos, sharpe_ratios, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    
    # 在柱子上标注数值
    for i, (bar, sr) in enumerate(zip(bars, sharpe_ratios)):
        height = bar.get_height()
        y_pos = height + 0.04 if height >= 0 else height - 0.06
        va = 'bottom' if height >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{sr:.2f}', ha='center', va=va, fontsize=16, fontweight='bold')
    
    # 设置x轴标签
    ax.set_xticks(x_pos)
    ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=16, fontweight='bold')
    
    ax.set_ylabel('Sharpe Ratio', fontsize=18, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 调整y轴范围，避免数值与边界重合
    y_min = min(sharpe_ratios) - 0.15
    y_max = max(sharpe_ratios) + 0.1
    ax.set_ylim(y_min, y_max)
    
    # 设置y轴刻度字体加粗
    ax.tick_params(axis='y', labelsize=16)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    # 设置背景色
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'real_cumulative_returns.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'real_cumulative_returns.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ real_cumulative_returns.pdf")


def plot_correlation_matrix(summary_df, output_dir):
    """绘制策略相关性矩阵（Figure 3 - 基于年化收益率）"""
    print("生成Figure 3: 策略相关性热力图...")
    
    # 1. 加载EATA portfolio数据
    eata_full_dir = Path('../results/eata_full_62stocks')
    eata_returns = {}
    
    for pf_file in sorted(eata_full_dir.glob('*_portfolio.csv')):
        ticker = pf_file.stem.replace('_portfolio', '')
        df = pd.read_csv(pf_file, index_col=0)
        portfolio = df['value']
        
        if len(portfolio) > 1:
            total_return = (portfolio.iloc[-1] - portfolio.iloc[0]) / portfolio.iloc[0]
            years = 4
            annual_return = (1 + total_return) ** (1/years) - 1
            eata_returns[ticker] = annual_return
    
    print(f"  ✅ EATA: {len(eata_returns)} 支股票")
    
    # 2. 加载传统策略数据
    strategy_files = {
        'Buy & Hold': 'buy_and_hold_62stocks_volatility.csv',
        'MACD': 'macd_62stocks_volatility.csv',
        'ARIMA': 'arima_62stocks_volatility.csv',
        'GP': 'gp_62stocks_volatility.csv',
        'GBDT': 'gbdt_62stocks_volatility.csv',
        'LSTM': 'lstm_62stocks_volatility.csv',
        'Transformer': 'transformer_62stocks_volatility.csv',
    }
    
    strategy_returns = {}
    comparison_dir = Path('../results/comparison_study')
    
    for strategy_name, filename in strategy_files.items():
        file_path = comparison_dir / filename
        if file_path.exists():
            df = pd.read_csv(file_path)
            returns_dict = {}
            for _, row in df.iterrows():
                ticker = row['ticker']
                ar = row.get('annual_return', 0)
                if abs(ar) > 1:
                    ar = ar / 100
                returns_dict[ticker] = ar
            strategy_returns[strategy_name] = returns_dict
            print(f"  ✅ {strategy_name}: {len(returns_dict)} 支股票")
    
    # 3. 加载FinRL策略数据（从detailed_outputs计算）
    finrl_strategies = {
        'PPO (FinRL)': 'finrl_ppo',
        'A2C (FinRL)': 'finrl_a2c',
        'TD3 (FinRL)': 'finrl_td3',
        'DDPG (FinRL)': 'finrl_ddpg',
    }
    
    detailed_dir = Path('../results/comparison_study/baseline_100stocks/detailed_outputs')
    
    for strategy_name, strategy_key in finrl_strategies.items():
        returns_dict = {}
        files = list(detailed_dir.glob(f'*-{strategy_key}-portfolio.csv'))
        
        for file in files:
            ticker = file.stem.split('-')[0]
            try:
                df = pd.read_csv(file)
                if 'portfolio_value' in df.columns and len(df) > 1:
                    pv = df['portfolio_value'].values
                    total_return = (pv[-1] - pv[0]) / pv[0]
                    years = 4
                    annual_return = (1 + total_return) ** (1/years) - 1
                    returns_dict[ticker] = annual_return
            except:
                continue
        
        if returns_dict:
            strategy_returns[strategy_name] = returns_dict
            print(f"  ✅ {strategy_name}: {len(returns_dict)} 支股票")
    
    # 3. 找到共同股票
    common_tickers = set(eata_returns.keys())
    for strategy_data in strategy_returns.values():
        common_tickers &= set(strategy_data.keys())
    common_tickers = sorted(list(common_tickers))
    print(f"  ✅ 共同股票: {len(common_tickers)} 支")
    
    # 4. 构建数据矩阵
    all_strategies = ['EATA'] + list(strategy_returns.keys())
    returns_matrix = []
    
    # EATA行
    eata_row = [eata_returns.get(t, 0) for t in common_tickers]
    returns_matrix.append(eata_row)
    
    # FinRL策略行
    for strategy in strategy_returns.keys():
        strategy_row = [strategy_returns[strategy].get(t, 0) for t in common_tickers]
        returns_matrix.append(strategy_row)
    
    # 转换为DataFrame并计算相关性
    returns_df = pd.DataFrame(returns_matrix, 
                             index=all_strategies,
                             columns=common_tickers).T
    
    corr_df = returns_df.corr()
    print(f"  ✅ 相关性矩阵: {corr_df.shape}")
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # 使用RdYlBu_r配色方案 - 先不显示数字
    sns.heatmap(corr_df, annot=False, cmap='RdYlBu_r', center=0,
                square=True, linewidths=1.0, linecolor='white',
                cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1, ax=ax)
    
    # 手动添加文本标注 - 确保每个格子都有
    for i in range(len(corr_df)):
        for j in range(len(corr_df.columns)):
            value = corr_df.iloc[i, j]
            text = ax.text(j + 0.5, i + 0.5, f'{value:.2f}',
                          ha='center', va='center',
                          color='black', fontsize=12, fontweight='bold')
    
    # 设置坐标轴标签字体
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16, fontweight='bold', rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=16, fontweight='bold', rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ correlation_heatmap.pdf")


def plot_return_distribution(eata_metrics, output_dir):
    """绘制EATA收益分布图（N=62）"""
    print("生成收益分布图...")
    
    # 提取年化收益率
    returns = eata_metrics['Annual Return (AR)'].values
    
    # 计算统计量
    mean_return = np.mean(returns)
    skewness = pd.Series(returns).skew()
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制直方图
    n, bins, patches = ax.hist(returns, bins=30, density=True, alpha=0.7, 
                                color='steelblue', edgecolor='black')
    
    # 添加核密度估计曲线
    from scipy import stats
    kde = stats.gaussian_kde(returns)
    x_range = np.linspace(returns.min(), returns.max(), 100)
    ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
    
    # 添加统计信息
    ax.axvline(mean_return, color='green', linestyle='--', linewidth=2, 
               label=f'Mean = {mean_return:.2f}%')
    
    ax.set_xlabel('Annual Return (%)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Density', fontsize=16, fontweight='bold')
    ax.legend(fontsize=13, frameon=True, fancybox=True)
    ax.grid(alpha=0.3)
    
    # 设置刻度字体
    ax.tick_params(axis='both', labelsize=14)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'real_return_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'real_return_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ real_return_distribution.pdf")


def plot_risk_return_all_strategies(output_dir):
    """绘制所有策略的风险-收益散点图（N=62）- 使用真实数据"""
    print("生成风险-收益散点图（所有策略）...")
    
    # 定义每个策略的颜色和标记（与原图一致）
    strategy_styles = {
        'eata': {'color': '#1f77b4', 'marker': 'o', 's': 80, 'label': 'EATA'},
        'buy_and_hold': {'color': '#ff7f0e', 'marker': 's', 's': 80, 'label': 'Buy&Hold'},
        'macd': {'color': '#2ca02c', 'marker': '^', 's': 100, 'label': 'MACD'},
        'arima': {'color': '#d62728', 'marker': 'v', 's': 100, 'label': 'ARIMA'},
        'gp': {'color': '#9467bd', 'marker': 'D', 's': 70, 'label': 'GP'},
        'gbdt': {'color': '#8c564b', 'marker': 'p', 's': 100, 'label': 'GBDT'},
        'lstm': {'color': '#e377c2', 'marker': 'h', 's': 100, 'label': 'LSTM'},
        'transformer': {'color': '#7f7f7f', 'marker': '*', 's': 150, 'label': 'Transformer'},
        'finrl_ppo': {'color': '#bcbd22', 'marker': 'X', 's': 120, 'label': 'FinRL-PPO'},
        'finrl_a2c': {'color': '#17becf', 'marker': 'P', 's': 120, 'label': 'FinRL-A2C'},
        'finrl_td3': {'color': '#ff9896', 'marker': 'd', 's': 100, 'label': 'FinRL-TD3'},
        'finrl_ddpg': {'color': '#98df8a', 'marker': '8', 's': 120, 'label': 'FinRL-DDPG'},
    }
    
    # 创建图表（与原图尺寸一致）
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 加载所有策略的真实数据
    strategies_to_plot = []
    
    # 1. EATA数据
    eata_df = pd.read_csv('../results/comparison_study/eata_62stocks_final_metrics.csv')
    eata_returns = eata_df['Annual Return (AR)'].values * 100
    eata_volatility = eata_df['Volatility (Annual)'].values * 100
    strategies_to_plot.append(('eata', eata_volatility, eata_returns))
    
    # 2. 其他7个策略的数据
    for strategy in ['buy_and_hold', 'macd', 'arima', 'gp', 'gbdt', 'lstm', 'transformer']:
        try:
            df = pd.read_csv(f'../results/comparison_study/{strategy}_62stocks_volatility.csv')
            vol = df['volatility'].values
            ret = df['annual_return'].values
            strategies_to_plot.append((strategy, vol, ret))
        except:
            print(f'⚠️  未找到 {strategy} 的数据')
    
    # 3. FinRL策略数据（从detailed_outputs计算）
    finrl_strategies = {
        'finrl_ppo': 'finrl_ppo',
        'finrl_a2c': 'finrl_a2c',
        'finrl_td3': 'finrl_td3',
        'finrl_ddpg': 'finrl_ddpg',
    }
    
    detailed_dir = Path('../results/comparison_study/baseline_100stocks/detailed_outputs')
    
    for strategy_key, strategy_name in finrl_strategies.items():
        vol_list = []
        ret_list = []
        files = list(detailed_dir.glob(f'*-{strategy_name}-portfolio.csv'))
        
        for file in files:
            try:
                df = pd.read_csv(file)
                if 'portfolio_value' in df.columns and len(df) > 1:
                    pv = df['portfolio_value'].values
                    
                    # 计算年化收益率
                    total_return = (pv[-1] - pv[0]) / pv[0]
                    years = 4
                    annual_return = ((1 + total_return) ** (1/years) - 1) * 100
                    
                    # 计算波动率
                    daily_returns = np.diff(pv) / pv[:-1]
                    volatility = np.std(daily_returns) * np.sqrt(252) * 100
                    
                    vol_list.append(volatility)
                    ret_list.append(annual_return)
            except:
                continue
        
        if vol_list:
            strategies_to_plot.append((strategy_key, np.array(vol_list), np.array(ret_list)))
            print(f'  ✅ {strategy_key}: {len(vol_list)} 支股票')
    
    # 绘制所有策略
    for strategy, vol, ret in strategies_to_plot:
        if strategy not in strategy_styles:
            continue
        style = strategy_styles[strategy]
        ax.scatter(vol, ret, 
                  c=style['color'], marker=style['marker'], s=style['s'], 
                  alpha=0.7, edgecolors='white', linewidth=0.5,
                  label=style['label'], zorder=3)
    
    # 添加零线
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.8)
    
    ax.set_xlabel('Volatility (%)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Annualized Return (%)', fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=13, framealpha=0.9, ncol=2, frameon=True, fancybox=True)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 设置刻度字体
    ax.tick_params(axis='both', labelsize=14)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    
    # 设置坐标轴范围
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'real_risk_return.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'real_risk_return.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ real_risk_return.pdf (所有策略真实数据)")


def main():
    """主函数"""
    print("="*80)
    print("生成对比实验图表 - 基于62支股票的真实数据")
    print("="*80)
    
    # 加载数据
    print("\n加载数据...")
    summary_df, eata_metrics, stocks = load_data()
    print(f"✅ 加载了 {len(summary_df)} 个策略的汇总数据")
    print(f"✅ 加载了 {len(eata_metrics)} 支股票的EATA详细数据")
    print(f"✅ 共用股票列表: {len(stocks)} 支")
    
    # 输出目录
    output_dir = Path('../paper/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成所有图表
    print("\n生成图表...")
    
    # 优先生成论文关键图表
    print("\n【论文关键图表】")
    plot_correlation_matrix(summary_df, output_dir)  # Figure 3
    
    # 其他辅助图表
    print("\n【辅助图表】")
    plot_cumulative_returns(summary_df, output_dir)
    plot_return_distribution(eata_metrics, output_dir)
    plot_risk_return_all_strategies(output_dir)
    
    print("\n" + "="*80)
    print("✅ 所有图表生成完成！")
    print("="*80)
    print(f"\n输出位置: {output_dir.absolute()}")
    print("\n论文关键图表:")
    print("  ✅ correlation_heatmap.pdf/png")
    print("\n辅助图表:")
    print("  - real_cumulative_returns.pdf/png")
    print("  - real_return_distribution.pdf/png")
    print("  - real_risk_return.pdf/png")


if __name__ == '__main__':
    main()
