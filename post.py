"""
统一的后处理脚本
从 results/ 获取实验数据，生成论文所需的图表
- 图表保存到 paper/figures/
- LaTeX表格保存到 paper/tables/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# 设置绘图风格
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('default')
        sns.set_style("darkgrid")
sns.set_palette("Set2")

# 路径配置
RESULTS_DIR = Path("results")
PAPER_FIGURES_DIR = Path("paper/figures")
PAPER_TABLES_DIR = Path("paper/tables")

# 确保输出目录存在
PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load_ablation_results(experiment_id=None):
    """
    从 results/ablation_study/ 加载消融实验结果
    
    Args:
        experiment_id: 实验ID，如果为None则加载最新的实验
    
    Returns:
        DataFrame: 消融实验结果
    """
    csv_dir = RESULTS_DIR / "ablation_study" / "csv_results"
    
    if experiment_id:
        csv_file = csv_dir / f"performance_summary_{experiment_id}.csv"
    else:
        # 找到最新的CSV文件
        csv_files = list(csv_dir.glob("performance_summary_*.csv"))
        if not csv_files:
            raise FileNotFoundError("未找到消融实验结果文件")
        csv_file = max(csv_files, key=lambda p: p.stat().st_mtime)
    
    print(f"加载消融实验结果: {csv_file}")
    return pd.read_csv(csv_file)


def generate_ablation_performance_comparison(df, save_format='pdf'):
    """
    生成消融实验性能对比图
    
    Args:
        df: 消融实验结果DataFrame
        save_format: 保存格式 ('pdf', 'png', 或 'both')
    """
    # 计算各变体的平均性能
    summary = df.groupby('variant').agg({
        'annual_return': 'mean',
        'sharpe_ratio': 'mean',
        'max_drawdown': 'mean',
        'win_rate': 'mean'
    }).round(4)
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Ablation Study: Performance Comparison', fontsize=16, fontweight='bold')
    
    metrics = [
        ('annual_return', 'Annual Return', axes[0, 0]),
        ('sharpe_ratio', 'Sharpe Ratio', axes[0, 1]),
        ('max_drawdown', 'Max Drawdown', axes[1, 0]),
        ('win_rate', 'Win Rate', axes[1, 1])
    ]
    
    for metric, title, ax in metrics:
        summary_sorted = summary.sort_values(metric, ascending=False)
        bars = ax.barh(summary_sorted.index, summary_sorted[metric])
        
        # 高亮Full变体
        for i, variant in enumerate(summary_sorted.index):
            if 'Full' in variant:
                bars[i].set_color('#2ecc71')
            else:
                bars[i].set_color('#3498db')
        
        ax.set_xlabel(title, fontsize=12)
        ax.set_title(f'{title} by Variant', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # 添加数值标签
        for i, v in enumerate(summary_sorted[metric]):
            ax.text(v, i, f' {v:.4f}', va='center', fontsize=10)
    
    plt.tight_layout()
    
    # 保存图表
    if save_format in ['pdf', 'both']:
        plt.savefig(PAPER_FIGURES_DIR / 'ablation_performance_comparison.pdf', 
                    dpi=300, bbox_inches='tight')
    if save_format in ['png', 'both']:
        plt.savefig(PAPER_FIGURES_DIR / 'ablation_performance_comparison.png', 
                    dpi=300, bbox_inches='tight')
    
    print(f"图表已保存到: {PAPER_FIGURES_DIR}")
    plt.close()


def generate_ablation_results_table(df):
    """
    生成消融实验结果的LaTeX表格
    
    Args:
        df: 消融实验结果DataFrame
    """
    # 计算各变体的平均性能和标准差
    summary = df.groupby('variant').agg({
        'annual_return': ['mean', 'std'],
        'sharpe_ratio': ['mean', 'std'],
        'max_drawdown': ['mean', 'std'],
        'win_rate': ['mean', 'std']
    }).round(4)
    
    # 生成LaTeX表格
    latex_content = r"""\begin{table}[htbp]
\centering
\caption{Ablation Study Results: Performance Metrics by Variant}
\label{tab:ablation_results}
\begin{tabular}{lcccc}
\toprule
\textbf{Variant} & \textbf{Annual Return} & \textbf{Sharpe Ratio} & \textbf{Max Drawdown} & \textbf{Win Rate} \\
\midrule
"""
    
    # 按年化收益排序
    summary_sorted = summary.sort_values(('annual_return', 'mean'), ascending=False)
    
    for variant in summary_sorted.index:
        ar_mean = summary.loc[variant, ('annual_return', 'mean')]
        ar_std = summary.loc[variant, ('annual_return', 'std')]
        sr_mean = summary.loc[variant, ('sharpe_ratio', 'mean')]
        sr_std = summary.loc[variant, ('sharpe_ratio', 'std')]
        md_mean = summary.loc[variant, ('max_drawdown', 'mean')]
        md_std = summary.loc[variant, ('max_drawdown', 'std')]
        wr_mean = summary.loc[variant, ('win_rate', 'mean')]
        wr_std = summary.loc[variant, ('win_rate', 'std')]
        
        # 高亮Full变体
        if 'Full' in variant:
            variant_name = r'\textbf{' + variant + '}'
        else:
            variant_name = variant
        
        latex_content += f"{variant_name} & "
        latex_content += f"{ar_mean:.2%} $\\pm$ {ar_std:.2%} & "
        latex_content += f"{sr_mean:.3f} $\\pm$ {sr_std:.3f} & "
        latex_content += f"{md_mean:.2%} $\\pm$ {md_std:.2%} & "
        latex_content += f"{wr_mean:.2%} $\\pm$ {wr_std:.2%} \\\\\n"
    
    latex_content += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    # 保存LaTeX表格
    table_file = PAPER_TABLES_DIR / 'ablation_results.tex'
    with open(table_file, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    print(f"LaTeX表格已保存到: {table_file}")


def generate_stock_performance_heatmap(df, save_format='pdf'):
    """
    生成各变体在不同股票上的性能热力图
    
    Args:
        df: 消融实验结果DataFrame
        save_format: 保存格式
    """
    # 创建透视表
    pivot = df.pivot_table(
        values='annual_return',
        index='variant',
        columns='ticker',
        aggfunc='mean'
    )
    
    # 创建热力图
    plt.figure(figsize=(16, 8))
    sns.heatmap(pivot, annot=True, fmt='.2%', cmap='RdYlGn', center=0,
                linewidths=0.5, cbar_kws={'label': 'Annual Return'})
    plt.title('Annual Return Heatmap: Variants vs Stocks', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Stock Ticker', fontsize=12)
    plt.ylabel('Variant', fontsize=12)
    plt.tight_layout()
    
    # 保存图表
    if save_format in ['pdf', 'both']:
        plt.savefig(PAPER_FIGURES_DIR / 'stock_performance_heatmap.pdf', 
                    dpi=300, bbox_inches='tight')
    if save_format in ['png', 'both']:
        plt.savefig(PAPER_FIGURES_DIR / 'stock_performance_heatmap.png', 
                    dpi=300, bbox_inches='tight')
    
    print(f"热力图已保存到: {PAPER_FIGURES_DIR}")
    plt.close()


def load_comparison_results():
    """
    从 results/comparison_study/ 加载对比实验结果
    
    Returns:
        DataFrame: 对比实验结果
    """
    comparison_dir = RESULTS_DIR / "comparison_study" / "raw_results"
    
    if not comparison_dir.exists():
        print(f"⚠️ 对比实验结果目录不存在: {comparison_dir}")
        return None
    
    # 找到所有JSON结果文件
    json_files = list(comparison_dir.glob("baseline_results_*.json"))
    if not json_files:
        print(f"⚠️ 未找到对比实验结果文件")
        return None
    
    print(f"加载对比实验结果: 找到 {len(json_files)} 个文件")
    
    # 解析结果（这里可以根据实际需要扩展）
    results = []
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"⚠️ 加载文件失败 {json_file}: {e}")
    
    return results


def generate_all_outputs(experiment_id=None, save_format='both', include_comparison=True):
    """
    生成所有论文所需的图表
    
    Args:
        experiment_id: 实验ID，如果为None则使用最新的实验
        save_format: 保存格式 ('pdf', 'png', 或 'both')
        include_comparison: 是否包含对比实验结果
    """
    print("=" * 80)
    print("开始生成论文图表...")
    print("=" * 80)
    
    # 1. 处理消融实验结果
    print("\n【消融实验】")
    print("-" * 80)
    try:
        df = load_ablation_results(experiment_id)
        print(f"加载了 {len(df)} 条实验记录")
        print(f"变体数量: {df['variant'].nunique()}")
        print(f"股票数量: {df['ticker'].nunique()}")
        
        # 生成消融实验图表
        print("\n[1/3] 生成性能对比图...")
        generate_ablation_performance_comparison(df, save_format)
        
        print("\n[2/3] 生成结果表格...")
        generate_ablation_results_table(df)
        
        print("\n[3/3] 生成股票性能热力图...")
        generate_stock_performance_heatmap(df, save_format)
    except FileNotFoundError as e:
        print(f"⚠️ 跳过消融实验: {e}")
        print("   （消融实验尚未运行或结果文件不存在）")
    
    # 2. 处理对比实验结果
    if include_comparison:
        print("\n【对比实验】")
        print("-" * 80)
        comparison_results = load_comparison_results()
        if comparison_results:
            print(f"✅ 加载了 {len(comparison_results)} 个对比实验结果")
            print("   （对比实验图表生成功能待扩展）")
            print("   提示: 对比实验的详细图表请使用 experiments/comparison_experiments/algorithms/post.py")
        else:
            print("⚠️ 跳过对比实验（无结果数据）")
    
    print("\n" + "=" * 80)
    print("图表生成完成！")
    print(f"图表位置: {PAPER_FIGURES_DIR.absolute()}")
    print(f"表格位置: {PAPER_TABLES_DIR.absolute()}")
    print("=" * 80)


if __name__ == "__main__":
    # 生成所有论文输出
    generate_all_outputs(save_format='both')
