#!/usr/bin/env python3
"""
学术论文实验数据处理流水线
Academic Paper Experiment Data Processing Pipeline

功能：
1. 从原始实验结果中提取结构化数据
2. 生成符合学术标准的CSV数据文件
3. 使用ggplot2风格绘制图表到figures/
4. 生成LaTeX表格到tables/

使用方法：
python experiment_pipeline.py --mode all
python experiment_pipeline.py --mode figures
python experiment_pipeline.py --mode tables
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置ggplot风格
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    plt.style.use('seaborn')
sns.set_palette("husl")

class ExperimentPipeline:
    """学术论文实验数据处理流水线"""
    
    def __init__(self, base_dir="/Users/zjt/Desktop/EATA-RL-main"):
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / "comparison_results"
        self.figures_dir = self.base_dir / "figures"
        self.tables_dir = self.base_dir / "tables" 
        self.data_dir = self.base_dir / "data"
        
        # 创建输出目录
        for dir_path in [self.figures_dir, self.tables_dir, self.data_dir]:
            dir_path.mkdir(exist_ok=True)
            
        # 实验参数配置
        self.strategies = [
            'eata', 'buy_and_hold', 'macd', 'transformer', 
            'ppo', 'gp', 'lstm', 'lightgbm', 'arima',
            # FinRL强化学习策略
            'finrl_ppo', 'finrl_a2c', 'finrl_sac', 'finrl_td3', 'finrl_ddpg',
            # InvestorBench LLM策略
            'investorbench_gpt35', 'investorbench_gpt4', 'investorbench_llama2', 'investorbench_finbert'
        ]
        
        self.strategy_names = {
            'eata': 'EATA',
            'buy_and_hold': 'Buy & Hold',
            'macd': 'MACD',
            'transformer': 'Transformer',
            'ppo': 'PPO',
            'gp': 'Genetic Programming',
            'lstm': 'LSTM',
            'lightgbm': 'LightGBM',
            'arima': 'ARIMA',
            # FinRL策略
            'finrl_ppo': 'FinRL-PPO',
            'finrl_a2c': 'FinRL-A2C',
            'finrl_sac': 'FinRL-SAC',
            'finrl_td3': 'FinRL-TD3',
            'finrl_ddpg': 'FinRL-DDPG',
            # InvestorBench策略
            'investorbench_gpt35': 'GPT-3.5',
            'investorbench_gpt4': 'GPT-4',
            'investorbench_llama2': 'Llama2',
            'investorbench_finbert': 'FinBERT'
        }
        
    def extract_experiment_data(self):
        """从JSON结果文件中提取实验数据"""
        print("🔍 提取实验数据...")
        
        # 查找最新的完整实验结果
        json_files = list(self.results_dir.glob("comparison_results_final_*.json"))
        if not json_files:
            raise FileNotFoundError("未找到实验结果JSON文件")
            
        # 选择最新的文件
        latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
        print(f"📁 使用实验结果文件: {latest_file.name}")
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
            
        # 提取实验参数
        experiment_info = self._extract_experiment_params(latest_file.name)
        
        # 转换为结构化数据
        experiment_data = []
        
        # 处理列表格式的结果
        if isinstance(results, list):
            for result in results:
                if isinstance(result, dict) and result.get('success', False):
                    # 从config中提取参数
                    config = result.get('config', {})
                    
                    row = {
                        'ticker': config.get('stock', 'UNKNOWN'),
                        'strategy': result.get('algorithm', 'UNKNOWN').lower(),
                        'strategy_name': self.strategy_names.get(result.get('algorithm', 'UNKNOWN').lower(), result.get('algorithm', 'UNKNOWN')),
                        'annual_return': float(result.get('annualized_return', 0)) * 100,  # 转换为百分比
                        'sharpe_ratio': float(result.get('sharpe_ratio', 0)),
                        'max_drawdown': float(result.get('max_drawdown', 0)),
                        'win_rate': 0,  # 这个格式中没有胜率数据
                        'volatility': float(result.get('volatility', 0)),
                        'calmar_ratio': 0,  # 需要计算
                        'sortino_ratio': 0,  # 需要计算
                        'total_return': float(result.get('total_return', 0)),
                        'num_trades': int(result.get('num_trades', 0)),
                        'experiment_time': float(result.get('experiment_time', 0)),
                        'lookback': config.get('lookback', 50),
                        'lookahead': config.get('lookahead', 10),
                        'stride': config.get('stride', 1),
                        'depth': config.get('depth', 300),
                        **experiment_info
                    }
                    experiment_data.append(row)
        else:
            # 处理字典格式的结果（原有逻辑）
            for ticker, ticker_results in results.items():
                if ticker == 'summary':
                    continue
                    
                for strategy, metrics in ticker_results.items():
                    if isinstance(metrics, dict) and 'Annual Return (AR)' in metrics:
                        row = {
                            'ticker': ticker,
                            'strategy': strategy,
                            'strategy_name': self.strategy_names.get(strategy, strategy),
                            'annual_return': float(metrics.get('Annual Return (AR)', 0)),
                            'sharpe_ratio': float(metrics.get('Sharpe Ratio', 0)),
                            'max_drawdown': float(metrics.get('Max Drawdown (MDD)', 0)),
                            'win_rate': float(metrics.get('Win Rate', 0)),
                            'volatility': float(metrics.get('Volatility (Annual)', 0)),
                            'calmar_ratio': float(metrics.get('Calmar Ratio', 0)),
                            'sortino_ratio': float(metrics.get('Sortino Ratio', 0)),
                            **experiment_info
                        }
                        experiment_data.append(row)
        
        df = pd.DataFrame(experiment_data)
        
        # 保存原始数据
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_data_file = self.data_dir / f"experiment_raw_data_{timestamp}.csv"
        df.to_csv(raw_data_file, index=False, encoding='utf-8-sig')
        print(f"💾 原始数据已保存: {raw_data_file}")
        
        return df
    
    def _extract_experiment_params(self, filename):
        """从文件名中提取实验参数"""
        # 从文件名解析时间戳等信息
        parts = filename.replace('.json', '').split('_')
        
        return {
            'experiment_date': parts[-2] if len(parts) >= 2 else 'unknown',
            'experiment_time': parts[-1] if len(parts) >= 1 else 'unknown',
            'lookback': 50,  # 默认参数，可以从配置文件读取
            'lookahead': 10,
            'stride': 1,
            'depth': 300
        }
    
    def generate_summary_statistics(self, df):
        """生成汇总统计数据"""
        print("📊 生成汇总统计...")
        
        # 按策略汇总
        strategy_summary = df.groupby(['strategy', 'strategy_name']).agg({
            'annual_return': ['mean', 'std', 'count'],
            'sharpe_ratio': 'mean',
            'max_drawdown': 'mean',
            'win_rate': 'mean',
            'volatility': 'mean',
            'calmar_ratio': 'mean',
            'sortino_ratio': 'mean'
        }).round(4)
        
        # 展平列名
        strategy_summary.columns = ['_'.join(col).strip() for col in strategy_summary.columns]
        strategy_summary = strategy_summary.reset_index()
        
        # 按年化收益排序
        strategy_summary = strategy_summary.sort_values('annual_return_mean', ascending=False)
        
        # 保存汇总数据
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.data_dir / f"strategy_summary_{timestamp}.csv"
        strategy_summary.to_csv(summary_file, index=False, encoding='utf-8-sig')
        print(f"💾 汇总数据已保存: {summary_file}")
        
        return strategy_summary
    
    def generate_figures(self, df, summary_df):
        """生成学术论文图表"""
        print("🎨 生成图表...")
        
        # 设置图表样式
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11
        })
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 策略性能对比柱状图
        self._plot_strategy_performance(summary_df, timestamp)
        
        # 2. 风险收益散点图
        self._plot_risk_return_scatter(summary_df, timestamp)
        
        # 3. 策略性能分布箱线图
        self._plot_performance_distribution(df, timestamp)
        
        # 4. 相关性热力图
        self._plot_correlation_heatmap(df, timestamp)
        
        print(f"✅ 图表已保存到 {self.figures_dir}")
    
    def _plot_strategy_performance(self, summary_df, timestamp):
        """策略性能对比柱状图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 年化收益对比
        bars1 = ax1.bar(range(len(summary_df)), summary_df['annual_return_mean'], 
                       color=sns.color_palette("husl", len(summary_df)))
        ax1.set_title('Annual Return Comparison Across Strategies', fontweight='bold')
        ax1.set_xlabel('Strategy')
        ax1.set_ylabel('Annual Return (%)')
        ax1.set_xticks(range(len(summary_df)))
        ax1.set_xticklabels(summary_df['strategy_name'], rotation=45, ha='right')
        
        # 添加数值标签
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}%', ha='center', va='bottom')
        
        # 夏普比率对比
        bars2 = ax2.bar(range(len(summary_df)), summary_df['sharpe_ratio_mean'],
                       color=sns.color_palette("husl", len(summary_df)))
        ax2.set_title('Sharpe Ratio Comparison Across Strategies', fontweight='bold')
        ax2.set_xlabel('Strategy')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.set_xticks(range(len(summary_df)))
        ax2.set_xticklabels(summary_df['strategy_name'], rotation=45, ha='right')
        
        # 添加数值标签
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / f'strategy_performance_comparison_{timestamp}.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / f'strategy_performance_comparison_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_risk_return_scatter(self, summary_df, timestamp):
        """风险收益散点图"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(summary_df['volatility_mean'], summary_df['annual_return_mean'],
                           s=100, alpha=0.7, c=range(len(summary_df)), cmap='viridis')
        
        # 添加策略标签
        for i, row in summary_df.iterrows():
            ax.annotate(row['strategy_name'], 
                       (row['volatility_mean'], row['annual_return_mean']),
                       xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax.set_xlabel('Volatility (Annual)')
        ax.set_ylabel('Annual Return (%)')
        ax.set_title('Risk-Return Profile of Trading Strategies', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / f'risk_return_scatter_{timestamp}.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / f'risk_return_scatter_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_distribution(self, df, timestamp):
        """策略性能分布箱线图"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 按策略分组的年化收益分布
        strategy_order = df.groupby('strategy_name')['annual_return'].mean().sort_values(ascending=False).index
        
        box_plot = ax.boxplot([df[df['strategy_name'] == strategy]['annual_return'].values 
                              for strategy in strategy_order],
                             labels=strategy_order, patch_artist=True)
        
        # 设置颜色
        colors = sns.color_palette("husl", len(strategy_order))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title('Distribution of Annual Returns Across Strategies', fontweight='bold')
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Annual Return (%)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / f'performance_distribution_{timestamp}.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / f'performance_distribution_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_heatmap(self, df, timestamp):
        """相关性热力图"""
        # 计算策略间相关性
        pivot_df = df.pivot_table(index='ticker', columns='strategy_name', values='annual_return')
        correlation_matrix = pivot_df.corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        heatmap = sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm',
                             center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        
        ax.set_title('Strategy Performance Correlation Matrix', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / f'strategy_correlation_{timestamp}.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / f'strategy_correlation_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_latex_tables(self, summary_df, df):
        """生成LaTeX表格"""
        print("📝 生成LaTeX表格...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 策略性能汇总表
        self._generate_performance_table(summary_df, timestamp)
        
        # 2. 详细统计表
        self._generate_detailed_stats_table(summary_df, timestamp)
        
        # 3. 前5名策略对比表
        self._generate_top_strategies_table(summary_df, timestamp)
        
        print(f"✅ LaTeX表格已保存到 {self.tables_dir}")
    
    def _generate_performance_table(self, summary_df, timestamp):
        """生成策略性能汇总表"""
        # 选择关键指标
        table_df = summary_df[['strategy_name', 'annual_return_mean', 'sharpe_ratio_mean', 
                              'max_drawdown_mean', 'win_rate_mean', 'annual_return_count']].copy()
        
        # 重命名列
        table_df.columns = ['Strategy', 'Annual Return (%)', 'Sharpe Ratio', 
                           'Max Drawdown', 'Win Rate (%)', 'Sample Size']
        
        # 格式化数值
        table_df['Annual Return (%)'] = table_df['Annual Return (%)'].apply(lambda x: f"{x:.2f}")
        table_df['Sharpe Ratio'] = table_df['Sharpe Ratio'].apply(lambda x: f"{x:.3f}")
        table_df['Max Drawdown'] = table_df['Max Drawdown'].apply(lambda x: f"{x:.3f}")
        table_df['Win Rate (%)'] = table_df['Win Rate (%)'].apply(lambda x: f"{x:.2f}")
        
        # 生成LaTeX代码
        latex_code = self._df_to_latex_table(
            table_df, 
            caption="Strategy Performance Summary",
            label="tab:strategy_performance",
            position="htbp"
        )
        
        # 保存文件
        with open(self.tables_dir / f'strategy_performance_{timestamp}.tex', 'w', encoding='utf-8') as f:
            f.write(latex_code)
    
    def _generate_detailed_stats_table(self, summary_df, timestamp):
        """生成详细统计表"""
        # 选择所有统计指标
        table_df = summary_df[['strategy_name', 'annual_return_mean', 'annual_return_std',
                              'sharpe_ratio_mean', 'volatility_mean', 'calmar_ratio_mean']].copy()
        
        # 重命名列
        table_df.columns = ['Strategy', 'Mean Return (%)', 'Std Return (%)', 
                           'Sharpe Ratio', 'Volatility', 'Calmar Ratio']
        
        # 格式化数值
        for col in ['Mean Return (%)', 'Std Return (%)', 'Volatility']:
            table_df[col] = table_df[col].apply(lambda x: f"{x:.2f}")
        for col in ['Sharpe Ratio', 'Calmar Ratio']:
            table_df[col] = table_df[col].apply(lambda x: f"{x:.3f}")
        
        # 生成LaTeX代码
        latex_code = self._df_to_latex_table(
            table_df,
            caption="Detailed Strategy Statistics",
            label="tab:detailed_stats",
            position="htbp"
        )
        
        # 保存文件
        with open(self.tables_dir / f'detailed_statistics_{timestamp}.tex', 'w', encoding='utf-8') as f:
            f.write(latex_code)
    
    def _generate_top_strategies_table(self, summary_df, timestamp):
        """生成前5名策略对比表"""
        # 选择前5名策略
        top5_df = summary_df.head(5)[['strategy_name', 'annual_return_mean', 
                                     'sharpe_ratio_mean', 'max_drawdown_mean']].copy()
        
        # 重命名列
        top5_df.columns = ['Strategy', 'Annual Return (%)', 'Sharpe Ratio', 'Max Drawdown']
        
        # 格式化数值
        top5_df['Annual Return (%)'] = top5_df['Annual Return (%)'].apply(lambda x: f"{x:.2f}")
        top5_df['Sharpe Ratio'] = top5_df['Sharpe Ratio'].apply(lambda x: f"{x:.3f}")
        top5_df['Max Drawdown'] = top5_df['Max Drawdown'].apply(lambda x: f"{x:.3f}")
        
        # 添加排名
        top5_df.insert(0, 'Rank', range(1, len(top5_df) + 1))
        
        # 生成LaTeX代码
        latex_code = self._df_to_latex_table(
            top5_df,
            caption="Top 5 Performing Strategies",
            label="tab:top_strategies",
            position="htbp"
        )
        
        # 保存文件
        with open(self.tables_dir / f'top_strategies_{timestamp}.tex', 'w', encoding='utf-8') as f:
            f.write(latex_code)
    
    def _df_to_latex_table(self, df, caption, label, position="htbp"):
        """将DataFrame转换为LaTeX表格"""
        # 生成表格头部
        num_cols = len(df.columns)
        col_spec = "l" + "c" * (num_cols - 1)
        
        latex_code = f"""\\begin{{table}}[{position}]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{{col_spec}}}
\\toprule
"""
        
        # 添加表头
        header = " & ".join(df.columns) + " \\\\\n"
        latex_code += header
        latex_code += "\\midrule\n"
        
        # 添加数据行
        for _, row in df.iterrows():
            row_str = " & ".join(str(val) for val in row.values) + " \\\\\n"
            latex_code += row_str
        
        # 添加表格尾部
        latex_code += """\\bottomrule
\\end{tabular}
\\end{table}
"""
        
        return latex_code
    
    def run_full_pipeline(self):
        """运行完整的实验数据处理流水线"""
        print("🚀 启动学术论文实验数据处理流水线...")
        print("=" * 60)
        
        try:
            # 1. 提取实验数据
            df = self.extract_experiment_data()
            print(f"📊 提取到 {len(df)} 条实验记录")
            
            # 2. 生成汇总统计
            summary_df = self.generate_summary_statistics(df)
            print(f"📈 生成 {len(summary_df)} 个策略的汇总统计")
            
            # 3. 生成图表
            self.generate_figures(df, summary_df)
            
            # 4. 生成LaTeX表格
            self.generate_latex_tables(summary_df, df)
            
            print("=" * 60)
            print("✅ 实验数据处理流水线执行完成！")
            print(f"📁 图表输出目录: {self.figures_dir}")
            print(f"📁 表格输出目录: {self.tables_dir}")
            print(f"📁 数据输出目录: {self.data_dir}")
            
            return df, summary_df
            
        except Exception as e:
            print(f"❌ 流水线执行失败: {str(e)}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='学术论文实验数据处理流水线')
    parser.add_argument('--mode', choices=['all', 'figures', 'tables', 'data'], 
                       default='all', help='处理模式')
    parser.add_argument('--base_dir', default='/Users/zjt/Desktop/EATA-RL-main',
                       help='项目根目录')
    
    args = parser.parse_args()
    
    # 创建流水线实例
    pipeline = ExperimentPipeline(args.base_dir)
    
    if args.mode == 'all':
        pipeline.run_full_pipeline()
    elif args.mode == 'data':
        df = pipeline.extract_experiment_data()
        pipeline.generate_summary_statistics(df)
    elif args.mode == 'figures':
        df = pipeline.extract_experiment_data()
        summary_df = pipeline.generate_summary_statistics(df)
        pipeline.generate_figures(df, summary_df)
    elif args.mode == 'tables':
        df = pipeline.extract_experiment_data()
        summary_df = pipeline.generate_summary_statistics(df)
        pipeline.generate_latex_tables(summary_df, df)


if __name__ == "__main__":
    main()
