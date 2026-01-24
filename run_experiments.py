#!/usr/bin/env python3
"""
改进的实验运行脚本
Enhanced Experiment Runner

功能：
1. 按参数组合运行实验，每个组合输出独立的CSV文件
2. 文件名包含完整参数信息
3. 输出详细的每轮次实验数据
4. 支持批量实验和单个实验

文件命名规范：
experiment_results_lookback{lb}_lookahead{la}_stride{s}_depth{d}_{ticker}_{timestamp}.csv
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import json
from itertools import product

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

try:
    from comparison_experiments.algorithms.baseline import BaselineRunner
except ImportError:
    print("⚠️ 无法导入BaselineRunner，将使用简化版本")
    BaselineRunner = None

try:
    from comparison_experiments.algorithms.data_utils import get_available_tickers
except ImportError:
    def get_available_tickers():
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']


class EnhancedExperimentRunner:
    """增强的实验运行器"""
    
    def __init__(self, base_dir="/Users/zjt/Desktop/EATA-RL-main"):
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / "experiment_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # 默认参数网格
        self.param_grid = {
            'lookback': [30, 50, 100],
            'lookahead': [5, 10, 20], 
            'stride': [1, 2],
            'depth': [200, 300, 500]
        }
        
        # 策略列表
        self.strategies = [
            'eata', 'buy_and_hold', 'macd', 'transformer',
            'ppo', 'gp', 'lstm', 'lightgbm', 'arima'
        ]
        
        # 测试股票
        self.test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
    def generate_param_combinations(self, custom_grid=None):
        """生成参数组合"""
        grid = custom_grid or self.param_grid
        
        param_names = list(grid.keys())
        param_values = list(grid.values())
        
        combinations = []
        for combo in product(*param_values):
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)
            
        return combinations
    
    def run_single_experiment(self, params, ticker, strategies=None, num_runs=1):
        """运行单个参数组合的实验"""
        strategies = strategies or self.strategies
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 生成文件名
        param_str = "_".join([f"{k}{v}" for k, v in params.items()])
        filename = f"experiment_results_{param_str}_{ticker}_{timestamp}.csv"
        filepath = self.results_dir / filename
        
        print(f"🧪 运行实验: {ticker} | 参数: {params}")
        
        # 存储所有轮次的结果
        all_results = []
        
        for run_id in range(num_runs):
            print(f"  📊 第 {run_id + 1}/{num_runs} 轮...")
            
            try:
                # 运行baseline实验
                runner = BaselineRunner()
                results = runner.run_real_data_experiment(
                    ticker=ticker,
                    strategies=strategies,
                    **params  # 传入EATA参数
                )
                
                # 处理结果
                for strategy, metrics in results.items():
                    if strategy == 'summary':
                        continue
                        
                    if isinstance(metrics, dict):
                        row = {
                            'run_id': run_id + 1,
                            'ticker': ticker,
                            'strategy': strategy,
                            'timestamp': timestamp,
                            **params,  # 添加所有参数
                            **metrics  # 添加所有指标
                        }
                        all_results.append(row)
                        
            except Exception as e:
                print(f"    ❌ 第 {run_id + 1} 轮失败: {str(e)}")
                continue
        
        # 保存结果到CSV
        if all_results:
            df = pd.DataFrame(all_results)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"  💾 结果已保存: {filename}")
            return df
        else:
            print(f"  ❌ 实验失败，无结果保存")
            return None
    
    def run_parameter_sweep(self, tickers=None, strategies=None, custom_grid=None, num_runs=3):
        """运行参数扫描实验"""
        tickers = tickers or self.test_tickers
        strategies = strategies or self.strategies
        
        # 生成参数组合
        param_combinations = self.generate_param_combinations(custom_grid)
        
        print(f"🚀 开始参数扫描实验")
        print(f"📊 参数组合数: {len(param_combinations)}")
        print(f"📈 测试股票: {tickers}")
        print(f"🔧 测试策略: {strategies}")
        print(f"🔄 每组合运行轮次: {num_runs}")
        print("=" * 60)
        
        total_experiments = len(param_combinations) * len(tickers)
        completed = 0
        
        all_experiment_files = []
        
        for i, params in enumerate(param_combinations):
            print(f"\n📋 参数组合 {i+1}/{len(param_combinations)}: {params}")
            
            for ticker in tickers:
                try:
                    df = self.run_single_experiment(
                        params=params,
                        ticker=ticker, 
                        strategies=strategies,
                        num_runs=num_runs
                    )
                    
                    if df is not None:
                        all_experiment_files.append(df)
                    
                    completed += 1
                    progress = (completed / total_experiments) * 100
                    print(f"  ✅ 进度: {completed}/{total_experiments} ({progress:.1f}%)")
                    
                except Exception as e:
                    print(f"  ❌ 实验失败 {ticker}: {str(e)}")
                    continue
        
        print("\n" + "=" * 60)
        print(f"🎉 参数扫描实验完成！")
        print(f"📁 结果文件保存在: {self.results_dir}")
        print(f"📊 成功完成: {len(all_experiment_files)} 个实验")
        
        return all_experiment_files
    
    def run_single_param_set(self, lookback=50, lookahead=10, stride=1, depth=300, 
                           tickers=None, strategies=None, num_runs=5):
        """运行单个参数集的实验"""
        params = {
            'lookback': lookback,
            'lookahead': lookahead, 
            'stride': stride,
            'depth': depth
        }
        
        tickers = tickers or self.test_tickers
        strategies = strategies or self.strategies
        
        print(f"🎯 运行单参数集实验")
        print(f"🔧 参数: {params}")
        print(f"📈 股票: {tickers}")
        print(f"🔄 运行轮次: {num_runs}")
        print("=" * 40)
        
        results = []
        for ticker in tickers:
            df = self.run_single_experiment(
                params=params,
                ticker=ticker,
                strategies=strategies, 
                num_runs=num_runs
            )
            if df is not None:
                results.append(df)
        
        return results
    
    def create_master_summary(self):
        """创建主汇总文件"""
        print("📋 创建主汇总文件...")
        
        # 查找所有实验结果文件
        csv_files = list(self.results_dir.glob("experiment_results_*.csv"))
        
        if not csv_files:
            print("❌ 未找到实验结果文件")
            return None
        
        print(f"📁 找到 {len(csv_files)} 个结果文件")
        
        # 合并所有结果
        all_data = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                all_data.append(df)
            except Exception as e:
                print(f"⚠️ 读取文件失败 {file.name}: {str(e)}")
                continue
        
        if all_data:
            master_df = pd.concat(all_data, ignore_index=True)
            
            # 保存主汇总文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            master_file = self.results_dir / f"master_experiment_summary_{timestamp}.csv"
            master_df.to_csv(master_file, index=False, encoding='utf-8-sig')
            
            print(f"💾 主汇总文件已保存: {master_file.name}")
            print(f"📊 总记录数: {len(master_df)}")
            
            # 生成简要统计
            self._print_summary_stats(master_df)
            
            return master_df
        
        return None
    
    def _print_summary_stats(self, df):
        """打印汇总统计"""
        print("\n📊 实验汇总统计:")
        print(f"  总实验次数: {len(df)}")
        print(f"  测试股票数: {df['ticker'].nunique()}")
        print(f"  测试策略数: {df['strategy'].nunique()}")
        print(f"  参数组合数: {len(df.groupby(['lookback', 'lookahead', 'stride', 'depth']))}")
        
        # 按策略统计平均性能
        strategy_stats = df.groupby('strategy')['Annual Return (AR)'].agg(['mean', 'count']).round(4)
        strategy_stats = strategy_stats.sort_values('mean', ascending=False)
        
        print("\n🏆 策略平均表现 (按年化收益排序):")
        for strategy, row in strategy_stats.iterrows():
            print(f"  {strategy:12s}: {row['mean']:8.2f}% (n={row['count']:3d})")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='增强的实验运行器')
    parser.add_argument('--mode', choices=['single', 'sweep', 'summary'], 
                       default='single', help='运行模式')
    parser.add_argument('--lookback', type=int, default=50, help='回望窗口')
    parser.add_argument('--lookahead', type=int, default=10, help='预测窗口')
    parser.add_argument('--stride', type=int, default=1, help='步长')
    parser.add_argument('--depth', type=int, default=300, help='深度')
    parser.add_argument('--tickers', nargs='+', default=['AAPL', 'MSFT', 'GOOGL'], 
                       help='测试股票')
    parser.add_argument('--strategies', nargs='+', 
                       default=['eata', 'buy_and_hold', 'macd', 'transformer'],
                       help='测试策略')
    parser.add_argument('--runs', type=int, default=3, help='运行轮次')
    parser.add_argument('--base_dir', default='/Users/zjt/Desktop/EATA-RL-main',
                       help='项目根目录')
    
    args = parser.parse_args()
    
    # 创建实验运行器
    runner = EnhancedExperimentRunner(args.base_dir)
    
    if args.mode == 'single':
        # 运行单参数集实验
        runner.run_single_param_set(
            lookback=args.lookback,
            lookahead=args.lookahead,
            stride=args.stride,
            depth=args.depth,
            tickers=args.tickers,
            strategies=args.strategies,
            num_runs=args.runs
        )
    elif args.mode == 'sweep':
        # 运行参数扫描
        runner.run_parameter_sweep(
            tickers=args.tickers,
            strategies=args.strategies,
            num_runs=args.runs
        )
    elif args.mode == 'summary':
        # 创建汇总
        runner.create_master_summary()


if __name__ == "__main__":
    main()
