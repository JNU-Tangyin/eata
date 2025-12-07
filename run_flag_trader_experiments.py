#!/usr/bin/env python3
"""
FLAG-TRADER风格对比实验运行器
FLAG-TRADER Style Comparison Experiment Runner

功能：
1. 运行EATA vs FinRL vs InvestorBench的全面对比实验
2. 复现FLAG-TRADER论文中的实验设置
3. 生成学术论文级别的结果分析
4. 支持多种实验配置和评估指标

参考论文: FLAG-TRADER: Fusion LLM-Agent with Gradient-based Reinforcement Learning for Financial Trading

使用方法:
python run_flag_trader_experiments.py --experiment_type full
python run_flag_trader_experiments.py --experiment_type finrl_only
python run_flag_trader_experiments.py --experiment_type llm_only
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import json
import time
from typing import Dict, List, Tuple, Optional

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

try:
    from comparison_experiments.algorithms.baseline import BaselineRunner
    from comparison_experiments.algorithms.data_utils import get_available_tickers
    from experiment_pipeline import ExperimentPipeline
except ImportError as e:
    print(f"⚠️ 导入模块失败: {e}")
    print("请确保在EATA项目根目录下运行此脚本")
    sys.exit(1)


class FlagTraderExperimentRunner:
    """FLAG-TRADER风格实验运行器"""
    
    def __init__(self, base_dir="/Users/zjt/Desktop/EATA-RL-main"):
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / "flag_trader_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # 实验配置 - 基于FLAG-TRADER论文
        self.experiment_configs = {
            'full': {
                'name': 'Full Comparison (EATA vs FinRL vs InvestorBench)',
                'strategies': [
                    'eata',  # 我们的方法
                    # 传统基线
                    'buy_and_hold', 'macd',
                    # 机器学习基线
                    'lstm', 'transformer', 'lightgbm',
                    # FinRL强化学习方法
                    'finrl_ppo', 'finrl_a2c', 'finrl_sac', 'finrl_td3',
                    # InvestorBench LLM方法
                    'investorbench_gpt35', 'investorbench_gpt4'
                ],
                'description': '完整对比实验，包含所有类型的基线方法'
            },
            'finrl_focus': {
                'name': 'FinRL Focused Comparison',
                'strategies': [
                    'eata',
                    'finrl_ppo', 'finrl_a2c', 'finrl_sac', 'finrl_td3', 'finrl_ddpg',
                    'buy_and_hold', 'ppo'  # 对照组
                ],
                'description': '专注于FinRL强化学习方法的对比'
            },
            'llm_focus': {
                'name': 'LLM Focused Comparison',
                'strategies': [
                    'eata',
                    'investorbench_gpt35', 'investorbench_gpt4', 
                    'investorbench_llama2', 'investorbench_finbert',
                    'transformer', 'lstm'  # 对照组
                ],
                'description': '专注于LLM方法的对比'
            },
            'academic': {
                'name': 'Academic Paper Comparison',
                'strategies': [
                    'eata',  # 提出的方法
                    'finrl_ppo', 'finrl_sac',  # FinRL代表
                    'investorbench_gpt35',  # LLM代表
                    'transformer', 'lstm',  # 深度学习基线
                    'buy_and_hold', 'macd'  # 传统基线
                ],
                'description': '学术论文标准对比实验'
            }
        }
        
        # 测试股票 - 选择不同市场特征的股票
        self.test_tickers = {
            'tech_growth': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
            'finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
            'diverse': ['AAPL', 'JPM', 'JNJ', 'XOM', 'WMT']
        }
        
        # 实验参数
        self.experiment_params = {
            'lookback': 50,
            'lookahead': 10,
            'stride': 1,
            'depth': 300,
            'num_runs': 3  # 每个配置运行3次取平均
        }
    
    def run_experiment_suite(self, experiment_type: str = 'academic',
                           ticker_set: str = 'diverse',
                           custom_tickers: Optional[List[str]] = None,
                           **kwargs) -> Dict:
        """运行实验套件"""
        
        if experiment_type not in self.experiment_configs:
            raise ValueError(f"不支持的实验类型: {experiment_type}")
        
        config = self.experiment_configs[experiment_type]
        tickers = custom_tickers or self.test_tickers.get(ticker_set, self.test_tickers['diverse'])
        
        print(f"🚀 启动FLAG-TRADER风格实验: {config['name']}")
        print(f"📊 实验配置: {config['description']}")
        print(f"📈 测试股票: {tickers}")
        print(f"🔧 测试策略: {config['strategies']}")
        print("=" * 80)
        
        # 更新实验参数
        params = self.experiment_params.copy()
        params.update(kwargs)
        
        # 运行实验
        start_time = time.time()
        results = self._run_baseline_experiments(
            strategies=config['strategies'],
            tickers=tickers,
            **params
        )
        
        experiment_time = time.time() - start_time
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"flag_trader_results_{experiment_type}_{timestamp}.json"
        
        experiment_summary = {
            'experiment_type': experiment_type,
            'experiment_config': config,
            'tickers': tickers,
            'parameters': params,
            'experiment_time': experiment_time,
            'timestamp': timestamp,
            'results': results
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(experiment_summary, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n✅ 实验完成！耗时: {experiment_time/60:.1f} 分钟")
        print(f"📁 结果已保存: {results_file}")
        
        # 生成快速报告
        self._generate_quick_report(experiment_summary)
        
        return experiment_summary
    
    def _run_baseline_experiments(self, strategies: List[str], tickers: List[str], 
                                 num_runs: int = 3, **params) -> Dict:
        """运行基线实验"""
        
        runner = BaselineRunner()
        all_results = {}
        
        total_experiments = len(tickers) * len(strategies) * num_runs
        completed = 0
        
        for ticker in tickers:
            print(f"\n📊 处理股票: {ticker}")
            ticker_results = {}
            
            # 获取股票数据 (这里需要实现数据获取逻辑)
            try:
                df = self._get_stock_data(ticker)
                if df is None or len(df) < 100:
                    print(f"⚠️ {ticker} 数据不足，跳过")
                    continue
                
                print(f"📈 数据量: {len(df)} 条记录")
                
                # 运行所有策略
                strategy_results = runner.run_all_strategies(
                    df=df,
                    ticker=ticker,
                    train_ratio=0.7,
                    selected_strategies=strategies
                )
                
                # 多次运行取平均 (对于需要训练的策略)
                if num_runs > 1:
                    strategy_results = self._run_multiple_times(
                        runner, df, ticker, strategies, num_runs
                    )
                
                ticker_results = strategy_results
                completed += len(strategies)
                
                progress = (completed / total_experiments) * 100
                print(f"📊 总体进度: {completed}/{total_experiments} ({progress:.1f}%)")
                
            except Exception as e:
                print(f"❌ {ticker} 实验失败: {e}")
                continue
            
            all_results[ticker] = ticker_results
        
        return all_results
    
    def _run_multiple_times(self, runner: BaselineRunner, df: pd.DataFrame, 
                           ticker: str, strategies: List[str], num_runs: int) -> Dict:
        """多次运行实验取平均值"""
        
        print(f"🔄 运行 {num_runs} 次实验取平均...")
        
        all_runs = []
        for run_id in range(num_runs):
            print(f"  📊 第 {run_id + 1}/{num_runs} 轮...")
            
            try:
                run_results = runner.run_all_strategies(
                    df=df,
                    ticker=ticker,
                    train_ratio=0.7,
                    selected_strategies=strategies
                )
                all_runs.append(run_results)
            except Exception as e:
                print(f"    ❌ 第 {run_id + 1} 轮失败: {e}")
                continue
        
        # 计算平均结果
        if not all_runs:
            return {}
        
        averaged_results = {}
        for strategy in strategies:
            strategy_metrics = []
            
            for run_result in all_runs:
                if strategy in run_result and run_result[strategy]['success']:
                    metrics = run_result[strategy]['metrics']
                    if metrics is not None:
                        strategy_metrics.append(metrics)
            
            if strategy_metrics:
                # 计算平均指标
                avg_metrics = pd.concat(strategy_metrics, axis=1).mean(axis=1)
                std_metrics = pd.concat(strategy_metrics, axis=1).std(axis=1)
                
                averaged_results[strategy] = {
                    'metrics': avg_metrics,
                    'metrics_std': std_metrics,
                    'success': True,
                    'num_successful_runs': len(strategy_metrics),
                    'description': f"{strategy} (平均 {len(strategy_metrics)} 次运行)"
                }
            else:
                averaged_results[strategy] = {
                    'metrics': None,
                    'success': False,
                    'description': f"{strategy} (所有运行均失败)"
                }
        
        return averaged_results
    
    def _get_stock_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """获取股票数据 (模拟实现)"""
        try:
            # 这里应该实现真实的数据获取逻辑
            # 可以从EATA项目的数据源获取，或使用yfinance等
            
            # 模拟数据生成 (实际使用时应替换为真实数据)
            np.random.seed(hash(ticker) % 2**32)
            n_days = 500
            
            dates = pd.date_range(start='2022-01-01', periods=n_days, freq='D')
            
            # 生成模拟的股价数据
            returns = np.random.normal(0.001, 0.02, n_days)
            prices = 100 * np.cumprod(1 + returns)
            
            df = pd.DataFrame({
                'date': dates,
                'open': prices * (1 + np.random.normal(0, 0.005, n_days)),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, n_days)
            })
            
            # 确保high >= close >= low
            df['high'] = np.maximum(df['high'], df['close'])
            df['low'] = np.minimum(df['low'], df['close'])
            
            return df
            
        except Exception as e:
            print(f"❌ 获取 {ticker} 数据失败: {e}")
            return None
    
    def _generate_quick_report(self, experiment_summary: Dict):
        """生成快速报告"""
        
        print("\n" + "="*80)
        print("📋 实验结果快速报告")
        print("="*80)
        
        results = experiment_summary['results']
        config = experiment_summary['experiment_config']
        
        # 统计成功的实验
        total_experiments = 0
        successful_experiments = 0
        strategy_performance = {}
        
        for ticker, ticker_results in results.items():
            for strategy, result in ticker_results.items():
                total_experiments += 1
                if result.get('success', False):
                    successful_experiments += 1
                    
                    metrics = result.get('metrics')
                    if metrics is not None and hasattr(metrics, 'get'):
                        annual_return = metrics.get('annualized_return', 0)
                        sharpe_ratio = metrics.get('sharpe_ratio', 0)
                        
                        if strategy not in strategy_performance:
                            strategy_performance[strategy] = {
                                'returns': [],
                                'sharpes': [],
                                'count': 0
                            }
                        
                        strategy_performance[strategy]['returns'].append(annual_return)
                        strategy_performance[strategy]['sharpes'].append(sharpe_ratio)
                        strategy_performance[strategy]['count'] += 1
        
        print(f"📊 实验统计:")
        print(f"  总实验数: {total_experiments}")
        print(f"  成功实验: {successful_experiments}")
        print(f"  成功率: {successful_experiments/total_experiments*100:.1f}%")
        
        print(f"\n🏆 策略性能排名 (按平均年化收益):")
        
        # 计算平均性能并排序
        strategy_avg_performance = []
        for strategy, perf in strategy_performance.items():
            if perf['count'] > 0:
                avg_return = np.mean(perf['returns'])
                avg_sharpe = np.mean(perf['sharpes'])
                strategy_avg_performance.append({
                    'strategy': strategy,
                    'avg_return': avg_return,
                    'avg_sharpe': avg_sharpe,
                    'count': perf['count']
                })
        
        # 按年化收益排序
        strategy_avg_performance.sort(key=lambda x: x['avg_return'], reverse=True)
        
        print(f"{'排名':<4} {'策略':<20} {'年化收益':<12} {'夏普比率':<10} {'实验次数':<8}")
        print("-" * 60)
        
        for i, perf in enumerate(strategy_avg_performance[:10]):  # 显示前10名
            rank = i + 1
            strategy = perf['strategy']
            avg_return = perf['avg_return']
            avg_sharpe = perf['avg_sharpe']
            count = perf['count']
            
            print(f"{rank:<4} {strategy:<20} {avg_return:>10.2%} {avg_sharpe:>8.3f} {count:>6}")
        
        # EATA性能分析
        if 'eata' in strategy_performance:
            eata_perf = strategy_performance['eata']
            if eata_perf['count'] > 0:
                eata_rank = next((i+1 for i, p in enumerate(strategy_avg_performance) 
                                if p['strategy'] == 'eata'), None)
                
                print(f"\n🎯 EATA性能分析:")
                print(f"  排名: {eata_rank}/{len(strategy_avg_performance)}")
                print(f"  平均年化收益: {np.mean(eata_perf['returns']):.2%}")
                print(f"  平均夏普比率: {np.mean(eata_perf['sharpes']):.3f}")
                print(f"  成功实验数: {eata_perf['count']}")
        
        print(f"\n📁 详细结果文件: {self.results_dir}")
        print("💡 使用 experiment_pipeline.py 生成完整的学术报告")
    
    def generate_academic_report(self, results_file: str):
        """生成学术论文级别的报告"""
        
        print("📝 生成学术报告...")
        
        # 使用现有的实验管道生成报告
        pipeline = ExperimentPipeline(str(self.base_dir))
        
        try:
            # 将FLAG-TRADER结果转换为标准格式
            self._convert_results_format(results_file)
            
            # 运行完整的报告生成流程
            df, summary_df = pipeline.run_full_pipeline()
            
            print("✅ 学术报告生成完成")
            return df, summary_df
            
        except Exception as e:
            print(f"❌ 学术报告生成失败: {e}")
            return None, None
    
    def _convert_results_format(self, results_file: str):
        """将FLAG-TRADER结果转换为标准实验管道格式"""
        # 这里需要实现格式转换逻辑
        # 将FLAG-TRADER的JSON结果转换为experiment_pipeline.py期望的格式
        pass


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='FLAG-TRADER风格对比实验')
    parser.add_argument('--experiment_type', 
                       choices=['full', 'finrl_focus', 'llm_focus', 'academic'],
                       default='academic', help='实验类型')
    parser.add_argument('--ticker_set', 
                       choices=['tech_growth', 'finance', 'diverse'],
                       default='diverse', help='股票集合')
    parser.add_argument('--tickers', nargs='+', help='自定义股票列表')
    parser.add_argument('--num_runs', type=int, default=3, help='每个配置运行次数')
    parser.add_argument('--lookback', type=int, default=50, help='回望窗口')
    parser.add_argument('--lookahead', type=int, default=10, help='预测窗口')
    parser.add_argument('--generate_report', action='store_true', help='生成学术报告')
    parser.add_argument('--base_dir', default='/Users/zjt/Desktop/EATA-RL-main', help='项目根目录')
    
    args = parser.parse_args()
    
    # 创建实验运行���
    runner = FlagTraderExperimentRunner(args.base_dir)
    
    # 运行实验
    results = runner.run_experiment_suite(
        experiment_type=args.experiment_type,
        ticker_set=args.ticker_set,
        custom_tickers=args.tickers,
        num_runs=args.num_runs,
        lookback=args.lookback,
        lookahead=args.lookahead
    )
    
    # 生成学术报告
    if args.generate_report:
        results_file = runner.results_dir / f"flag_trader_results_{args.experiment_type}_*.json"
        runner.generate_academic_report(str(results_file))


if __name__ == "__main__":
    main()
