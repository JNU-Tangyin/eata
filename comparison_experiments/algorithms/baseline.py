"""
Baseline策略运行器 - 统一入口
管理所有baseline策略的运行和结果收集
"""

import os
import sys
import warnings

# 在导入任何其他模块之前设置环境变量
# 修复环境变量格式
os.environ['PYTHONWARNINGS'] = 'ignore'

# 设置环境变量来禁用各种库的详细输出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 禁用TensorFlow日志
os.environ['DARTS_LOGGING_LEVEL'] = 'ERROR'  # 设置Darts日志级别
os.environ['PYTORCH_LIGHTNING_LOGGING_LEVEL'] = 'ERROR'  # 禁用PyTorch Lightning详细输出

# 忽略 urllib3 在 LibreSSL 环境下关于 NotOpenSSL 的兼容性提示
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL 1.1.1+")
warnings.filterwarnings("ignore", message=".*urllib3.*OpenSSL.*", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

# 尝试导入并忽略NotOpenSSLWarning
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except ImportError:
    pass

# 禁用PyTorch MPS pin_memory警告
warnings.filterwarnings("ignore", message="'pin_memory' argument is set as true but not supported on MPS now")
warnings.filterwarnings("ignore", message=".*pin_memory.*MPS.*", category=UserWarning)

# 禁用statsmodels收敛警告
warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed to converge")
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
# 禁用statsmodels ConvergenceWarning
try:
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:
    pass
warnings.filterwarnings("ignore", message=".*Maximum Likelihood optimization.*")

# 禁用sklearn FutureWarning
warnings.filterwarnings("ignore", message="`BaseEstimator._validate_data` is deprecated")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# 禁用LightGBM警告
warnings.filterwarnings("ignore", message="Only training set found, disabling early stopping")
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

# 禁用Darts导入信息
warnings.filterwarnings("ignore", message=".*StatsForecast.*could not be imported.*")
warnings.filterwarnings("ignore", message=".*XGBoost.*could not be imported.*")

# 禁用PyTorch Transformer警告
warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*batch_first.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# 禁用PyTorch Lightning详细输出
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")
warnings.filterwarnings("ignore", message=".*GPU available.*")
warnings.filterwarnings("ignore", message=".*TPU available.*")
warnings.filterwarnings("ignore", message=".*HPU available.*")

import pandas as pd
import numpy as np
import importlib
import traceback
from typing import Dict, List, Tuple, Optional

# 策略配置
STRATEGY_CONFIGS = {
    'buy_and_hold': {
        'module': 'buy_and_hold',
        'function': 'run_buy_and_hold',
        'requires_training': False,
        'description': '买入持有策略'
    },
    'macd': {
        'module': 'macd',
        'function': 'run_macd_strategy', 
        'requires_training': False,
        'description': 'MACD交叉策略'
    },
    'arima': {
        'module': 'arima',
        'function': 'run_arima_strategy',
        'requires_training': True,
        'description': 'ARIMA时间序列预测'
    },
    'gp': {
        'module': 'gp',
        'function': 'run_gp_strategy',
        'requires_training': True,
        'description': '遗传编程策略'
    },
    'lightgbm': {
        'module': 'lgb_strategy',
        'function': 'run_lightgbm_strategy',
        'requires_training': True,
        'description': 'LightGBM机器学习策略'
    },
    'lstm': {
        'module': 'lstm',
        'function': 'run_lstm_strategy',
        'requires_training': True,
        'description': 'LSTM神经网络策略'
    },
    'transformer': {
        'module': 'transformer',
        'function': 'run_transformer_strategy',
        'requires_training': True,
        'description': 'Transformer模型策略'
    },
    'ppo': {
        'module': 'ppo',
        'function': 'run_ppo_strategy',
        'requires_training': True,
        'description': 'PPO强化学习策略'
    },
    'eata': {
        'module': 'eata',
        'function': 'run_eata_strategy',
        'requires_training': True,
        'description': 'EATA强化学习策略'
    }
}


class BaselineRunner:
    """Baseline策略运行器"""
    
    def __init__(self):
        self.results = {}
        self.failed_strategies = {}
    
    def run_strategy(self, strategy_name: str, df: pd.DataFrame, 
                    train_df: Optional[pd.DataFrame] = None, 
                    test_df: Optional[pd.DataFrame] = None,
                    ticker: str = 'UNKNOWN') -> Tuple[bool, Optional[pd.Series], Optional[pd.DataFrame]]:
        """
        运行单个策略
        
        Args:
            strategy_name: 策略名称
            df: 完整数据（用于不需要训练的策略）
            train_df: 训练数据
            test_df: 测试数据
            ticker: 股票代码
            
        Returns:
            tuple: (success, metrics, backtest_results)
        """
        if strategy_name not in STRATEGY_CONFIGS:
            print(f"❌ 未知策略: {strategy_name}")
            return False, None, None
        
        config = STRATEGY_CONFIGS[strategy_name]
        
        try:
            # 动态导入策略模块 - 修复相对导入问题
            try:
                # 首先尝试相对导入
                module = importlib.import_module(f".{config['module']}", package=__package__)
            except (TypeError, ImportError):
                # 如果相对导入失败，尝试绝对导入
                try:
                    module = importlib.import_module(config['module'])
                except ImportError:
                    # 如果还是失败，尝试从当前目录导入
                    import sys
                    import os
                    current_dir = os.path.dirname(__file__)
                    if current_dir not in sys.path:
                        sys.path.insert(0, current_dir)
                    module = importlib.import_module(config['module'])
            
            strategy_func = getattr(module, config['function'])
            
            print(f"\n🚀 运行策略: {config['description']}")
            
            # 根据策略类型调用不同参数
            if config['requires_training']:
                if train_df is None or test_df is None:
                    raise ValueError(f"策略 {strategy_name} 需要训练数据和测试数据")
                metrics, backtest_results = strategy_func(train_df, test_df, ticker)
            else:
                if df is None:
                    raise ValueError(f"策略 {strategy_name} 需要完整数据")
                metrics, backtest_results = strategy_func(df)
            
            return True, metrics, backtest_results
            
        except Exception as e:
            error_msg = f"策略 {strategy_name} 运行失败: {str(e)}"
            print(f"❌ {error_msg}")
            print(f"   详细错误: {traceback.format_exc()}")
            
            self.failed_strategies[strategy_name] = {
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            
            return False, None, None
    
    def run_all_strategies(self, df: pd.DataFrame, ticker: str = 'TEST',
                          train_ratio: float = 0.7,  # 改为70%训练，30%测试，给EATA更多测试数据
                          selected_strategies: Optional[List[str]] = None) -> Dict:
        """
        运行所有策略
        
        Args:
            df: 完整数据
            ticker: 股票代码
            train_ratio: 训练集比例
            selected_strategies: 选择的策略列表，None表示运行所有策略
            
        Returns:
            dict: 所有策略的结果
        """
        print(f"🎯 开始运行Baseline策略对比 - 股票: {ticker}")
        print(f"📊 数据量: {len(df)} 条记录")
        
        # 分割数据
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        print(f"📈 训练集: {len(train_df)} 条记录")
        print(f"📉 测试集: {len(test_df)} 条记录")
        
        # 确定要运行的策略
        strategies_to_run = selected_strategies if selected_strategies else list(STRATEGY_CONFIGS.keys())
        print(f"🎲 将运行 {len(strategies_to_run)} 个策略: {strategies_to_run}")
        
        results = {}
        
        for strategy_name in strategies_to_run:
            if strategy_name not in STRATEGY_CONFIGS:
                print(f"⚠️ 跳过未知策略: {strategy_name}")
                continue
                
            config = STRATEGY_CONFIGS[strategy_name]
            success, metrics, backtest_results = self.run_strategy(
                strategy_name=strategy_name,
                df=df if not config['requires_training'] else None,
                train_df=train_df if config['requires_training'] else None,
                test_df=test_df if config['requires_training'] else None,
                ticker=ticker
            )
            
            if success:
                results[strategy_name] = {
                    'metrics': metrics,
                    'backtest_results': backtest_results,
                    'description': config['description'],
                    'success': True
                }
            else:
                results[strategy_name] = {
                    'metrics': None,
                    'backtest_results': None,
                    'description': config['description'],
                    'success': False
                }
        
        self.results[ticker] = results
        return results
    
    def generate_comparison_report(self, results: Dict) -> str:
        """生成对比报告"""
        report = []
        report.append("🏆 Baseline策略对比报告")
        report.append("=" * 80)
        
        successful_strategies = {k: v for k, v in results.items() if v['success']}
        failed_strategies = {k: v for k, v in results.items() if not v['success']}
        
        if successful_strategies:
            report.append(f"\n📊 成功策略 ({len(successful_strategies)}/{len(results)}):")
            report.append("-" * 80)
            report.append(f"{'策略':<15} {'年化收益':<12} {'夏普比率':<10} {'最大回撤':<10} {'总收益':<10}")
            report.append("-" * 80)
            
            # 按年化收益排序
            sorted_strategies = sorted(
                successful_strategies.items(),
                key=lambda x: x[1]['metrics'].get('annualized_return', 0),
                reverse=True
            )
            
            for strategy_name, result in sorted_strategies:
                metrics = result['metrics']
                annual_return = metrics.get('annualized_return', 0)
                sharpe_ratio = metrics.get('sharpe_ratio', 0)
                max_drawdown = metrics.get('max_drawdown', 0)
                total_return = metrics.get('total_return', 0)
                
                report.append(
                    f"{strategy_name:<15} {annual_return:>10.2%} {sharpe_ratio:>9.2f} "
                    f"{max_drawdown:>9.2%} {total_return:>9.2%}"
                )
        
        if failed_strategies:
            report.append(f"\n❌ 失败策略 ({len(failed_strategies)}):")
            for strategy_name, result in failed_strategies.items():
                report.append(f"   - {strategy_name}: {result['description']}")
        
        return "\n".join(report)
    
    def save_results(self, results: Dict, ticker: str, output_dir: str = "comparison_results"):
        """保存结果到文件"""
        import os
        import json
        from datetime import datetime
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存JSON结果
        json_file = os.path.join(output_dir, f"baseline_results_{ticker}_{timestamp}.json")
        
        # 转换为可序列化的格式
        serializable_results = {}
        for strategy_name, result in results.items():
            if result['success'] and result['metrics'] is not None:
                serializable_results[strategy_name] = {
                    'metrics': result['metrics'].to_dict(),
                    'description': result['description'],
                    'success': result['success']
                }
            else:
                serializable_results[strategy_name] = {
                    'metrics': None,
                    'description': result['description'],
                    'success': result['success']
                }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        # 保存文本报告
        report_file = os.path.join(output_dir, f"baseline_report_{ticker}_{timestamp}.txt")
        report = self.generate_comparison_report(results)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n💾 结果已保存:")
        print(f"   JSON: {json_file}")
        print(f"   报告: {report_file}")


def run_real_data_experiment(ticker: str, selected_strategies=None):
    """使用真实股票数据运行baseline策略对比"""
    from data_utils import load_real_stock_data, add_technical_indicators
    
    # 加载真实股票数据
    print(f"📊 加载真实股票数据: {ticker}")
    df = load_real_stock_data(ticker)
    print(f"✅ 数据加载完成: {len(df)} 条记录，时间范围: {df['date'].min()} 到 {df['date'].max()}")
    
    # 添加技术指标
    df = add_technical_indicators(df)
    print(f"✅ 技术指标计算完成: {len(df.columns)} 列")
    
    # 运行策略
    runner = BaselineRunner()
    results = runner.run_all_strategies(df, ticker=ticker, selected_strategies=selected_strategies)
    
    # 生成报告
    report = runner.generate_comparison_report(results)
    print(f"\n{report}")
    
    # 保存结果
    runner.save_results(results, ticker)
    
    return results




def get_available_tickers():
    """获取数据库中可用的股票列表"""
    import sqlite3
    import os
    from pathlib import Path
    
    # 构建数据库路径
    project_root = Path(__file__).resolve().parents[2]
    db_path = project_root / "stock.db"
    
    if not os.path.exists(db_path):
        print(f"❌ 数据库文件不存在: {db_path}")
        return []
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT code FROM downloaded ORDER BY code")
    tickers = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    return tickers


def main():
    """主函数 - 支持命令行参数和真实数据"""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='运行Baseline策略对比实验')
    parser.add_argument('ticker', nargs='?', default='AAPL', 
                       help='股票代码 (默认: AAPL)')
    parser.add_argument('--strategies', type=str, 
                       help='指定运行的策略，用逗号分隔 (例如: arima,ppo,macd)')
    parser.add_argument('--list-tickers', action='store_true',
                       help='列出所有可用的股票代码')
    
    # 如果没有命令行参数，使用默认多股票模式
    if len(sys.argv) == 1:
        print(f"🎯 开始运行Baseline策略对比实验（多股票默认模式）")
        print(f"💡 提示: 使用 python baseline.py --help 查看更多选项")
        # 设置默认参数并继续执行多股票逻辑
        class DefaultArgs:
            ticker = 'AAPL'
            strategies = None
            list_tickers = False
        args = DefaultArgs()
    else:
        args = parser.parse_args()
    
    # 列出可用股票
    if args.list_tickers:
        tickers = get_available_tickers()
        print("📊 数据库中可用的股票代码:")
        for i, ticker in enumerate(tickers, 1):
            print(f"  {i:2d}. {ticker}")
        print(f"\n总计: {len(tickers)} 支股票")
        return
    
    # 验证股票代码
    available_tickers = get_available_tickers()
    if args.ticker not in available_tickers:
        print(f"❌ 股票代码 {args.ticker} 不在数据库中")
        print(f"📊 可用股票: {', '.join(available_tickers[:10])}...")
        print("💡 使用 --list-tickers 查看所有可用股票")
        return
    
    # 解析策略列表
    selected_strategies = None
    if args.strategies:
        selected_strategies = [s.strip() for s in args.strategies.split(',')]
        print(f"🎯 将运行指定策略: {selected_strategies}")
    
    # 运行真实数据实验 - 支持多个股票
    print(f"🚀 开始运行Baseline策略对比实验")
    
    # 默认总是测试多个股票，这里上限设置为 100 支
    available_tickers = get_available_tickers()
    
    # 优选的股票组合（优先考虑这几只）
    preferred_tickers = ['AAPL', 'MSFT', 'GOOGL']
    test_tickers = []
    
    # 添加用户指定的股票（如果不在优选列表中）
    if args.ticker not in preferred_tickers:
        test_tickers.append(args.ticker)
    
    # 添加优选股票（如果可用），最多 100 支
    for ticker in preferred_tickers:
        if ticker in available_tickers and ticker not in test_tickers:
            test_tickers.append(ticker)
            if len(test_tickers) >= 100:  # 最多 100 个股票
                break
    
    # 如果还不够 100 个，从可用股票中补充
    if len(test_tickers) < 100:
        for ticker in available_tickers:
            if ticker not in test_tickers:
                test_tickers.append(ticker)
                if len(test_tickers) >= 100:
                    break
    
    print(f"📈 将测试股票: {test_tickers}")
    
    all_results = {}
    
    try:
        for ticker in test_tickers:
            print(f"\n{'='*60}")
            print(f"🎯 正在测试股票: {ticker}")
            print(f"{'='*60}")
            
            results = run_real_data_experiment(ticker, selected_strategies)
            all_results[ticker] = results
            
            print(f"✅ {ticker} 测试完成")
        
        # 汇总输出所有结果
        print(f"\n{'='*80}")
        print(f"🏆 多股票策略对比汇总")
        print(f"{'='*80}")
        
        # 获取所有策略名称
        all_strategies = set()
        for results in all_results.values():
            if results:
                all_strategies.update(results.keys())
        
        # 按股票显示结果
        for ticker, results in all_results.items():
            print(f"\n📊 {ticker} 股票结果:")
            print("-" * 60)
            if results:
                # 按年化收益排序显示策略
                strategy_performance = []
                for strategy_name, result in results.items():
                    if result and 'metrics' in result:
                        ann_return = result['metrics']['annualized_return']
                        strategy_performance.append((strategy_name, ann_return, result))
                
                strategy_performance.sort(key=lambda x: x[1], reverse=True)
                
                for strategy_name, ann_return, result in strategy_performance:
                    metrics = result['metrics']
                    print(f"   {strategy_name:12s}: {metrics['annualized_return']:8.2%} "
                          f"(夏普: {metrics['sharpe_ratio']:5.2f}, "
                          f"回撤: {metrics['max_drawdown']:6.2%})")
            else:
                print("   ❌ 无有效结果")
        
        # 策略横向对比
        print(f"\n{'='*80}")
        print(f"📈 策略横向对比 (按平均年化收益排序)")
        print(f"{'='*80}")
        
        strategy_summary = {}
        for ticker, results in all_results.items():
            if results:
                for strategy_name, result in results.items():
                    if result and 'metrics' in result:
                        if strategy_name not in strategy_summary:
                            strategy_summary[strategy_name] = []
                        strategy_summary[strategy_name].append(result['metrics']['annualized_return'])
        
        # 计算平均表现并排序
        strategy_avg = []
        for strategy, returns in strategy_summary.items():
            avg_return = sum(returns) / len(returns)
            strategy_avg.append((strategy, avg_return, len(returns)))
        
        strategy_avg.sort(key=lambda x: x[1], reverse=True)
        
        print(f"{'策略':12s} {'平均年化收益':>12s} {'测试股票数':>8s}")
        print("-" * 40)
        for strategy, avg_return, count in strategy_avg:
            print(f"{strategy:12s} {avg_return:11.2%} {count:7d}")
        
        print(f"\n🎉 多股票实验完成！结果已保存到 comparison_results/ 目录")
        return all_results
        
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    main()
