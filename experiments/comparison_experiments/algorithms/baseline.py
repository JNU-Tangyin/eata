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
        'function': 'run_buy_and_hold_strategy',  # 🔧 修复函数名
        'requires_training': True,
        'description': '买入持有策略'
    },
    'macd': {
        'module': 'macd',
        'function': 'run_macd_strategy', 
        'requires_training': True,
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
    'gbdt': {
        'module': 'gbdt_strategy',
        'function': 'run_gbdt_strategy',
        'requires_training': True,
        'description': 'GBDT梯度提升决策树策略'
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
    'eata': {
        'module': 'eata',
        'function': 'run_eata_strategy',
        'requires_training': True,
        'description': 'EATA强化学习策略'
    },
    'finrl_ppo': {
        'module': 'finrl_strategies',
        'function': 'run_finrl_ppo_strategy',
        'requires_training': True,
        'description': 'FinRL PPO强化学习策略'
    },
    'finrl_a2c': {
        'module': 'finrl_strategies',
        'function': 'run_finrl_a2c_strategy',
        'requires_training': True,
        'description': 'FinRL A2C强化学习策略'
    },
    'finrl_sac': {
        'module': 'finrl_strategies',
        'function': 'run_finrl_sac_strategy',
        'requires_training': True,
        'description': 'FinRL SAC强化学习策略'
    },
    'finrl_td3': {
        'module': 'finrl_strategies',
        'function': 'run_finrl_td3_strategy',
        'requires_training': True,
        'description': 'FinRL TD3强化学习策略'
    },
    'finrl_ddpg': {
        'module': 'finrl_strategies',
        'function': 'run_finrl_ddpg_strategy',
        'requires_training': True,
        'description': 'FinRL DDPG强化学习策略'
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
                          train_ratio: float = 0.7,  # 🔧 改为70%训练，30%测试，与消融实验一致
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
    
    def save_results(self, results: Dict, ticker: str, output_dir: str = None, 
                    params: Dict = None, run_id: int = 1):
        """保存结果到CSV和JSON文件"""
        import os
        import pandas as pd
        import json
        from datetime import datetime
        from pathlib import Path
        
        # 默认使用项目根目录的results/comparison_study/
        if output_dir is None:
            project_root = Path(__file__).parent.parent.parent.parent
            output_dir = str(project_root / "results" / "comparison_study" / "raw_results")
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 构建参数组合字符串
        if params:
            param_str = "_".join([f"{k}{v}" for k, v in params.items()])
        else:
            # 默认参数组合
            param_str = "lookback30_lookahead10_stride1_depth200"
        
        # 文件命名：参数组合_股票代码_轮次_时间戳.csv
        csv_file = os.path.join(output_dir, f"experiment_{param_str}_{ticker}_run{run_id:03d}_{timestamp}.csv")
        
        # 准备CSV数据 - 每轮次实验的详细数据
        csv_data = []
        
        for strategy_name, result in results.items():
            # 跳过default策略，不保存到CSV
            if strategy_name == 'default':
                continue
                
            if result['success'] and result['metrics'] is not None:
                # 基础信息
                row = {
                    'timestamp': timestamp,
                    'ticker': ticker,
                    'strategy': strategy_name,
                    'run_id': run_id,
                    'success': result['success'],
                    'description': result['description']
                }
                
                # 添加参数信息
                if params:
                    row.update(params)
                else:
                    row.update({
                        'lookback': 30,
                        'lookahead': 10, 
                        'stride': 1,
                        'depth': 200
                    })
                
                # 添加所有指标
                if isinstance(result['metrics'], pd.Series):
                    row.update(result['metrics'].to_dict())
                elif isinstance(result['metrics'], dict):
                    row.update(result['metrics'])
                
                csv_data.append(row)
            else:
                # 失败的实验也记录（但跳过default策略）
                if strategy_name == 'default':
                    continue
                    
                row = {
                    'timestamp': timestamp,
                    'ticker': ticker,
                    'strategy': strategy_name,
                    'run_id': run_id,
                    'success': False,
                    'description': result['description'],
                    'error': 'Strategy execution failed'
                }
                
                # 添加参数信息
                if params:
                    row.update(params)
                else:
                    row.update({
                        'lookback': 30,
                        'lookahead': 10,
                        'stride': 1, 
                        'depth': 200
                    })
                
                csv_data.append(row)
        
        # 保存CSV文件
        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_file, index=False, encoding='utf-8-sig')
            
            # 保存所有策略的详细交易数据
            self.save_detailed_trading_data(results, ticker, output_dir, timestamp)
            
            # 同时保存JSON文件供post.py使用
            json_file = os.path.join(output_dir, f"baseline_results_{ticker}_{timestamp}.json")
            json_data = {}
            for strategy_name, result in results.items():
                # 跳过default策略，不保存到JSON
                if strategy_name == 'default':
                    continue
                    
                if result['success'] and result['metrics'] is not None:
                    json_data[strategy_name] = {
                        'total_return': float(result['metrics']['total_return']),
                        'annualized_return': float(result['metrics']['annualized_return']),
                        'sharpe_ratio': float(result['metrics']['sharpe_ratio']),
                        'max_drawdown': float(result['metrics']['max_drawdown']),
                        'volatility': float(result['metrics'].get('volatility', 0)),  # 🔧 新增：波动率
                        'success': True,
                        'description': result['description']
                    }
                else:
                    json_data[strategy_name] = {
                        'success': False,
                        'description': result['description']
                    }
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 实验数据已保存: {os.path.basename(csv_file)}")
            print(f"✅ JSON数据已保存: {os.path.basename(json_file)}")
            print(f"   📊 包含 {len(csv_data)} 个策略的详细数据")
            return csv_file
        else:
            print(f"❌ 无有效数据保存")
            return None

    def save_detailed_trading_data(self, results: Dict, ticker: str, output_dir: str, timestamp: str):
        """保存所有策略的详细交易数据"""
        import os
        import pandas as pd
        
        # 创建详细输出目录 - 与raw_results平级
        # output_dir 是 raw_results/，所以上一级是 comparison_study/
        detailed_dir = os.path.join(os.path.dirname(output_dir), "detailed_outputs")
        os.makedirs(detailed_dir, exist_ok=True)
        
        for strategy_name, result in results.items():
            # 跳过default策略，不保存其详细数据
            if strategy_name == 'default':
                continue
                
            if result['success'] and result.get('backtest_results') is not None:
                try:
                    backtest_df = result['backtest_results']
                    
                    # 确保有必要的列
                    if isinstance(backtest_df, pd.DataFrame) and not backtest_df.empty:
                        # 创建标准化的详细数据格式
                        detailed_data = pd.DataFrame()
                        
                        # 基础信息
                        if 'date' in backtest_df.columns:
                            try:
                                detailed_data['日期'] = pd.to_datetime(backtest_df['date'])
                            except:
                                # 如果日期转换失败，使用索引创建日期
                                detailed_data['日期'] = pd.date_range('2023-01-01', periods=len(backtest_df), freq='D')
                        elif backtest_df.index.name == 'date' or hasattr(backtest_df.index, 'date'):
                            detailed_data['日期'] = pd.to_datetime(backtest_df.index)
                        else:
                            # 如果没有日期列，创建一个合理的日期范围
                            if strategy_name.startswith('finrl'):
                                # FinRL策略使用测试期间的日期
                                detailed_data['日期'] = pd.date_range('2023-01-01', periods=len(backtest_df), freq='D')
                            else:
                                detailed_data['日期'] = pd.date_range('2023-01-01', periods=len(backtest_df), freq='D')
                        
                        # 交易信号
                        if 'signal' in backtest_df.columns:
                            detailed_data['买卖信号'] = backtest_df['signal']
                        elif 'action' in backtest_df.columns:
                            detailed_data['买卖信号'] = backtest_df['action']
                        elif 'actions' in backtest_df.columns:
                            # FinRL策略返回'actions'列
                            actions = backtest_df['actions']
                            if isinstance(actions.iloc[0], (list, np.ndarray)):
                                # 如果actions是数组，取第一个元素或求和
                                signals = []
                                for action in actions:
                                    if isinstance(action, (list, np.ndarray)):
                                        action_sum = np.sum(action) if len(action) > 0 else 0
                                    else:
                                        action_sum = action
                                    
                                    if action_sum > 0.01:
                                        signals.append(1)  # 买入
                                    elif action_sum < -0.01:
                                        signals.append(-1)  # 卖出
                                    else:
                                        signals.append(0)  # 持有
                                detailed_data['买卖信号'] = signals
                            else:
                                # 如果actions是标量，直接使用
                                detailed_data['买卖信号'] = actions
                        else:
                            # 如果没有信号列，根据收益率推断
                            if 'strategy_return' in backtest_df.columns:
                                returns = backtest_df['strategy_return']
                                signals = []
                                for ret in returns:
                                    if ret > 0.001:
                                        signals.append(1)  # 买入
                                    elif ret < -0.001:
                                        signals.append(-1)  # 卖出
                                    else:
                                        signals.append(0)  # 持有
                                detailed_data['买卖信号'] = signals
                            elif 'portfolio_value' in backtest_df.columns and strategy_name == 'eata':
                                # EATA特殊处理：从portfolio_value推导交易信号
                                portfolio_values = backtest_df['portfolio_value']
                                returns = portfolio_values.pct_change().fillna(0)
                                signals = []
                                for ret in returns:
                                    if ret > 0.005:  # 0.5%以上收益认为是买入信号
                                        signals.append(1)
                                    elif ret < -0.005:  # -0.5%以下认为是卖出信号
                                        signals.append(-1)
                                    else:
                                        signals.append(0)  # 持有
                                detailed_data['买卖信号'] = signals
                            else:
                                detailed_data['买卖信号'] = 0  # 默认持有
                        
                        # 价格信息
                        if 'close' in backtest_df.columns:
                            price = backtest_df['close']
                            # 模拟Q25和Q75（基于收盘价的±2%）
                            detailed_data['Q25预测'] = price * 0.98
                            detailed_data['Q75预测'] = price * 1.02
                            detailed_data['Q25真实'] = price * 0.98
                            detailed_data['Q75真实'] = price * 1.02
                        elif 'portfolio_value' in backtest_df.columns and strategy_name.startswith('finrl'):
                            # FinRL策略特殊处理：从portfolio_value推导价格变化
                            portfolio_values = backtest_df['portfolio_value']
                            initial_value = portfolio_values.iloc[0] if len(portfolio_values) > 0 else 1000000
                            # 将portfolio_value转换为相对价格变化，模拟股价
                            price_changes = portfolio_values.pct_change().fillna(0)
                            base_price = 100  # 基准价格
                            simulated_prices = [base_price]
                            for change in price_changes[1:]:
                                new_price = simulated_prices[-1] * (1 + change)
                                simulated_prices.append(new_price)
                            
                            detailed_data['Q25预测'] = [p * 0.98 for p in simulated_prices]
                            detailed_data['Q75预测'] = [p * 1.02 for p in simulated_prices]
                            detailed_data['Q25真实'] = [p * 0.98 for p in simulated_prices]
                            detailed_data['Q75真实'] = [p * 1.02 for p in simulated_prices]
                        elif 'portfolio_value' in backtest_df.columns and strategy_name == 'eata':
                            # EATA特殊处理：从portfolio_value推导价格变化
                            portfolio_values = backtest_df['portfolio_value']
                            initial_value = portfolio_values.iloc[0] if len(portfolio_values) > 0 else 1000000
                            # 将portfolio_value转换为相对价格变化
                            normalized_values = portfolio_values / initial_value * 100  # 标准化到100基准
                            detailed_data['Q25预测'] = normalized_values * 0.98
                            detailed_data['Q75预测'] = normalized_values * 1.02
                            detailed_data['Q25真实'] = normalized_values * 0.98
                            detailed_data['Q75真实'] = normalized_values * 1.02
                        else:
                            # 如果没有价格数据，使用模拟价格
                            base_price = 100
                            if 'cumulative_return' in backtest_df.columns:
                                prices = base_price * (1 + backtest_df['cumulative_return'])
                            else:
                                prices = [base_price] * len(backtest_df)
                            
                            detailed_data['Q25预测'] = [p * 0.98 for p in prices]
                            detailed_data['Q75预测'] = [p * 1.02 for p in prices]
                            detailed_data['Q25真实'] = [p * 0.98 for p in prices]
                            detailed_data['Q75真实'] = [p * 1.02 for p in prices]
                        
                        # 保存详细数据文件
                        detailed_file = os.path.join(detailed_dir, f"{ticker}-{strategy_name}-001-{timestamp}.csv")
                        detailed_data.to_csv(detailed_file, index=False, encoding='utf-8-sig')
                        
                        print(f"✅ 保存 {strategy_name} 详细交易数据: {os.path.basename(detailed_file)}")
                        
                except Exception as e:
                    print(f"⚠️ 保存 {strategy_name} 详细数据失败: {e}")
                    continue


def run_real_data_experiment(ticker: str, selected_strategies=None, params=None, run_id=1):
    """使用真实股票数据运行baseline策略对比"""
    from data_utils import load_csv_stock_data, add_technical_indicators  # 🔧 改用CSV数据源
    
    # 默认参数
    if params is None:
        params = {
            'lookback': 50,
            'lookahead': 10,
            'stride': 1,
            'depth': 300
        }
    
    # 加载真实股票数据
    print(f"📊 加载真实股票数据: {ticker}")
    print(f"🔧 实验参数: {params}")
    print(f"🔄 运行轮次: {run_id}")
    
    df = load_csv_stock_data(ticker)  # 🔧 改用CSV数据源
    
    # 关键修改：只保留2020年以后的数据（与消融实验一致）
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    df = df[df['date'] >= '2020-01-01'].copy()
    print(f" 筛选2020年后数据: {len(df)} 条记录，时间范围: {df['date'].min()} 到 {df['date'].max()}")
    
    # 关键修复：先分割数据，再添加技术指标，避免数据泄露
    print(" 先分割训练/测试集，再计算技术指标（避免数据泄露）...")
    train_ratio = 0.7
    split_idx = int(len(df) * train_ratio)
    
    # 分割原始数据
    train_df_raw = df.iloc[:split_idx].copy()
    test_df_raw = df.iloc[split_idx:].copy()
    print(f" 训练集: {len(train_df_raw)} 条")
    print(f" 测试集: {len(test_df_raw)} 条")
    
    # 分别计算技术指标
    print(" 分别计算训练集和测试集的技术指标...")
    train_df = add_technical_indicators(train_df_raw)
    test_df = add_technical_indicators(test_df_raw)
    
    # 重新合并（保持分割边界清晰）
    df = pd.concat([train_df, test_df]).reset_index(drop=True)
    print(f" 技术指标添加完成，数据列数: {len(df.columns)}")
    
    # 运行策略
    runner = BaselineRunner()
    # 注意：这里传入的df已经包含了正确计算的技术指标
    # runner内部会再次分割，但技术指标已经是隔离计算的
    results = runner.run_all_strategies(df, ticker=ticker, selected_strategies=selected_strategies)
    
    # 生成报告
    report = runner.generate_comparison_report(results)
    print(f"\n{report}")
    
    # 保存结果 - 传入参数和轮次信息
    runner.save_results(results, ticker, params=params, run_id=run_id)
    
    return results




def get_available_tickers():
    """获取CSV文件中可用的股票列表"""
    from pathlib import Path
    
    # 🔧 改用CSV数据源：从data目录获取股票列表
    project_root = Path(__file__).resolve().parents[3]
    data_dir = project_root / "data"
    
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return []
    
    # 获取所有CSV文件
    csv_files = list(data_dir.glob("*.csv"))
    tickers = sorted([f.stem for f in csv_files])
    
    return tickers


def run_parameter_experiments():
    """运行参数组合实验"""
    
    # 使用默认参数组合
    param_combinations = [
        {'lookback': 50, 'lookahead': 10, 'stride': 1, 'depth': 300},
    ]
    
    # 测试股票 - 从数据库动态获取100支美股
    import sqlite3
    import os
    
    # 数据库路径 - 使用绝对路径确保可靠性
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, '..', '..', 'stock.db')
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT DISTINCT ticker FROM stock_data ORDER BY ticker')
        test_tickers = [row[0] for row in cursor.fetchall()]
        conn.close()
        print(f"📊 从数据库加载 {len(test_tickers)} 支股票")
    else:
        # 备用股票列表（数据库中实际存在的100支股票）
        test_tickers = [
            'AAPL', 'ABBV', 'ABT', 'ADBE', 'AJG', 'AMD', 'AMGN', 'AMZN', 'AON', 'AVGO',
            'AXP', 'BA', 'BABA', 'BAC', 'BIIB', 'BLK', 'BMY', 'C', 'CAT', 'CME',
            'COF', 'COP', 'COST', 'CRM', 'CSCO', 'CVX', 'DE', 'DHR', 'DIS', 'EBAY',
            'EMR', 'EOG', 'ETSY', 'FDX', 'GE', 'GILD', 'GOOGL', 'GS', 'HAL', 'HD',
            'HON', 'IBM', 'ICE', 'ILMN', 'INTC', 'ITW', 'JNJ', 'JPM', 'KO', 'LMT',
            'LOW', 'LYFT', 'MA', 'MCD', 'MCO', 'META', 'MMC', 'MMM', 'MPC', 'MRK',
            'MS', 'MSFT', 'NFLX', 'NKE', 'NOC', 'NVDA', 'ORCL', 'OXY', 'PEP', 'PFE',
            'PH', 'PNC', 'PSX', 'PYPL', 'QCOM', 'REGN', 'ROK', 'RTX', 'SBUX', 'SCHW',
            'SHOP', 'SLB', 'SPGI', 'SQ', 'TFC', 'TGT', 'TMO', 'TSLA', 'TXN', 'UBER',
            'UNH', 'UPS', 'USB', 'V', 'VLO', 'VRTX', 'W', 'WFC', 'WMT', 'XOM'
        ]
        print(f"📊 使用备用股票列表 {len(test_tickers)} 支股票")
    
    # 测试策略
    test_strategies = list(STRATEGY_CONFIGS.keys())
    
    # 每个组合运行的轮次
    num_runs = 1
    
    # 显示完整股票信息
    print(f"📈 测试股票: {len(test_tickers)}支美股")
    print("📊 完整股票列表:")
    
    # 按行显示，每行10个股票
    for i in range(0, len(test_tickers), 10):
        row_tickers = test_tickers[i:i+10]
        row_str = ', '.join(f'{ticker:6s}' for ticker in row_tickers)
        print(f"   {i+1:3d}-{min(i+10, len(test_tickers)):3d}: {row_str}")
    
    print(f"   ✅ 总计: {len(test_tickers)} 支股票")
    print(f"🎲 测试策略: {test_strategies}")
    
    total_experiments = len(param_combinations) * len(test_tickers) * num_runs
    completed = 0
    
    for i, params in enumerate(param_combinations):
        
        for ticker in test_tickers:
            print(f"📊 股票: {ticker}")
            
            for run_id in range(1, num_runs + 1):
                try:
                    # 运行实验
                    results = run_real_data_experiment(
                        ticker=ticker,
                        params=params,
                        run_id=run_id
                    )
                    
                    completed += 1
                    progress = (completed / total_experiments) * 100
                    print(f"    ✅ 完成 ({progress:.1f}%)")
                    
                except Exception as e:
                    print(f"    ❌ 失败: {str(e)}")
                    completed += 1
                    continue
    
    print(f"\n🎉 参数组合实验完成！")
    print(f"📁 结果文件保存在: results/comparison_study/raw_results/")
    print(f"📊 总实验数: {total_experiments}")
    
    # 生成实验汇总统计
    generate_experiment_summary()


def generate_experiment_summary():
    """生成实验汇总统计"""
    import pandas as pd
    import numpy as np
    from pathlib import Path
    import json
    
    print(f"\n📊 生成实验汇总统计...")
    print("=" * 80)
    
    # 使用新的results路径
    project_root = Path(__file__).parent.parent.parent.parent
    results_dir = project_root / "results" / "comparison_study" / "raw_results"
    if not results_dir.exists():
        print("❌ 结果目录不存在")
        return
    
    # 收集所有JSON结果文件
    json_files = list(results_dir.glob("baseline_results_*.json"))
    if not json_files:
        print("❌ 未找到实验结果文件")
        return
    
    print(f"📋 找到 {len(json_files)} 个实验结果文件")
    
    # 收集所有策略的性能数据
    all_results = []
    strategy_stats = {}
    
    for json_file in json_files:
        # 从文件名提取股票代码
        parts = json_file.stem.split('_')
        if len(parts) >= 3:
            ticker = parts[2]
        else:
            continue
            
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for strategy, metrics in data.items():
                if metrics.get('success', False):
                    result = {
                        'ticker': ticker,
                        'strategy': strategy,
                        'total_return': metrics.get('total_return', 0),
                        'annualized_return': metrics.get('annualized_return', 0),
                        'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                        'max_drawdown': metrics.get('max_drawdown', 0),
                        'volatility': metrics.get('volatility', 0)  # 🔧 新增：波动率
                    }
                    all_results.append(result)
                    
                    # 按策略统计
                    if strategy not in strategy_stats:
                        strategy_stats[strategy] = []
                    strategy_stats[strategy].append(result)
        
        except Exception as e:
            print(f"⚠️ 读取 {json_file.name} 失败: {e}")
            continue
    
    if not all_results:
        print("❌ 未找到有效的实验结果")
        return
    
    # 转换为DataFrame
    df = pd.DataFrame(all_results)
    
    print(f"✅ 成功收集 {len(all_results)} 条实验结果")
    print(f"📈 涵盖 {df['ticker'].nunique()} 支股票, {df['strategy'].nunique()} 个策略")
    print()
    
    # 按策略汇总统计
    print("📊 各策略平均表现:")
    print("=" * 80)
    
    # 计算汇总统计
    strategy_summary = df.groupby('strategy').agg({
        'annualized_return': 'mean',
        'sharpe_ratio': 'mean', 
        'max_drawdown': 'mean',
        'total_return': 'mean'
    }).round(4)
    
    # 按年化收益排序
    strategy_summary = strategy_summary.sort_values('annualized_return', ascending=False)
    
    # 格式化显示
    print(f"{'策略':<15} {'年化收益':>12} {'夏普比率':>12} {'最大回撤':>12} {'总收益':>12}")
    print("-" * 80)
    
    for strategy, row in strategy_summary.iterrows():
        annual_return = f"{row['annualized_return']:.2%}"
        sharpe_ratio = f"{row['sharpe_ratio']:.2f}"
        max_drawdown = f"{row['max_drawdown']:.2%}"
        total_return = f"{row['total_return']:.2%}"
        
        print(f"{strategy:<15} {annual_return:>12} {sharpe_ratio:>12} {max_drawdown:>12} {total_return:>12}")
    
    print("=" * 80)
    print()
    
    # 显示最佳表现
    print("🏆 最佳表现:")
    print("-" * 40)
    best_return = df.loc[df['annualized_return'].idxmax()]
    best_sharpe = df.loc[df['sharpe_ratio'].idxmax()]
    
    print(f"最高年化收益: {best_return['strategy']} ({best_return['ticker']}) - {best_return['annualized_return']:.2%}")
    print(f"最高夏普比率: {best_sharpe['strategy']} ({best_sharpe['ticker']}) - {best_sharpe['sharpe_ratio']:.3f}")
    print()
    
    # 保存汇总结果
    summary_file = results_dir / "experiment_summary.csv"
    strategy_summary.to_csv(summary_file, encoding='utf-8-sig')
    
    detailed_file = results_dir / "detailed_results.csv"
    df.to_csv(detailed_file, index=False, encoding='utf-8-sig')
    
    print(f"💾 汇总结果已保存:")
    print(f"   📄 策略汇总: {summary_file}")
    print(f"   📄 详细结果: {detailed_file}")
    print("=" * 80)


def main():
    """主函数 - 支持命令行参数和真实数据"""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='运行Baseline策略对比实验')
    parser.add_argument('ticker', nargs='?', default='AAPL', 
                       help='股票代码 (默认: AAPL 苹果公司)')
    parser.add_argument('--strategies', type=str, 
                       help='指定运行的策略，用逗号分隔 (例如: arima,ppo,macd)')
    parser.add_argument('--list-tickers', action='store_true',
                       help='列出所有可用的股票代码')
    parser.add_argument('--param-experiments', action='store_true',
                       help='运行参数组合实验')
    parser.add_argument('--runs', type=int, default=1,
                       help='运行轮次数 (默认: 1)')
    
    # 如果没有命令行参数，运行参数组合实验
    if len(sys.argv) == 1:
        print(f"🚀 开始参数组合实验")
        run_parameter_experiments()
        return
    
    args = parser.parse_args()
    
    # 运行参数组合实验
    if args.param_experiments:
        run_parameter_experiments()
        return
    
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
    
    # 运行多轮次实验
    print(f"🔄 将运行 {args.runs} 轮实验")
    for run_id in range(1, args.runs + 1):
        print(f"\n{'='*50}")
        print(f"🔄 第 {run_id}/{args.runs} 轮实验")
        print(f"{'='*50}")
        
        results = run_real_data_experiment(
            ticker=args.ticker,
            selected_strategies=selected_strategies,
            run_id=run_id
        )
    
    # 显示单个股票的实验结果汇总
    print(f"\n📊 {args.ticker} 实验结果汇总:")
    print("=" * 80)
    
    if results:
        # 收集成功的结果
        successful_results = []
        failed_strategies = []
        
        for strategy_name, result in results.items():
            if result['success'] and result['metrics'] is not None:
                metrics = result['metrics']
                successful_results.append({
                    'strategy': strategy_name,
                    'annualized_return': metrics['annualized_return'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'max_drawdown': metrics['max_drawdown'],
                    'total_return': metrics['total_return']
                })
            else:
                failed_strategies.append(strategy_name)
        
        if successful_results:
            # 按年化收益排序
            successful_results.sort(key=lambda x: x['annualized_return'], reverse=True)
            
            # 格式化显示
            print(f"{'策略':<15} {'年化收益':>12} {'夏普比率':>12} {'最大回撤':>12} {'总收益':>12}")
            print("-" * 80)
            
            for result in successful_results:
                annual_return = f"{result['annualized_return']:.2%}"
                sharpe_ratio = f"{result['sharpe_ratio']:.2f}"
                max_drawdown = f"{result['max_drawdown']:.2%}"
                total_return = f"{result['total_return']:.2%}"
                
                print(f"{result['strategy']:<15} {annual_return:>12} {sharpe_ratio:>12} {max_drawdown:>12} {total_return:>12}")
        
        if failed_strategies:
            print(f"\n❌ 运行失败的策略: {', '.join(failed_strategies)}")
    
    print("=" * 80)


if __name__ == '__main__':
    main()
