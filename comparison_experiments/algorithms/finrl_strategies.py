"""

FinRL官方仓库: https://github.com/AI4Finance-Foundation/FinRL
FinRL论文: "FinRL: Deep Reinforcement Learning Framework to Automate Trading in Quantitative Finance"
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta

# 死锁防护 - 设置线程数为1
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 禁用GPU避免CUDA死锁
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['MKL_THREADING_LAYER'] = 'sequential'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# 禁用不必要的警告，但保留重要错误信息
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    # FinRL核心模块
    import finrl
    from finrl import config
    from finrl.config import INDICATORS
    from finrl.config_tickers import DOW_30_TICKER
    
    # FinRL环境和预处理
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
    from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
    from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
    
    # FinRL智能体
    from finrl.agents.stablebaselines3.models import DRLAgent
    from finrl.meta.data_processor import DataProcessor
    
    # Stable Baselines3 (FinRL的核心依赖)
    from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
    from stable_baselines3.common.callbacks import BaseCallback
    
    print("✅ FinRL原版框架导入成功")
    try:
        print(f"📦 FinRL版本: {finrl.__version__}")
    except AttributeError:
        print("📦 FinRL版本: 已安装 (版本信息不可用)")
    
    # 设置PyTorch线程数防止死锁
    try:
        import torch
        torch.set_num_threads(1)
        print("🔒 PyTorch线程数已设置为1 (防死锁)")
    except ImportError:
        pass
    
except ImportError as e:
    print("❌ FinRL原版框架导入失败！")
    print(f"错误详情: {e}")
    print("\n🔧 解决方案:")
    print("1. 安装FinRL: pip install finrl")
    print("2. 安装依赖: pip install stable-baselines3[extra]")
    print("3. 安装数据源: pip install yfinance")
    print("4. 检查Python版本 >= 3.8")
    raise ImportError("FinRL原版框架未安装，请先安装FinRL及其依赖")


class AuthenticFinRLConfig:
    """FinRL原版配置 - 使用官方推荐参数"""
    
    def __init__(self):
        # FinRL官方推荐的技术指标
        self.TECHNICAL_INDICATORS = [
            'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 
            'dx_30', 'close_30_sma', 'close_60_sma'
        ]
        
        # FinRL官方环境参数
        self.ENV_PARAMS = {
            "hmax": 100,                    # 最大持仓量
            "initial_amount": 1000000,      # 初始资金
            "buy_cost_pct": 0.001,          # 买入手续费
            "sell_cost_pct": 0.001,         # 卖出手续费
            "reward_scaling": 1e-4,         # 奖励缩放
            "print_verbosity": 5            # 日志详细程度
        }
        
        # FinRL官方算法参数
        self.ALGORITHM_PARAMS = {
            'PPO': {
                'n_steps': 2048,
                'ent_coef': 0.01,
                'learning_rate': 0.00025,
                'batch_size': 128,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'max_grad_norm': 0.5
            },
            'A2C': {
                'learning_rate': 0.0007,
                'n_steps': 5,
                'gamma': 0.99,
                'gae_lambda': 1.0,
                'ent_coef': 0.01,
                'vf_coef': 0.25,
                'max_grad_norm': 0.5
            },
            'SAC': {
                'learning_rate': 0.0003,
                'buffer_size': 100000,
                'learning_starts': 100,
                'batch_size': 256,
                'tau': 0.005,
                'gamma': 0.99,
                'train_freq': 1,
                'gradient_steps': 1,
                'ent_coef': 'auto'
            },
            'TD3': {
                'learning_rate': 0.001,
                'buffer_size': 1000000,
                'learning_starts': 25000,
                'batch_size': 100,
                'tau': 0.005,
                'gamma': 0.99,
                'train_freq': (1, "episode"),
                'gradient_steps': -1,
                'policy_delay': 2,
                'target_policy_noise': 0.2,
                'target_noise_clip': 0.5
            },
            'DDPG': {
                'learning_rate': 0.001,
                'buffer_size': 1000000,
                'learning_starts': 100,
                'batch_size': 100,
                'tau': 0.005,
                'gamma': 0.99,
                'train_freq': (1, "episode"),
                'gradient_steps': -1
            }
        }


class AuthenticFinRLDataProcessor:
    """FinRL原版数据处理器 - 使用官方数据处理流程"""
    
    def __init__(self):
        self.config = AuthenticFinRLConfig()
        
    def prepare_data(self, df: pd.DataFrame, ticker: str, 
                    start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """使用FinRL官方数据处理流程"""
        
        print(f"📊 使用FinRL原版数据处理器处理 {ticker} 数据...")
        
        # 1. 数据格式标准化 - 严格按照FinRL要求
        processed_df = self._standardize_data_format(df, ticker)
        
        # 2. 使用FinRL官方特征工程
        processed_df = self._apply_finrl_feature_engineering(processed_df)
        
        # 3. 数据验证 - 确保符合FinRL标准
        self._validate_finrl_data(processed_df)
        
        print(f"✅ 数据处理完成，最终数据形状: {processed_df.shape}")
        return processed_df
    
    def _standardize_data_format(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """标准化为FinRL格式"""
        
        finrl_df = df.copy()
        
        # 确保必需列存在
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'tic']
        
        # 处理日期列
        if 'date' not in finrl_df.columns:
            if isinstance(finrl_df.index, pd.DatetimeIndex):
                finrl_df.reset_index(inplace=True)
                finrl_df.rename(columns={'index': 'date'}, inplace=True)
            else:
                raise ValueError("数据必须包含日期信息")
        
        # 确保日期格式正确
        finrl_df['date'] = pd.to_datetime(finrl_df['date'])
        
        # 添加股票代码
        finrl_df['tic'] = ticker
        
        # 验证OHLCV数据完整性
        ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in ohlcv_columns:
            if col not in finrl_df.columns:
                raise ValueError(f"缺少必需的列: {col}")
        
        # 数据质量检查
        if finrl_df[ohlcv_columns].isnull().any().any():
            raise ValueError("OHLCV数据包含缺失值")
        
        # 确保价格数据的逻辑一致性
        invalid_rows = (finrl_df['high'] < finrl_df['low']) | \
                      (finrl_df['high'] < finrl_df['close']) | \
                      (finrl_df['low'] > finrl_df['close'])
        
        if invalid_rows.any():
            print(f"⚠️ 发现 {invalid_rows.sum()} 行价格数据不一致，已修正")
            # 修正不一致的数据
            finrl_df.loc[invalid_rows, 'high'] = finrl_df.loc[invalid_rows, ['open', 'close']].max(axis=1)
            finrl_df.loc[invalid_rows, 'low'] = finrl_df.loc[invalid_rows, ['open', 'close']].min(axis=1)
        
        # 按FinRL要求排序
        finrl_df = finrl_df.sort_values(['date', 'tic']).reset_index(drop=True)
        
        return finrl_df[required_columns]
    
    def _apply_finrl_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用FinRL官方特征工程"""
        
        print("🔧 应用FinRL官方特征工程...")
        
        # 使用FinRL官方FeatureEngineer
        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=self.config.TECHNICAL_INDICATORS,
            use_vix=False,  # 单股票不使用VIX
            use_turbulence=False,  # 单股票不使用turbulence
            user_defined_feature=False
        )
        
        # 应用特征工程
        processed_df = fe.preprocess_data(df)
        
        print(f"📈 添加了 {len(self.config.TECHNICAL_INDICATORS)} 个技术指标")
        return processed_df
    
    def _validate_finrl_data(self, df: pd.DataFrame):
        """验证数据是否符合FinRL标准"""
        
        # 检查必需列
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'tic']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少FinRL必需列: {missing_cols}")
        
        # 检查技术指标
        for indicator in self.config.TECHNICAL_INDICATORS:
            if indicator not in df.columns:
                raise ValueError(f"缺少技术指标: {indicator}")
        
        # 检查数据完整性
        if df.isnull().any().any():
            null_cols = df.columns[df.isnull().any()].tolist()
            raise ValueError(f"数据包含缺失值: {null_cols}")
        
        print("✅ 数据验证通过，符合FinRL标准")


class AuthenticFinRLAgent:
    """FinRL原版智能体 - 使用官方DRLAgent"""
    
    def __init__(self, algorithm: str = 'PPO'):
        self.algorithm = algorithm
        self.config = AuthenticFinRLConfig()
        self.model = None
        self.env = None
        
        if algorithm not in self.config.ALGORITHM_PARAMS:
            raise ValueError(f"不支持的算法: {algorithm}")
    
    def create_environment(self, df: pd.DataFrame) -> DummyVecEnv:
        """创建FinRL官方交易环境"""
        
        print(f"🏗️ 创建FinRL官方交易环境...")
        
        # 计算状态空间和动作空间
        stock_dimension = len(df.tic.unique())
        state_space = 1 + 2 * stock_dimension + len(self.config.TECHNICAL_INDICATORS) * stock_dimension
        action_space = stock_dimension
        
        # 使用FinRL官方环境参数
        env_kwargs = {
            "hmax": self.config.ENV_PARAMS["hmax"],
            "initial_amount": self.config.ENV_PARAMS["initial_amount"],
            "num_stock_shares": [0] * stock_dimension,  # 初始持股数量
            "buy_cost_pct": [self.config.ENV_PARAMS["buy_cost_pct"]] * stock_dimension,
            "sell_cost_pct": [self.config.ENV_PARAMS["sell_cost_pct"]] * stock_dimension,
            "reward_scaling": self.config.ENV_PARAMS["reward_scaling"],
            "state_space": state_space,
            "action_space": action_space,
            "tech_indicator_list": self.config.TECHNICAL_INDICATORS,
            "print_verbosity": self.config.ENV_PARAMS["print_verbosity"],
            "stock_dim": stock_dimension  # 添加缺失的stock_dim参数
        }
        
        # 创建FinRL官方环境
        env = StockTradingEnv(df=df, **env_kwargs)
        
        # 包装为向量化环境
        env = DummyVecEnv([lambda: env])
        
        print(f"✅ 环境创建成功 - 状态空间: {state_space}, 动作空间: {action_space}")
        
        self.env = env
        return env
    
    def train(self, env: DummyVecEnv, total_timesteps: int = 50000) -> Any:
        """使用FinRL官方训练流程"""
        
        print(f"🚀 开始训练FinRL {self.algorithm} 模型...")
        
        # 获取算法参数
        algo_params = self.config.ALGORITHM_PARAMS[self.algorithm].copy()
        
        # 创建模型
        if self.algorithm == 'PPO':
            model = PPO("MlpPolicy", env, verbose=0, **algo_params)
        elif self.algorithm == 'A2C':
            model = A2C("MlpPolicy", env, verbose=0, **algo_params)
        elif self.algorithm == 'SAC':
            # SAC通常需要更多训练步数
            model = SAC("MlpPolicy", env, verbose=0, **algo_params)
        elif self.algorithm == 'TD3':
            # TD3需要噪声
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(
                mean=np.zeros(n_actions), 
                sigma=0.1 * np.ones(n_actions)
            )
            model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=0, **algo_params)
        elif self.algorithm == 'DDPG':
            # DDPG需要噪声
            n_actions = env.action_space.shape[-1]
            action_noise = OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(n_actions),
                sigma=0.1 * np.ones(n_actions)
            )
            model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=0, **algo_params)
        
        # 训练模型
        model.learn(total_timesteps=total_timesteps)
        
        print(f"✅ {self.algorithm} 模型训练完成")
        
        self.model = model
        return model
    
    def backtest(self, test_env: DummyVecEnv, test_df: pd.DataFrame = None) -> Tuple[pd.Series, pd.DataFrame]:
        """使用FinRL官方回测流程"""
        
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train()方法")
        
        print(f"📊 开始FinRL {self.algorithm} 回测...")
        
        # 重置环境
        obs = test_env.reset()
        
        # 存储回测结果
        portfolio_values = []
        actions_taken = []
        dates = []
        
        done = False
        step_count = 0
        
        # 记录初始资产价值
        if hasattr(test_env.envs[0], 'asset_memory') and test_env.envs[0].asset_memory:
            initial_value = test_env.envs[0].asset_memory[-1]
            portfolio_values.append(initial_value)
            actions_taken.append(0)  # 初始动作为0
            dates.append(0)
        
        while not done:
            # 使用训练好的模型预测动作
            action, _states = self.model.predict(obs, deterministic=True)
            
            # 执行动作
            obs, rewards, done, info = test_env.step(action)
            
            # 在执行动作后记录资产价值
            if hasattr(test_env.envs[0], 'asset_memory') and test_env.envs[0].asset_memory:
                portfolio_value = test_env.envs[0].asset_memory[-1]
                portfolio_values.append(portfolio_value)
                actions_taken.append(action[0] if isinstance(action, np.ndarray) else action)
                
                # 记录日期
                if hasattr(test_env.envs[0], 'date'):
                    dates.append(test_env.envs[0].date)
                else:
                    dates.append(step_count + 1)
            
            step_count += 1
            
            # 防止无限循环
            if step_count > 10000:
                print("⚠️ 回测步数超过限制，强制结束")
                break
        
        if not portfolio_values:
            raise ValueError("回测过程中未收集到有效数据")
        
        # 调试信息
        print(f"🔍 调试信息: 收集到 {len(portfolio_values)} 个资产价值")
        if len(portfolio_values) > 0:
            print(f"🔍 初始资产: {portfolio_values[0]:.2f}")
            print(f"🔍 最终资产: {portfolio_values[-1]:.2f}")
        
        # 计算性能指标
        if len(portfolio_values) < 2:
            print("⚠️ 资产价值数据不足，使用默认值")
            returns = pd.Series([0])
            total_return = 0
            annualized_return = 0
        else:
            # 检查资产价值是否完全没有变化
            unique_values = set(portfolio_values)
            if len(unique_values) == 1:
                print("⚠️ 检测到资产价值完全没有变化，可能是FinRL环境问题")
                print(f"🔍 所有资产价值都是: {portfolio_values[0]:.2f}")
                returns = pd.Series([0])
                total_return = 0
                annualized_return = 0
                final_portfolio_value = portfolio_values[0]
            else:
                # 检查最后一个值是否被重置为初始值
                if len(portfolio_values) > 2 and portfolio_values[-1] == portfolio_values[0]:
                    print("⚠️ 检测到环境重置，使用倒数第二个值作为最终资产")
                    final_portfolio_value = portfolio_values[-2]
                    # 移除最后一个重置值
                    portfolio_values = portfolio_values[:-1]
                else:
                    final_portfolio_value = portfolio_values[-1]
            
                returns = pd.Series(portfolio_values).pct_change().dropna()
                
                # 年化收益率 - 使用修正后的最终价值
                total_return = (final_portfolio_value / portfolio_values[0]) - 1
                trading_days = len(portfolio_values)
                if trading_days > 1:
                    annualized_return = (1 + total_return) ** (252 / trading_days) - 1
                else:
                    annualized_return = total_return
            
            print(f"🔍 收益计算: 初始={portfolio_values[0]:.2f}, 最终={final_portfolio_value:.2f}")
            print(f"🔍 总收益: {total_return:.4f} ({total_return*100:.2f}%)")
            print(f"🔍 年化收益: {annualized_return:.4f} ({annualized_return*100:.2f}%)")
        
        # 其他指标
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        volatility = returns.std() * np.sqrt(252)
        
        # 创建指标Series
        metrics = pd.Series({
            'annualized_return': annualized_return,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'num_trades': len([a for a in actions_taken if np.abs(a).sum() > 0.01]),
            'final_portfolio_value': final_portfolio_value if 'final_portfolio_value' in locals() else portfolio_values[-1],
            'algorithm': self.algorithm
        })
        
        # 创建回测结果DataFrame - 确保长度一致
        if test_df is not None:
            min_length = min(len(test_df), len(portfolio_values), len(actions_taken))
            date_index = test_df.index[:min_length]
        else:
            min_length = min(len(portfolio_values), len(actions_taken))
            date_index = range(min_length)
            
        backtest_results = pd.DataFrame({
            'date': date_index,
            'portfolio_value': portfolio_values[:min_length],
            'returns': returns[:min_length] if len(returns) >= min_length else [0] * min_length,
            'actions': actions_taken[:min_length]
        })
        
        print(f"✅ 回测完成 - 年化收益: {annualized_return:.2%}, 夏普比率: {sharpe_ratio:.3f}")
        
        return metrics, backtest_results
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """计算最大回撤"""
        peak = portfolio_values[0]
        max_dd = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd


class AuthenticFinRLRunner:
    """FinRL原版运行器 - 完整的FinRL工作流程"""
    
    def __init__(self):
        self.data_processor = AuthenticFinRLDataProcessor()
        self.supported_algorithms = ['PPO', 'A2C', 'SAC', 'TD3', 'DDPG']
    
    def run_finrl_strategy(self, algorithm: str, train_df: pd.DataFrame, 
                          test_df: pd.DataFrame, ticker: str = 'STOCK',
                          total_timesteps: int = 50000,
                          **kwargs) -> Tuple[pd.Series, pd.DataFrame]:
        """运行完整的FinRL策略 - 使用原版框架"""
        
        if algorithm.upper() not in self.supported_algorithms:
            raise ValueError(f"不支持的算法: {algorithm}. 支持的算法: {self.supported_algorithms}")
        
        algorithm = algorithm.upper()
        print(f"🚀 运行FinRL原版 {algorithm} 策略 - 股票: {ticker}")
        
        try:
            # 1. 数据预处理 - 使用FinRL官方流程
            print("📊 步骤1: 数据预处理...")
            train_processed = self.data_processor.prepare_data(train_df, ticker)
            test_processed = self.data_processor.prepare_data(test_df, ticker)
            
            # 2. 创建智能体
            print("🤖 步骤2: 创建FinRL智能体...")
            agent = AuthenticFinRLAgent(algorithm)
            
            # 3. 创建训练环境
            print("🏗️ 步骤3: 创建训练环境...")
            train_env = agent.create_environment(train_processed)
            
            # 4. 训练模型
            print("🚀 步骤4: 训练模型...")
            model = agent.train(train_env, total_timesteps)
            
            # 5. 创建测试环境
            print("🧪 步骤5: 创建测试环境...")
            test_env = agent.create_environment(test_processed)
            
            # 6. 回测
            print("📊 步骤6: 执行回测...")
            metrics, backtest_results = agent.backtest(test_env, test_df)
            
            print(f"✅ FinRL {algorithm} 策略执行完成!")
            return metrics, backtest_results
            
        except Exception as e:
            print(f"❌ FinRL {algorithm} 策略执行失败: {e}")
            raise


# 全局FinRL运行器实例
authentic_finrl_runner = AuthenticFinRLRunner()

# 导出函数 - 与baseline.py兼容
def run_finrl_ppo_strategy(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                          ticker: str = 'STOCK', **kwargs) -> Tuple[pd.Series, pd.DataFrame]:
    """FinRL原版PPO策略"""
    return authentic_finrl_runner.run_finrl_strategy('PPO', train_df, test_df, ticker, **kwargs)

def run_finrl_a2c_strategy(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                          ticker: str = 'STOCK', **kwargs) -> Tuple[pd.Series, pd.DataFrame]:
    """FinRL原版A2C策略"""
    return authentic_finrl_runner.run_finrl_strategy('A2C', train_df, test_df, ticker, **kwargs)

def run_finrl_sac_strategy(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                          ticker: str = 'STOCK', **kwargs) -> Tuple[pd.Series, pd.DataFrame]:
    """FinRL原版SAC策略"""
    # SAC需要更多训练步数
    kwargs.setdefault('total_timesteps', 100000)
    return authentic_finrl_runner.run_finrl_strategy('SAC', train_df, test_df, ticker, **kwargs)

def run_finrl_td3_strategy(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                          ticker: str = 'STOCK', **kwargs) -> Tuple[pd.Series, pd.DataFrame]:
    """FinRL原版TD3策略"""
    return authentic_finrl_runner.run_finrl_strategy('TD3', train_df, test_df, ticker, **kwargs)

def run_finrl_ddpg_strategy(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                           ticker: str = 'STOCK', **kwargs) -> Tuple[pd.Series, pd.DataFrame]:
    """FinRL原版DDPG策略"""
    # DDPG需要更多训练步数
    kwargs.setdefault('total_timesteps', 100000)
    return authentic_finrl_runner.run_finrl_strategy('DDPG', train_df, test_df, ticker, **kwargs)


if __name__ == "__main__":
    # 测试FinRL原版集成
    print("🧪 测试FinRL原版集成...")
    
    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
    n_days = len(dates)
    
    # 生成真实的股价数据
    returns = np.random.normal(0.001, 0.02, n_days)
    prices = 100 * np.cumprod(1 + returns)
    
    test_data = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.normal(0, 0.005, n_days)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, n_days)
    })
    
    # 确保价格逻辑正确
    test_data['high'] = np.maximum(test_data['high'], test_data['close'])
    test_data['low'] = np.minimum(test_data['low'], test_data['close'])
    
    # 分割训练和测试数据
    split_point = int(len(test_data) * 0.7)
    train_data = test_data.iloc[:split_point]
    test_data = test_data.iloc[split_point:]
    
    try:
        print("🚀 测试FinRL PPO策略...")
        metrics, results = run_finrl_ppo_strategy(train_data, test_data, 'TEST', total_timesteps=1000)
        
        print("✅ FinRL原版集成测试成功!")
        print(f"📊 年化收益: {metrics['annualized_return']:.2%}")
        print(f"📈 夏普比率: {metrics['sharpe_ratio']:.3f}")
        print(f"📉 最大回撤: {metrics['max_drawdown']:.2%}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print("请检查FinRL安装是否正确")
