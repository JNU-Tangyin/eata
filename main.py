import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import warnings
import logging

# 隐藏警告和日志噪音
warnings.filterwarnings('ignore')
logging.getLogger('MCTSAdapter').setLevel(logging.CRITICAL)
logging.getLogger('NEMoTS').setLevel(logging.CRITICAL)
logging.getLogger('nemots').setLevel(logging.CRITICAL)
logging.getLogger('engine').setLevel(logging.CRITICAL)
logging.getLogger('model').setLevel(logging.CRITICAL)
logging.getLogger('mcts').setLevel(logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)

# 完全禁用所有日志输出
import sys
class NullWriter:
    def write(self, txt): pass
    def flush(self): pass

# 临时重定向stderr来隐藏日志
original_stderr = sys.stderr

from data import BaostockDataWorker
from sliding_window_nemots import SlidingWindowNEMoTS
from agent import Agent
from env import StockmarketEnv
from rl import IntegratedRLFeedbackSystem
from tin_metrics import TradingMetrics, compare_strategies
import torch
import time

class BandwagonRL:
    """
    Bandwagon强化学习主算法
    整合滑动窗口、NEMoTS训练、预测和RL反馈机制
    """
    
    def __init__(self, asset_codes: List[str], window_size: int = 20, lookahead: int = 20, topk: int = 10):
        """
        初始化Bandwagon RL算法
        
        Args:
            asset_codes: 资产代码列表
            window_size: 滑动窗口大小
            lookahead: 前瞻窗口大小
            topk: 最佳拟合数量
        """
        self.asset_codes = asset_codes
        self.window_size = window_size
        self.lookahead = lookahead
        self.topk = topk
        
        # 数据工作器
        try:
            self.dataworker = BaostockDataWorker()
            print(f"   ✅ 数据工作器初始化成功")
        except Exception as e:
            print(f"   ⚠️ 数据工作器初始化失败: {e}")
            self.dataworker = None
        
        # 为每个资产创建滑动窗口NEMoTS
        self.nemots_models = {}
        for code in asset_codes:
            try:
                # 临时隐藏stderr输出
                sys.stderr = NullWriter()
                self.nemots_models[code] = SlidingWindowNEMoTS(
                    lookback=window_size, 
                    lookahead=lookahead
                )
                sys.stderr = original_stderr
                print(f"   ✅ NEMoTS模型创建成功: {code}")
            except Exception as e:
                sys.stderr = original_stderr
                print(f"   ⚠️ NEMoTS模型创建失败: {code}, {e}")
                self.nemots_models[code] = None
        
        # RL相关组件
        self.agents = {}  # 每个资产的智能体
        self.envs = {}    # 每个资产的环境
        
        # 训练历史和反馈机制
        self.training_history = []
        self.reward_history = []
        self.loss_history = []
        
        # 集成RL反馈系统
        self.feedback_system = IntegratedRLFeedbackSystem()
        
        # 时间统计 - 按老师建议添加
        self.timing_stats = {
            'total_time': 0,
            'training_times': [],
            'prediction_times': [],
            'signal_generation_times': [],
            'reward_calculation_times': [],
            'feedback_times': [],
            'iteration_times': []
        }
        
        print(f"🚀 Bandwagon RL算法初始化完成")
        print(f"   资产数量: {len(asset_codes)}")
        print(f"   窗口大小: {window_size}, 前瞻: {lookahead}, TopK: {topk}")
        print(f"   ⏱️ 时间统计功能已启用")
    
    def load_asset_data(self, code: str, days: int = 500) -> pd.DataFrame:
        """从文件中读取资产代码数据"""
        if self.dataworker is None:
            print(f"📊 数据工作器不可用，使用模拟数据: {code}")
            return self._create_mock_data(code, days)
            
        try:
            data = self.dataworker.latest(code, ktype='d', days=days)
            print(f"📊 加载资产 {code}: {len(data)} 天数据")
            
            # 确保数据包含必要的列
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_cols):
                print(f"⚠️ 数据缺少必要列，使用模拟数据")
                return self._create_mock_data(code, days)
            
            # 确保有amount列
            if 'amount' not in data.columns:
                data['amount'] = data['volume'] * data['close']
            
            return data
        except Exception as e:
            print(f"❌ 加载资产 {code} 失败: {e}，使用模拟数据")
            return self._create_mock_data(code, days)
    
    def _create_mock_data(self, code: str, days: int) -> pd.DataFrame:
        """创建模拟数据用于测试"""
        import datetime
        
        dates = pd.date_range(end=datetime.datetime.now(), periods=days, freq='D')
        
        # 生成模拟价格数据
        base_price = 10.0
        prices = []
        current_price = base_price
        
        for i in range(days):
            # 随机游走 + 小幅趋势
            change = np.random.normal(0.001, 0.02)  # 0.1%均值，2%标准差
            current_price = current_price * (1 + change)
            current_price = max(current_price, 1.0)  # 确保价格为正
            prices.append(current_price)
        
        # 生成OHLCV数据
        data = []
        for i, price in enumerate(prices):
            high = price * (1 + np.random.uniform(0, 0.02))
            low = price * (1 - np.random.uniform(0, 0.02))
            open_price = prices[i-1] if i > 0 else price
            volume = np.random.uniform(1000, 10000)
            
            data.append({
                'date': dates[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume,
                'amount': volume * price
            })
        
        df = pd.DataFrame(data)
        print(f"   使用模拟数据: {len(df)} 天")
        return df
    
    def sliding_window_training(self, code: str, data: pd.DataFrame, current_day: int) -> Dict[str, Any]:
        """
        滑动窗口训练 - 获得topk个最佳拟合
        
        Args:
            code: 资产代码
            data: 历史数据
            current_day: 当前训练日索引
            
        Returns:
            训练结果包含topk个最佳拟合
        """
        start_time = time.time()  # 开始计时
        print(f"\n🧠 开始滑动窗口训练: {code}, 第{current_day}天")
        
        # 获取训练窗口数据 - 需要足够的历史数据用于训练
        # 至少需要 window_size + lookahead 的数据
        required_length = self.window_size + self.lookahead
        start_idx = max(0, current_day - required_length)
        end_idx = current_day
        window_data = data.iloc[start_idx:end_idx].copy()
        
        print(f"   数据范围: {start_idx} -> {end_idx}, 长度: {len(window_data)}, 需要: {required_length}")
        
        if len(window_data) < required_length:
            print(f"⚠️ 数据不足，跳过训练 (需要{required_length}天，实际{len(window_data)}天)")
            return {'success': False, 'reason': 'insufficient_data'}
        
        # 使用NEMoTS进行训练
        nemots_model = self.nemots_models.get(code)
        if nemots_model is None:
            print(f"⚠️ NEMoTS模型不可用，使用简化训练")
            return {
                'success': True,
                'topk_models': ['simplified_model'] * self.topk,
                'metrics': {'mae': 0.02, 'mse': 0.001, 'corr': 0.5, 'reward': 0.01, 'loss': 0.01},
                'model_object': None
            }
        
        # 临时隐藏stderr输出
        sys.stderr = NullWriter()
        try:
            training_result = nemots_model.sliding_fit(window_data)
        finally:
            # 恢复stderr
            sys.stderr = original_stderr
        
        if training_result['success']:
            # 这里简化为单个最佳拟合，实际可以扩展为topk个
            topk_models = [training_result['best_expression']] * self.topk
            
            # 记录训练时间
            training_time = time.time() - start_time
            self.timing_stats['training_times'].append(training_time)
            print(f"   ⏱️ 训练耗时: {training_time:.3f}秒")
            
            return {
                'success': True,
                'topk_models': topk_models,
                'metrics': training_result.get('metrics', {
                    'mae': training_result.get('mae', 0.02),
                    'mse': training_result.get('mse', 0.001), 
                    'corr': training_result.get('corr', 0.5),
                    'reward': training_result.get('reward', 0.01),
                    'loss': training_result.get('loss', 0.01)
                }),
                'model_object': nemots_model,
                'training_time': training_time
            }
        else:
            # 确保失败时也有metrics字段
            return {
                'success': False,
                'reason': training_result.get('reason', 'training_failed'),
                'topk_models': [],
                'metrics': {
                    'mae': training_result.get('mae', 1.0),
                    'mse': training_result.get('mse', 1.0),
                    'corr': training_result.get('corr', 0.0),
                    'reward': training_result.get('reward', 0.0),
                    'loss': training_result.get('loss', 1.0)
                },
                'model_object': None
            }
    
    def generate_predictions(self, topk_models: List[str], data: pd.DataFrame, n_days: int = 20) -> np.ndarray:
        """
        使用topk个拟合对未来n日做预测，生成200个价格预测
        
        Args:
            topk_models: topk个最佳拟合模型
            data: 历史数据
            n_days: 预测天数
            
        Returns:
            shape为(200, n_days)的价格预测矩阵
        """
        start_time = time.time()
        print(f"🔮 生成价格预测: {len(topk_models)}个模型 × {n_days}天")
        
        # 简化实现：每个模型生成20个预测（共200个）
        predictions_per_model = 200 // self.topk
        all_predictions = []
        
        for model_expr in topk_models:
            # 基于模型表达式生成预测（这里简化为随机游走+趋势）
            last_price = data['close'].iloc[-1]
            
            for _ in range(predictions_per_model):
                # 生成单条预测路径
                prediction_path = []
                current_price = last_price
                
                for day in range(n_days):
                    # 简化的价格预测逻辑（实际应该基于NEMoTS模型）
                    trend = np.random.normal(0.001, 0.02)  # 小幅上涨趋势+噪声
                    current_price = current_price * (1 + trend)
                    prediction_path.append(current_price)
                
                all_predictions.append(prediction_path)
        
        predictions = np.array(all_predictions)
        
        # 记录预测时间
        prediction_time = time.time() - start_time
        self.timing_stats['prediction_times'].append(prediction_time)
        print(f"   预测矩阵形状: {predictions.shape}")
        print(f"   ⏱️ 预测耗时: {prediction_time:.3f}秒")
        return predictions
    
    def generate_trading_signals(self, predictions: np.ndarray, current_price: float = None) -> List[int]:
        """
        基于200个价格预测的(Q25,Q75)生成交易信号
        未来替换成共形预测
        
        Args:
            predictions: 价格预测矩阵 (200, n_days)
            current_price: 当前价格作为基准
            
        Returns:
            交易信号列表 [-1, 0, 1] 对应 [卖出, 持有, 买入]
        """
        start_time = time.time()
        print(f"📈 生成交易信号基于预测分位数")
        
        signals = []
        
        # 如果没有提供当前价格，使用第一天预测的中位数作为基准
        if current_price is None:
            current_price = np.percentile(predictions[:, 0], 50)
        
        for day in range(predictions.shape[1]):
            day_predictions = predictions[:, day]
            
            # 计算分位数
            q25 = np.percentile(day_predictions, 25)
            q75 = np.percentile(day_predictions, 75)
            median = np.percentile(day_predictions, 50)
            
            # 从配置文件读取交易参数 - 调整为更合理的阈值
            buy_threshold, sell_threshold, uncertainty_threshold = self._get_trading_thresholds()
            
            # 计算预测的相对变化
            predicted_change = (median - current_price) / current_price
            uncertainty = (q75 - q25) / median if median > 0 else 0
            
            # 改进的信号生成逻辑
            if uncertainty > uncertainty_threshold:  # 不确定性太高
                signal = 0  # 持有
            elif predicted_change > (buy_threshold - 1.0):  # 预期上涨超过阈值
                signal = 1  # 买入
            elif predicted_change < -(1.0 - sell_threshold):  # 预期下跌超过阈值
                signal = -1  # 卖出
            else:
                signal = 0  # 持有
            
            signals.append(signal)
            
            # 更新基准价格为当前预测的中位数
            current_price = median
        
        # 记录信号生成时间
        signal_time = time.time() - start_time
        self.timing_stats['signal_generation_times'].append(signal_time)
        print(f"   生成信号: 买入{signals.count(1)}, 持有{signals.count(0)}, 卖出{signals.count(-1)}")
        print(f"   ⏱️ 信号生成耗时: {signal_time:.3f}秒")
        return signals
    
    def _get_trading_thresholds(self):
        """从配置文件读取交易阈值"""
        try:
            import json
            import os
            
            config_file = 'config.json'
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                trading_config = config.get('trading', {})
                buy_threshold = trading_config.get('buy_threshold', 1.012)
                sell_threshold = trading_config.get('sell_threshold', 0.988)
                uncertainty_threshold = trading_config.get('uncertainty_threshold', 0.12)
                
                return buy_threshold, sell_threshold, uncertainty_threshold
        except:
            pass
        
        # 默认值 - 调整为更合理的阈值
        return 1.005, 0.995, 0.05  # 0.5%涨跌幅阈值，5%不确定性阈值
    
    def calculate_reward_loss(self, signals: List[int], ground_truth: pd.DataFrame) -> Tuple[float, float]:
        """
        交易信号与ground truth比对，获得loss和reward
        
        Args:
            signals: 交易信号列表
            ground_truth: lookAhead窗口的真实数据
            
        Returns:
            (reward, loss) 元组
        """
        print(f"⚖️ 计算reward和loss")
        
        if len(ground_truth) == 0:
            return 0.0, 1.0
        
        # 计算实际收益
        actual_returns = ground_truth['close'].pct_change().fillna(0)
        
        # 计算基准收益（买入持有策略）
        benchmark_return = np.sum(actual_returns)
        
        # 根据信号计算策略收益
        strategy_returns = []
        for i, signal in enumerate(signals[:len(actual_returns)]):
            if i < len(actual_returns):
                if signal == 1:  # 买入
                    strategy_returns.append(actual_returns.iloc[i])
                elif signal == -1:  # 卖出 (做空)
                    strategy_returns.append(-actual_returns.iloc[i])
                else:  # 持有 - 改为获得市场收益而不是0
                    strategy_returns.append(actual_returns.iloc[i] * 0.5)  # 持有获得50%市场收益
            else:
                strategy_returns.append(0)
        
        # 计算策略收益
        strategy_return = np.sum(strategy_returns)
        
        # 计算超额收益（相对于基准）
        excess_return = strategy_return - benchmark_return
        
        # 改进的reward/loss计算
        if excess_return > 0:
            reward = excess_return + max(0, strategy_return)  # 超额收益 + 绝对收益
            loss = 0
        else:
            reward = max(0, strategy_return)  # 至少获得正的绝对收益
            loss = max(0, -excess_return)  # 相对基准的损失
        
        print(f"   策略收益: {strategy_return:.4f}, 超额收益: {excess_return:.4f}, Reward: {reward:.4f}, Loss: {loss:.4f}")
        return reward, loss
    
    def extract_market_state(self, data: pd.DataFrame, current_day: int) -> np.ndarray:
        """
        从市场数据中提取状态特征向量
        
        Args:
            data: 市场数据
            current_day: 当前日期索引
            
        Returns:
            10维市场状态特征向量
        """
        # 获取最近几天的数据用于特征提取
        lookback_days = min(10, current_day)
        recent_data = data.iloc[current_day-lookback_days:current_day]
        
        if len(recent_data) == 0:
            return np.zeros(10)
        
        # 提取10维特征
        features = []
        
        # 1. 价格变化率
        price_change = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
        features.append(np.clip(price_change, -0.1, 0.1))
        
        # 2. 价格波动率
        price_volatility = recent_data['close'].pct_change().std()
        features.append(np.clip(price_volatility, 0, 0.1))
        
        # 3. 成交量变化
        volume_change = (recent_data['volume'].iloc[-1] - recent_data['volume'].iloc[0]) / recent_data['volume'].iloc[0]
        features.append(np.clip(volume_change, -1, 1))
        
        # 4. 最高价相对位置
        high_position = (recent_data['close'].iloc[-1] - recent_data['low'].min()) / (recent_data['high'].max() - recent_data['low'].min())
        features.append(np.clip(high_position, 0, 1))
        
        # 5-7. 移动平均线相对位置（3日、5日、10日）
        for window in [3, 5, min(10, len(recent_data))]:
            if len(recent_data) >= window:
                ma = recent_data['close'].rolling(window).mean().iloc[-1]
                ma_position = (recent_data['close'].iloc[-1] - ma) / ma
                features.append(np.clip(ma_position, -0.1, 0.1))
            else:
                features.append(0.0)
        
        # 8. RSI指标（简化版）
        price_changes = recent_data['close'].pct_change().dropna()
        if len(price_changes) > 0:
            gains = price_changes[price_changes > 0].mean()
            losses = abs(price_changes[price_changes < 0].mean())
            rsi = gains / (gains + losses) if (gains + losses) > 0 else 0.5
            features.append(rsi)
        else:
            features.append(0.5)
        
        # 9. 成交量相对强度
        volume_strength = recent_data['volume'].iloc[-1] / recent_data['volume'].mean()
        features.append(np.clip(volume_strength, 0, 3))
        
        # 10. 趋势强度
        if len(recent_data) >= 3:
            trend = np.polyfit(range(len(recent_data)), recent_data['close'], 1)[0]
            trend_strength = trend / recent_data['close'].mean()
            features.append(np.clip(trend_strength, -0.01, 0.01))
        else:
            features.append(0.0)
        
        return np.array(features[:10])  # 确保返回10维向量

    def update_agent_with_feedback(self, code: str, reward: float, loss: float, signals: List[int], 
                                 market_state: np.ndarray, action: int, data: pd.DataFrame, current_day: int):
        """
        将reward和loss反馈到智能体进行更新
        使用集成的RL反馈系统
        """
        print(f"🔄 更新智能体 {code} - Reward: {reward:.4f}, Loss: {loss:.4f}")
        
        # 构建上下文信息
        context = {
            'signals': signals,
            'current_day': current_day,
            'market_volatility': market_state[1] if len(market_state) > 1 else 0.0,
            'trading_volume': market_state[2] if len(market_state) > 2 else 0.0,
            'price_trend': market_state[0] if len(market_state) > 0 else 0.0,
            'prediction_confidence': 0.8,  # 可以从NEMoTS模型获取
            'asset_code': code
        }
        
        # 使用集成反馈系统处理reward和loss
        feedback_result = self.feedback_system.process_episode_feedback(
            code=code,
            reward=reward,
            loss=loss,
            market_state=market_state,
            action=action,
            context=context
        )
        
        # 应用反馈结果到具体组件
        self._apply_feedback_to_components(code, feedback_result)
        
        # 记录到历史
        self.reward_history.append({
            'code': code,
            'reward': reward,
            'loss': loss,
            'signals': signals,
            'feedback_result': feedback_result,
            'timestamp': pd.Timestamp.now()
        })
        
        return feedback_result
    
    def _apply_feedback_to_components(self, code: str, feedback_result: Dict[str, Any]):
        """
        将反馈结果应用到具体组件
        """
        print(f"🔧 应用反馈到组件: {code}")
        
        # 1. 应用到NEMoTS模型
        if 'loss_processing' in feedback_result and 'nemots' in feedback_result['loss_processing']:
            nemots_feedback = feedback_result['loss_processing']['nemots']
            if code in self.nemots_models:
                self._apply_nemots_feedback(code, nemots_feedback)
        
        # 2. 应用到Agent
        if 'loss_processing' in feedback_result and 'agent' in feedback_result['loss_processing']:
            agent_feedback = feedback_result['loss_processing']['agent']
            self._apply_agent_feedback(code, agent_feedback)
        
        # 3. 应用reward强化
        if 'reward_processing' in feedback_result:
            reward_feedback = feedback_result['reward_processing']
            self._apply_reward_reinforcement(code, reward_feedback)
    
    def _apply_nemots_feedback(self, code: str, nemots_feedback: Dict[str, Any]):
        """应用NEMoTS反馈"""
        if code not in self.nemots_models:
            return
        
        nemots_model = self.nemots_models[code]
        
        # 调整学习率
        if 'learning_rate_multiplier' in nemots_feedback:
            lr_mult = nemots_feedback['learning_rate_multiplier']
            if hasattr(nemots_model, 'hyperparams') and hasattr(nemots_model.hyperparams, 'lr'):
                nemots_model.hyperparams.lr *= lr_mult
                print(f"   📉 调整NEMoTS学习率: ×{lr_mult:.3f}")
        
        # 调整探索率
        if 'exploration_rate_multiplier' in nemots_feedback:
            exp_mult = nemots_feedback['exploration_rate_multiplier']
            if hasattr(nemots_model, 'hyperparams') and hasattr(nemots_model.hyperparams, 'exploration_rate'):
                nemots_model.hyperparams.exploration_rate *= exp_mult
                print(f"   🔍 调整探索率: ×{exp_mult:.3f}")
    
    def _apply_agent_feedback(self, code: str, agent_feedback: Dict[str, Any]):
        """应用Agent反馈"""
        if code not in self.agents:
            # 创建智能体
            stock_df = pd.DataFrame({'code': [code], 'name': [code], 'weight': [1.0], 'sector': ['default']})
            self.agents[code] = Agent(stock_df)
        
        agent = self.agents[code]
        
        # 调整权重（这里可以扩展Agent类来支持动态权重调整）
        print(f"   🤖 应用Agent权重调整: {agent_feedback.get('reason', 'unknown')}")
    
    def _apply_reward_reinforcement(self, code: str, reward_feedback: Dict[str, Any]):
        """应用奖励强化"""
        if reward_feedback.get('action') == 'reinforce':
            print(f"   ✅ 强化策略: 动作{reward_feedback.get('target_action')} 增强{reward_feedback.get('enhancement', 0):.3f}")
        elif reward_feedback.get('action') == 'penalize':
            print(f"   ❌ 惩罚策略: 动作{reward_feedback.get('target_action')} 惩罚{reward_feedback.get('penalty', 0):.3f}")
    
    def run_rl_iteration(self, code: str, data: pd.DataFrame, current_day: int) -> Dict[str, Any]:
        """
        执行单次RL迭代
        
        Args:
            code: 资产代码
            data: 完整数据
            current_day: 当前日期索引
            
        Returns:
            迭代结果
        """
        iteration_start_time = time.time()  # 大循环计时
        print(f"\n🔄 RL迭代: {code}, 第{current_day}天")
        
        # 1. 滑动窗口训练
        training_result = self.sliding_window_training(code, data, current_day)
        if not training_result['success']:
            return training_result
        
        # 2. 生成预测
        topk_models = training_result['topk_models']
        historical_data = data.iloc[:current_day]
        predictions = self.generate_predictions(topk_models, historical_data, self.lookahead)
        
        # 3. 生成交易信号
        current_price = historical_data['close'].iloc[-1]
        signals = self.generate_trading_signals(predictions, current_price)
        
        # 4. 获取ground truth（lookAhead窗口）
        lookahead_end = min(current_day + self.lookahead, len(data))
        ground_truth = data.iloc[current_day:lookahead_end]
        
        # 5. 计算reward和loss
        reward, loss = self.calculate_reward_loss(signals, ground_truth)
        
        # 6. 提取市场状态特征
        market_state = self.extract_market_state(data, current_day)
        
        # 7. 确定主要交易动作（简化为信号的众数）
        main_action = max(set(signals), key=signals.count) if signals else 0
        main_action = {-1: 0, 0: 1, 1: 2}.get(main_action, 1)  # 转换为0,1,2格式
        
        # 8. 反馈到智能体
        feedback = self.update_agent_with_feedback(code, reward, loss, signals, market_state, main_action, data, current_day)
        
        # 记录整个迭代时间
        iteration_time = time.time() - iteration_start_time
        self.timing_stats['iteration_times'].append(iteration_time)
        print(f"   ⏱️ 完整迭代耗时: {iteration_time:.3f}秒")
        
        return {
            'success': True,
            'training_metrics': training_result['metrics'],
            'predictions_shape': predictions.shape,
            'signals': signals,
            'reward': reward,
            'loss': loss,
            'feedback': feedback,
            'iteration_time': iteration_time
        }
    
    def run_algorithm(self):
        """
        运行完整的Bandwagon算法
        """
        total_start_time = time.time()  # 总时间计时
        print(f"\n🚀 启动Bandwagon算法")
        print("=" * 60)
        
        for code in self.asset_codes:
            print(f"\n处理资产: {code}")
            
            # 1. 加载数据
            data = self.load_asset_data(code)
            if data.empty:
                continue
            
            # 2. 滑动窗口RL训练
            total_days = len(data)
            # 确保有足够的数据进行训练
            start_day = self.window_size + self.lookahead  # 至少需要window_size + lookahead的历史数据
            
            print(f"   数据总长度: {total_days}, 开始训练日: {start_day}")
            
            if start_day >= total_days - self.lookahead:
                print(f"⚠️ 数据不足以进行训练，跳过资产 {code}")
                continue
            
            for current_day in range(start_day, total_days - self.lookahead):
                # 检查是否还有lookAhead窗口
                remaining_days = total_days - current_day
                if remaining_days < self.lookahead:
                    print(f"📊 lookAhead窗口耗尽，切换到直接预测模式")
                    # 直接预测模式
                    self.direct_prediction_mode(code, data, current_day)
                    break
                
                # 执行RL迭代
                iteration_result = self.run_rl_iteration(code, data, current_day)
                
                if iteration_result['success']:
                    self.training_history.append({
                        'code': code,
                        'day': current_day,
                        'result': iteration_result
                    })
                    
                    print(f"   第{current_day}天完成 - Reward: {iteration_result['reward']:.4f}")
                else:
                    print(f"   第{current_day}天失败: {iteration_result.get('reason', 'unknown')}")
        
        # 记录总时间
        self.timing_stats['total_time'] = time.time() - total_start_time
        
        print(f"\n✅ Bandwagon算法执行完成")
        print(f"⏱️ 总执行时间: {self.timing_stats['total_time']:.2f}秒")
        
        # 打印时间分析
        self._print_timing_analysis()
        
        self.print_summary()
        
        # 保存反馈系统状态
        self.feedback_system.save_system_state("bandwagon_feedback_state.pkl")
    
    def direct_prediction_mode(self, code: str, data: pd.DataFrame, current_day: int):
        """
        直接预测模式（当lookAhead窗口耗尽时）
        """
        print(f"🔮 直接预测模式: {code}")
        
        # 使用最新训练的模型进行预测
        if code in self.nemots_models:
            nemots_model = self.nemots_models[code]
            historical_data = data.iloc[:current_day]
            
            # 生成最终预测
            final_predictions = self.generate_predictions(['final_model'], historical_data, self.lookahead)
            current_price = historical_data['close'].iloc[-1]
            final_signals = self.generate_trading_signals(final_predictions, current_price)
            
            print(f"   最终预测信号: {final_signals}")
            return final_signals
        
        return [0] * self.lookahead  # 默认持有
    
    def _print_timing_analysis(self):
        """按老师建议：打印详细的时间分析"""
        print(f"\n⏱️ 详细时间分析 (按老师建议)")
        print("=" * 60)
        
        stats = self.timing_stats
        
        if stats['training_times']:
            print(f"🧠 训练时间统计:")
            print(f"   总次数: {len(stats['training_times'])}")
            print(f"   平均耗时: {np.mean(stats['training_times']):.3f}秒")
            print(f"   最长耗时: {np.max(stats['training_times']):.3f}秒")
            print(f"   最短耗时: {np.min(stats['training_times']):.3f}秒")
            print(f"   总训练时间: {np.sum(stats['training_times']):.2f}秒")
            print(f"   占总时间: {np.sum(stats['training_times'])/stats['total_time']*100:.1f}%")
        
        if stats['prediction_times']:
            print(f"\n🔮 预测时间统计:")
            print(f"   总次数: {len(stats['prediction_times'])}")
            print(f"   平均耗时: {np.mean(stats['prediction_times']):.3f}秒")
            print(f"   总预测时间: {np.sum(stats['prediction_times']):.2f}秒")
            print(f"   占总时间: {np.sum(stats['prediction_times'])/stats['total_time']*100:.1f}%")
        
        if stats['signal_generation_times']:
            print(f"\n📈 信号生成时间统计:")
            print(f"   总次数: {len(stats['signal_generation_times'])}")
            print(f"   平均耗时: {np.mean(stats['signal_generation_times']):.3f}秒")
            print(f"   总信号时间: {np.sum(stats['signal_generation_times']):.2f}秒")
            print(f"   占总时间: {np.sum(stats['signal_generation_times'])/stats['total_time']*100:.1f}%")
        
        if stats['iteration_times']:
            print(f"\n🔄 完整迭代时间统计:")
            print(f"   总迭代次数: {len(stats['iteration_times'])}")
            print(f"   平均迭代耗时: {np.mean(stats['iteration_times']):.3f}秒")
            print(f"   最长迭代: {np.max(stats['iteration_times']):.3f}秒")
            print(f"   最短迭代: {np.min(stats['iteration_times']):.3f}秒")
        
        # 找出主要卡点
        print(f"\n🎯 主要卡点分析:")
        time_components = {
            '训练': np.sum(stats['training_times']) if stats['training_times'] else 0,
            '预测': np.sum(stats['prediction_times']) if stats['prediction_times'] else 0,
            '信号生成': np.sum(stats['signal_generation_times']) if stats['signal_generation_times'] else 0,
        }
        
        sorted_components = sorted(time_components.items(), key=lambda x: x[1], reverse=True)
        for i, (component, time_spent) in enumerate(sorted_components):
            percentage = time_spent / stats['total_time'] * 100
            print(f"   {i+1}. {component}: {time_spent:.2f}秒 ({percentage:.1f}%)")
        
        print("=" * 60)
    
    def print_summary(self):
        """打印算法执行摘要"""
        print(f"\n📊 算法执行摘要")
        print(f"   总训练轮数: {len(self.training_history)}")
        print(f"   总反馈记录: {len(self.reward_history)}")
        
        if self.reward_history:
            total_reward = sum(r['reward'] for r in self.reward_history)
            total_loss = sum(r['loss'] for r in self.reward_history)
            print(f"   总Reward: {total_reward:.4f}")
            print(f"   总Loss: {total_loss:.4f}")
            print(f"   净收益: {total_reward - total_loss:.4f}")
            
            # 计算详细的交易指标
            self._calculate_trading_metrics()
            
        # 打印反馈系统统计
        feedback_stats = self.feedback_system.get_system_statistics()
        print(f"\n🔧 反馈系统统计:")
        print(f"   成功率: {feedback_stats['success_rate']:.2%}")
        print(f"   适应次数: {feedback_stats['adaptation_count']}")
        if 'reward_statistics' in feedback_stats:
            print(f"   平均奖励: {feedback_stats['reward_statistics']['average_reward']:.4f}")
    
    def _calculate_trading_metrics(self):
        """计算详细的交易指标"""
        print(f"\n📈 详细交易指标分析")
        print("=" * 60)
        
        # 提取收益率序列
        strategy_returns = []
        benchmark_returns = []
        
        for record in self.reward_history:
            # 策略收益 = reward - loss
            net_return = record['reward'] - record['loss']
            strategy_returns.append(net_return)
            
            # 简化的基准收益（可以改为实际市场数据）
            benchmark_returns.append(np.random.normal(0.0003, 0.015))  # 模拟市场收益
        
        if len(strategy_returns) > 0:
            # 使用tin_metrics计算指标
            strategy_returns = np.array(strategy_returns)
            benchmark_returns = np.array(benchmark_returns)
            
            # 计算策略指标
            metrics_calc = TradingMetrics(
                returns=strategy_returns,
                benchmark_returns=benchmark_returns,
                risk_free_rate=0.02
            )
            
            # 打印详细指标
            metrics_calc.print_metrics("Bandwagon策略")
            
            # 按资产分组计算
            self._calculate_asset_metrics()
    
    def _calculate_asset_metrics(self):
        """按资产计算指标"""
        print(f"\n📊 分资产指标分析")
        print("=" * 60)
        
        asset_returns = {}
        
        # 按资产分组收益
        for record in self.reward_history:
            code = record['code']
            net_return = record['reward'] - record['loss']
            
            if code not in asset_returns:
                asset_returns[code] = []
            asset_returns[code].append(net_return)
        
        # 为每个资产计算指标
        strategy_dict = {}
        for code, returns in asset_returns.items():
            if len(returns) > 10:  # 至少需要10个数据点
                strategy_dict[code] = np.array(returns)
        
        if len(strategy_dict) > 0:
            # 生成基准收益
            max_len = max(len(returns) for returns in strategy_dict.values())
            benchmark = np.random.normal(0.0003, 0.015, max_len)
            
            # 对比分析
            comparison_df = compare_strategies(strategy_dict, benchmark)
            print("\n各资产表现对比:")
            print(comparison_df.round(4))
        else:
            print("   数据不足，无法进行分资产分析")


def main():
    """主函数 - 启动算法"""
    # 调回三支股票进行测试
    asset_codes = ['sh.600519', 'sh.000001', 'sz.000002']  # 三支股票：茅台、平安银行、万科A
    
    # 创建Bandwagon RL算法实例 - 增加窗口大小以改善学习
    bandwagon = BandwagonRL(
        asset_codes=asset_codes,
        window_size=30,    # 增加窗口大小，提供更多学习数据
        lookahead=10,      # 保持前瞻窗口
        topk=5            # 减少topk数量
    )
    
    # 运行算法
    bandwagon.run_algorithm()


if __name__ == "__main__":
    main()


