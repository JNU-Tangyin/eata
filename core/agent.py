import numpy as np
import pandas as pd
import torch
from typing import Optional, Dict, Any, List, Tuple
from scipy.stats import wasserstein_distance

# 导入新的NEMoTS核心模块
from core.eata_agent.engine import Engine
from core.eata_agent.args import Args

# 导入RL反馈系统
from rl import IntegratedRLFeedbackSystem

class Agent:
    def __init__(self, df: pd.DataFrame, lookback: int = 100, lookahead: int = 20, stride: int = 1, depth: int = 300):
        self.stock_list = df
        self.lookback = lookback
        self.lookahead = lookahead
        self.stride = stride
        self.depth = depth
        self.hyperparams = self._create_hyperparams()
        self.engine = Engine(self.hyperparams)
        self.previous_best_tree = None
        self.previous_best_expression = None
        self.is_trained = False
        self.training_history = []
        self.__name__ = 'EATA_Agent_v3.1_RL_Enhanced'
        
        # 初始化RL反馈系统
        self.rl_feedback_system = IntegratedRLFeedbackSystem()
        
        # RL相关状态追踪
        self.episode_count = 0
        self.last_market_state = None
        self.last_action = None
        self.last_reward = None
        self.last_loss = None

        print("EATA Agent (RL增强模式) 初始化完成")
        print(f"   - Lookback={self.lookback}, Lookahead={self.lookahead}")
        print(f"   - Stride={self.stride}, Depth={self.depth}")
        print("   - 决策规则: 固定 Q25/Q75 共识规则 + RL反馈增强")
        print("   - RL反馈系统: 已激活")

    def _create_hyperparams(self) -> Args:
        """创建超参数配置 - 增强版"""
        args = Args()
        # 优先使用GPU进行加速
        if torch.cuda.is_available():
            args.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            args.device = torch.device("mps")
        else:
            args.device = torch.device("cpu")
        args.seed = 42
        args.seq_in = self.lookback
        args.seq_out = self.lookahead
        args.stride = self.stride
        args.depth = self.depth
        args.used_dimension = 1
        args.features = 'M'
        args.symbolic_lib = "NEMoTS"
        args.max_len = 35
        args.max_module_init = 10
        args.num_transplant = 5
        args.num_runs = 5
        args.eta = 1.0
        args.num_aug = 3
        args.exploration_rate = 1 / np.sqrt(2)
        args.transplant_step = 800
        args.norm_threshold = 1e-5
        args.epoch = 10
        args.round = 2
        args.train_size = 32  # 减少训练阈值，让训练更频繁发生
        args.lr = 1e-5
        args.weight_decay = 0.0001
        args.clip = 5.0
        args.buffer_size = 128
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        return args

    def _prepare_data(self, df: pd.DataFrame) -> np.ndarray:
        """准备单个滑动窗口的数据"""
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        if not all(col in df.columns for col in feature_cols):
            raise ValueError(f"输入数据缺少必要列: 需要 {feature_cols}")
        
        data = df[feature_cols].values
        diff = np.diff(data, axis=0)
        last_row = data[:-1]
        last_row[last_row == 0] = 1e-9
        change_rates = diff / last_row
        
        change_rates[:, :4] = np.clip(change_rates[:, :4], -0.1, 0.1)
        change_rates[:, 4:] = np.clip(change_rates[:, 4:], -0.5, 0.5)

        if len(change_rates) < self.lookback + self.lookahead:
            raise ValueError(f"数据长度不足：需要{self.lookback + self.lookahead}，实际可用{len(change_rates)}")
        
        window_data = change_rates[-(self.lookback + self.lookahead):]
        return window_data

    def _predict_distribution(self, top_10_exps: List[str], lookback_data: np.ndarray) -> np.ndarray:
        """为Top-10表达式生成未来预测分布"""
        all_predictions = []
        lookback_data_transposed = lookback_data.T

        eval_vars = {"np": np}
        for i in range(lookback_data_transposed.shape[0]):
            eval_vars[f'x{i}'] = lookback_data_transposed[i, :]

        for exp in top_10_exps:
            try:
                corrected_expression = exp.replace("exp", "np.exp").replace("cos", "np.cos").replace("sin", "np.sin").replace("sqrt", "np.sqrt").replace("log", "np.log")
                historical_fit = eval(corrected_expression, {"__builtins__": None}, eval_vars)

                if not isinstance(historical_fit, np.ndarray) or historical_fit.ndim == 0:
                    historical_fit = np.repeat(historical_fit, self.lookback)
                
                time_axis = np.arange(self.lookback)
                coeffs = np.polyfit(time_axis, historical_fit, 1)
                trend_line = np.poly1d(coeffs)

                future_time_axis = np.arange(self.lookback, self.lookback + self.lookahead)
                future_predictions = trend_line(future_time_axis)
                all_predictions.extend(future_predictions)

            except Exception as e:
                print(f"表达式 '{exp}' 预测失败: {e}。使用简单趋势预测。")
                # 使用简单的价格趋势而不是填充0
                if len(lookback_data_transposed) > 0:
                    # 使用收盘价的简单线性趋势
                    close_prices = lookback_data_transposed[3, :]  # 假设第4列是收盘价
                    time_axis = np.arange(len(close_prices))
                    if len(close_prices) > 1:
                        coeffs = np.polyfit(time_axis, close_prices, 1)
                        trend_line = np.poly1d(coeffs)
                        future_time_axis = np.arange(len(close_prices), len(close_prices) + self.lookahead)
                        future_predictions = trend_line(future_time_axis)
                        # 转换为收益率
                        if len(close_prices) > 0:
                            last_price = close_prices[-1]
                            returns = (future_predictions - last_price) / last_price
                            all_predictions.extend(returns)
                        else:
                            all_predictions.extend([0.001] * self.lookahead)  # 小的正收益率
                    else:
                        all_predictions.extend([0.001] * self.lookahead)  # 小的正收益率
                else:
                    all_predictions.extend([0.001] * self.lookahead)  # 小的正收益率
        
        return np.array(all_predictions)

    def _calculate_rl_reward_and_signal(self, prediction_distribution: np.ndarray, lookahead_ground_truth: np.ndarray, shares_held: int) -> Tuple[float, int]:
        """
        计算RL奖励和交易信号
        - RL奖励: 基于预测分布与真实分布的瓦瑟斯坦距离。
        - 交易信号: 基于固定的Q25/Q75规则。
        """
        try:
            if prediction_distribution.size == 0:
                return 0.0, 0

            # 交易信号决策
            strategy = [25, 75]
            q_low, q_high = np.percentile(prediction_distribution, strategy)
            
            print(f"  [调试] 预测分布: min={prediction_distribution.min():.6f}, max={prediction_distribution.max():.6f}")
            print(f"  [调试] Q25={q_low:.6f}, Q75={q_high:.6f}, median={np.median(prediction_distribution):.6f}")
            
            intended_signal = 0
            if q_low > 0:
                intended_signal = 1
                print(f"  [决策] 预测分布的 25% 分位数 > 0，生成意图信号: 买入")
            elif q_high < 0:
                intended_signal = -1
                print(f"  [决策] 预测分布的 75% 分位数 < 0，生成意图信号: 卖出")
            else:
                if prediction_distribution.min() >= 0:
                    median_val = np.median(prediction_distribution)
                    threshold = (prediction_distribution.max() - prediction_distribution.min()) * 0.3
                    if median_val > threshold:
                        intended_signal = 1
                        print(f"  [决策] 全正分布，中位数{median_val:.6f} > 阈值{threshold:.6f}，生成意图信号: 买入")
                    else:
                        print(f"  [决策] 全正分布，中位数{median_val:.6f} <= 阈值{threshold:.6f}，生成意图信号: 持有")
                else:
                    print("  [决策] 预测分布跨越零点，信号不明确，生成意图信号: 持有")

            # RL奖励计算
            actual_returns = lookahead_ground_truth.T[3, :] 
            
            # 调试信息：检查输入数据
            print(f"  [RL调试] 预测分布形状: {prediction_distribution.shape}, 范围: [{prediction_distribution.min():.6f}, {prediction_distribution.max():.6f}]")
            print(f"  [RL调试] 真实收益形状: {actual_returns.shape}, 范围: [{actual_returns.min():.6f}, {actual_returns.max():.6f}]")
            
            # 检查输入数据有效性
            if len(prediction_distribution) == 0 or len(actual_returns) == 0:
                print(f"  ⚠️ 空的输入数据，返回默认RL奖励0.0")
                return 0.0, intended_signal
                
            if np.all(np.isnan(prediction_distribution)) or np.all(np.isnan(actual_returns)):
                print(f"  ⚠️ 输入数据全为nan，返回默认RL奖励0.0")
                return 0.0, intended_signal
            
            # 检查是否使用简单距离函数（EATA-Simple变体）
            if hasattr(self, '_variant_distance_function') and self._variant_distance_function == 'simple_mae':
                # 简单MAE距离：使用收益差的平均绝对误差
                distance = np.mean(np.abs(prediction_distribution - actual_returns))
                print(f"  [简单RL调试] MAE距离: {distance}")
            else:
                # 默认使用Wasserstein距离
                distance = wasserstein_distance(prediction_distribution, actual_returns)
                print(f"  [RL调试] 瓦瑟斯坦距离: {distance}")
            
            # 处理异常的距离值
            if np.isnan(distance) or np.isinf(distance):
                print(f"  ⚠️ 异常的瓦瑟斯坦距离: {distance}")
                print(f"  [诊断] 预测分布统计: mean={np.mean(prediction_distribution):.6f}, std={np.std(prediction_distribution):.6f}")
                print(f"  [诊断] 真实收益统计: mean={np.mean(actual_returns):.6f}, std={np.std(actual_returns):.6f}")
                return 0.0, intended_signal
            elif distance < 0:
                print(f"  ⚠️ 负的瓦瑟斯坦距离: {distance}，这不应该发生")
                return 0.0, intended_signal
            
            rl_reward = 1 / (1 + distance)
            print(f"  [RL调试] 计算的RL奖励: {rl_reward:.6f}")
            
            # 最终检查
            if np.isnan(rl_reward) or np.isinf(rl_reward):
                print(f"  ⚠️ 最终RL奖励异常: {rl_reward}，返回0.0")
                rl_reward = 0.0
            
            return rl_reward, intended_signal
        except Exception as e:
            print(f"--- 🚨 在 _calculate_rl_reward_and_signal 中捕获到致命错误 🚨 ---")
            print(f"错误信息: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, 0

    def _process_rl_feedback(self, rl_reward: float, mae: float, trading_signal: int, lookback_data: np.ndarray):
        """
        处理RL反馈 - 师弟建议的闭环机制
        """
        try:
            # 更新episode计数
            self.episode_count += 1
            
            # 准备市场状态特征（简化版，10维）
            market_state = np.zeros(10)
            if lookback_data.size > 0:
                # 使用最近的价格变化作为市场状态特征
                recent_data = lookback_data[-10:] if len(lookback_data) >= 10 else lookback_data
                market_state[:len(recent_data.flatten()[:10])] = recent_data.flatten()[:10]
            
            # 准备上下文信息
            context = {
                'mae': mae,
                'episode_count': self.episode_count,
                'market_volatility': np.std(lookback_data) if lookback_data.size > 0 else 0.0,
                'prediction_confidence': 1.0 / (1.0 + mae) if mae > 0 else 1.0
            }
            
            # 处理RL反馈
            feedback_result = self.rl_feedback_system.process_episode_feedback(
                code=f"EATA_Episode_{self.episode_count}",
                reward=rl_reward,
                loss=mae,  # 使用MAE作为loss
                market_state=market_state,
                action=trading_signal,
                context=context
            )
            
            # 应用反馈到NEMoTS超参数（师弟建议的核心功能）
            if 'loss_processing' in feedback_result and 'nemots' in feedback_result['loss_processing']:
                nemots_updates = feedback_result['loss_processing']['nemots']
                self._apply_nemots_feedback(nemots_updates)
            
            print(f"🔧 RL反馈处理完成 - Episode {self.episode_count}")
            print(f"   净收益: {feedback_result.get('net_outcome', 0):.4f}")
            print(f"   适应类型: {feedback_result.get('system_adaptation', {}).get('type', 'unknown')}")
            
        except Exception as e:
            print(f"⚠️ RL反馈处理失败: {e}")

    def _apply_nemots_feedback(self, nemots_updates: Dict[str, Any]):
        """
        应用RL反馈到NEMoTS超参数 - 师弟建议的核心功能
        """
        try:
            print(f"🎯 应用NEMoTS参数调整...")
            
            # 应用探索率调整
            if 'exploration_rate_multiplier' in nemots_updates:
                old_rate = self.hyperparams.exploration_rate
                self.hyperparams.exploration_rate *= nemots_updates['exploration_rate_multiplier']
                self.hyperparams.exploration_rate = np.clip(self.hyperparams.exploration_rate, 0.1, 2.0)
                print(f"   探索率: {old_rate:.3f} -> {self.hyperparams.exploration_rate:.3f}")
            
            # 应用学习率调整
            if 'learning_rate_multiplier' in nemots_updates:
                old_lr = self.hyperparams.lr
                self.hyperparams.lr *= nemots_updates['learning_rate_multiplier']
                self.hyperparams.lr = np.clip(self.hyperparams.lr, 1e-6, 1e-3)
                print(f"   学习率: {old_lr:.6f} -> {self.hyperparams.lr:.6f}")
            
            # 应用运行次数调整
            if 'num_runs_multiplier' in nemots_updates:
                old_runs = self.hyperparams.num_runs
                self.hyperparams.num_runs = int(self.hyperparams.num_runs * nemots_updates['num_runs_multiplier'])
                self.hyperparams.num_runs = np.clip(self.hyperparams.num_runs, 1, 10)
                print(f"   运行次数: {old_runs} -> {self.hyperparams.num_runs}")
            
            # 更新引擎参数
            if hasattr(self.engine, 'model'):
                self.engine.model.exploration_rate = self.hyperparams.exploration_rate
        
        except Exception as e:
            print(f"NEMoTS参数调整失败: {e}")

    def criteria(self, test_df: pd.DataFrame, shares_held: int = 0) -> Tuple[int, float]:
        """
        核心决策方法 - 增强版
        """
        try:
            # 使用滑动窗口NEMoTS进行预测
            from sliding_window_nemots import SlidingWindowNEMoTS
            
            # 检查是否有变体参数需要传递
            variant_kwargs = {}
            if hasattr(self, '_variant_alpha'):
                variant_kwargs['alpha'] = self._variant_alpha
            if hasattr(self, '_variant_num_transplant'):
                variant_kwargs['num_transplant'] = self._variant_num_transplant
            if hasattr(self, '_variant_num_aug'):
                variant_kwargs['num_aug'] = self._variant_num_aug
            if hasattr(self, '_variant_exploration_rate'):
                variant_kwargs['exploration_rate'] = self._variant_exploration_rate
            
            # 🔧 方案1：检查Engine上的直接传递参数
            if hasattr(self.engine, '_variant_exploration_rate'):
                variant_kwargs['exploration_rate'] = self.engine._variant_exploration_rate
                print(f"🔧 [方案1] 从Engine获取exploration_rate: {self.engine._variant_exploration_rate}")
            
            nemots = SlidingWindowNEMoTS(
                lookback=self.lookback,
                lookahead=self.lookahead,
                stride=self.stride,
                depth=self.depth,
                previous_best_tree=getattr(self, '_previous_best_tree', None),
                external_engine=self.engine,
                **variant_kwargs
            )

            # 准备数据
            full_window_data = self._prepare_data(test_df)
            lookback_data = full_window_data[:self.lookback, :]
            lookahead_data = full_window_data[-self.lookahead:, :]

            # 运行NEMoTS
            result = nemots.sliding_fit(test_df)
            
            # 提取结果
            best_exp = result.get('best_expression', '0')
            top_10_exps = result.get('top_10_expressions', ['0'] * 10)
            mae = result.get('mae', 0.0)
            mcts_score = result.get('mcts_score', 0.0)
            new_best_tree = result.get('best_tree', None)

            # 保存状态
            self._previous_best_tree = new_best_tree
            self.is_trained = True
            
            record = {'mae': mae, 'mcts_score': mcts_score}
            self.training_history.append(record)
            print(f"NEMoTS运行完成: MAE={mae:.4f}, MCTS Score={mcts_score:.4f}")

            # 生成预测分布
            prediction_distribution = self._predict_distribution(top_10_exps, lookback_data)
            print(f"生成了 {len(prediction_distribution)} 个预测点。")

            # 计算RL奖励和交易信号
            rl_reward, trading_signal = self._calculate_rl_reward_and_signal(
                prediction_distribution, lookahead_data, shares_held
            )
            print(f"RL奖励 (基于真实信号): {rl_reward:.4f}, 意图交易信号: {trading_signal}")

            # RL反馈处理
            self._process_rl_feedback(rl_reward, mae, trading_signal, lookback_data)

            return trading_signal, rl_reward

        except Exception as e:
            print(f"NEMoTS Agent 'criteria' 失败: {e}")
            import traceback
            traceback.print_exc()
            return 0, 0

    # choose_action, vote, strength 方法保持不变
    @classmethod
    def choose_action(cls, s: tuple) -> int:
        try:
            _, s1, _, _ = s
            temp_agent = Agent(pd.DataFrame())
            action, _ = temp_agent.criteria(s1, shares_held=0)
            return action
        except Exception as e:
            print(f"动作选择失败: {e}")
            return 0

    def vote(self) -> int:
        print("'vote' 方法被简化，仅返回中性信号。请在 predict.py 中实现多股票循环。")
        return 50

    def strength(self, w1: float, w2: float, w3: float, w4: float) -> pd.Series:
        print("'strength' 方法被简化，返回固定值。")
        self.stock_list['strength'] = [50] * len(self.stock_list)
        return self.stock_list['strength']
