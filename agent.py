import numpy as np
import pandas as pd
import torch
from typing import Optional, Dict, Any

# 导入新的NEMoTS核心模块
from eata_agent.engine import Engine
from eata_agent.args import Args

class Agent:
    def __init__(self, df: pd.DataFrame, lookback: int = 20, lookahead: int = 5):
        """
        新版 NEMoTS Agent
        @param df: 股票列表 (在当前设计中暂未使用，但保留接口)
        @param lookback: 训练回看窗口大小
        @param lookahead: 预测窗口大小
        """
        self.stock_list = df
        self.lookback = lookback
        self.lookahead = lookahead

        # 1. 创建超参数配置
        self.hyperparams = self._create_hyperparams()

        # 2. 初始化核心引擎
        self.engine = Engine(self.hyperparams)

        # 3. 语法树继承机制
        self.previous_best_tree = None
        self.previous_best_expression = None

        # 4. 训练状态
        self.is_trained = False
        self.training_history = []
        
        self.__name__ = 'EATA_Agent_v2'
        print("🤖 新版 EATA Agent 初始化完成")
        print(f"   Lookback={self.lookback}, Lookahead={self.lookahead}")

    def _create_hyperparams(self) -> Args:
        """创建超参数配置"""
        args = Args()
        args.device = torch.device("cpu")
        args.seed = 42
        args.seq_in = self.lookback
        args.seq_out = self.lookahead
        args.used_dimension = 1
        args.features = 'M'
        args.symbolic_lib = "NEMoTS"
        args.max_len = 25
        args.max_module_init = 10
        # 重量级默认参数 (冷启动)
        args.num_transplant = 5
        args.num_runs = 2
        args.eta = 1.0
        args.num_aug = 5
        args.exploration_rate = 1 / np.sqrt(2)
        args.transplant_step = 500 # 重量级
        args.norm_threshold = 1e-5
        args.epoch = 10
        args.round = 2
        args.train_size = 64
        args.lr = 1e-5
        args.weight_decay = 0.0001
        args.clip = 5.0
        args.buffer_size = 64
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        return args

    def _prepare_data(self, df: pd.DataFrame) -> np.ndarray:
        """准备单个滑动窗口的数据，使用变化率进行标准化"""
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        if not all(col in df.columns for col in feature_cols):
            raise ValueError(f"输入数据缺少必要列: 需要 {feature_cols}")
        
        data = df[feature_cols].values
        # 使用diff和clip安全地计算变化率
        diff = np.diff(data, axis=0)
        last_row = data[:-1]
        # 防止除以零
        last_row[last_row == 0] = 1e-9
        change_rates = diff / last_row
        
        # 对价格和成交量/额应用不同的clip
        change_rates[:, :4] = np.clip(change_rates[:, :4], -0.1, 0.1) # 价格
        change_rates[:, 4:] = np.clip(change_rates[:, 4:], -0.5, 0.5) # 量、额

        if len(change_rates) < self.lookback + self.lookahead:
            raise ValueError(f"数据长度不足：需要{self.lookback + self.lookahead}，实际可用{len(change_rates)}")
        
        # 取最后一个窗口
        window_data = change_rates[-(self.lookback + self.lookahead):]
        return window_data

    def criteria(self, d: pd.DataFrame) -> int:
        """
        核心决策函数：运行NEMoTS并生成交易信号
        @input d: window_size的df
        @output: 交易信号 1(买入)/-1(卖出)/0(持有)
        """
        try:
            # 1. 动态调整参数 (冷/热启动)
            if self.previous_best_tree is not None:
                print("检测到已有语法树，切换到轻量化参数...")
                self.engine.model.num_transplant = 2
                self.engine.model.transplant_step = 100
                self.engine.model.num_aug = 2
            else:
                print("首次运行，使用重量级参数...")
                self.engine.model.num_transplant = 5
                self.engine.model.transplant_step = 500
                self.engine.model.num_aug = 5

            # 2. 准备数据
            window_data = self._prepare_data(d)

            # 3. 运行引擎
            print("调用核心引擎 engine.simulate...")
            best_exp, _, _, loss, mae, mse, corr, _, reward, new_best_tree = self.engine.simulate(
                window_data, previous_best_tree=self.previous_best_tree
            )

            # 4. 保存状态用于下一次继承
            self.previous_best_expression = str(best_exp)
            self.previous_best_tree = new_best_tree
            self.is_trained = True
            
            # 5. 记录历史
            record = {'mae': mae, 'corr': corr, 'reward': reward}
            self.training_history.append(record)
            print(f"NEMoTS运行完成: MAE={mae:.4f}, Corr={corr:.4f}, Reward={reward:.4f}")

            # 6. 根据结果生成信号
            if mae < 0.01 and not np.isnan(corr):
                if corr > 0.1: return 1
                if corr < -0.1: return -1
            elif reward > 0.6:
                return 1
            elif reward < 0.4:
                return -1
            return 0

        except Exception as e:
            print(f"⚠️ NEMoTS Agent 'criteria' 失败: {e}")
            return 0 # 出错时返回持有

    @classmethod
    def choose_action(cls, s: tuple) -> int:
        """RL兼容接口, 直接调用criteria"""
        try:
            _, s1, _, _ = s # s1是股票日线数据
            # 注意：这里每次都创建一个新的Agent实例，无法实现语法树继承。
            # 这是一个待优化的点，在真实RL环境中需要一个持久化的Agent。
            temp_agent = Agent(pd.DataFrame()) 
            return temp_agent.criteria(s1)
        except Exception as e:
            print(f"⚠️ 动作选择失败: {e}")
            return 0

    def vote(self) -> int:
        """(简化)对ETF总体信号进行投票"""
        # 在实际应用中，这里应该循环多支股票并综合其criteria结果
        print("⚠️ 'vote' 方法被简化，仅返回中性信号。请在 predict.py 中实现多股票循环。")
        return 50

    def strength(self, w1: float, w2: float, w3: float, w4: float) -> pd.Series:
        """(简化)计算股票强度"""
        print("⚠️ 'strength' 方法被简化，返回固定值。")
        self.stock_list['strength'] = [50] * len(self.stock_list)
        return self.stock_list['strength']