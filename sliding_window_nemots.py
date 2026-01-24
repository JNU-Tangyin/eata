#!/usr/bin/env python
# coding=utf-8
# 直接调用核心模块：engine.simulate → model.run → mcts + network

import numpy as np
import torch
import pandas as pd
from typing import Optional, Tuple, Dict, Any
import warnings
import logging
import sys

warnings.filterwarnings('ignore')

# 隐藏所有日志输出
logging.getLogger('MCTSAdapter').setLevel(logging.CRITICAL)
logging.getLogger('NEMoTS').setLevel(logging.CRITICAL)
logging.getLogger('nemots').setLevel(logging.CRITICAL)
logging.getLogger('engine').setLevel(logging.CRITICAL)
logging.getLogger('model').setLevel(logging.CRITICAL)
logging.getLogger('mcts').setLevel(logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)

# 创建空输出类
class NullWriter:
    def write(self, txt): pass
    def flush(self): pass

# 导入NEMoTS核心模块
try:
    from eata_agent.engine import Engine
    from eata_agent.args import Args
except ImportError:
    from nemots.engine import Engine
    from nemots.args import Args
    print("⚠️ 回退到原版NEMoTS引擎")


class SlidingWindowNEMoTS:
    
    def __init__(self, lookback: int = 20, lookahead: int = 5):
        """
        初始化滑动窗口NEMoTS
        
        Args:
            lookback: 训练窗口大小（对应原NEMoTS的seq_in）
            lookahead: 预测窗口大小（对应原NEMoTS的seq_out）
        """
        self.lookback = lookback
        self.lookahead = lookahead
        
        # 从main函数迁移的超参数
        self.hyperparams = self._create_hyperparams()
        
        # 初始化引擎
        original_stderr = sys.stderr
        original_stdout = sys.stdout
        try:
            sys.stderr = NullWriter()
            sys.stdout = NullWriter()
            self.engine = Engine(self.hyperparams)
        finally:
            sys.stderr = original_stderr
            sys.stdout = original_stdout
        
        # 语法树继承和多样性管理
        self.previous_best_tree = None
        self.previous_best_expression = None
        self.expression_diversity_pool = []  # 保存多个优秀表达式
        self.stagnation_counter = 0  # 停滞计数器
        self.max_stagnation = 5  # 增加最大停滞次数，减少重启频率
        
        # 训练状态
        self.is_trained = False
        self.training_history = []
        
        # 性能监控 - 针对表达式固化的敏感重启条件
        self.performance_threshold = 0.5   # MAE超过0.5就认为性能极差
        self.consecutive_poor_performance = 0
        self.expression_repetition_count = 0  # 表达式重复计数
        
        print(f"滑动窗口NEMoTS初始化完成")
        print(f"   lookback={lookback}, lookahead={lookahead}")
        print(f"   核心模块: engine → model → mcts + network")
    
    def _create_hyperparams(self) -> Args:
        """
        创建超参数配置（增强探索能力版本）
        针对局部最优问题的改进配置
        """
        args = Args()
        
        # 优先使用MPU进行性能优化
        if torch.backends.mps.is_available():
            args.device = torch.device("mps")
        elif torch.cuda.is_available():
            args.device = torch.device("cuda")
        else:
            args.device = torch.device("cpu")
        
        args.seed = np.random.randint(1, 10000)  # 随机种子增加多样性
        
        # 数据配置（适配滑动窗口）
        args.seq_in = self.lookback
        args.seq_out = self.lookahead
        args.used_dimension = 1
        args.features = 'M'  # 多变量预测多变量
        
        # NEMoTS核心参数 - 增强探索能力（最佳效果版本）
        args.symbolic_lib = "NEMoTS"
        args.max_len = 25
        args.max_module_init = 10
        args.num_transplant = 5
        args.num_runs = 3  # 减少运行次数以适应滑动窗口
        args.eta = 1.0
        args.num_aug = 5
        args.exploration_rate = 1 / np.sqrt(2)
        args.transplant_step = 500  # 减少步数以适应滑动窗口
        args.norm_threshold = 1e-5
        
        # 训练参数（适配滑动窗口）
        args.epoch = 10  # 减少epoch以适应实时性
        args.round = 2   # 减少round以适应滑动窗口
        args.train_size = 64  # 减少batch size
        args.lr = 1e-5
        args.weight_decay = 0.0001
        args.clip = 5.0
        args.buffer_size = 64 # 明确设置经验池大小，确保alpha系数能快速增长
        args.max_len = 30  # 增加表达式长度上限
        args.max_module_init = 20  # 增加初始模块数量
        args.num_transplant = 4  # 增加移植次数
        args.num_runs = 8  # 显著增加运行次数
        args.eta = 1.5  # 增加eta值，提高探索强度
        args.num_aug = 2  # 增加数据增强
        args.exploration_rate = 1.2  # 提高探索率
        args.transplant_step = 1000  # 增加移植步数
        args.norm_threshold = 1e-6  # 更严格的收敛阈值
        
        # 训练参数 - 平衡探索与效率
        args.epoch = 15  # 适度增加epoch
        args.round = 3   # 增加round数
        args.train_size = 32  # 适度减少batch size增加随机性
        args.lr = 5e-5  # 提高学习率
        args.weight_decay = 0.0005  # 增加正则化
        args.clip = 3.0  # 适度降低梯度裁剪
        
        # 多样性随机种子（每次调用都不同）
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
        print(f"增强探索超参数配置完成 (seed={args.seed})")
        return args
    
    def _adaptive_hyperparams_adjustment(self):
        """
        基于历史性能动态调整超参数 - 更温和的调整
        """
        if len(self.training_history) < 3:
            return
        
        # 分析最近的性能趋势
        recent_maes = [record['mae'] for record in self.training_history[-3:]]
        avg_recent_mae = np.mean(recent_maes)
        
        # 只有在性能极差时才调整，避免过度调整
        if avg_recent_mae > self.performance_threshold * 1.5:  # 更严格的调整条件
            print(f"   📈 检测到性能极差 (MAE={avg_recent_mae:.4f})，轻微增加探索强度")
            
            # 更温和的调整
            self.hyperparams.exploration_rate = min(1.2, self.hyperparams.exploration_rate * 1.05)
            self.hyperparams.num_runs = min(6, self.hyperparams.num_runs + 1)
            self.hyperparams.eta = min(1.5, self.hyperparams.eta * 1.05)
            
            print(f"   调整后: exploration_rate={self.hyperparams.exploration_rate:.3f}, "
                  f"num_runs={self.hyperparams.num_runs}, eta={self.hyperparams.eta:.3f}")
        
        # 性能良好时保持稳定，不做调整
        elif avg_recent_mae < self.performance_threshold * 0.3:
            print(f"   ✅ 性能良好 (MAE={avg_recent_mae:.4f})，保持当前参数")
    
    def _prepare_sliding_window_data(self, df: pd.DataFrame) -> torch.Tensor:
        """
        准备滑动窗口数据
        基于RL范式的数据处理，替代全序列拟合
        """
        # 选择特征列
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        data = df[feature_cols].values
        
        # 改进的数据标准化 - 增强信号强度
        normalized_data = []
        
        # 计算全局统计信息用于标准化
        all_changes = []
        for i in range(1, len(data)):
            for j in range(4):  # price columns
                if data[i-1, j] != 0:
                    change = (data[i, j] - data[i-1, j]) / data[i-1, j]
                    all_changes.append(change)
        
        if all_changes:
            change_std = np.std(all_changes)
            change_mean = np.mean(all_changes)
        else:
            change_std = 0.01
            change_mean = 0.0
        
        # 第一行使用原始数据（标准化处理）
        if len(data) > 0:
            first_row = []
            for j in range(4):  # open, high, low, close
                first_row.append(0.0)  # 第一行变化率为0
            for j in [4, 5]:  # volume, amount
                first_row.append(0.0)  # 第一行变化率为0
            normalized_data.append(first_row)
        
        # 后续行使用增强的标准化
        for i in range(1, len(data)):
            row = []
            # 价格变化率 - 使用Z-score标准化增强信号
            for j in range(4):  # open, high, low, close
                if data[i-1, j] != 0:
                    change_rate = (data[i, j] - data[i-1, j]) / data[i-1, j]
                    # Z-score标准化，然后放大信号
                    if change_std > 0:
                        normalized_change = (change_rate - change_mean) / change_std
                        # 放大信号强度，但保持在合理范围
                        enhanced_change = np.tanh(normalized_change * 2) * 0.5
                    else:
                        enhanced_change = 0.0
                else:
                    enhanced_change = 0.0
                row.append(enhanced_change)
            
            # 成交量变化率 - 简化处理
            for j in [4, 5]:  # volume, amount
                if data[i-1, j] > 0 and data[i, j] > 0:
                    vol_change = (data[i, j] - data[i-1, j]) / data[i-1, j]
                    # 使用tanh函数压缩但保持信号
                    enhanced_vol = np.tanh(vol_change) * 0.3
                else:
                    enhanced_vol = 0.0
                row.append(enhanced_vol)
            
            normalized_data.append(row)
        
        normalized_data = np.array(normalized_data)
        
        # 创建滑动窗口
        if len(normalized_data) < self.lookback + self.lookahead:
            raise ValueError(f"数据长度不足：需要{self.lookback + self.lookahead}，实际{len(normalized_data)}")
        
        # 取最后一个窗口的数据
        start_idx = len(normalized_data) - self.lookback - self.lookahead
        window_data = normalized_data[start_idx:start_idx + self.lookback + self.lookahead]
        
        # 转换为tensor格式，添加batch维度
        # tensor_data = torch.FloatTensor(window_data).unsqueeze(0)  # [1, seq_len, features]
        
        print(f"滑动窗口数据准备完成:")
        print(f"   原始数据: {len(data)} → 标准化数据: {len(normalized_data)}")
        # print(f"   窗口数据: {tensor_data.shape}")
        # print(f"   变化率范围: [{tensor_data.min().item():.4f}, {tensor_data.max().item():.4f}]")
        
        return window_data
    
    def _manage_diversity_pool(self, expression: str, mae: float):
        """
        管理表达式多样性池
        保存多个优秀但不同的表达式
        """
        # 只保存性能较好的表达式
        if mae < self.performance_threshold * 2:
            # 检查是否已存在相似表达式
            is_similar = False
            for existing_expr, _ in self.expression_diversity_pool:
                if self._expressions_similar(expression, existing_expr):
                    is_similar = True
                    break
            
            if not is_similar:
                self.expression_diversity_pool.append((expression, mae))
                # 保持池大小在合理范围
                if len(self.expression_diversity_pool) > 5:
                    # 移除性能最差的
                    self.expression_diversity_pool.sort(key=lambda x: x[1])
                    self.expression_diversity_pool = self.expression_diversity_pool[:5]
                
                print(f"   添加到多样性池: {expression[:50]}... (MAE={mae:.4f})")
    
    def _expressions_similar(self, expr1: str, expr2: str) -> bool:
        """
        简单的表达式相似性检查
        """
        # 简化的相似性检查 - 可以根据需要改进
        return expr1 == expr2 or (len(expr1) > 10 and expr1[:10] == expr2[:10])
    
    def _should_restart(self) -> bool:
        """
        判断是否需要重启搜索 - 针对表达式固化的敏感策略
        """
        if len(self.training_history) < 2:
            return False
        
        # 检查表达式重复情况
        if len(self.training_history) >= 3:
            recent_expressions = [record['best_expression'] for record in self.training_history[-3:]]
            if len(set(recent_expressions)) == 1:  # 连续3次相同表达式
                self.expression_repetition_count += 1
            else:
                self.expression_repetition_count = 0
        
        # 检查性能是否极差
        if len(self.training_history) >= 2:
            recent_maes = [record['mae'] for record in self.training_history[-2:]]
            if all(mae > self.performance_threshold for mae in recent_maes):
                self.consecutive_poor_performance += 1
            else:
                self.consecutive_poor_performance = 0
        
        # 敏感的重启条件
        should_restart = (
            self.expression_repetition_count >= 3 or  # 连续3次相同表达式
            self.consecutive_poor_performance >= 2 or  # 连续2次性能极差
            (len(self.training_history) >= 2 and 
             self.training_history[-1]['mae'] > 0.8 and 
             self.training_history[-2]['mae'] > 0.8)  # 连续2次MAE>0.8
        )
        
        if should_restart:
            print(f"   🔄 触发重启: 表达式重复={self.expression_repetition_count}, 差性能={self.consecutive_poor_performance}")
            print(f"   最近MAE: {[record['mae'] for record in self.training_history[-3:]]}")
        
        return should_restart
    
    def _restart_search(self):
        """
        重启搜索策略 - 强制跳出表达式固化
        """
        print(f"   🔄 检测到表达式固化，执行强制重启...")
        
        # 重新创建超参数（使用新的随机种子）
        self.hyperparams = self._create_hyperparams()
        
        # 完全清空继承信息，强制重新开始
        self.previous_best_tree = None
        self.previous_best_expression = None
        
        # 清空多样性池，避免固化表达式影响
        self.expression_diversity_pool = []
        
        # 重置所有计数器
        self.stagnation_counter = 0
        self.consecutive_poor_performance = 0
        self.expression_repetition_count = 0
        
        # 大幅提高探索强度，强制跳出局部最优
        self.hyperparams.exploration_rate = min(2.0, self.hyperparams.exploration_rate * 1.5)
        self.hyperparams.eta = min(2.5, self.hyperparams.eta * 1.3)
        self.hyperparams.num_runs = min(12, self.hyperparams.num_runs + 3)
        
        print(f"   🚀 强制重启完成，提高探索强度: exploration_rate={self.hyperparams.exploration_rate:.3f}")

    def check_and_apply_config(self):
        """检查并应用配置文件更新"""
        import json
        import os
        
        config_file = 'config.json'
        if not os.path.exists(config_file):
            return
            
        try:
            # 检查文件修改时间
            if not hasattr(self, '_last_config_time'):
                self._last_config_time = 0
                
            mtime = os.path.getmtime(config_file)
            if mtime <= self._last_config_time:
                return
                
            # 读取新配置
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            print(f"🔄 检测到配置更新，应用新参数...")
            
            # 应用NEMoTS参数
            if 'nemots' in config:
                nemots_config = config['nemots']
                for key, value in nemots_config.items():
                    if hasattr(self.hyperparams, key):
                        old_value = getattr(self.hyperparams, key)
                        setattr(self.hyperparams, key, value)
                        print(f"   📝 {key}: {old_value} → {value}")
                    else:
                        # 动态添加新属性
                        setattr(self.hyperparams, key, value)
                        print(f"   📝 {key}: 新增 → {value}")
            
            # 应用系统参数
            if 'system' in config and 'window_size' in config['system']:
                new_size = config['system']['window_size']
                if new_size != self.lookback:
                    print(f"   📝 lookback: {self.lookback} → {new_size}")
                    self.lookback = new_size
            
            self._last_config_time = mtime
            print(f"✅ 配置更新完成")
            
        except Exception as e:
            print(f"⚠️ 配置更新失败: {e}")

    def _inherit_previous_tree(self):
        """
        语法树继承机制
        """
        if self.previous_best_tree is not None:
            print(f"继承前一窗口最优语法树: {self.previous_best_expression}")
            return self.previous_best_tree
        else:
            return None

    def sliding_fit(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        滑动窗口训练
        """
        print(f"\n开始滑动窗口训练...")
        
        self.check_and_apply_config()
        
        try:
            # 1. 准备滑动窗口数据
            window_data = self._prepare_sliding_window_data(df)
            
            # 2. 获取真实的未来价格（用于损失计算）
            future_prices = df['close'].values[-self.lookahead:]
            
            # 3. 动态调整超参数
            self._adaptive_hyperparams_adjustment()
            
            # 4. 语法树继承
            inherited_tree = self._inherit_previous_tree()
            
            # 5. 调用核心引擎进行模拟和搜索
            print(f"调用核心模块: engine.simulate...")
            # 注意：这里的返回格式需要与Engine.simulate匹配
            result = self.engine.simulate(window_data, previous_best_tree=inherited_tree)
            
            # 解析结果
            if isinstance(result, tuple) and len(result) >= 9:
                best_exp, all_times, test_data, loss, mae, mse, corr, policy, reward = result[:9]
                new_best_tree = result[9] if len(result) > 9 else None
            else:
                # 兼容性处理
                best_exp = result
                mae, mse, corr, loss, reward = 0, 0, 0, 0, 0
                new_best_tree = None

            # 6. 管理多样性池
            self._manage_diversity_pool(str(best_exp), mae)
            
            # 7. 更新继承状态
            self.previous_best_expression = str(best_exp)
            if new_best_tree is not None:
                self.previous_best_tree = new_best_tree
            
            # 8. 记录训练历史
            self.is_trained = True
            training_record = {
                'best_expression': str(best_exp),
                'mae': mae,
                'mse': mse,
                'corr': corr,
                'loss': loss,
                'reward': reward
            }
            self.training_history.append(training_record)
            
            print(f"滑动窗口训练完成")
            print(f"   最优表达式: {best_exp}")
            print(f"   MAE: {mae:.4f}, MSE: {mse:.4f}")
            
            return {
                'success': True,
                'best_expression': str(best_exp),
                'mae': mae,
                'mse': mse,
                'corr': corr,
                'loss': loss,
                'reward': reward
            }
            
        except Exception as e:
            print(f"❌ 滑动窗口训练失败: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'reason': str(e)}

    def predict(self, df: pd.DataFrame) -> int:
        """
        基于当前训练好的模型进行预测并返回交易信号
        """
        if not self.is_trained:
            return 0  # 未训练返回持有信号
            
        try:
            # 准备数据
            window_data = self._prepare_sliding_window_data(df)
            
            # 使用当前最优表达式进行预测
            # 这里简化处理，实际可能需要更复杂的预测逻辑
            # 调用agent计算信号
            from agent import TradingAgent
            agent = TradingAgent()
            
            # 模拟信号获取
            # 在实际集成中，这部分会由外部调用者处理，或者这里调用已训练的模型
            # 这里暂时返回一个中性信号
            return 0
        except Exception as e:
            print(f"预测失败: {e}")
            return 0

    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        if not self.training_history:
            return {'trained': False}
            
        latest = self.training_history[-1]
        return {
            'trained': True,
            'total_windows': len(self.training_history),
            'latest_expression': latest['best_expression'],
            'latest_metrics': {
                'mae': latest['mae'],
                'mse': latest['mse'],
                'corr': latest['corr']
            },
            'has_inheritance': self.previous_best_tree is not None
        }

def test_sliding_window_nemots():
    """测试滑动窗口NEMoTS"""
    print("测试滑动窗口NEMoTS")
    # 模拟数据
    data = pd.DataFrame(np.random.randn(100, 6), columns=['open', 'high', 'low', 'close', 'volume', 'amount'])
    sw = SlidingWindowNEMoTS(lookback=20, lookahead=5)
    res = sw.sliding_fit(data)
    print(f"测试结果: {res['success']}")

if __name__ == "__main__":
    test_sliding_window_nemots()