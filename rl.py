#!/usr/bin/env python
# coding=utf-8
"""
强化学习反馈系统
负责处理reward利用和loss反馈机制
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import pickle
import os

class RewardUtilizationSystem:
    """
    Reward利用系统
    负责将reward转化为对模型的具体改进
    """
    
    def __init__(self, learning_rate: float = 0.001, memory_size: int = 1000):
        """
        初始化reward利用系统
        
        Args:
            learning_rate: 学习率
            memory_size: 经验回放缓冲区大小
        """
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        
        # 经验回放缓冲区
        self.experience_buffer = deque(maxlen=memory_size)
        
        # 策略网络（简化的神经网络，用于学习最优策略）
        self.policy_net = self._build_policy_network()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # 奖励历史统计
        self.reward_stats = {
            'total_rewards': 0.0,
            'episode_count': 0,
            'best_reward': float('-inf'),
            'recent_rewards': deque(maxlen=100)
        }
        
        print(f"🎯 Reward利用系统初始化完成")
        print(f"   学习率: {learning_rate}")
        print(f"   经验缓冲区大小: {memory_size}")
    
    def _build_policy_network(self) -> nn.Module:
        """构建策略网络"""
        class PolicyNetwork(nn.Module):
            def __init__(self):
                super(PolicyNetwork, self).__init__()
                # 输入: 市场状态特征 (假设10维)
                # 输出: 动作概率分布 (3维: 买入/持有/卖出)
                self.fc1 = nn.Linear(10, 64)
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, 3)
                self.softmax = nn.Softmax(dim=-1)
                
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = self.fc3(x)
                return self.softmax(x)
        
        return PolicyNetwork()
    
    def store_experience(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
        """
        存储经验到回放缓冲区
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
        """
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'timestamp': pd.Timestamp.now()
        }
        self.experience_buffer.append(experience)
        
        # 更新奖励统计
        self.reward_stats['total_rewards'] += reward
        self.reward_stats['episode_count'] += 1
        self.reward_stats['recent_rewards'].append(reward)
        
        if reward > self.reward_stats['best_reward']:
            self.reward_stats['best_reward'] = reward
            print(f"🏆 新的最佳奖励: {reward:.4f}")
    
    def utilize_reward(self, code: str, reward: float, loss: float, 
                      market_state: np.ndarray, action: int) -> Dict[str, Any]:
        """
        核心方法：利用reward改进策略
        
        Args:
            code: 资产代码
            reward: 获得的奖励
            loss: 损失值
            market_state: 市场状态特征
            action: 执行的动作
            
        Returns:
            策略更新结果
        """
        print(f"🎯 利用reward改进策略: {code}")
        print(f"   Reward: {reward:.4f}, Loss: {loss:.4f}, Action: {action}")
        
        # 1. 存储经验
        # 这里简化next_state为当前state（实际应该是下一时刻的状态）
        self.store_experience(market_state, action, reward, market_state)
        
        # 2. 计算策略梯度
        net_reward = reward - loss  # 净奖励
        
        # 3. 根据reward调整策略
        update_result = {}
        
        if net_reward > 0:
            # 正奖励：增强当前策略
            update_result = self._reinforce_strategy(market_state, action, net_reward)
            print(f"   ✅ 正奖励 {net_reward:.4f}: 增强策略")
        elif net_reward < 0:
            # 负奖励：惩罚当前策略
            update_result = self._penalize_strategy(market_state, action, abs(net_reward))
            print(f"   ❌ 负奖励 {net_reward:.4f}: 惩罚策略")
        else:
            # 零奖励：保持当前策略
            update_result = {'action': 'maintain', 'adjustment': 0.0}
            print(f"   ➖ 零奖励: 保持策略")
        
        # 4. 执行策略网络更新（如果有足够经验）
        if len(self.experience_buffer) >= 32:  # 批量大小
            network_update = self._update_policy_network()
            update_result['network_update'] = network_update
        
        return update_result
    
    def _reinforce_strategy(self, state: np.ndarray, action: int, reward: float) -> Dict[str, Any]:
        """增强策略（正奖励时调用）"""
        # 增加该动作在该状态下的概率
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_probs = self.policy_net(state_tensor)
            current_prob = action_probs[0, action].item()
        
        # 计算增强幅度（基于奖励大小）
        enhancement = min(0.1, reward * 0.05)  # 最大增强10%
        
        return {
            'action': 'reinforce',
            'target_action': action,
            'current_prob': current_prob,
            'enhancement': enhancement,
            'reward': reward
        }
    
    def _penalize_strategy(self, state: np.ndarray, action: int, penalty: float) -> Dict[str, Any]:
        """惩罚策略（负奖励时调用）"""
        # 降低该动作在该状态下的概率
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_probs = self.policy_net(state_tensor)
            current_prob = action_probs[0, action].item()
        
        # 计算惩罚幅度
        penalty_amount = min(0.1, penalty * 0.05)  # 最大惩罚10%
        
        return {
            'action': 'penalize',
            'target_action': action,
            'current_prob': current_prob,
            'penalty': penalty_amount,
            'loss': penalty
        }
    
    def _update_policy_network(self) -> Dict[str, Any]:
        """更新策略网络"""
        if len(self.experience_buffer) < 32:
            return {'updated': False, 'reason': 'insufficient_data'}
        
        # 采样批量经验
        batch_size = min(32, len(self.experience_buffer))
        batch = np.random.choice(list(self.experience_buffer), batch_size, replace=False)
        
        states = torch.FloatTensor([exp['state'] for exp in batch])
        actions = torch.LongTensor([exp['action'] for exp in batch])
        rewards = torch.FloatTensor([exp['reward'] for exp in batch])
        
        # 计算策略梯度
        self.optimizer.zero_grad()
        
        action_probs = self.policy_net(states)
        selected_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze()
        
        # 策略梯度损失（REINFORCE算法）
        loss = -torch.mean(torch.log(selected_probs) * rewards)
        
        loss.backward()
        self.optimizer.step()
        
        return {
            'updated': True,
            'loss': loss.item(),
            'batch_size': batch_size,
            'avg_reward': rewards.mean().item()
        }
    
    def get_reward_statistics(self) -> Dict[str, Any]:
        """获取奖励统计信息"""
        recent_avg = np.mean(list(self.reward_stats['recent_rewards'])) if self.reward_stats['recent_rewards'] else 0.0
        
        return {
            'total_rewards': self.reward_stats['total_rewards'],
            'episode_count': self.reward_stats['episode_count'],
            'average_reward': self.reward_stats['total_rewards'] / max(1, self.reward_stats['episode_count']),
            'best_reward': self.reward_stats['best_reward'],
            'recent_average': recent_avg,
            'experience_buffer_size': len(self.experience_buffer)
        }


class LossFeedbackSystem:
    """
    Loss反馈系统
    负责将loss反馈到智能体的具体位置和方法
    """
    
    def __init__(self, feedback_threshold: float = 0.01, adaptation_rate: float = 0.1):
        """
        初始化loss反馈系统
        
        Args:
            feedback_threshold: 反馈阈值，超过此值才触发反馈
            adaptation_rate: 适应速率
        """
        self.feedback_threshold = feedback_threshold
        self.adaptation_rate = adaptation_rate
        
        # Loss历史记录
        self.loss_history = deque(maxlen=1000)
        
        # 反馈目标组件
        self.feedback_targets = {
            'nemots_hyperparams': True,    # NEMoTS超参数
            'agent_weights': True,         # Agent权重
            'prediction_confidence': True, # 预测置信度
            'risk_management': True        # 风险管理参数
        }
        
        print(f"🔄 Loss反馈系统初始化完成")
        print(f"   反馈阈值: {feedback_threshold}")
        print(f"   适应速率: {adaptation_rate}")
    
    def process_loss_feedback(self, code: str, loss: float, loss_source: str, 
                            context: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理loss反馈的核心方法
        
        Args:
            code: 资产代码
            loss: 损失值
            loss_source: 损失来源 ('prediction', 'trading', 'risk')
            context: 上下文信息
            
        Returns:
            反馈处理结果
        """
        print(f"🔄 处理Loss反馈: {code}")
        print(f"   Loss: {loss:.4f}, 来源: {loss_source}")
        
        # 记录loss历史
        loss_record = {
            'code': code,
            'loss': loss,
            'source': loss_source,
            'context': context,
            'timestamp': pd.Timestamp.now()
        }
        self.loss_history.append(loss_record)
        
        feedback_results = {}
        
        # 只有当loss超过阈值时才进行反馈
        if loss > self.feedback_threshold:
            print(f"   ⚠️ Loss超过阈值 {self.feedback_threshold}，触发反馈机制")
            
            # 1. 反馈到NEMoTS超参数
            if self.feedback_targets['nemots_hyperparams']:
                nemots_feedback = self._feedback_to_nemots(loss, loss_source, context)
                feedback_results['nemots'] = nemots_feedback
            
            # 2. 反馈到Agent权重
            if self.feedback_targets['agent_weights']:
                agent_feedback = self._feedback_to_agent(loss, loss_source, context)
                feedback_results['agent'] = agent_feedback
            
            # 3. 反馈到预测置信度
            if self.feedback_targets['prediction_confidence']:
                confidence_feedback = self._feedback_to_prediction_confidence(loss, loss_source, context)
                feedback_results['confidence'] = confidence_feedback
            
            # 4. 反馈到风险管理
            if self.feedback_targets['risk_management']:
                risk_feedback = self._feedback_to_risk_management(loss, loss_source, context)
                feedback_results['risk'] = risk_feedback
        else:
            print(f"   ✅ Loss在阈值内，无需反馈")
            feedback_results['action'] = 'no_feedback_needed'
        
        return feedback_results
    
    def _feedback_to_nemots(self, loss: float, source: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """反馈到NEMoTS模型参数"""
        print(f"   🧠 反馈到NEMoTS模型")
        
        # 根据loss调整NEMoTS超参数
        adjustments = {}
        
        if source == 'prediction':
            # 预测loss高 -> 降低学习率，增加正则化
            lr_adjustment = -self.adaptation_rate * loss
            reg_adjustment = self.adaptation_rate * loss * 0.5
            
            adjustments = {
                'learning_rate_multiplier': 1 + lr_adjustment,
                'regularization_multiplier': 1 + reg_adjustment,
                'exploration_rate_multiplier': 1 + self.adaptation_rate * loss,  # 增加探索
                'reason': f'预测loss={loss:.4f}过高，降低学习率，增加探索'
            }
        
        elif source == 'trading':
            # 交易loss高 -> 调整模型复杂度
            complexity_adjustment = -self.adaptation_rate * loss
            
            adjustments = {
                'max_len_multiplier': 1 + complexity_adjustment,  # 降低模型复杂度
                'num_runs_multiplier': 1 + self.adaptation_rate * loss,  # 增加运行次数
                'reason': f'交易loss={loss:.4f}过高，降低模型复杂度'
            }
        
        return adjustments
    
    def _feedback_to_agent(self, loss: float, source: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """反馈到Agent权重系统"""
        print(f"   🤖 反馈到Agent权重")
        
        # 调整Agent的strength计算权重
        weight_adjustments = {}
        
        if loss > 0.05:  # 高loss
            # 降低当前策略的权重，增加保守性
            weight_adjustments = {
                'stock_strength_weight': -self.adaptation_rate * loss,
                'market_strength_weight': self.adaptation_rate * loss * 0.5,  # 增加市场权重
                'risk_aversion_multiplier': 1 + self.adaptation_rate * loss,
                'reason': f'Loss={loss:.4f}过高，增加保守性'
            }
        elif loss > 0.02:  # 中等loss
            # 微调权重
            weight_adjustments = {
                'stock_strength_weight': -self.adaptation_rate * loss * 0.5,
                'sector_strength_weight': self.adaptation_rate * loss * 0.3,
                'reason': f'Loss={loss:.4f}中等，微调权重'
            }
        
        return weight_adjustments
    
    def _feedback_to_prediction_confidence(self, loss: float, source: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """反馈到预测置信度系统"""
        print(f"   🎯 反馈到预测置信度")
        
        # 根据loss调整预测置信度阈值
        confidence_adjustments = {
            'confidence_threshold_adjustment': -self.adaptation_rate * loss,  # 降低置信度阈值
            'uncertainty_penalty_multiplier': 1 + self.adaptation_rate * loss,  # 增加不确定性惩罚
            'prediction_weight_decay': self.adaptation_rate * loss * 0.1,  # 预测权重衰减
            'reason': f'Loss={loss:.4f}，降低预测置信度'
        }
        
        return confidence_adjustments
    
    def _feedback_to_risk_management(self, loss: float, source: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """反馈到风险管理系统"""
        print(f"   ⚠️ 反馈到风险管理")
        
        # 根据loss调整风险管理参数
        risk_adjustments = {
            'stop_loss_tightening': self.adaptation_rate * loss,  # 收紧止损
            'position_size_reduction': self.adaptation_rate * loss * 0.5,  # 减少仓位
            'volatility_threshold_adjustment': -self.adaptation_rate * loss,  # 降低波动率阈值
            'max_drawdown_limit_tightening': self.adaptation_rate * loss * 0.3,  # 收紧最大回撤限制
            'reason': f'Loss={loss:.4f}，收紧风险控制'
        }
        
        return risk_adjustments
    
    def get_loss_statistics(self) -> Dict[str, Any]:
        """获取loss统计信息"""
        if not self.loss_history:
            return {'no_data': True}
        
        losses = [record['loss'] for record in self.loss_history]
        recent_losses = losses[-100:] if len(losses) > 100 else losses
        
        return {
            'total_loss_events': len(self.loss_history),
            'average_loss': np.mean(losses),
            'max_loss': np.max(losses),
            'recent_average_loss': np.mean(recent_losses),
            'loss_trend': 'increasing' if len(recent_losses) > 10 and recent_losses[-5:] > recent_losses[:5] else 'stable',
            'feedback_trigger_rate': len([l for l in losses if l > self.feedback_threshold]) / len(losses)
        }


class IntegratedRLFeedbackSystem:
    """
    集成的RL反馈系统
    整合reward利用和loss反馈
    """
    
    def __init__(self):
        """初始化集成反馈系统"""
        self.reward_system = RewardUtilizationSystem()
        self.loss_system = LossFeedbackSystem()
        
        # 系统状态
        self.system_state = {
            'total_episodes': 0,
            'successful_episodes': 0,
            'failed_episodes': 0,
            'adaptation_history': []
        }
        
        print(f"🔧 集成RL反馈系统初始化完成")
    
    def process_episode_feedback(self, code: str, reward: float, loss: float, 
                               market_state: np.ndarray, action: int, 
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个episode的完整反馈
        
        Args:
            code: 资产代码
            reward: 奖励值
            loss: 损失值
            market_state: 市场状态
            action: 执行的动作
            context: 上下文信息
            
        Returns:
            完整的反馈处理结果
        """
        print(f"\n🔧 处理Episode反馈: {code}")
        print(f"   Reward: {reward:.4f}, Loss: {loss:.4f}, Action: {action}")
        
        # 更新系统状态
        self.system_state['total_episodes'] += 1
        if reward > loss:
            self.system_state['successful_episodes'] += 1
        else:
            self.system_state['failed_episodes'] += 1
        
        # 1. 处理reward利用
        reward_result = self.reward_system.utilize_reward(
            code, reward, loss, market_state, action
        )
        
        # 2. 处理loss反馈
        loss_result = self.loss_system.process_loss_feedback(
            code, loss, 'trading', context
        )
        
        # 3. 整合结果
        integrated_result = {
            'episode_id': self.system_state['total_episodes'],
            'code': code,
            'reward_processing': reward_result,
            'loss_processing': loss_result,
            'net_outcome': reward - loss,
            'system_adaptation': self._calculate_system_adaptation(reward, loss),
            'timestamp': pd.Timestamp.now()
        }
        
        # 记录适应历史
        self.system_state['adaptation_history'].append(integrated_result)
        
        return integrated_result
    
    def _calculate_system_adaptation(self, reward: float, loss: float) -> Dict[str, Any]:
        """计算系统适应性调整"""
        net_outcome = reward - loss
        
        if net_outcome > 0.02:  # 显著正收益
            adaptation = {
                'type': 'positive_reinforcement',
                'strength': min(1.0, net_outcome * 10),
                'action': 'enhance_current_strategy'
            }
        elif net_outcome < -0.02:  # 显著负收益
            adaptation = {
                'type': 'negative_feedback',
                'strength': min(1.0, abs(net_outcome) * 10),
                'action': 'adjust_strategy_parameters'
            }
        else:  # 中性结果
            adaptation = {
                'type': 'neutral',
                'strength': 0.0,
                'action': 'maintain_current_strategy'
            }
        
        return adaptation
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        reward_stats = self.reward_system.get_reward_statistics()
        loss_stats = self.loss_system.get_loss_statistics()
        
        success_rate = (self.system_state['successful_episodes'] / 
                       max(1, self.system_state['total_episodes']))
        
        return {
            'system_state': self.system_state,
            'success_rate': success_rate,
            'reward_statistics': reward_stats,
            'loss_statistics': loss_stats,
            'adaptation_count': len(self.system_state['adaptation_history'])
        }
    
    def save_system_state(self, filepath: str):
        """保存系统状态"""
        state_data = {
            'system_state': self.system_state,
            'reward_stats': self.reward_system.get_reward_statistics(),
            'loss_stats': self.loss_system.get_loss_statistics()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state_data, f)
        
        print(f"💾 系统状态已保存到: {filepath}")
    
    def load_system_state(self, filepath: str):
        """加载系统状态"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                state_data = pickle.load(f)
            
            self.system_state = state_data['system_state']
            print(f"📂 系统状态已从 {filepath} 加载")
        else:
            print(f"⚠️ 状态文件 {filepath} 不存在")


# 使用示例
def example_usage():
    """使用示例"""
    print("🔧 RL反馈系统使用示例")
    print("=" * 50)
    
    # 创建集成反馈系统
    feedback_system = IntegratedRLFeedbackSystem()
    
    # 模拟市场状态（10维特征）
    market_state = np.random.randn(10)
    
    # 模拟几个episode的反馈
    for i in range(5):
        code = f"sh.60000{i}"
        reward = np.random.uniform(0, 0.1)  # 0-10%的奖励
        loss = np.random.uniform(0, 0.05)   # 0-5%的损失
        action = np.random.choice([0, 1, 2])  # 随机动作
        
        context = {
            'market_volatility': np.random.uniform(0.01, 0.05),
            'trading_volume': np.random.uniform(1000, 10000),
            'prediction_confidence': np.random.uniform(0.5, 0.9)
        }
        
        # 处理反馈
        result = feedback_system.process_episode_feedback(
            code, reward, loss, market_state, action, context
        )
        
        print(f"\nEpisode {i+1} 处理完成:")
        print(f"  净收益: {result['net_outcome']:.4f}")
        print(f"  适应类型: {result['system_adaptation']['type']}")
    
    # 打印系统统计
    stats = feedback_system.get_system_statistics()
    print(f"\n📊 系统统计:")
    print(f"  总Episodes: {stats['system_state']['total_episodes']}")
    print(f"  成功率: {stats['success_rate']:.2%}")
    print(f"  平均奖励: {stats['reward_statistics']['average_reward']:.4f}")


if __name__ == "__main__":
    example_usage()
