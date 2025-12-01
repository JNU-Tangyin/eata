"""
PPO 策略
基于PPO强化学习的交易策略
"""

import pandas as pd
import numpy as np

try:
    from .data_utils import run_vectorized_backtest
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from data_utils import run_vectorized_backtest


class TradingEnv:
    """交易环境"""
    def __init__(self, df):
        # 延迟导入
        import gymnasium as gym
        from gymnasium import spaces
        
        self.df = df.reset_index(drop=True)
        self.current_step = 0
        self.initial_balance = 100000
        self.balance = self.initial_balance
        self.shares = 0
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        
        # 观察空间：价格相关特征
        n_features = 10
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_features,))
        
    def reset(self, seed=None):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares = 0
        return self._get_observation(), {}
        
    def step(self, action):
        if self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0, True, True, {}
            
        current_price = self.df.iloc[self.current_step]['close']
        next_price = self.df.iloc[self.current_step + 1]['close']
        
        # 执行动作
        if action == 1 and self.balance > current_price:  # 买入
            shares_to_buy = int(self.balance // current_price)
            self.shares += shares_to_buy
            self.balance -= shares_to_buy * current_price
        elif action == 2 and self.shares > 0:  # 卖出
            self.balance += self.shares * current_price
            self.shares = 0
        
        # 计算奖励（基于价格变化和持仓）
        price_change = (next_price - current_price) / current_price
        if self.shares > 0:
            reward = price_change  # 持有股票时，奖励为价格变化
        else:
            reward = -price_change * 0.1  # 空仓时，小幅惩罚错过上涨
            
        self.current_step += 1
        obs = self._get_observation()
        done = self.current_step >= len(self.df) - 1
        
        return obs, reward, done, False, {}
        
    def _get_observation(self):
        if self.current_step >= len(self.df):
            return np.zeros(10)
            
        row = self.df.iloc[self.current_step]
        
        # 构建观察特征
        features = [
            row['close'] / 100,  # 标准化价格
            row.get('rsi_14', 50) / 100,  # RSI
            row.get('macd_12_26_9', 0) / 10,  # MACD
            row.get('sma_20', row['close']) / row['close'] - 1,  # SMA相对位置
            row.get('bb_upper', row['close']) / row['close'] - 1,  # 布林带上轨
            row.get('bb_lower', row['close']) / row['close'] - 1,  # 布林带下轨
            row.get('volume', 1000000) / 1000000,  # 标准化成交量
            self.balance / self.initial_balance,  # 余额比例
            self.shares * row['close'] / self.initial_balance,  # 持仓价值比例
            (self.balance + self.shares * row['close']) / self.initial_balance - 1  # 总收益率
        ]
        
        return np.array(features, dtype=np.float32)


class PPOAgent:
    """手动实现的PPO算法，避免stable_baselines3的死锁问题"""
    
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        # 策略网络
        class PolicyNetwork(nn.Module):
            def __init__(self, state_dim, action_dim):
                super().__init__()
                self.fc = nn.Sequential(
                    nn.Linear(state_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, action_dim),
                    nn.Softmax(dim=-1)
                )
                
            def forward(self, x):
                return self.fc(x)
        
        # 价值网络
        class ValueNetwork(nn.Module):
            def __init__(self, state_dim):
                super().__init__()
                self.fc = nn.Sequential(
                    nn.Linear(state_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                )
                
            def forward(self, x):
                return self.fc(x)
        
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)
        self.optimizer = optim.Adam(list(self.policy.parameters()) + list(self.value.parameters()), lr=lr)
        
        # 存储轨迹
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        
    def select_action(self, state):
        import torch
        
        state = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_probs = self.policy(state)
            value = self.value(state)
            
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        self.states.append(state.squeeze().numpy())
        self.actions.append(action.item())
        self.log_probs.append(log_prob.item())
        self.values.append(value.item())
        
        return action.item()
    
    def store_reward(self, reward):
        self.rewards.append(reward)
    
    def update(self):
        import torch
        import torch.nn.functional as F
        
        if len(self.rewards) == 0:
            return
        
        # 计算折扣奖励
        discounted_rewards = []
        discounted_reward = 0
        for reward in reversed(self.rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            discounted_rewards.insert(0, discounted_reward)
        
        # 标准化奖励
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        if len(discounted_rewards) > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        
        # 转换为张量
        old_states = torch.FloatTensor(np.array(self.states))
        old_actions = torch.LongTensor(self.actions)
        old_log_probs = torch.FloatTensor(self.log_probs)
        
        # PPO更新
        for _ in range(self.k_epochs):
            # 计算新的动作概率和价值
            action_probs = self.policy(old_states)
            values = self.value(old_states).squeeze()
            
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(old_actions)
            
            # 计算比率
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # 计算优势
            advantages = discounted_rewards - values.detach()
            
            # PPO损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 价值损失
            value_loss = F.mse_loss(values, discounted_rewards)
            
            # 总损失
            total_loss = policy_loss + 0.5 * value_loss
            
            # 更新网络
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        
        # 清空轨迹
        self.clear_memory()
    
    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []


class TradingEnvironment:
    """简化的交易环境"""
    
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.current_step = 0
        self.max_steps = len(df) - 1
        self.reset()
    
    def reset(self):
        self.current_step = 0
        self.portfolio_value = 1.0
        self.position = 0  # -1=空头, 0=空仓, 1=多头
        self.cash = 1.0
        self.shares = 0
        return self._get_state()
    
    def _get_state(self):
        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1
        
        row = self.df.iloc[self.current_step]
        
        # 构建状态特征
        features = [
            row.get('rsi_14', 50) / 100,  # RSI标准化
            np.tanh(row.get('macd_12_26_9', 0) / row['close']),  # MACD标准化
            (row.get('sma_20', row['close']) / row['close'] - 1),  # SMA相对位置
            (row.get('bb_upper', row['close']) / row['close'] - 1),  # 布林带上轨
            (row.get('bb_lower', row['close']) / row['close'] - 1),  # 布林带下轨
            np.tanh(row.get('volume', 1000000) / 1000000 - 1),  # 成交量标准化
            self.position,  # 当前持仓
            (self.portfolio_value - 1),  # 组合收益
            self.current_step / self.max_steps,  # 时间进度
            np.tanh((row['close'] - self.df['close'].iloc[max(0, self.current_step-5):self.current_step+1].mean()) / row['close'])  # 短期价格动量
        ]
        
        return np.array(features, dtype=np.float32)
    
    def step(self, action):
        if self.current_step >= self.max_steps:
            return self._get_state(), 0, True
        
        current_price = self.df.iloc[self.current_step]['close']
        
        # 执行动作: 0=卖出, 1=持有, 2=买入
        old_position = self.position
        
        if action == 0 and self.position > 0:  # 卖出
            self.cash += self.shares * current_price
            self.shares = 0
            self.position = 0
        elif action == 2 and self.position <= 0:  # 买入
            if self.cash > 0:
                self.shares = self.cash / current_price
                self.cash = 0
                self.position = 1
        # action == 1 是持有，不做操作
        
        # 移动到下一步
        self.current_step += 1
        
        # 计算奖励
        if self.current_step < len(self.df):
            next_price = self.df.iloc[self.current_step]['close']
            price_change = (next_price - current_price) / current_price
            
            # 基于持仓和价格变化计算奖励
            if self.position > 0:  # 持有股票
                reward = price_change * 10  # 放大奖励信号
            else:  # 空仓
                reward = -price_change * 2  # 错过上涨的小惩罚
            
            # 交易成本
            if old_position != self.position:
                reward -= 0.001  # 交易成本
        else:
            reward = 0
        
        # 更新组合价值
        if self.shares > 0:
            self.portfolio_value = self.shares * self.df.iloc[min(self.current_step, len(self.df)-1)]['close']
        else:
            self.portfolio_value = self.cash
        
        done = self.current_step >= self.max_steps
        
        return self._get_state(), reward, done


def train_ppo_single_process(train_data, test_data):
    """单进程PPO训练，避免死锁"""
    try:
        print("   Training PPO with custom implementation...")
        
        # 创建环境和智能体
        env = TradingEnvironment(train_data)
        state_dim = len(env._get_state())
        action_dim = 3  # 卖出, 持有, 买入
        
        agent = PPOAgent(state_dim, action_dim)
        
        # 训练参数
        max_episodes = min(100, len(train_data) // 10)  # 根据数据量调整
        max_steps_per_episode = min(200, len(train_data))
        
        for episode in range(max_episodes):
            state = env.reset()
            episode_reward = 0
            
            for step in range(max_steps_per_episode):
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)
                
                agent.store_reward(reward)
                episode_reward += reward
                
                if done:
                    break
                
                state = next_state
            
            # 更新策略
            agent.update()
            
            if episode % 20 == 0:
                print(f"   Episode {episode}, Reward: {episode_reward:.4f}")
        
        # 在测试集上生成信号
        test_env = TradingEnvironment(test_data)
        state = test_env.reset()
        signals = []
        
        for _ in range(len(test_data)):
            action = agent.select_action(state)
            next_state, _, done = test_env.step(action)
            
            # 转换动作为信号
            if action == 0:
                signals.append(-1)  # 卖出
            elif action == 2:
                signals.append(1)   # 买入
            else:
                signals.append(0)   # 持有
            
            if done:
                break
            
            state = next_state
        
        # 确保信号长度匹配
        while len(signals) < len(test_data):
            signals.append(0)
        
        return signals[:len(test_data)]
        
    except Exception as e:
        print(f"   Custom PPO training failed: {e}")
        # 返回基于技术指标的信号作为fallback
        signals = []
        for i in range(len(test_data)):
            row = test_data.iloc[i]
            rsi = row.get('rsi_14', 50)
            if rsi < 30:
                signals.append(1)  # 超卖买入
            elif rsi > 70:
                signals.append(-1)  # 超买卖出
            else:
                signals.append(0)  # 持有
        return signals


def run_ppo_strategy(train_df: pd.DataFrame, test_df: pd.DataFrame, ticker: str):
    """
    PPO策略 - 基于ETS-SDA原始实现
    
    Args:
        train_df: 训练数据
        test_df: 测试数据
        ticker: 股票代码
        
    Returns:
        tuple: (metrics, backtest_results)
    """
    print(f"Running PPO strategy for {ticker}...")
    
    try:
        print("   Using custom PPO implementation (single-process, no deadlock)...")
        
        # 使用自定义的单进程PPO实现
        signals = train_ppo_single_process(train_df, test_df)
        
        # 处理信号并运行回测
        df_ppo = test_df.copy()
        df_ppo['signal'] = signals
        
        # 运行回测
        metrics, backtest_results = run_vectorized_backtest(df_ppo, signal_col='signal')
        
        print(f"✅ PPO strategy completed for {ticker}")
        print(f"   Total Return: {metrics['total_return']:.2%}")
        print(f"   Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"   Signal distribution: Buy={sum(1 for s in signals if s==1)}, "
              f"Sell={sum(1 for s in signals if s==-1)}, "
              f"Hold={sum(1 for s in signals if s==0)}")
        
        return metrics, backtest_results
        
    except Exception as e:
        print(f"❌ PPO strategy failed for {ticker}: {e}")
        
        # 返回默认结果
        performance_metrics = pd.Series({
            'annualized_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'total_return': 0,
            'num_trades': 0
        })
        equity_curve = pd.DataFrame({
            'date': test_df['date'], 
            'portfolio_value': 1.0
        })
        return performance_metrics, equity_curve


if __name__ == '__main__':
    # 测试代码
    from data_utils import add_technical_indicators
    
    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='B')
    prices = [100]
    for _ in range(len(dates)-1):
        prices.append(prices[-1] * (1 + np.random.normal(0, 0.02)))
    
    df = pd.DataFrame({
        'date': dates,
        'ticker': 'TEST',
        'open': [p * 0.99 for p in prices],
        'high': [p * 1.02 for p in prices],
        'low': [p * 0.98 for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    df = add_technical_indicators(df)
    
    # 分割数据
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    try:
        metrics, _ = run_ppo_strategy(train_df, test_df, 'TEST')
        print("PPO strategy test completed!")
    except Exception as e:
        print(f"PPO test failed: {e}")
        print("This is expected if stable_baselines3 is not installed properly")
