import argparse # 新增：导入argparse模块
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import quantstats as qs
import logging

# 屏蔽Numpy数值计算警告 (例如除以0，log(0)等)
np.seterr(all='ignore')
# 屏蔽RuntimeWarning数学运算警告
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered")
warnings.filterwarnings("ignore", category=RuntimeWarning)
# 屏蔽Matplotlib找不到字体的警告
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


# 核心改动：直接导入我们改造后的Agent
from agent import Agent
from data import DataStorage # 导入数据存储类
from performance_metrics import TradingMetrics # 导入我们新增的指标计算模块
from rl import IntegratedRLFeedbackSystem # 导入强化学习反馈系统

class Predictor:
    def __init__(self, lookback=100, lookahead=20, stride=2, depth=200):
        """
        新版预测器，核心职责是初始化和调用Agent。
        """
        self.lookback = lookback
        self.lookahead = lookahead
        self.stride = stride
        self.depth = depth
        self.agent = Agent(df=pd.DataFrame(), lookback=lookback, lookahead=lookahead, stride=stride, depth=depth)
        self.feedback_system = IntegratedRLFeedbackSystem() # 初始化反馈系统
        print(f"🤖 新版 Predictor 初始化完成，参数: lookback={lookback}, lookahead={lookahead}, stride={stride}, depth={depth}")

    def extract_market_state(self, data: pd.DataFrame) -> np.ndarray:
        """
        从滑动窗口数据中提取10维市场状态特征向量，用于RL反馈
        """
        if len(data) == 0:
            return np.zeros(10)
        
        features = []
        # 1. 价格变化率
        price_change = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]
        features.append(np.clip(price_change, -0.1, 0.1))
        
        # 2. 价格波动率
        price_volatility = data['close'].pct_change().std()
        features.append(np.clip(price_volatility, 0, 0.1))
        
        # 3. 成交量变化
        if data['volume'].iloc[0] != 0:
            volume_change = (data['volume'].iloc[-1] - data['volume'].iloc[0]) / data['volume'].iloc[0]
        else:
            volume_change = 0
        features.append(np.clip(volume_change, -1, 1))
        
        # 4. 最高价相对位置
        low_min = data['low'].min()
        high_max = data['high'].max()
        if high_max != low_min:
            high_position = (data['close'].iloc[-1] - low_min) / (high_max - low_min)
        else:
            high_position = 0.5
        features.append(np.clip(high_position, 0, 1))
        
        # 5-7. 移动平均线相对位置
        for window in [3, 5, 10]:
            if len(data) >= window:
                ma = data['close'].rolling(window).mean().iloc[-1]
                if ma != 0:
                    ma_position = (data['close'].iloc[-1] - ma) / ma
                else:
                    ma_position = 0
                features.append(np.clip(ma_position, -0.1, 0.1))
            else:
                features.append(0.0)
        
        # 8. RSI简易特征
        price_diff = data['close'].diff()
        gain = price_diff.clip(lower=0).mean()
        loss = (-price_diff.clip(upper=0)).mean()
        if (gain + loss) != 0:
            rsi = gain / (gain + loss)
        else:
            rsi = 0.5
        features.append(rsi)
        
        # 9. 成交量相对强度
        vol_mean = data['volume'].mean()
        if vol_mean != 0:
            volume_strength = data['volume'].iloc[-1] / vol_mean
        else:
            volume_strength = 1.0
        features.append(np.clip(volume_strength, 0, 3))
        
        # 10. 趋势强度
        if len(data) > 1:
            trend = np.polyfit(range(len(data)), data['close'], 1)[0]
            if data['close'].mean() != 0:
                trend_strength = trend / data['close'].mean()
            else:
                trend_strength = 0
        else:
            trend_strength = 0
        features.append(np.clip(trend_strength, -0.01, 0.01))
        
        return np.array(features[:10])

    def update_feedback(self, ticker: str, reward: float, loss: float, action: int, window_df: pd.DataFrame):
        """
        处理反馈并更新Agent参数
        """
        market_state = self.extract_market_state(window_df)
        
        context = {
            'ticker': ticker,
            'prediction_confidence': 0.8 # 暂时固定，未来可从Agent获取
        }
        
        # 处理反馈
        result = self.feedback_system.process_episode_feedback(
            code=ticker,
            reward=reward,
            loss=loss,
            market_state=market_state,
            action=action,
            context=context
        )
        
        # 应用反馈到NEMoTS模型
        if 'loss_processing' in result and 'nemots' in result['loss_processing']:
            nemots_feedback = result['loss_processing']['nemots']
            
            # 访问Agent内部的engine和model
            if hasattr(self.agent, 'engine') and hasattr(self.agent.engine, 'model'):
                # 调整学习率
                if 'learning_rate_multiplier' in nemots_feedback:
                    lr_mult = nemots_feedback['learning_rate_multiplier']
                    # 注意：Agent.engine.model.args 保存了超参数
                    if hasattr(self.agent.engine.model, 'args'):
                         self.agent.engine.model.args.lr *= lr_mult
                         print(f"   📉 [RL反馈] 调整NEMoTS学习率: ×{lr_mult:.3f}")
                
                # 调整探索率
                if 'exploration_rate_multiplier' in nemots_feedback:
                    exp_mult = nemots_feedback['exploration_rate_multiplier']
                    if hasattr(self.agent.engine.model, 'args'):
                        self.agent.engine.model.args.exploration_rate *= exp_mult
                        print(f"   🔍 [RL反馈] 调整NEMoTS探索率: ×{exp_mult:.3f}")

    def predict(self, df: pd.DataFrame, shares_held: int) -> tuple[int, float]:
        """
        使用Agent对单个数据窗口进行预测。
        现在返回一个包含action和rl_reward的元组。
        """
        print("\n[Predictor] -> 调用 Agent.criteria 进行决策...")
        action, rl_reward = self.agent.criteria(df, shares_held=shares_held)
        action_name = {-1: '卖出', 0: '持有', 1: '买入'}[action]
        print(f"[Predictor] <- Agent决策结果: {action} ({action_name}), RL Reward: {rl_reward:.4f}")
        return action, rl_reward


def run_eata_core_backtest(
    stock_df: pd.DataFrame,
    ticker: str,
    lookback: int = 50,
    lookahead: int = 10,
    stride: int = 1,
    depth: int = 300,
):
    """在给定单支股票数据上运行 EATA 本体的核心回测逻辑。

    注意：
    - 这里只负责滑动窗口 + 交易 + 指标计算，不做画图、不生成 HTML 报告。
    - 返回值与 baseline 侧对接用：metrics(dict) + portfolio_df(DataFrame[value] 索引为日期)。
    """

    # 初始化 Predictor / Agent
    predictor = Predictor(lookback=lookback, lookahead=lookahead, stride=stride, depth=depth)

    stock_df = stock_df.copy()
    stock_df['date'] = pd.to_datetime(stock_df['date'])
    stock_df.sort_values(by='date', inplace=True)
    stock_df.reset_index(drop=True, inplace=True)

    # 窗口与回测参数
    window_len = predictor.agent.lookback + predictor.agent.lookahead + 1
    
    # 动态调整测试窗口数量，适应数据长度
    max_possible_windows = len(stock_df) - window_len + 1
    num_test_windows = min(1000, max_possible_windows)  # 最多1000次，但不超过数据允许的范围
    
    if num_test_windows < 50:  # 至少需要50次测试才有意义
        raise ValueError(f"股票 {ticker} 的数据不足，只能进行 {num_test_windows} 次窗口测试（最少需要50次）")
    
    print(f"📊 EATA将进行 {num_test_windows} 次窗口测试（数据长度: {len(stock_df)}）")

    initial_cash = 1_000_000
    cash = initial_cash
    shares = 0
    stance = 0  # 1: 多头, -1: 空头, 0: 空仓
    portfolio_values = []
    all_trade_dates = []

    # 滑动窗口回测
    for i in range(0, num_test_windows, 2):
        offset = num_test_windows - 1 - i
        start_index = -(window_len + offset)
        end_index = -offset if offset > 0 else None

        window_df = stock_df.iloc[start_index:end_index].copy()
        window_df.reset_index(drop=True, inplace=True)

        # 调用 Agent 决策
        action, rl_reward = predictor.predict(df=window_df, shares_held=shares)

        # --- 新增：RL 反馈闭环 ---
        # 1. 提取市场状态
        market_state = predictor.extract_market_state(window_df)
        
        # 2. 定义 Loss (简化逻辑：若RL Reward较低，视为产生了Loss)
        loss = max(0, 0.5 - rl_reward)
        
        # 3. 调用反馈系统
        feedback_result = predictor.feedback_system.process_episode_feedback(
            code=ticker,
            reward=rl_reward,
            loss=loss,
            market_state=market_state,
            action={-1: 0, 0: 1, 1: 2}.get(action, 1),
            context={'ticker': ticker}
        )
        
        # 4. 应用反馈到 Agent 参数
        if 'loss_processing' in feedback_result and 'nemots' in feedback_result['loss_processing']:
            nemots_feedback = feedback_result['loss_processing']['nemots']
            # 直接更新 Agent 的超参数
            if hasattr(predictor.agent, 'hyperparams'):
                if 'learning_rate_multiplier' in nemots_feedback:
                    lr_mult = nemots_feedback['learning_rate_multiplier']
                    predictor.agent.hyperparams.lr *= lr_mult
                    print(f"   📉 [RL反馈] 调整学习率: ×{lr_mult:.3f}")
                
                if 'exploration_rate_multiplier' in nemots_feedback:
                    exp_mult = nemots_feedback['exploration_rate_multiplier']
                    predictor.agent.hyperparams.exploration_rate *= exp_mult
                    print(f"   🔍 [RL反馈] 调整探索率: ×{exp_mult:.3f}")
        # --- 闭环结束 ---

        # 交易发生在 lookback 之后的第一天
        trade_day_index = predictor.agent.lookback
        trade_price = window_df.loc[trade_day_index, 'open']

        # 更新姿态
        if action != 0:
            stance = action

        # 按姿态执行交易（支持多/空）
        if stance == 1:  # 多头
            if shares < 0:  # 先平空
                cash -= abs(shares) * trade_price
                shares = 0
            if shares == 0 and cash > 0:
                shares_to_buy = cash // trade_price
                shares += shares_to_buy
                cash -= shares_to_buy * trade_price

        elif stance == -1:  # 空头
            if shares > 0:  # 先平多
                cash += shares * trade_price
                shares = 0
            if shares == 0:
                value_to_short = cash
                shares_to_short = value_to_short // trade_price
                shares -= shares_to_short
                cash += shares_to_short * trade_price

        # 在 lookahead 期间记录资产轨迹
        lookahead_period_df = window_df.iloc[
            trade_day_index : trade_day_index + predictor.agent.lookahead
        ]
        for _, day in lookahead_period_df.iterrows():
            daily_value = cash + shares * day['close']
            portfolio_values.append(daily_value)
            all_trade_dates.append(day['date'])

    if not portfolio_values:
        raise ValueError(f"股票 {ticker} 未产生任何资产记录")

    portfolio_df = pd.DataFrame({'value': portfolio_values}, index=pd.to_datetime(all_trade_dates))
    portfolio_df = portfolio_df[~portfolio_df.index.duplicated(keep='last')]

    # 计算指标：使用 TradingMetrics，与本体保持一致
    stock_df_indexed = stock_df.set_index('date')
    benchmark_prices = stock_df_indexed.loc[portfolio_df.index, 'close']
    daily_returns = portfolio_df['value'].pct_change().dropna()
    buy_and_hold_returns = benchmark_prices.pct_change().dropna()

    metrics_calc = TradingMetrics(
        returns=daily_returns.values,
        benchmark_returns=buy_and_hold_returns.values,
    )
    metrics = metrics_calc.get_all_metrics()

    return metrics, portfolio_df


if __name__ == "__main__":
    # 新增：解析命令行参数
    parser = argparse.ArgumentParser(description="EATA Project Core Function Test, Backtest, and Evaluation (Multi-stock Version)")
    parser.add_argument('--project_name', type=str, default='default',
                        help='Name of the current project/experiment for distinguishing output files.')
    args = parser.parse_args()

    print("🚀 启动 EATA 项目核心功能测试、回测与评估 (多股票版)")
    print("=======================================================")

    try:
        # 1. 从 stock.db 加载所有数据
        print("\n[Main] 从 stock.db 的 raw_data 表加载所有数据...")
        import sqlite3
        
        conn = sqlite3.connect('stock.db')
        query = """
        SELECT code, date, open, high, low, close, volume, amount 
        FROM raw_data 
        ORDER BY code, date
        """
        all_data = pd.read_sql_query(query, conn)
        conn.close()
        
        if all_data.empty:
            raise Exception("数据库中没有找到数据。")
        
        print(f"✅ 数据加载完成: {len(all_data)} 条记录")

        # 2. 指定测试的三支股票（与baseline对比实验一致）
        target_tickers = ['AAPL', 'MSFT', 'GOOGL']
        all_available_tickers = all_data['code'].unique()
        
        # 筛选出实际可用的股票
        available_tickers = [ticker for ticker in target_tickers if ticker in all_available_tickers]
        if not available_tickers:
            print(f"❌ 目标股票 {target_tickers} 在数据中不可用")
            print(f"📊 可用股票: {list(all_available_tickers)[:10]}...")
            exit(1)
        
        print(f"[Main] 将测试指定的 {len(available_tickers)} 支股票: {available_tickers}")
        all_tickers = available_tickers

        # 3. 初始化一个列表来存储所有股票的最终指标
        all_results = []

        # 4. 外层循环：遍历每一支股票
        for ticker_idx, ticker in enumerate(all_tickers):
            print(f"\n\n{'='*15} 开始回测股票: {ticker} ({ticker_idx + 1}/{len(all_tickers)}) {'='*15}")
            
            # --- 每个股票都使用全新的Agent ---
            # 重新初始化Predictor，使用指定参数: lookback=50, lookahead=10, stride=1, depth=300
            predictor = Predictor(lookback=50, lookahead=10, stride=1, depth=300)
            
            stock_df = all_data[all_data['code'] == ticker].copy()
            stock_df['date'] = pd.to_datetime(stock_df['date']) # 确保date列是datetime类型
            stock_df.sort_values(by='date', inplace=True)
            stock_df.reset_index(drop=True, inplace=True)
            
            # 确保数据足够长
            window_len = predictor.agent.lookback + predictor.agent.lookahead + 1
            num_test_windows = 1000 # 默认1000个窗口
            
            if len(stock_df) < window_len + num_test_windows - 1:
                print(f"  [WARN] 股票 {ticker} 的数据不足，无法进行 {num_test_windows} 次窗口测试。跳过。")
                continue

            print(f"[Main] 已选择股票 {ticker} 进行测试，共 {len(stock_df)} 条记录。")
            print(f"\n[Main] 将在最新的数据上运行 {num_test_windows} 个连续的滑动窗口进行回测...")

            # 5. 初始化模拟账户和记录器
            initial_cash = 1_000_000
            cash = initial_cash
            shares = 0
            stance = 0 # 新增：交易姿态，1为多头，-1为空头，0为空仓
            portfolio_values = [] # 记录每日总资产
            all_trade_dates = [] # 记录所有回测区间的日期
            rl_rewards_history = [] # 记录每个窗口的RL奖励
            action_spans = [] # 新增：记录每个窗口的动作和时间范围，用于绘图

            # --- 初始持仓逻辑已被移除，回测将从100%现金开始 ---

            # 6. 循环执行回测 (核心改造：增加步长2，实现跳跃窗口)
            for i in range(0, num_test_windows, 2):
                window_number = i + 1
                
                # 从数据尾部向前切片，模拟在最新数据上进行的回测
                offset = num_test_windows - 1 - i
                start_index = -(window_len + offset)
                end_index = -offset if offset > 0 else None
                
                window_df = stock_df.iloc[start_index:end_index].copy()
                window_df.reset_index(drop=True, inplace=True)
    
                # --- 深度诊断：检查滑动窗口的日期范围 ---
                # 我们只对一支有问题的股票进行诊断，以减少日志量
                if ticker == 'AMZN':
                    if i < 5 or i >= num_test_windows - 5:
                        print(f"        [深度诊断 i={i}] window_df 日期: {window_df['date'].iloc[0].date()} -> {window_df['date'].iloc[-1].date()}")
                # --- 结束诊断 ---
    
                print(f"\n[Main] === 第 {window_number}/{num_test_windows} 次预测 ({'冷启动' if i == 0 else '热启动'}) ===")                
                # 获取Agent的交易决策，并传入当前持仓状态
                action, rl_reward = predictor.predict(df=window_df, shares_held=shares)
                rl_rewards_history.append(rl_reward)

                # --- 新增：RL 反馈闭环 (针对 standalone 模式) ---
                # 1. 提取市场状态
                market_state = predictor.extract_market_state(window_df)
                
                # 2. 定义 Loss
                loss = max(0, 0.5 - rl_reward)
                
                # 3. 调用反馈系统
                feedback_result = predictor.feedback_system.process_episode_feedback(
                    code=ticker,
                    reward=rl_reward,
                    loss=loss,
                    market_state=market_state,
                    action={-1: 0, 0: 1, 1: 2}.get(action, 1),
                    context={'ticker': ticker, 'mode': 'standalone'}
                )
                
                # 4. 应用反馈到 Agent 参数
                if 'loss_processing' in feedback_result and 'nemots' in feedback_result['loss_processing']:
                    nemots_feedback = feedback_result['loss_processing']['nemots']
                    if hasattr(predictor.agent, 'hyperparams'):
                        if 'learning_rate_multiplier' in nemots_feedback:
                            lr_mult = nemots_feedback['learning_rate_multiplier']
                            predictor.agent.hyperparams.lr *= lr_mult
                            print(f"   📉 [RL反馈] 独立运行模式 - 调整学习率: ×{lr_mult:.3f}")
                        
                        if 'exploration_rate_multiplier' in nemots_feedback:
                            exp_mult = nemots_feedback['exploration_rate_multiplier']
                            predictor.agent.hyperparams.exploration_rate *= exp_mult
                            print(f"   🔍 [RL反馈] 独立运行模式 - 调整探索率: ×{exp_mult:.3f}")
                # --- 闭环结束 ---

                # --- 新增：记录动作区间 ---
                lookahead_period_df_for_span = window_df.iloc[predictor.agent.lookback : predictor.agent.lookback + predictor.agent.lookahead]
                if not lookahead_period_df_for_span.empty:
                    start_date = lookahead_period_df_for_span['date'].iloc[0]
                    end_date = lookahead_period_df_for_span['date'].iloc[-1]
                    action_spans.append({'start': start_date, 'end': end_date, 'action': action})
                # --- 结束新增 ---
                
                # --- 模拟交易与资产记录 (已升级支持做空) ---
                # 交易发生在lookback期之后的第一天
                trade_day_index = predictor.agent.lookback
                trade_price = window_df.loc[trade_day_index, 'open']

                # 核心逻辑：更新交易“姿态”
                # Agent的“0”信号意味着“保持姿态”，非“0”信号则更新为新姿态
                if action != 0:
                    stance = action

                # 根据“姿态”执行交易
                if stance == 1: # 姿态: 做多
                    if shares < 0: # 如果当前是空头，先平仓
                        cash_needed_to_cover = abs(shares) * trade_price
                        cash -= cash_needed_to_cover
                        print(f"  [交易] 平空仓: 买回 {abs(shares)} 股 at {trade_price:.2f}")
                        shares = 0
                    
                    if shares == 0 and cash > 0: # 如果是空仓，则全仓买入
                        shares_to_buy = cash // trade_price
                        shares += shares_to_buy
                        cash -= shares_to_buy * trade_price
                        print(f"  [交易] 建多仓: 买入 {shares_to_buy} 股 at {trade_price:.2f}")

                elif stance == -1: # 姿态: 做空
                    if shares > 0: # 如果当前是多头，先平仓
                        cash += shares * trade_price
                        print(f"  [交易] 平多仓: 卖出 {shares} 股 at {trade_price:.2f}")
                        shares = 0

                    if shares == 0: # 如果是空仓，则建立等同于当前现金价值的空头仓位
                        value_to_short = cash # 使用当前现金作为做空的名义价值
                        shares_to_short = value_to_short // trade_price
                        shares -= shares_to_short # 持股变为负数
                        cash += shares_to_short * trade_price # 卖出借来的股票，现金增加
                        print(f"  [交易] 建空仓: 卖空 {shares_to_short} 股 at {trade_price:.2f}")
                
                # 在lookahead期间，逐日更新并记录资产
                lookahead_period_df = window_df.iloc[trade_day_index : trade_day_index + predictor.agent.lookahead]
                for _, day in lookahead_period_df.iterrows():
                    # 核心公式：总资产 = 现金 + 持股价值。持股为负时，自动扣除空头负债。
                    daily_value = cash + shares * day['close']
                    portfolio_values.append(daily_value)
                    all_trade_dates.append(day['date'])
                
                print(f"  [资产] 窗口结束时总资产: {portfolio_values[-1]:.2f}")

            print(f"\n🎉 EATA 项目回测完成 ({ticker})！")
            
            # 7. 计算并展示专业指标
            print("\n[Main] 正在计算策略表现指标...")
            portfolio_df = pd.DataFrame({'value': portfolio_values}, index=pd.to_datetime(all_trade_dates))

            # 修复: QuantStats不允许重复的索引。删除重复日期，保留最后一次的记录。
            portfolio_df = portfolio_df[~portfolio_df.index.duplicated(keep='last')]

            # --- 核心修复：为资产曲线和指标计算增加统一的“第0天”起点 ---
            # 1. 找到回测期开始的前一个交易日
            if portfolio_df.empty:
                print(f"  [WARN] 股票 {ticker} 没有产生任何交易记录，无法生成图表和报告。")
                continue
            first_trade_date = portfolio_df.index[0]
            first_date_loc_series = stock_df.index[stock_df['date'] == first_trade_date]
            if first_date_loc_series.empty:
                print(f"  [WARN] 无法在原始数据中定位到首次交易日期 {first_trade_date}，跳过T0点对齐。")
                start_day_minus_one_loc = -1
            else:
                first_date_loc = first_date_loc_series[0]
                start_day_minus_one_loc = first_date_loc - 1


            if start_day_minus_one_loc >= 0:
                start_date_t0 = stock_df.loc[start_day_minus_one_loc, 'date']
                
                # 2. 创建一个代表“第0天”的DataFrame
                start_row = pd.DataFrame({'value': [initial_cash]}, index=[start_date_t0])
                
                # 3. 将“第0天”拼接到Agent的资产数据前
                portfolio_df = pd.concat([start_row, portfolio_df])
                print(f"  [绘图修复] 已为资产曲线添加共同起点: {start_date_t0.date()}，初始资产: {initial_cash}")
            else:
                print("  [绘图修复] 警告：无法找到回测前一日，资产曲线可能没有T0起点。")
            # --- 结束修复 ---

            daily_returns = portfolio_df['value'].pct_change().dropna()

            # 计算基准策略（买入并持有） - 更稳健的方法
            # 1. 确保原始数据以日期为索引，以便高效查找
            stock_df_indexed = stock_df.set_index('date')

            # 2. 从原始数据中，提取与我们策略回测期间完全对应的收盘价
            benchmark_prices = stock_df_indexed.loc[portfolio_df.index, 'close']

            # 3. 计算基准收益率
            buy_and_hold_returns = benchmark_prices.pct_change().dropna()

            metrics = TradingMetrics(returns=daily_returns.values, benchmark_returns=buy_and_hold_returns.values)
            metrics.print_metrics(f"EATA Agent 策略表现 ({ticker})") # 打印时带上股票代码

            # 8. 绘制并保存资产曲线图
            print("\n[Main] 正在绘制资产曲线图...")
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, ax = plt.subplots(figsize=(16, 8))

            # --- 新增：绘制背景颜色 ---
            for span in action_spans:
                if span['action'] == 1: # 买入
                    ax.axvspan(span['start'], span['end'], facecolor='#90ee90', alpha=0.2, linewidth=0)
                elif span['action'] == -1: # 卖出
                    ax.axvspan(span['start'], span['end'], facecolor='#ffcccb', alpha=0.2, linewidth=0)
            # --- 结束新增 ---

            # --- 核心修复：使用统一起点后的数据进行绘图 ---
            # 1. 绘制Agent策略曲线 (现在包含了T0点)
            ax.plot(portfolio_df.index, portfolio_df['value'], label='EATA Agent Strategy', color='royalblue', linewidth=2)

            # 2. 绘制买入并持有基准曲线 (基于同样包含T0的benchmark_prices)
            #    使用更清晰的归一化方法计算，确保起点一致
            benchmark_value = (benchmark_prices / benchmark_prices.iloc[0]) * initial_cash
            ax.plot(benchmark_value.index, benchmark_value.values, label='Buy and Hold Benchmark', color='grey', linestyle='--', linewidth=2)
            # --- 结束修复 ---
            
            ax.set_title(f'EATA Agent vs. Buy and Hold Performance ({ticker})', fontsize=18)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Portfolio Value', fontsize=12)
            ax.legend(fontsize=12)
            plt.tight_layout()
            
            # 保存图表 (文件名包含股票代码和项目名称)
            figure_path = f'asset_curve_{args.project_name}_{ticker}_1.png'
            plt.savefig(figure_path)
            plt.close(fig) # 关闭图表，释放内存
            print(f"\n📈 资产曲线图已成功保存到: {figure_path}")

            # 9. 生成 QuantStats 报告
            print("\n[Main] 正在生成 QuantStats 详细报告...")
            try:
                # 确保索引是 DatetimeIndex 以兼容 QuantStats
                daily_returns.index = pd.to_datetime(daily_returns.index)
                buy_and_hold_returns.index = pd.to_datetime(buy_and_hold_returns.index)
                
                report_path = f'EATA_Strategy_Report_{args.project_name}_{ticker}_1.html' # 文件名包含股票代码和项目名称
                qs.reports.html(daily_returns, benchmark=buy_and_hold_returns, output=report_path, title=f'{ticker} - EATA Agent Performance')
                print(f"\n📊 QuantStats 报告已成功保存到: {report_path}")
            except Exception as e:
                print(f"\n⚠️ 生成 QuantStats 报告失败 ({ticker}): {e}")

            # 10. 新增：绘制并保存RL奖励趋势图
            print("\n[Main] 正在绘制RL奖励趋势图...")
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, ax = plt.subplots(figsize=(16, 8))
            
            reward_series = pd.Series(rl_rewards_history)
            moving_avg = reward_series.rolling(window=50).mean()

            ax.plot(reward_series.index, reward_series, label='Raw RL Reward', color='lightsteelblue', alpha=0.7)
            ax.plot(moving_avg.index, moving_avg, label='50-Window Moving Average', color='crimson', linewidth=2)
            
            ax.set_title(f'RL Reward Trend Over Windows ({ticker})', fontsize=18)
            ax.set_xlabel('Window Number', fontsize=12)
            ax.set_ylabel('RL Reward', fontsize=12)
            ax.legend(fontsize=12)
            plt.tight_layout()
            
            # 保存图表 (文件名包含股票代码和项目名称)
            reward_figure_path = f'rl_reward_trend_{args.project_name}_{ticker}_1.png'
            plt.savefig(reward_figure_path)
            plt.close(fig) # 关闭图表，释放内存
            print(f"\n📉 RL奖励趋势图已成功保存到: {reward_figure_path}")

            # 收集当前股票的指标，用于最终汇总
            current_metrics = metrics.get_all_metrics()
            current_metrics['Ticker'] = ticker # 添加股票代码
            all_results.append(current_metrics)

        # 11. 打印最终的汇总结果
        print(f"\n\n{'='*60}")
        print(f"🏆 EATA策略三股票回测汇总")
        print(f"参数: lookback=50, lookahead=10, stride=1, depth=300")
        print(f"{'='*60}")
        
        # 简化对比表格
        if all_results:
            print(f"{'股票':8s} {'年化收益':>10s} {'夏普比率':>8s} {'最大回撤':>8s} {'盈利因子':>8s}")
            print("-" * 50)
            for result in all_results:
                ticker = result['Ticker']
                annual_return = result['Annual Return (AR)'] * 100
                sharpe = result['Sharpe Ratio']
                max_dd = result['Max Drawdown (MDD)'] * 100
                profit_factor = result['Profit Factor']
                print(f"{ticker:8s} {annual_return:9.2f}% {sharpe:7.2f} {max_dd:7.2f}% {profit_factor:7.2f}")
        
        print(f"\n{'='*25} 详细指标汇总 {'='*25}")
        results_df = pd.DataFrame(all_results)
        # 格式化百分比列
        for col in ['Annual Return (AR)', 'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown (MDD)', 'Calmar Ratio', 'Win Rate', 'Volatility (Annual)', 'Alpha', 'IRR']:
            if col in results_df.columns:
                results_df[col] = results_df[col].apply(lambda x: f"{x*100:.2f}%")
        # 格式化其他数值列
        for col in ['Beta', 'Profit Factor']:
            if col in results_df.columns:
                results_df[col] = results_df[col].apply(lambda x: f"{x:.2f}")
        
        print(results_df.to_string()) # 使用to_string()防止截断
        print("="*60)

    except Exception as e:
        print(f"\n❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()
