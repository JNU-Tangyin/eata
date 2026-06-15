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
from core.agent import Agent
from core.data import DataStorage # 导入数据存储类
from core.performance_metrics import TradingMetrics # 导入我们新增的指标计算模块

# 导入复杂度计算工具（处理导入失败的情况）
try:
    import sys
    import os
    # 确保utils目录在路径中
    utils_path = os.path.join(os.path.dirname(__file__), 'utils')
    if utils_path not in sys.path:
        sys.path.insert(0, utils_path)
    from expression_complexity import count_ast_nodes, estimate_method_complexity
except ImportError:
    # 如果导入失败，提供简单的替代实现
    def count_ast_nodes(expr):
        """简单的复杂度估计：基于表达式长度"""
        return len(str(expr).split())
    def estimate_method_complexity(method):
        return 1

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
        print(f"🤖 新版 Predictor 初始化完成，参数: lookback={lookback}, lookahead={lookahead}, stride={stride}, depth={depth}")

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
    variant_params: dict = None,
    pre_configured_agent: Agent = None,
    variant_mode: str = None,  # 🎯 架构级变体模式标记
):
    """在给定单支股票数据上运行 EATA 本体的核心回测逻辑。

    注意：
    - 这里只负责滑动窗口 + 交易 + 指标计算，不做画图、不生成 HTML 报告。
    - 返回值与 baseline 侧对接用：metrics(dict) + portfolio_df(DataFrame[value] 索引为日期)。
    """

    # 初始化 Predictor / Agent，支持预配置的Agent实例
    if pre_configured_agent is not None:
        print(f"🔄 使用预配置的Agent实例 (变体已应用修改)")
        predictor = Predictor(lookback=lookback, lookahead=lookahead, stride=stride, depth=depth)
        predictor.agent = pre_configured_agent
        print(f"   Agent类型: {type(pre_configured_agent).__name__}")
        print(f"   Agent修改状态: 已预配置")
    else:
        print(f"🔄 创建新的Agent实例")
        predictor = Predictor(lookback=lookback, lookahead=lookahead, stride=stride, depth=depth)
    
    # 🔧 方案1：参数化方法调用 - 提取关键参数用于直接传递
    variant_profit_loss_weight = None
    variant_exploration_rate = None
    
    print(f"🔧 [方案1] variant_params检查: {variant_params}")
    
    if variant_params:
        print(f"🔧 [方案1] 收到变体参数: {variant_params}")
        
        # 提取关键参数用于直接传递
        variant_profit_loss_weight = variant_params.get('profit_loss_weight')
        variant_exploration_rate = variant_params.get('exploration_rate')
        variant_alpha = variant_params.get('alpha')  # 修复：提取alpha参数
        variant_skip_mcts = variant_params.get('skip_mcts')  # 🔧 新增：提取skip_mcts参数
        variant_skip_nn = variant_params.get('skip_nn')  # 🔧 新增：提取skip_nn参数
        variant_skip_memory = variant_params.get('skip_memory')  # 🔧 新增：提取skip_memory参数
        
        # 🎯 新增：提取Simple变体的目标函数参数
        variant_objective_function = variant_params.get('objective_function')  # MSE/KL/JS/CVaR
        variant_distance_calculator = variant_params.get('distance_calculator')  # 距离计算函数
        variant_custom_score_function = variant_params.get('custom_score_function')  # 🎯 自定义score函数
        
        print(f"🔧 [方案1] 提取的关键参数:")
        print(f"   - profit_loss_weight: {variant_profit_loss_weight}")
        print(f"   - exploration_rate: {variant_exploration_rate}")
        print(f"   - alpha: {variant_alpha}")
        print(f"   - skip_mcts: {variant_skip_mcts}")
        print(f"   - skip_nn: {variant_skip_nn}")
        print(f"   - skip_memory: {variant_skip_memory}")
        print(f"   - objective_function: {variant_objective_function}")
        print(f"   - distance_calculator: {variant_distance_calculator}")
        print(f"   - custom_score_function: {variant_custom_score_function}")
        
        # 设置Agent上的变体参数标识，供criteria()使用
        if variant_profit_loss_weight is not None:
            predictor.agent.engine._variant_profit_loss_weight = variant_profit_loss_weight
        if variant_exploration_rate is not None:
            predictor.agent.engine._variant_exploration_rate = variant_exploration_rate
            print(f"   ✅ exploration_rate={variant_exploration_rate} 已注入到 engine")
        if variant_alpha is not None:
            predictor.agent._variant_alpha = variant_alpha  # 修复：注入alpha到agent
            print(f"   ✅ alpha={variant_alpha} 已注入到 agent")
        if variant_skip_mcts is not None:
            predictor.agent.engine.model._variant_skip_mcts = variant_skip_mcts  # 🔧 新增：注入skip_mcts到model
            print(f"   ✅ skip_mcts={variant_skip_mcts} 已注入到 model")
        if variant_skip_nn is not None:
            predictor.agent.engine.model._variant_skip_nn = variant_skip_nn  # 🔧 新增：注入skip_nn到model
            print(f"   ✅ skip_nn={variant_skip_nn} 已注入到 model")
        if variant_skip_memory is not None:
            predictor.agent.engine.model._variant_skip_memory = variant_skip_memory  # 🔧 新增：注入skip_memory到model
            print(f"   ✅ skip_memory={variant_skip_memory} 已注入到 model")
        
        # 🎯 新增：注入Simple变体的目标函数参数
        if variant_objective_function is not None:
            predictor.agent._variant_objective = variant_objective_function
            print(f"   ✅ objective_function={variant_objective_function} 已注入到 agent")
        if variant_distance_calculator is not None:
            predictor.agent._variant_distance_calculator = variant_distance_calculator
            print(f"   ✅ distance_calculator 已注入到 agent")
        if variant_custom_score_function is not None:
            # 🎯 关键修复：将自定义score函数注入到模型中，让MCTS使用它
            predictor.agent.engine.model._variant_custom_score_function = variant_custom_score_function
            print(f"   ✅ custom_score_function 已注入到 model (将影响MCTS搜索)")
        
    # 🎯 架构级变体模式设置：通过环境变量控制
    if variant_mode:
        print(f"🔧 [消融实验] 设置变体模式: {variant_mode}")
        
        # 设置环境变量启用消融实验模式
        import os
        os.environ['ABLATION_EXPERIMENT_MODE'] = 'true'
        print(f"   ✅ 环境变量ABLATION_EXPERIMENT_MODE已设置为true")
        
        # 设置神经网络容器的变体模式（确保网络重建后能恢复）
        if hasattr(predictor.agent.engine.model, 'p_v_net_ctx'):
            predictor.agent.engine.model.p_v_net_ctx._variant_mode = variant_mode
            # 同时设置当前网络实例
            if hasattr(predictor.agent.engine.model.p_v_net_ctx, 'pv_net'):
                predictor.agent.engine.model.p_v_net_ctx.pv_net._variant_mode = variant_mode
            print(f"   ✅ 神经网络层面变体模式已设置: {variant_mode}")
        else:
            print(f"   ⚠️ 无法访问神经网络，变体模式设置失败")
        
        # 🎯 强制启用消融实验模式（确保隔离方案正确工作）
        if variant_params:
            print(f"🔧 [强制] 通过环境变量启用消融实验模式")
            import os
            os.environ['ABLATION_EXPERIMENT_MODE'] = 'true'
        
        # 仍然使用新的统一参数应用器处理其他参数
        try:
            from ablation_study.variant_system import VariantParameterApplier
            success = VariantParameterApplier.apply_to_agent(predictor.agent, variant_params)
            
            if success:
                print(f"✅ [方案1] 其他变体参数应用成功")
            else:
                print(f"⚠️ [方案1] 其他变体参数应用部分失败，但继续执行")
                
        except ImportError:
            # 回退到旧的应用方式
            print(f"🔄 [方案1] 回退到旧的参数应用方式")
            from variant_modifier import _apply_variant_modifications
            _apply_variant_modifications(predictor.agent, variant_params)
            
    else:
        print(f"ℹ️ [方案1] 无变体参数，使用默认配置")

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
    rl_rewards_history = []  # 收集RL rewards
    
    # 🔧 新增：收敛曲线记录
    convergence_history = []  # 记录每个窗口后的累积Sharpe Ratio
    window_timestamps = []  # 记录每个窗口的时间戳
    
    # 🔧 新增：表达式和复杂度记录（用于Pareto Frontier图）
    discovered_expressions = []  # 记录发现的所有表达式
    expression_complexities = []  # 记录对应的复杂度
    expression_performances = []  # 记录对应的性能（Sharpe Ratio）

    # 🔧 方案A：滑动窗口回测使用完整数据集（与主实验一致）
    # 从数据集末尾往前滑动（与原始逻辑一致）
    max_test_windows = len(stock_df) - window_len + 1
    num_test_windows = min(1000, max_test_windows)
    
    if num_test_windows < 50:
        raise ValueError(f"数据不足，只能进行 {num_test_windows} 次窗口测试（最少需要50次）")
    
    print(f"📊 完整数据回测：将进行 {num_test_windows} 次窗口测试（stride=2）")
    print(f"   回测方式：从末尾往前滑动（与原始逻辑一致）")
    
    # 🔧 滑动窗口回测（从末尾往前，stride=2，与原始逻辑一致）
    window_count = 0
    for i in range(0, num_test_windows, 2):
        offset = num_test_windows - 1 - i
        start_index = -(window_len + offset)
        end_index = -offset if offset > 0 else None
        
        # 使用完整数据集进行回测
        window_df = stock_df.iloc[start_index:end_index].copy()
        window_count += 1
        window_df.reset_index(drop=True, inplace=True)

        # 调用 Agent 决策
        action, rl_reward = predictor.predict(df=window_df, shares_held=shares)
        rl_rewards_history.append(rl_reward)  # 收集RL reward
        
        # 🔧 新增：记录当前窗口发现的表达式和复杂度
        if hasattr(predictor.agent, 'last_discovered_expression'):
            expr = predictor.agent.last_discovered_expression
            if expr and expr != '0':
                complexity = count_ast_nodes(expr)
                discovered_expressions.append(expr)
                expression_complexities.append(complexity)
                # 性能将在最后统一计算

        # 交易发生在 lookback 之后的第一天
        trade_day_index = predictor.agent.lookback
        trade_price = window_df.loc[trade_day_index, 'open']

        # 🔧 信号强度过滤：只有足够强的信号才执行交易
        # 这可以减少在趋势市场中的过度交易，提高持仓稳定性
        original_action = action
        filtered_signal = False
        if action != 0:
            # 根据不同情况设置阈值
            if stance == 0:
                # 情况1: 从空仓建仓 - 使用基础阈值
                signal_strength_threshold = 0.3
            elif action == stance:
                # 情况2: 确认继续持有当前方向 - 使用较低阈值（容易通过）
                signal_strength_threshold = 0.2
            else:
                # 情况3: 切换方向（多转空或空转多）- 使用更高阈值
                signal_strength_threshold = 0.5
            
            # 应用信号强度过滤
            if abs(rl_reward) < signal_strength_threshold:
                action = 0  # 弱信号不执行，保持当前仓位
                filtered_signal = True
                if window_count <= 20:  # 前20个窗口打印详细日志
                    print(f"   🔽 Window {window_count}: 信号过滤 - RL reward {rl_reward:.4f} < 阈值 {signal_strength_threshold:.2f}, 保持当前仓位 (stance={stance})")
        
        # 更新姿态
        if action != 0:
            stance = action
            if window_count <= 20:  # 前20个窗口打印详细日志
                print(f"   ✅ Window {window_count}: 执行信号 - action={action}, RL reward={rl_reward:.4f}, 新仓位={stance}")

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
        
        # 🔧 新增：计算当前累积Sharpe Ratio
        if len(portfolio_values) > 10:  # 至少需要10个数据点才能计算Sharpe
            temp_portfolio = pd.Series(portfolio_values)
            temp_returns = temp_portfolio.pct_change().dropna()
            if len(temp_returns) > 0 and temp_returns.std() > 0:
                cumulative_sharpe = (temp_returns.mean() / temp_returns.std()) * np.sqrt(252)
                convergence_history.append(cumulative_sharpe)
                window_timestamps.append(window_count)  # 记录窗口索引作为时间戳

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
    
    # 添加平均RL reward到指标中
    if rl_rewards_history:
        # 过滤掉nan和inf值
        valid_rewards = [r for r in rl_rewards_history if not (np.isnan(r) or np.isinf(r))]
        if valid_rewards:
            avg_rl_reward = np.mean(valid_rewards)
            print(f"📊 平均RL奖励: {avg_rl_reward:.6f} (有效样本: {len(valid_rewards)}/{len(rl_rewards_history)})")
        else:
            avg_rl_reward = 0.0
            print(f"⚠️ 所有RL奖励都是无效值 (nan/inf)，设置为0.0")
        metrics['Average RL Reward'] = avg_rl_reward
    else:
        metrics['Average RL Reward'] = 0.0
        print(f"⚠️ 没有收集到RL奖励历史，设置为0.0")
    
    # 🔧 新增：添加收敛历史数据
    metrics['Convergence History'] = convergence_history
    metrics['Window Timestamps'] = window_timestamps
    if convergence_history:
        print(f"📈 收敛曲线记录: {len(convergence_history)} 个数据点")
    
    # 🔧 新增：添加表达式和复杂度数据（用于Pareto Frontier图）
    metrics['Discovered Expressions'] = discovered_expressions
    metrics['Expression Complexities'] = expression_complexities
    if discovered_expressions:
        print(f"🔍 表达式记录: {len(discovered_expressions)} 个表达式")
        print(f"   平均复杂度: {np.mean(expression_complexities):.1f} 节点")
        print(f"   复杂度范围: {min(expression_complexities)}-{max(expression_complexities)} 节点")
    
    # 🔧 调试：打印关键指标
    print(f"📊 关键指标: AR={metrics.get('Annual Return (AR)', 'N/A'):.4f}, Sharpe={metrics.get('Sharpe Ratio', 'N/A'):.4f}, MDD={metrics.get('Max Drawdown (MDD)', 'N/A'):.4f}")

    return metrics, portfolio_df


if __name__ == "__main__":
    # 新增：解析命令行参数
    parser = argparse.ArgumentParser(description="EATA Project Core Function Test, Backtest, and Evaluation (Multi-stock Version)")
    parser.add_argument('--project_name', type=str, default='default',
                        help='Name of the current project/experiment for distinguishing output files.')
    parser.add_argument('--stock_list', type=str, default=None,
                        help='Path to file containing list of stock tickers (one per line)')
    parser.add_argument('--save_portfolio', action='store_true',
                        help='Save portfolio and convergence data for each stock')
    args = parser.parse_args()

    print("🚀 启动 EATA 项目核心功能测试、回测与评估 (多股票版)")
    print("=======================================================")

    try:
        # 1. 从 data/ 目录加载CSV文件
        print("\n[Main] 从 data/ 目录加载CSV文件...")
        from pathlib import Path
        import os
        
        # 获取项目根目录（predict.py所在目录）
        project_root = Path(__file__).parent
        data_dir = project_root / 'data'
        
        print(f"   数据目录: {data_dir}")
        csv_files = list(data_dir.glob('*.csv'))
        
        if not csv_files:
            raise Exception("data/ 目录中没有找到CSV文件。")
        
        print(f"   找到 {len(csv_files)} 个CSV文件")
        
        # 加载所有CSV文件
        all_data_list = []
        for csv_file in csv_files:
            ticker = csv_file.stem  # 文件名即股票代码
            try:
                df = pd.read_csv(csv_file)
                # 标准化列名为小写
                df.columns = df.columns.str.lower()
                df['code'] = ticker  # 添加股票代码列
                all_data_list.append(df)
            except Exception as e:
                print(f"   ⚠️ 加载 {ticker} 失败: {e}")
                continue
        
        if not all_data_list:
            raise Exception("没有成功加载任何数据。")
        
        all_data = pd.concat(all_data_list, ignore_index=True)
        all_data['date'] = pd.to_datetime(all_data['date'])
        all_data = all_data.sort_values(['code', 'date']).reset_index(drop=True)
        
        # 添加amount列（如果不存在）- 与原始逻辑保持一致，使用0
        if 'amount' not in all_data.columns:
            all_data['amount'] = 0
            print(f"   ✅ 添加amount列 (amount = 0，与原始逻辑一致)")
        
        print(f"✅ 数据加载完成: {len(all_data)} 条记录，{all_data['code'].nunique()} 支股票")

        # 2. 读取股票列表
        if args.stock_list:
            # 从文件读取股票列表
            stock_list_path = project_root / args.stock_list if not Path(args.stock_list).is_absolute() else Path(args.stock_list)
            print(f"\n[Main] 从文件读取股票列表: {stock_list_path}")
            with open(stock_list_path, 'r') as f:
                target_tickers = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            print(f"   ✅ 读取到 {len(target_tickers)} 支股票")
        else:
            # 默认测试三支股票
            target_tickers = ['AAPL', 'MSFT', 'GOOGL']
            print(f"\n[Main] 使用默认测试股票: {target_tickers}")
        
        all_available_tickers = all_data['code'].unique()
        
        # 筛选出实际可用的股票
        available_tickers = [ticker for ticker in target_tickers if ticker in all_available_tickers]
        if not available_tickers:
            print(f"❌ 目标股票 {target_tickers} 在数据中不可用")
            print(f"📊 可用股票: {list(all_available_tickers)[:10]}...")
            exit(1)
        
        print(f"[Main] 将测试 {len(available_tickers)} 支股票")
        all_tickers = available_tickers

        # 3. 初始化一个列表来存储所有股票的最终指标
        all_results = []

        # 4. 外层循环：遍历每一支股票
        for ticker_idx, ticker in enumerate(all_tickers):
            print(f"\n\n{'='*15} 开始回测股票: {ticker} ({ticker_idx + 1}/{len(all_tickers)}) {'='*15}")
            
            try:
                # 获取股票数据
                stock_df = all_data[all_data['code'] == ticker].copy()
                stock_df['date'] = pd.to_datetime(stock_df['date'])
                stock_df.sort_values(by='date', inplace=True)
                stock_df.reset_index(drop=True, inplace=True)
                
                # 使用run_eata_core_backtest函数（包含convergence数据）
                metrics, portfolio_df = run_eata_core_backtest(
                    stock_df=stock_df,
                    ticker=ticker,
                    lookback=50,
                    lookahead=10,
                    stride=1,
                    depth=300
                )
                
                print(f"✅ {ticker} 回测完成")
                
                # 收集当前股票的指标
                current_metrics = {
                    'Ticker': ticker,
                    'Annual Return (AR)': metrics.get('Annual Return (AR)', 0),
                    'Sharpe Ratio': metrics.get('Sharpe Ratio', 0),
                    'Max Drawdown (MDD)': metrics.get('Max Drawdown (MDD)', 0),
                    'Convergence History': metrics.get('Convergence History', []),
                    'Window Timestamps': metrics.get('Window Timestamps', [])
                }
                all_results.append(current_metrics)
                
                # 保存portfolio和convergence数据（用于Figure 3和Figure 6）
                if args.save_portfolio:
                    import json
                    
                    # 创建输出目录（使用绝对路径）
                    output_dir = project_root / 'results' / 'eata_full_62stocks'
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 保存portfolio数据（使用绝对路径）
                    portfolio_file = output_dir / f'{ticker}_portfolio.csv'
                    portfolio_df.to_csv(str(portfolio_file), index=True)
                    print(f"   ✅ 保存portfolio: {portfolio_file.relative_to(project_root)}")
                    
                    # 保存convergence history（使用绝对路径）
                    convergence_data = {
                        'ticker': ticker,
                        'convergence_history': current_metrics.get('Convergence History', []),
                        'window_timestamps': current_metrics.get('Window Timestamps', []),
                        'annual_return': current_metrics.get('Annual Return (AR)', 0),
                        'sharpe_ratio': current_metrics.get('Sharpe Ratio', 0),
                        'max_drawdown': current_metrics.get('Max Drawdown (MDD)', 0)
                    }
                    convergence_file = output_dir / f'{ticker}_convergence.json'
                    with open(str(convergence_file), 'w') as f:
                        json.dump(convergence_data, f, indent=2)
                    print(f"   ✅ 保存convergence: {convergence_file.relative_to(project_root)}")
                    
            except Exception as e:
                print(f"   ❌ {ticker} 失败: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        # 打印最终的汇总结果
        print(f"\n\n{'='*60}")
        print(f"🏆 EATA策略三股票回测汇总")
        print(f"参数: lookback=50, lookahead=10, stride=1, depth=300")
        print(f"{'='*60}")
        
        # 简化对比表格
        if all_results:
            print(f"{'股票':8s} {'年化收益':>10s} {'夏普比率':>8s} {'最大回撤':>8s}")
            print("-" * 50)
            for result in all_results:
                ticker = result['Ticker']
                annual_return = result['Annual Return (AR)'] * 100
                sharpe = result['Sharpe Ratio']
                max_dd = result['Max Drawdown (MDD)'] * 100
                print(f"{ticker:8s} {annual_return:9.2f}% {sharpe:7.2f} {max_dd:7.2f}%")
        
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
