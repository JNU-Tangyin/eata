#!/usr/bin/env python3
"""
EATA-RL 参数调优脚本
支持24种参数组合的自动化测试和时间记录

参数组合:
- lookback: [50, 100]
- lookahead: [10, 20] 
- stride: [1, 2, 5]
- depth (transplant_step): [300, 800]

总计: 2 × 2 × 3 × 2 = 24 种组合
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import product
import sqlite3
import argparse
import warnings

# 屏蔽各种警告信息
np.seterr(all='ignore')
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*OpenSSL.*')
warnings.filterwarnings('ignore', message='.*urllib3.*')
warnings.filterwarnings('ignore', message='.*Gym.*')
warnings.filterwarnings('ignore', message='.*findfont.*')
warnings.filterwarnings('ignore', message='.*SimHei.*')

# 屏蔽matplotlib字体警告
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['DejaVu Sans']  # 使用系统默认字体

# 屏蔽日志输出
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent import Agent
from data import DataStorage
from performance_metrics import TradingMetrics

# 进一步屏蔽调试信息
logging.getLogger().setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)

class ParameterTuner:
    def __init__(self, test_stock='AMZN', num_test_windows=500):
        """
        初始化参数调优器
        
        Args:
            test_stock: 测试股票代码
            num_test_windows: 测试窗口数量
        """
        self.test_stock = test_stock
        self.num_test_windows = num_test_windows
        self.results = []
        
        # 参数组合定义
        self.param_combinations = list(product(
            [50, 100],      # lookback
            [10, 20],       # lookahead  
            [1, 2, 5],      # stride
            [300, 800]      # depth (transplant_step)
        ))
        
        print(f"🔧 参数调优器初始化完成")
        print(f"   测试股票: {test_stock}")
        print(f"   测试窗口数: {num_test_windows}")
        print(f"   参数组合数: {len(self.param_combinations)}")
        print(f"   实际测试窗口数范围: {num_test_windows//5}-{num_test_windows} (取决于stride)")
        
    def get_current_params(self):
        """获取当前使用的参数配置"""
        return {
            'lookback': 100,
            'lookahead': 20, 
            'stride': 2,
            'depth': 800
        }
    
    def load_test_data(self):
        """加载测试数据"""
        # 静默加载数据，减少输出
        
        try:
            # 尝试从stock_large.db加载
            conn = sqlite3.connect('stock_large.db')
            query = f"SELECT * FROM raw_data WHERE code = '{self.test_stock}' ORDER BY date"
            stock_df = pd.read_sql_query(query, conn)
            conn.close()
            
            if stock_df.empty:
                raise Exception(f"在stock_large.db中未找到股票 {self.test_stock}")
                
        except Exception as e:
            print(f"从stock_large.db加载失败: {e}")
            # 尝试从stock.db加载
            try:
                data_storage = DataStorage()
                all_data = data_storage.load_raw()
                stock_df = all_data[all_data['code'] == self.test_stock].copy()
                
                if stock_df.empty:
                    raise Exception(f"在stock.db中也未找到股票 {self.test_stock}")
                    
            except Exception as e2:
                print(f"从stock.db加载也失败: {e2}")
                raise Exception("无法加载测试数据")
        
        stock_df['date'] = pd.to_datetime(stock_df['date'])
        stock_df.sort_values(by='date', inplace=True)
        stock_df.reset_index(drop=True, inplace=True)
        
        # 静默返回数据
        
        return stock_df
    
    def run_single_test(self, lookback, lookahead, stride, depth):
        """
        运行单个参数组合的测试
        
        Args:
            lookback: 回看窗口大小
            lookahead: 前瞻窗口大小  
            stride: 滑动步长
            depth: 搜索深度 (transplant_step)
            
        Returns:
            dict: 测试结果
        """
        print(f"\n🧪 测试组合: L{lookback}_A{lookahead}_S{stride}_D{depth}")
        
        start_time = time.time()
        
        try:
            # 加载数据
            stock_df = self.load_test_data()
            
            # 创建自定义Agent
            agent = Agent(df=pd.DataFrame(), lookback=lookback, lookahead=lookahead)
            
            # 修改Agent的depth参数
            agent.hyperparams.transplant_step = depth
            agent.engine.model.transplant_step = depth
            
            print(f"   Agent就绪, 开始{self.num_test_windows}窗口测试...")
            
            # 检查数据是否足够
            window_len = lookback + lookahead + 1
            if len(stock_df) < window_len + self.num_test_windows - 1:
                raise Exception(f"数据不足: 需要{window_len + self.num_test_windows - 1}, 实际{len(stock_df)}")
            
            # 初始化模拟账户
            initial_cash = 1_000_000
            cash = initial_cash
            shares = 0
            portfolio_values = []
            rl_rewards = []
            window_times = []
            
            # 运行回测 - 使用指定的stride
            actual_windows = 0
            for i in range(0, self.num_test_windows, stride):
                window_start_time = time.time()
                
                # 计算窗口数据
                offset = self.num_test_windows - 1 - i
                start_index = -(window_len + offset)
                end_index = -offset if offset > 0 else None
                
                window_df = stock_df.iloc[start_index:end_index].copy()
                window_df.reset_index(drop=True, inplace=True)
                
                # 获取预测
                action, rl_reward = agent.criteria(window_df, shares_held=shares)
                rl_rewards.append(rl_reward)
                
                # 模拟交易
                trade_day_index = lookback
                trade_price = window_df.loc[trade_day_index, 'open']
                
                if action == 1 and cash > trade_price:  # 买入
                    new_shares = int(cash // trade_price)
                    shares += new_shares
                    cash -= new_shares * trade_price
                elif action == -1 and shares > 0:  # 卖出
                    cash += shares * trade_price
                    shares = 0
                
                # 记录资产价值
                current_price = window_df.loc[trade_day_index, 'close']
                portfolio_value = cash + shares * current_price
                portfolio_values.append(portfolio_value)
                
                window_time = time.time() - window_start_time
                window_times.append(window_time)
                actual_windows += 1
                
                if actual_windows % 20 == 0:
                    avg_window_time = np.mean(window_times[-20:])
                    progress = actual_windows / (self.num_test_windows // stride) * 100
                    print(f"   进度: {progress:.0f}% ({actual_windows} 个窗口), 平均用时: {avg_window_time:.2f}s")
            
            # 计算性能指标
            if len(portfolio_values) > 1:
                returns = pd.Series(portfolio_values).pct_change().dropna()
                total_return = (portfolio_values[-1] - initial_cash) / initial_cash
                
                # 计算年化收益率 (假设252个交易日/年)
                num_periods = len(portfolio_values)
                years = num_periods / 252  # 交易日转年数
                if years > 0:
                    annualized_return = (1 + total_return) ** (1/years) - 1
                else:
                    annualized_return = 0
                
                if len(returns) > 0 and returns.std() > 0:
                    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
                else:
                    sharpe_ratio = 0
                
                # 计算最大回撤
                cumulative = pd.Series(portfolio_values)
                running_max = cumulative.cummax()
                drawdown = (running_max - cumulative) / running_max
                max_drawdown = drawdown.max()
                
                avg_rl_reward = np.mean(rl_rewards) if rl_rewards else 0
                avg_window_time = np.mean(window_times) if window_times else 0
            else:
                total_return = sharpe_ratio = max_drawdown = avg_rl_reward = avg_window_time = annualized_return = 0
            
            end_time = time.time()
            total_time = end_time - start_time
            
            result = {
                'lookback': lookback,
                'lookahead': lookahead,
                'stride': stride,
                'depth': depth,
                'total_time': total_time,
                'avg_time_per_window': avg_window_time,
                'total_return': total_return,
                'annualized_return': annualized_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'avg_rl_reward': avg_rl_reward,
                'final_portfolio_value': portfolio_values[-1] if portfolio_values else initial_cash,
                'num_windows_completed': actual_windows,
                'success': True,
                'error': None
            }
            
            print(f"   ✅ 测试完成! 用时: {total_time:.1f}s, 总收益: {total_return:.2%}, 夏普比率: {sharpe_ratio:.2f}")
            
        except Exception as e:
            end_time = time.time()
            total_time = end_time - start_time
            
            result = {
                'lookback': lookback,
                'lookahead': lookahead,
                'stride': stride,
                'depth': depth,
                'total_time': total_time,
                'success': False,
                'error': str(e),
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'avg_rl_reward': 0,
                'num_windows_completed': 0
            }
            
            print(f"   ❌ 测试失败: {e}, 用时: {total_time:.1f}s")
        
        return result
    
    def run_all_tests(self):
        """运行所有参数组合的测试"""
        print(f"\n🚀 开始运行所有 {len(self.param_combinations)} 种参数组合的测试...")
        print("=" * 80)
        
        current_params = self.get_current_params()
        print(f"📋 当前使用的参数配置:")
        print(f"   lookback: {current_params['lookback']}")
        print(f"   lookahead: {current_params['lookahead']}")
        print(f"   stride: {current_params['stride']}")
        print(f"   depth: {current_params['depth']}")
        print("=" * 80)
        
        total_start_time = time.time()
        
        for i, (lookback, lookahead, stride, depth) in enumerate(self.param_combinations):
            print(f"\n" + "="*80)
            print(f"🎯 [{self.test_stock}] 参数组合进度: {i + 1}/{len(self.param_combinations)} ({(i+1)/len(self.param_combinations)*100:.1f}%)")
            print(f"📋 当前测试: L{lookback}_A{lookahead}_S{stride}_D{depth}")
            print("="*80)
            
            # 标记当前配置
            is_current = (lookback == current_params['lookback'] and 
                         lookahead == current_params['lookahead'] and
                         stride == current_params['stride'] and 
                         depth == current_params['depth'])
            
            if is_current:
                print("   🎯 这是当前使用的参数配置!")
            
            result = self.run_single_test(lookback, lookahead, stride, depth)
            result['is_current_config'] = is_current
            result['test_order'] = i + 1
            
            self.results.append(result)
            
            # 显示当前测试结果
            if result['success']:
                print(f"✅ 测试完成: 收益{result['total_return']:.2%}, 夏普{result['sharpe_ratio']:.2f}, 用时{result['total_time']:.1f}s")
            else:
                print(f"❌ 测试失败: {result['error']}")
            print("="*80)
            
            # 保存中间结果
            if (i + 1) % 5 == 0:  # 每5个测试保存一次
                self.save_results(intermediate=True)
        
        total_time = time.time() - total_start_time
        print(f"\n🎉 所有测试完成! 总用时: {total_time:.1f}s ({total_time/60:.1f}分钟)")
        
        # 保存最终结果
        self.save_results(intermediate=False)
        self.print_summary()
    
    def save_results(self, intermediate=False):
        """保存测试结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if intermediate:
            filename = f"tuning_results_intermediate_{self.test_stock}_{timestamp}.json"
        else:
            filename = f"tuning_results_final_{self.test_stock}_{timestamp}.json"
        
        results_data = {
            'test_stock': self.test_stock,
            'num_test_windows': self.num_test_windows,
            'timestamp': timestamp,
            'total_combinations': len(self.param_combinations),
            'completed_tests': len(self.results),
            'results': self.results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        if not intermediate:
            print(f"📁 最终结果已保存到: {filename}")
    
    def print_summary(self):
        """打印测试结果摘要"""
        if not self.results:
            print("❌ 没有测试结果可显示")
            return
        
        print(f"\n📈 测试结果摘要 (股票: {self.test_stock})")
        print("=" * 120)
        
        # 按总收益排序
        successful_results = [r for r in self.results if r['success']]
        if not successful_results:
            print("❌ 没有成功的测试结果")
            return
        
        successful_results.sort(key=lambda x: x['total_return'], reverse=True)
        
        print(f"{'排名':<4} {'lookback':<8} {'lookahead':<9} {'stride':<6} {'depth':<5} {'总收益':<10} {'夏普比率':<8} {'最大回撤':<10} {'用时(s)':<8} {'窗口数':<6} {'当前':<4}")
        print("-" * 120)
        
        for i, result in enumerate(successful_results):
            current_mark = "✓" if result.get('is_current_config', False) else ""
            print(f"{i+1:<4} {result['lookback']:<8} {result['lookahead']:<9} {result['stride']:<6} {result['depth']:<5} "
                  f"{result['total_return']:<10.2%} {result['sharpe_ratio']:<8.2f} {result['max_drawdown']:<10.2%} "
                  f"{result['total_time']:<8.1f} {result['num_windows_completed']:<6} {current_mark:<4}")
        
        # 显示当前配置的排名
        current_config_result = next((r for r in self.results if r.get('is_current_config', False)), None)
        if current_config_result and current_config_result['success']:
            current_rank = successful_results.index(current_config_result) + 1
            print(f"\n🎯 当前配置排名: {current_rank}/{len(successful_results)}")
            print(f"   当前配置表现: 收益 {current_config_result['total_return']:.2%}, "
                  f"夏普 {current_config_result['sharpe_ratio']:.2f}, "
                  f"用时 {current_config_result['total_time']:.1f}s")
        
        # 时间统计
        avg_time = np.mean([r['total_time'] for r in successful_results])
        min_time = min([r['total_time'] for r in successful_results])
        max_time = max([r['total_time'] for r in successful_results])
        
        print(f"\n⏱️  时间统计:")
        print(f"   平均用时: {avg_time:.1f}s ({avg_time/60:.1f}分钟)")
        print(f"   最快用时: {min_time:.1f}s ({min_time/60:.1f}分钟)")
        print(f"   最慢用时: {max_time:.1f}s ({max_time/60:.1f}分钟)")
        
        # 最佳配置推荐
        best_result = successful_results[0]
        print(f"\n🏆 最佳配置推荐:")
        print(f"   lookback={best_result['lookback']}, lookahead={best_result['lookahead']}, "
              f"stride={best_result['stride']}, depth={best_result['depth']}")
        print(f"   预期收益: {best_result['total_return']:.2%}, 夏普比率: {best_result['sharpe_ratio']:.2f}")
        print(f"   预计用时: {best_result['total_time']:.1f}s ({best_result['total_time']/60:.1f}分钟)")
        
        # 失败统计
        failed_results = [r for r in self.results if not r['success']]
        if failed_results:
            print(f"\n❌ 失败的测试: {len(failed_results)}/{len(self.results)}")
            for result in failed_results:
                print(f"   lookback={result['lookback']}, lookahead={result['lookahead']}, "
                      f"stride={result['stride']}, depth={result['depth']}: {result['error']}")

def main():
    parser = argparse.ArgumentParser(description="EATA-RL 参数调优工具")
    parser.add_argument('--stock', type=str, default='AMZN', help='测试股票代码')
    parser.add_argument('--windows', type=int, default=100, help='测试窗口数量')
    parser.add_argument('--quick', action='store_true', help='快速测试模式(只测试几个关键组合)')
    parser.add_argument('--multi', action='store_true', help='多股票测试模式(测试AAPL, AAOI, ACIW)')
    parser.add_argument('--single', action='store_true', help='单股票测试模式')
    
    # 如果在PyCharm中直接运行（没有命令行参数），默认使用多股票完整测试
    import sys
    if len(sys.argv) == 1:  # 只有脚本名，没有其他参数
        print("🎯 检测到PyCharm直接运行，启用多股票完整测试模式")
        print("   测试配置: 3支不同趋势股票 × 24种参数组合 × 1000窗口")
        print("   股票组合: AAPL(上升) + AAOI(下跌) + ACIW(震荡)")
        print("   覆盖情况: 牛市 → 熊市 → 横盘市")
        print("   如需其他模式，请使用命令行参数:")
        print("   --single: 单股票模式")
        print("   --quick: 快速模式(4种关键组合)")
        print()
        # 设置默认参数
        args = argparse.Namespace(
            stock='AMZN',
            windows=1000,
            quick=False,  # 改为完整测试模式
            multi=True,
            single=False
        )
    else:
        args = parser.parse_args()
    
    print("🔧 EATA-RL 参数调优工具")
    print("=" * 50)
    
    # 多股票测试模式
    if args.multi:
        test_stocks = ['AAPL', 'AAOI', 'ACIW']  # 三支代表不同趋势的股票
        stock_descriptions = {
            'AAPL': '上升趋势 (+230.5%)',
            'AAOI': '下跌趋势 (-94.6%)', 
            'ACIW': '震荡趋势 (-6.6%)'
        }
        
        print("📊 多股票对比测试模式")
        print("测试股票:")
        for stock in test_stocks:
            print(f"  - {stock}: {stock_descriptions[stock]}")
        print("=" * 50)
        
        all_results = {}
        
        for i, stock in enumerate(test_stocks):
            print(f"\n" + "-" * 60)
            print(f"🎯 开始测试股票 {i+1}/{len(test_stocks)}: {stock}")
            print(f"   特征: {stock_descriptions[stock]}")
            
            # 计算预计时间
            if args.quick:
                total_combinations = 4
                est_time_per_combo = 3  # 分钟
            else:
                total_combinations = 24
                est_time_per_combo = 2  # 分钟
            
            est_total_time = total_combinations * est_time_per_combo
            print(f"   预计测试: {total_combinations}种参数组合")
            print(f"   预计用时: {est_total_time}分钟")
            print("-" * 60)
            
            tuner = ParameterTuner(test_stock=stock, num_test_windows=args.windows)
            
            if args.quick:
                print("⚡ 快速测试模式: 只测试4个关键组合")
                tuner.num_test_windows = args.windows  # 使用指定的窗口数
                tuner.param_combinations = [
                    (50, 10, 1, 300),   # AMZN最优参数
                    (100, 20, 2, 800),  # 当前参数
                    (100, 10, 1, 800),  # 高精度
                    (50, 20, 5, 300)    # 快速模式
                ]
            
            tuner.run_all_tests()
            all_results[stock] = tuner.results
            
            # 输出当前股票的详细结果表格
            print(f"\n" + "=" * 120)
            print(f"测试结果摘要 (股票: {stock})")
            print("=" * 120)
            
            # 过滤成功的结果并排序
            successful_results = [r for r in tuner.results if r['success']]
            if successful_results:
                # 按总收益率排序
                successful_results.sort(key=lambda x: x['total_return'], reverse=True)
                
                print(f"{'排名':<4} {'lookback':<8} {'lookahead':<9} {'stride':<6} {'depth':<5} {'总收益':<12} {'夏普比率':<8} {'最大回撤':<10} {'用时(s)':<8} {'窗口数':<6} {'当前':<4}")
                print("-" * 120)
                
                for rank, result in enumerate(successful_results, 1):
                    current_mark = "✓" if result.get('is_current_config', False) else ""
                    print(f"{rank:<4} {result['lookback']:<8} {result['lookahead']:<9} {result['stride']:<6} {result['depth']:<5} "
                          f"{result['total_return']:<12.2%} {result['sharpe_ratio']:<8.2f} {result['max_drawdown']:<10.2%} "
                          f"{result['total_time']:<8.1f} {result.get('num_windows_completed', 'N/A'):<6} {current_mark:<4}")
                
                print("-" * 120)
                best_result = successful_results[0]
                print(f"🏆 最佳配置: L{best_result['lookback']}_A{best_result['lookahead']}_S{best_result['stride']}_D{best_result['depth']}")
                print(f"📊 最佳表现: 收益{best_result['total_return']:.2%}, 夏普{best_result['sharpe_ratio']:.2f}, 回撤{best_result['max_drawdown']:.2%}")
            else:
                print("❌ 所有测试均失败，无有效结果")
            
            print("=" * 120)
            
            if i < len(test_stocks) - 1:
                print(f"\n⏳ 准备测试下一只股票 ({test_stocks[i+1]})...")
        
        # 注意：现在测试3支不同趋势的股票
        
        # 生成详细对比报告
        print("\n" + "=" * 100)
        print("🎉 多股票参数调优完成！生成详细对比报告")
        print("=" * 100)
        
        # 1. 总体统计
        total_tests = sum(len(results) for results in all_results.values())
        total_successful = sum(len([r for r in results if r['success']]) for results in all_results.values())
        total_time = sum(sum(r['total_time'] for r in results if r['success']) for results in all_results.values())
        
        print(f"\n📊 总体统计:")
        print(f"   总测试数: {total_tests}")
        print(f"   成功测试: {total_successful}")
        print(f"   成功率: {total_successful/total_tests*100:.1f}%")
        print(f"   总用时: {total_time:.1f}s ({total_time/3600:.1f}小时)")
        
        # 2. 各股票最佳配置对比
        print(f"\n🏆 各股票最佳配置对比:")
        print("-" * 110)
        print(f"{'股票':<8} {'市场特征':<20} {'最佳配置':<15} {'总收益':<10} {'年化收益':<10} {'夏普比率':<8} {'最大回撤':<10} {'用时(s)':<8}")
        print("-" * 110)
        
        best_results = {}
        print(f"调试: all_results包含的股票: {list(all_results.keys())}")
        print(f"调试: test_stocks: {test_stocks}")
        
        for stock in test_stocks:
            results = all_results[stock]
            successful_results = [r for r in results if r['success']]
            
            if successful_results:
                best_result = max(successful_results, key=lambda x: x['total_return'])
                best_results[stock] = best_result
                config_str = f"L{best_result['lookback']}_A{best_result['lookahead']}_S{best_result['stride']}_D{best_result['depth']}"
                print(f"{stock:<8} {stock_descriptions[stock]:<20} {config_str:<15} {best_result['total_return']:<10.2%} "
                      f"{best_result.get('annualized_return', 0):<10.2%} {best_result['sharpe_ratio']:<8.2f} {best_result['max_drawdown']:<10.2%} {best_result['total_time']:<8.1f}")
            else:
                print(f"{stock:<8} {stock_descriptions[stock]:<20} {'失败':<15} {'N/A':<10} {'N/A':<10} {'N/A':<8} {'N/A':<10} {'N/A':<8}")
        
        # 3. 参数影响分析
        print(f"\n📈 参数影响分析:")
        if best_results:
            # 分析最优参数的分布
            lookbacks = [r['lookback'] for r in best_results.values()]
            lookaheads = [r['lookahead'] for r in best_results.values()]
            strides = [r['stride'] for r in best_results.values()]
            depths = [r['depth'] for r in best_results.values()]
            
            from collections import Counter
            print(f"   最优lookback分布: {dict(Counter(lookbacks))}")
            print(f"   最优lookahead分布: {dict(Counter(lookaheads))}")
            print(f"   最优stride分布: {dict(Counter(strides))}")
            print(f"   最优depth分布: {dict(Counter(depths))}")
        
        # 4. 市场适应性分析
        print(f"\n🎯 市场适应性分析:")
        if best_results:
            # 按收益率排序
            sorted_stocks = sorted(best_results.keys(), key=lambda x: best_results[x]['total_return'], reverse=True)
            print(f"   收益率排名:")
            for i, stock in enumerate(sorted_stocks, 1):
                result = best_results[stock]
                print(f"   {i}. {stock} ({stock_descriptions[stock]}): {result['total_return']:.2%}")
            
            # 风险调整收益排名
            sorted_by_sharpe = sorted(best_results.keys(), key=lambda x: best_results[x]['sharpe_ratio'], reverse=True)
            print(f"   夏普比率排名:")
            for i, stock in enumerate(sorted_by_sharpe, 1):
                result = best_results[stock]
                print(f"   {i}. {stock}: {result['sharpe_ratio']:.2f}")
        
        # 5. 推荐配置
        print(f"\n💡 推荐配置:")
        if best_results:
            # 找出综合表现最好的配置
            best_overall = max(best_results.items(), key=lambda x: x[1]['sharpe_ratio'])
            stock, result = best_overall
            print(f"   综合最优配置 (基于{stock}的表现):")
            print(f"   lookback={result['lookback']}, lookahead={result['lookahead']}, stride={result['stride']}, depth={result['depth']}")
            print(f"   预期表现: 收益{result['total_return']:.2%}, 夏普{result['sharpe_ratio']:.2f}, 回撤{result['max_drawdown']:.2%}")
        
        # 6. 保存详细报告到文件
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"multi_stock_tuning_report_{timestamp}.txt"
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("EATA-RL 多股票参数调优报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("测试配置:\n")
            f.write(f"  股票数量: {len(test_stocks)}\n")
            f.write(f"  参数组合: 24种\n")
            f.write(f"  测试窗口: {args.windows}\n\n")
            
            f.write("各股票最佳结果:\n")
            for stock in test_stocks:
                if stock in best_results:
                    result = best_results[stock]
                    f.write(f"\n{stock} ({stock_descriptions[stock]}):\n")
                    f.write(f"  最佳配置: L{result['lookback']}_A{result['lookahead']}_S{result['stride']}_D{result['depth']}\n")
                    f.write(f"  收益率: {result['total_return']:.2%}\n")
                    f.write(f"  夏普比率: {result['sharpe_ratio']:.2f}\n")
                    f.write(f"  最大回撤: {result['max_drawdown']:.2%}\n")
                    f.write(f"  用时: {result['total_time']:.1f}s\n")
        
        print(f"\n📁 详细报告已保存到: {report_filename}")
        print("\n🎉 多股票参数调优全部完成！")
        
        return
    
    # 单股票测试模式
    tuner = ParameterTuner(test_stock=args.stock, num_test_windows=args.windows)
    
    if args.quick:
        print("⚡ 快速测试模式: 只测试4个关键组合")
        # 只测试几个关键组合，使用指定窗口数
        tuner.num_test_windows = args.windows
        tuner.param_combinations = [
            (50, 10, 1, 300),   # AMZN最优参数
            (100, 20, 2, 800),  # 当前参数
            (100, 10, 1, 800),  # 高精度
            (50, 20, 5, 300)    # 快速模式
        ]
    else:
        # 检查窗口数是否合理
        min_effective_windows = args.windows // 5  # stride=5时的最少窗口数
        if min_effective_windows < 100:
            print(f"⚠️  警告: 当stride=5时只有{min_effective_windows}个实际测试窗口")
            print(f"   建议使用至少500个窗口以确保充分测试")
        
        print(f"📊 完整测试模式: 24种组合 × {args.windows}窗口")
        estimated_time = len(tuner.param_combinations) * args.windows * 0.5 / 60  # 估算时间
        print(f"   预估总时间: {estimated_time:.0f}-{estimated_time*2:.0f}分钟")
    
    tuner.run_all_tests()

if __name__ == "__main__":
    main()
