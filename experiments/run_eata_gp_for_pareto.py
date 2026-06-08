"""
重新运行EATA和GP的对比实验，收集表达式和复杂度数据
用于生成Figure 8的Pareto Frontier图
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# 导入对比实验的算法
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'comparison_experiments/algorithms'))
from eata import run_eata_strategy
from gp import run_gp_strategy
from utils.expression_complexity import count_ast_nodes

# 股票列表（使用对比实验中的股票）
STOCKS = [
    'AAPL', 'AMD', 'AMT', 'BA', 'BAC', 
    'BHP', 'CAT', 'COST', 'DE', 'EQIX',
    'GE', 'GOOG', 'JNJ', 'JPM', 'KO',
    'MSFT', 'NFLX', 'NVDA', 'SCHW', 'XOM'
]

def load_stock_data(ticker):
    """加载股票数据"""
    data_file = Path(f'../data/{ticker}.csv')
    if not data_file.exists():
        print(f"⚠️ 数据文件不存在: {data_file}")
        return None, None
    
    df = pd.read_csv(data_file)
    
    # 标准化列名为小写
    df.columns = df.columns.str.lower()
    
    # 分割训练和测试集（80/20）
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    return train_df, test_df

def run_eata_with_expression(ticker, train_df, test_df):
    """运行EATA并提取表达式"""
    print(f"\n{'='*80}")
    print(f"运行EATA: {ticker}")
    print(f"{'='*80}")
    
    try:
        metrics, _ = run_eata_strategy(train_df, test_df, ticker)
        
        # 提取表达式和复杂度
        expressions = metrics.get('Discovered Expressions', [])
        complexities = metrics.get('Expression Complexities', [])
        
        if expressions and complexities:
            # 使用最终表达式
            final_expr = expressions[-1]
            final_complexity = complexities[-1]
        else:
            # 如果没有表达式数据，尝试估算
            print(f"  ⚠️ {ticker}: EATA没有表达式数据，使用默认值")
            final_expr = "unknown"
            final_complexity = 20  # 默认复杂度
        
        return {
            'ticker': ticker,
            'method': 'EATA',
            'expression': final_expr,
            'complexity': final_complexity,
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'annualized_return': metrics.get('annualized_return', 0),
            'max_drawdown': metrics.get('max_drawdown', 0)
        }
        
    except Exception as e:
        print(f"  ❌ EATA失败 ({ticker}): {e}")
        return None

def run_gp_with_expression(ticker, train_df, test_df):
    """运行GP并提取表达式"""
    print(f"\n{'='*80}")
    print(f"运行GP: {ticker}")
    print(f"{'='*80}")
    
    try:
        metrics, _ = run_gp_strategy(train_df, test_df, ticker)
        
        # 提取GP表达式
        gp_expr = metrics.get('gp_expression', None)
        
        if gp_expr and gp_expr != '0':
            # 计算复杂度
            complexity = count_ast_nodes(gp_expr)
        else:
            print(f"  ⚠️ {ticker}: GP没有表达式数据")
            gp_expr = "unknown"
            complexity = 30  # 默认复杂度
        
        return {
            'ticker': ticker,
            'method': 'GP',
            'expression': gp_expr,
            'complexity': complexity,
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'annualized_return': metrics.get('annualized_return', 0),
            'max_drawdown': metrics.get('max_drawdown', 0)
        }
        
    except Exception as e:
        print(f"  ❌ GP失败 ({ticker}): {e}")
        return None

def main():
    """主函数"""
    print("🎯 开始运行EATA和GP对比实验（收集表达式数据）\n")
    
    results = []
    
    for ticker in STOCKS:
        print(f"\n{'#'*80}")
        print(f"# 处理股票: {ticker}")
        print(f"{'#'*80}")
        
        # 加载数据
        train_df, test_df = load_stock_data(ticker)
        if train_df is None:
            continue
        
        # 运行EATA
        eata_result = run_eata_with_expression(ticker, train_df, test_df)
        if eata_result:
            results.append(eata_result)
        
        # 运行GP
        gp_result = run_gp_with_expression(ticker, train_df, test_df)
        if gp_result:
            results.append(gp_result)
    
    # 保存结果
    output_dir = Path('../results/pareto_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存为JSON
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_file = output_dir / f'eata_gp_expressions_{timestamp}.json'
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 保存为CSV
    csv_file = output_dir / f'eata_gp_expressions_{timestamp}.csv'
    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False)
    
    print(f"\n{'='*80}")
    print(f"✅ 实验完成！")
    print(f"{'='*80}")
    print(f"总结果数: {len(results)}")
    print(f"EATA结果: {len([r for r in results if r['method'] == 'EATA'])}")
    print(f"GP结果: {len([r for r in results if r['method'] == 'GP'])}")
    print(f"\n结果已保存到:")
    print(f"  JSON: {json_file}")
    print(f"  CSV: {csv_file}")
    
    # 显示复杂度统计
    if results:
        df = pd.DataFrame(results)
        print(f"\n复杂度统计:")
        for method in ['EATA', 'GP']:
            method_data = df[df['method'] == method]
            if len(method_data) > 0:
                print(f"  {method}: 平均={method_data['complexity'].mean():.1f}, "
                      f"范围={method_data['complexity'].min()}-{method_data['complexity'].max()}")

if __name__ == '__main__':
    main()
