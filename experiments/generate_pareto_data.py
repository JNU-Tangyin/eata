"""
为Figure 8生成Pareto Frontier数据
收集所有对比方法的性能和复杂度数据
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import json
from pathlib import Path
from utils.expression_complexity import count_ast_nodes, estimate_method_complexity

def collect_eata_data():
    """
    从EATA实验结果中收集表达式和复杂度数据
    """
    print("=" * 80)
    print("收集EATA数据")
    print("=" * 80)
    
    # 查找最新的消融实验结果（包含EATA-Full）
    ablation_dir = Path('../results/ablation_study/raw_results')
    json_files = list(ablation_dir.glob('ablation_results_*.json'))
    
    if not json_files:
        print("⚠️ 未找到消融实验结果文件")
        return []
    
    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
    print(f"读取文件: {latest_file}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    eata_data = []
    for result in data:
        if result['variant'] == 'EATA-Full':
            ticker = result['ticker']
            sharpe = result['sharpe_ratio']
            
            # 提取表达式和复杂度
            expressions = result.get('discovered_expressions', [])
            complexities = result.get('expression_complexities', [])
            
            if expressions and complexities:
                # 使用最终表达式
                final_expr = expressions[-1]
                final_complexity = complexities[-1]
                
                eata_data.append({
                    'method': 'EATA',
                    'ticker': ticker,
                    'expression': final_expr,
                    'complexity': final_complexity,
                    'sharpe_ratio': sharpe
                })
                print(f"  {ticker}: 复杂度={final_complexity}, Sharpe={sharpe:.3f}")
            else:
                # 如果没有表达式数据，使用估算值
                print(f"  {ticker}: 无表达式数据，跳过")
    
    print(f"✅ 收集了 {len(eata_data)} 个EATA数据点")
    return eata_data

def collect_baseline_data():
    """
    从对比实验结果中收集基线方法的数据
    """
    print("\n" + "=" * 80)
    print("收集基线方法数据")
    print("=" * 80)
    
    # 读取对比实验的详细结果
    comparison_file = Path('../results/comparison_study/raw_results/detailed_results.csv')
    
    if not comparison_file.exists():
        print(f"⚠️ 未找到对比实验结果文件: {comparison_file}")
        return []
    
    df = pd.read_csv(comparison_file)
    print(f"读取文件: {comparison_file}")
    print(f"总记录数: {len(df)}")
    
    baseline_data = []
    
    # 为每个方法分配固定复杂度
    for _, row in df.iterrows():
        method = row['strategy']
        ticker = row['ticker']
        sharpe = row['sharpe_ratio']
        
        # 跳过EATA（已单独处理）和无效数据
        if method == 'eata' or sharpe == 0 or np.isnan(sharpe):
            continue
        
        # 估算复杂度
        complexity = estimate_method_complexity(method)
        
        baseline_data.append({
            'method': method,
            'ticker': ticker,
            'expression': method,  # 非符号方法用方法名
            'complexity': complexity,
            'sharpe_ratio': sharpe
        })
    
    print(f"✅ 收集了 {len(baseline_data)} 个基线数据点")
    
    # 显示每个方法的统计
    df_baseline = pd.DataFrame(baseline_data)
    print("\n各方法统计:")
    for method in df_baseline['method'].unique():
        method_data = df_baseline[df_baseline['method'] == method]
        print(f"  {method:<15} 数据点: {len(method_data):3d}, "
              f"复杂度: {method_data['complexity'].iloc[0]:4d}, "
              f"Sharpe: {method_data['sharpe_ratio'].mean():.3f} ± {method_data['sharpe_ratio'].std():.3f}")
    
    return baseline_data

def save_pareto_data(all_data, output_file):
    """
    保存Pareto数据到CSV
    """
    df = pd.DataFrame(all_data)
    df.to_csv(output_file, index=False)
    print(f"\n✅ Pareto数据已保存到: {output_file}")
    print(f"   总数据点: {len(df)}")
    print(f"   方法数: {df['method'].nunique()}")
    print(f"   股票数: {df['ticker'].nunique()}")
    
    return df

def main():
    """主函数"""
    print("🎯 开始生成Pareto Frontier数据\n")
    
    # 收集EATA数据
    eata_data = collect_eata_data()
    
    # 收集基线数据
    baseline_data = collect_baseline_data()
    
    # 合并所有数据
    all_data = eata_data + baseline_data
    
    if not all_data:
        print("\n❌ 没有收集到任何数据！")
        print("   请确保已运行EATA实验并记录了表达式数据")
        return
    
    # 保存数据
    output_dir = Path('../results/pareto_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'pareto_data.csv'
    
    df = save_pareto_data(all_data, output_file)
    
    # 显示数据摘要
    print("\n" + "=" * 80)
    print("数据摘要")
    print("=" * 80)
    print(f"复杂度范围: {df['complexity'].min():.0f} - {df['complexity'].max():.0f}")
    print(f"Sharpe范围: {df['sharpe_ratio'].min():.3f} - {df['sharpe_ratio'].max():.3f}")
    print(f"\n前10个最佳性能点:")
    print(df.nlargest(10, 'sharpe_ratio')[['method', 'ticker', 'complexity', 'sharpe_ratio']])

if __name__ == '__main__':
    main()
