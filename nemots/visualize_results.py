#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
双目标预测结果可视化脚本
用于处理dual_target_train.py输出的预测结果，生成丰富的可视化图表和指标分析
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import argparse
from tracker import DualTargetVisualizer


def load_prediction_data(evaluation_json=None, pred_csv=None):
    """
    加载预测数据，支持从评估JSON文件和预测CSV文件中加载

    Args:
        evaluation_json: 评估结果JSON文件
        pred_csv: 预测结果CSV文件
    
    Returns:
        dict: 包含pred_q25, pred_q75, actual, dates的字典
    """
    data = {}
    
    # 从CSV读取预测数据
    if pred_csv and os.path.exists(pred_csv):
        df = pd.read_csv(pred_csv)
        print(f"CSV文件列名: {df.columns.tolist()}")
        
        # 尝试识别关键列
        date_col = next((col for col in df.columns if 'date' in col.lower()), None)
        q25_col = next((col for col in df.columns if 'q25' in col.lower() and 'pred' in col.lower()), None)
        q75_col = next((col for col in df.columns if 'q75' in col.lower() and 'pred' in col.lower()), None)
        actual_col = next((col for col in df.columns if any(kw in col.lower() for kw in ['actual', 'true', 'real', 'close'])), None)
        
        if not (q25_col and q75_col and actual_col):
            print(f"警告: 无法自动识别所有需要的列。请检查CSV文件格式。")
            print(f"需要包含Q25预测、Q75预测和实际价格列。")
            return None
            
        data['pred_q25'] = df[q25_col].values
        data['pred_q75'] = df[q75_col].values
        data['actual'] = df[actual_col].values
        data['dates'] = df[date_col].values if date_col else [f"T{i}" for i in range(len(data['pred_q25']))]
        
        print(f"从CSV加载了{len(data['pred_q25'])}条预测记录")
        
    # 从evaluation JSON读取评估结果
    elif evaluation_json and os.path.exists(evaluation_json):
        try:
            with open(evaluation_json, 'r') as f:
                eval_data = json.load(f)
                
            print(f"已加载评估结果JSON，包含指标: {list(eval_data.keys())}")
        except Exception as e:
            print(f"加载评估结果JSON时出错: {e}")
            return None
    
    return data


def main():
    parser = argparse.ArgumentParser(description='可视化双目标预测结果')
    parser.add_argument('--pred_csv', type=str, default='evaluation/predictions.csv',
                        help='预测结果CSV文件路径，应包含pred_q25, pred_q75, actual列')
    parser.add_argument('--eval_json', type=str, default='evaluation/dual_target_results.json',
                        help='评估结果JSON文件路径')
    parser.add_argument('--output_dir', type=str, default='evaluation/visualization',
                        help='可视化输出目录')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载预测数据
    prediction_data = load_prediction_data(args.eval_json, args.pred_csv)
    
    if not prediction_data:
        print("无预测数据可用，请检查输入文件。")
        sys.exit(1)
    
    # 创建可视化器并生成所有图表
    visualizer = DualTargetVisualizer(save_dir=args.output_dir)
    metrics = visualizer.generate_all_visualizations(
        pred_q25=prediction_data.get('pred_q25'),
        pred_q75=prediction_data.get('pred_q75'),
        actual=prediction_data.get('actual'),
        dates=prediction_data.get('dates')
    )
    
    print(f"\n===== 主要评估指标 =====")
    print(f"MAE (Q25/Q75平均): {metrics['基本指标']['avg_mae']:.4f}")
    print(f"MSE (Q25/Q75平均): {metrics['基本指标']['avg_mse']:.4f}")
    print(f"实际价格超出预测区间比例: {(metrics['区间指标']['below_q25_rate'] + metrics['区间指标']['above_q75_rate'])*100:.2f}%")
    print(f"平均卖出风险: {metrics['交易风险']['avg_sell_risk']:.2f}%")
    print(f"平均买入风险: {metrics['交易风险']['avg_buy_risk']:.2f}%")
    
    print(f"\n所有可视化结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
