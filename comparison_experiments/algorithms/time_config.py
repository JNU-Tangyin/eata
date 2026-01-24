"""
统一时间配置模块
确保所有策略使用相同的时间分割
"""

import pandas as pd
from typing import Dict, Tuple

def get_time_periods() -> Dict[str, Dict[str, str]]:
    """
    获取统一的时间配置
    
    Returns:
        Dict: 包含训练、验证、测试期的时间配置
    """
    return {
        'train': {
            'start': '2010-01-01',
            'end': '2017-12-31'
        },
        'valid': {
            'start': '2018-01-01', 
            'end': '2019-12-31'
        },
        'test': {
            'start': '2020-01-01',
            'end': '2022-12-12'
        }
    }

def split_data_by_time(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    根据统一时间配置分割数据
    
    Args:
        df: 包含date列的DataFrame
        
    Returns:
        tuple: (train_data, valid_data, test_data)
    """
    time_config = get_time_periods()
    
    # 确保date列是datetime类型
    df['date'] = pd.to_datetime(df['date'])
    
    # 按时间分割数据
    train_data = df[
        (df['date'] >= time_config['train']['start']) & 
        (df['date'] <= time_config['train']['end'])
    ].copy()
    
    valid_data = df[
        (df['date'] >= time_config['valid']['start']) & 
        (df['date'] <= time_config['valid']['end'])
    ].copy()
    
    test_data = df[
        (df['date'] >= time_config['test']['start']) & 
        (df['date'] <= time_config['test']['end'])
    ].copy()
    
    return train_data, valid_data, test_data

def get_test_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    获取测试期数据
    
    Args:
        df: 包含date列的DataFrame
        
    Returns:
        pd.DataFrame: 测试期数据
    """
    time_config = get_time_periods()
    
    # 确保date列是datetime类型
    df['date'] = pd.to_datetime(df['date'])
    
    # 获取测试期数据
    test_data = df[
        (df['date'] >= time_config['test']['start']) & 
        (df['date'] <= time_config['test']['end'])
    ].copy()
    
    return test_data

def print_time_config():
    """
    打印当前时间配置
    """
    config = get_time_periods()
    print("=== 统一时间配置 ===")
    print(f"训练期: {config['train']['start']} 到 {config['train']['end']} (8年)")
    print(f"验证期: {config['valid']['start']} 到 {config['valid']['end']} (2年)")
    print(f"测试期: {config['test']['start']} 到 {config['test']['end']} (3年)")
    print("=" * 50)

if __name__ == "__main__":
    # 测试时间配置
    print_time_config()
