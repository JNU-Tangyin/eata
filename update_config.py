#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快捷配置更新脚本
运行此脚本可以快速修改参数，无需重启主程序
"""

import json
import os

def load_config():
    """加载当前配置"""
    config_file = 'config.json'
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_config(config):
    """保存配置"""
    with open('config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def update_nemots_params(**kwargs):
    """更新NEMoTS参数"""
    config = load_config()
    if 'nemots' not in config:
        config['nemots'] = {}
    
    config['nemots'].update(kwargs)
    save_config(config)
    print(f"✅ NEMoTS参数已更新: {kwargs}")

def update_trading_params(**kwargs):
    """更新交易参数"""
    config = load_config()
    if 'trading' not in config:
        config['trading'] = {}
    
    config['trading'].update(kwargs)
    save_config(config)
    print(f"✅ 交易参数已更新: {kwargs}")

def update_system_params(**kwargs):
    """更新系统参数"""
    config = load_config()
    if 'system' not in config:
        config['system'] = {}
    
    config['system'].update(kwargs)
    save_config(config)
    print(f"✅ 系统参数已更新: {kwargs}")

def show_current_config():
    """显示当前配置"""
    config = load_config()
    print("📋 当前配置:")
    print("=" * 50)
    for section, params in config.items():
        print(f"\n{section.upper()}:")
        for key, value in params.items():
            print(f"  {key}: {value}")

def main():
    """主菜单"""
    print("🔧 Bandwagon参数热更新工具")
    print("=" * 50)
    
    while True:
        print("\n选择操作:")
        print("1. 显示当前配置")
        print("2. 更新NEMoTS参数")
        print("3. 更新交易参数")
        print("4. 更新系统参数")
        print("5. 快速预设")
        print("0. 退出")
        
        choice = input("\n请输入选择 (0-5): ").strip()
        
        if choice == '0':
            break
        elif choice == '1':
            show_current_config()
        elif choice == '2':
            print("\n🧠 NEMoTS参数更新")
            print("常用参数:")
            print("- exploration_rate: 探索率 (1.0-2.5)")
            print("- num_runs: 运行次数 (5-15)")
            print("- eta: 搜索强度 (1.0-2.5)")
            
            # 简化输入
            exp_rate = input("exploration_rate (回车跳过): ").strip()
            num_runs = input("num_runs (回车跳过): ").strip()
            eta = input("eta (回车跳过): ").strip()
            
            params = {}
            if exp_rate: params['exploration_rate'] = float(exp_rate)
            if num_runs: params['num_runs'] = int(num_runs)
            if eta: params['eta'] = float(eta)
            
            if params:
                update_nemots_params(**params)
            
        elif choice == '3':
            print("\n💰 交易参数更新")
            print("参数说明:")
            print("- buy_threshold: 买入阈值 (1.005-1.020)")
            print("- sell_threshold: 卖出阈值 (0.980-0.995)")
            print("- uncertainty_threshold: 不确定性阈值 (0.05-0.20)")
            
            buy = input("buy_threshold (回车跳过): ").strip()
            sell = input("sell_threshold (回车跳过): ").strip()
            uncertainty = input("uncertainty_threshold (回车跳过): ").strip()
            
            params = {}
            if buy: params['buy_threshold'] = float(buy)
            if sell: params['sell_threshold'] = float(sell)
            if uncertainty: params['uncertainty_threshold'] = float(uncertainty)
            
            if params:
                update_trading_params(**params)
                
        elif choice == '4':
            print("\n⚙️ 系统参数更新")
            window_size = input("window_size (回车跳过): ").strip()
            
            params = {}
            if window_size: params['window_size'] = int(window_size)
            
            if params:
                update_system_params(**params)
                
        elif choice == '5':
            print("\n🚀 快速预设")
            print("1. 激进探索 (高探索率)")
            print("2. 保守交易 (高阈值)")
            print("3. 平衡配置 (推荐)")
            
            preset = input("选择预设 (1-3): ").strip()
            
            if preset == '1':
                update_nemots_params(
                    exploration_rate=2.2,
                    num_runs=12,
                    eta=2.0
                )
            elif preset == '2':
                update_trading_params(
                    buy_threshold=1.020,
                    sell_threshold=0.980,
                    uncertainty_threshold=0.15
                )
            elif preset == '3':
                update_nemots_params(
                    exploration_rate=1.8,
                    num_runs=10,
                    eta=1.6
                )
                update_trading_params(
                    buy_threshold=1.015,
                    sell_threshold=0.985,
                    uncertainty_threshold=0.12
                )

if __name__ == "__main__":
    main()
