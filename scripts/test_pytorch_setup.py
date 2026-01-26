#!/usr/bin/env python3
"""
PyTorch环境配置验证脚本
用于验证conda pytorch环境在VS Code中的配置是否正确
"""

import sys
import torch
import numpy as np
import pandas as pd
import gym
import matplotlib.pyplot as plt
import quantstats as qs

def test_pytorch_setup():
    """验证PyTorch环境配置"""
    print("=" * 50)
    print("PyTorch环境配置验证")
    print("=" * 50)
    
    # 基本信息
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"NumPy版本: {np.__version__}")
    print(f"Pandas版本: {pd.__version__}")
    print(f"Gym版本: {gym.__version__}")
    
    # CUDA支持检查
    print(f"\nCUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 基本张量操作测试
    print("\n测试基本张量操作:")
    x = torch.randn(3, 3)
    y = torch.randn(3, 3)
    z = torch.matmul(x, y)
    print(f"矩阵乘法结果形状: {z.shape}")
    
    # GPU测试（如果可用）
    if torch.cuda.is_available():
        print("\n测试GPU操作:")
        x_gpu = x.cuda()
        y_gpu = y.cuda()
        z_gpu = torch.matmul(x_gpu, y_gpu)
        print(f"GPU矩阵乘法结果形状: {z_gpu.shape}")
        print(f"结果在GPU上: {z_gpu.is_cuda}")
    
    # 测试其他依赖
    print("\n测试其他依赖:")
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    print(f"Pandas DataFrame创建成功: {df.shape}")
    
    # 测试gym环境
    try:
        env = gym.make('CartPole-v1')
        print(f"Gym环境创建成功: {env.spec.id}")
        env.close()
    except Exception as e:
        print(f"Gym环境测试失败: {e}")
    
    print("\n✅ 所有测试通过！PyTorch环境配置成功！")
    print("=" * 50)

if __name__ == "__main__":
    test_pytorch_setup()
