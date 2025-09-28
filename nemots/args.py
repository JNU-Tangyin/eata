#!/usr/bin/env python
# coding=utf-8
# NEMoTS参数配置类
# 从main函数提取的超参数管理
# @date 2025.09.27

import torch
import numpy as np


class Args:
    """
    NEMoTS参数配置类
    集成main函数中的所有超参数设置
    """
    
    def __init__(self):
        # 设备配置
        self.device = torch.device("cpu")
        self.seed = 42
        
        # 数据配置
        self.data = 'custom'
        self.root_path = './dataset/'
        self.data_path = 'nvda_tech_daily.csv'
        self.embed = 'timeF'
        self.freq = 'h'
        self.features = 'M'  # M:multivariate predict multivariate
        self.target = 'OT'
        self.used_dimension = 1
        
        # NEMoTS核心参数
        self.symbolic_lib = "NEMoTS"
        self.max_len = 20
        self.max_module_init = 10
        self.num_transplant = 2
        self.num_runs = 5
        self.eta = 1.0
        self.num_aug = 0
        self.exploration_rate = 1 / np.sqrt(2)
        self.transplant_step = 1000
        self.norm_threshold = 1e-5
        
        # 序列长度配置
        self.seq_in = 84
        self.seq_out = 12
        
        # 训练参数
        self.epoch = 50
        self.round = 5
        self.train_size = 128
        self.lr = 1e-6
        self.weight_decay = 0.0001
        self.clip = 5.0
        
        # 记录配置
        self.recording = False
        self.tag = "records"
        self.logtag = "records_logtag"
    
    def update_for_sliding_window(self, lookback: int, lookahead: int):
        """
        为滑动窗口更新参数
        """
        self.seq_in = lookback
        self.seq_out = lookahead
        
        # 适配滑动窗口的参数调整
        self.num_runs = 3  # 减少运行次数
        self.epoch = 10    # 减少epoch
        self.round = 2     # 减少round
        self.train_size = 64  # 减少batch size
        self.transplant_step = 500  # 减少步数
        
        return self
    
    def __str__(self):
        """参数字符串表示"""
        params = []
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                params.append(f"{key}={value}")
        return "Args(" + ", ".join(params) + ")"
