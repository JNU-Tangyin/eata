"""
EATA-NoMem: 无进化记忆变体
移除历史知识传承机制，设置num_transplant=0
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import pandas as pd
from experiments.ablation_study.variant_system import BaseVariant, VariantConfig

class EATANoMem(BaseVariant):
    """
    无进化记忆的EATA变体
    通过设置num_transplant=0移除历史知识传承
    """
    
    def _create_config(self) -> VariantConfig:
        """
        创建NoMem变体配置
        
        Returns:
            VariantConfig: NoMem变体的配置
        """
        return VariantConfig(
            name="EATA-NoMem",
            description="无进化记忆变体，skip_memory=True，真正移除历史知识传承机制",
            train_size=32,  # 使用默认训练触发阈值
            extra_params={
                'skip_memory': True,  # 🔧 新增：完全禁用进化记忆
                'variant_type': 'no_memory',
                'focus': 'no_evolutionary_memory'
            }
        )
    
    def get_variant_info(self):
        """获取变体信息"""
        return {
            'name': self.config.name,
            'description': self.config.description,
            'modifications': {
                'num_transplant': self.config.num_transplant,
                'target_component': 'MCTS',
                'modification_type': 'parameter_override'
            },
            'hypothesis': '移除进化记忆机制，可能降低模型适应性但提升计算效率',
            'expected_performance': {
                'adaptation_speed': '-40%',
                'computational_efficiency': '+30%',
                'memory_usage': '-50%'
            }
        }
