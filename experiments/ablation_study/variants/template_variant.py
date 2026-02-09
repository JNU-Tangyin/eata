"""
通用消融实验变体模板
使用与对比实验相同的核心回测逻辑
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import pandas as pd
import numpy as np

class TemplateVariant:
    """
    消融实验变体模板
    """
    
    def __init__(self, df: pd.DataFrame, **kwargs):
        """
        初始化变体
        """
        self.name = "Template-Variant"
        self.description = "变体描述"
        self.modifications = {}
        
    def run_backtest(self, train_df: pd.DataFrame, test_df: pd.DataFrame, ticker: str):
        """
        运行回测 - 使用与对比实验相同的核心回测逻辑
        """
        try:
            print(f"运行{self.name}回测 - {ticker}")
            print(f"   修改: {self.description}")
            
            # 导入并使用与对比实验相同的核心回测函数
            import sys
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            sys.path.insert(0, project_root)
            
            from predict import run_eata_core_backtest
            
            # 合并训练和测试数据，与对比实验保持一致
            combined_df = pd.concat([train_df, test_df]).reset_index(drop=True)
            
            # 使用与对比实验完全相同的参数和调用方式
            core_metrics, portfolio_df = run_eata_core_backtest(
                stock_df=combined_df,
                ticker=ticker,
                lookback=50,
                lookahead=10,
                stride=1,
                depth=300
            )
            
            # 转换指标格式
            annual_return = core_metrics.get('Annual Return (AR)', 0.0)
            sharpe_ratio = core_metrics.get('Sharpe Ratio', 0.0)
            max_drawdown = core_metrics.get('Max Drawdown (MDD)', 0.0)
            win_rate = core_metrics.get('Win Rate', 0.0)
            volatility = core_metrics.get('Volatility (Annual)', 0.0)
            
            return {
                'variant': self.name,
                'ticker': ticker,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'volatility': volatility,
                'rl_reward': 0.0,
                'modifications': self.modifications
            }
            
        except Exception as e:
            print(f"{self.name}回测失败: {str(e)}")
            return {
                'variant': self.name,
                'ticker': ticker,
                'annual_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'volatility': 0.0,
                'rl_reward': 0.0,
                'modifications': self.modifications,
                'error': str(e)
            }
    
    def get_variant_info(self):
        """
        获取变体信息
        """
        return {
            'name': self.name,
            'description': self.description,
            'modifications': self.modifications
        }
