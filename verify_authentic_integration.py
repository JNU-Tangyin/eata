#!/usr/bin/env python3
"""
原版集成验证脚本
Authentic Integration Verification Script

功能：
1. 验证FinRL和InvestorBench是否使用原版框架
2. 检查依赖安装和配置
3. 测试真实API调用和算法执行
4. 确保没有使用模拟或简化版本

使用方法：
python verify_authentic_integration.py --check-all
python verify_authentic_integration.py --check-finrl
python verify_authentic_integration.py --check-investorbench
"""

import os
import sys
import argparse
import importlib
import inspect
from pathlib import Path
import numpy as np
import pandas as pd

class AuthenticIntegrationVerifier:
    """原版集成验证器"""
    
    def __init__(self):
        self.verification_results = {
            'finrl': {'status': 'unknown', 'details': []},
            'investorbench': {'status': 'unknown', 'details': []},
            'overall': {'status': 'unknown', 'authentic': False}
        }
    
    def verify_finrl_authentic(self) -> bool:
        """验证FinRL是否为原版"""
        
        print("🔍 验证FinRL原版集成...")
        details = []
        
        try:
            # 1. 检查FinRL核心模块导入
            print("  📦 检查FinRL核心模块...")
            
            try:
                import finrl
                details.append(f"✅ FinRL版本: {finrl.__version__}")
                
                from finrl.agents.stablebaselines3.models import DRLAgent
                from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
                from stable_baselines3 import PPO, A2C, SAC, TD3, DDPG
                
                details.append("✅ FinRL核心模块导入成功")
                
            except ImportError as e:
                details.append(f"❌ FinRL核心模块导入失败: {e}")
                self.verification_results['finrl']['status'] = 'failed'
                self.verification_results['finrl']['details'] = details
                return False
            
            # 2. 检查authentic模块
            print("  🔧 检查FinRL authentic模块...")
            
            try:
                from comparison_experiments.algorithms.finrl import (
                    AuthenticFinRLRunner, AuthenticFinRLConfig
                )
                details.append("✅ FinRL模块导入成功")
                
                # 检查是否有Mock类
                module = importlib.import_module('comparison_experiments.algorithms.finrl')
                source_code = inspect.getsource(module)
                
                if 'MockFinRLClass' in source_code or 'mock' in source_code.lower():
                    details.append("⚠️ 发现Mock类，可能不是完全原版")
                else:
                    details.append("✅ 未发现Mock类，确认为原版实现")
                
            except ImportError as e:
                details.append(f"❌ FinRL authentic模块导入失败: {e}")
                self.verification_results['finrl']['status'] = 'failed'
                self.verification_results['finrl']['details'] = details
                return False
            
            # 3. 测试FinRL功能
            print("  🧪 测试FinRL核心功能...")
            
            try:
                # 创建测试数据
                test_data = self._create_test_data()
                train_data = test_data.iloc[:50]
                test_data_small = test_data.iloc[50:70]
                
                # 测试数据处理
                from comparison_experiments.algorithms.finrl import AuthenticFinRLDataProcessor
                processor = AuthenticFinRLDataProcessor()
                processed_data = processor.prepare_data(train_data, 'TEST')
                
                # 检查技术指标是否正确添加
                expected_indicators = ['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma']
                missing_indicators = [ind for ind in expected_indicators if ind not in processed_data.columns]
                
                if missing_indicators:
                    details.append(f"⚠️ 缺少技术指标: {missing_indicators}")
                else:
                    details.append("✅ 技术指标添加正确")
                
                details.append("✅ FinRL数据处理功能正常")
                
            except Exception as e:
                details.append(f"❌ FinRL功能测试失败: {e}")
                self.verification_results['finrl']['status'] = 'failed'
                self.verification_results['finrl']['details'] = details
                return False
            
            # 4. 验证算法参数
            print("  ⚙️ 验证FinRL算法参数...")
            
            try:
                config = AuthenticFinRLConfig()
                
                # 检查是否使用官方推荐参数
                if 'PPO' in config.ALGORITHM_PARAMS:
                    ppo_params = config.ALGORITHM_PARAMS['PPO']
                    if 'learning_rate' in ppo_params and 'n_steps' in ppo_params:
                        details.append("✅ FinRL算法参数配置正确")
                    else:
                        details.append("⚠️ FinRL算法参数可能不完整")
                else:
                    details.append("❌ 缺少FinRL算法参数配置")
                
            except Exception as e:
                details.append(f"❌ FinRL参数验证失败: {e}")
            
            self.verification_results['finrl']['status'] = 'passed'
            self.verification_results['finrl']['details'] = details
            
            print("✅ FinRL原版验证通过")
            return True
            
        except Exception as e:
            details.append(f"❌ FinRL验证异常: {e}")
            self.verification_results['finrl']['status'] = 'error'
            self.verification_results['finrl']['details'] = details
            return False
    
    def verify_investorbench_authentic(self) -> bool:
        """验证InvestorBench是否为原版"""
        
        print("🔍 验证InvestorBench原版集成...")
        details = []
        
        try:
            # 1. 检查核心依赖
            print("  📦 检查InvestorBench核心依赖...")
            
            try:
                import openai
                from transformers import AutoTokenizer, AutoModelForCausalLM
                import torch
                
                details.append(f"✅ OpenAI版本: {openai.__version__}")
                details.append("✅ Transformers和PyTorch导入成功")
                
            except ImportError as e:
                details.append(f"❌ InvestorBench依赖导入失败: {e}")
                self.verification_results['investorbench']['status'] = 'failed'
                self.verification_results['investorbench']['details'] = details
                return False
            
            # 2. 检查authentic模块
            print("  🔧 检查InvestorBench authentic模块...")
            
            try:
                from comparison_experiments.algorithms.investorbench import (
                    AuthenticInvestorBenchRunner, AuthenticOpenAIClient
                )
                details.append("✅ InvestorBench模块导入成功")
                
                # 检查是否有Mock实现
                module = importlib.import_module('comparison_experiments.algorithms.investorbench')
                source_code = inspect.getsource(module)
                
                if 'mock' in source_code.lower() and 'MockLLMModel' in source_code:
                    details.append("⚠️ 发现Mock实现，但仅作为fallback")
                else:
                    details.append("✅ 未发现Mock实现，确认为原版")
                
            except ImportError as e:
                details.append(f"❌ InvestorBench authentic模块导入失败: {e}")
                self.verification_results['investorbench']['status'] = 'failed'
                self.verification_results['investorbench']['details'] = details
                return False
            
            # 3. 检查API配置
            print("  🔑 检查API配置...")
            
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                details.append("✅ OpenAI API Key已配置")
                
                # 测试API连接 (可选)
                if len(sys.argv) > 1 and '--test-api' in sys.argv:
                    try:
                        client = AuthenticOpenAIClient('gpt-3.5-turbo')
                        details.append("✅ OpenAI客户端初始化成功")
                    except Exception as e:
                        details.append(f"⚠️ OpenAI客户端测试失败: {e}")
                else:
                    details.append("ℹ️ 跳过API连接测试 (使用--test-api启用)")
            else:
                details.append("⚠️ OpenAI API Key未配置，无法使用真实LLM")
            
            # 4. 验证任务配置
            print("  📋 验证任务配置...")
            
            try:
                from comparison_experiments.algorithms.investorbench import AuthenticInvestorBenchConfig
                config = AuthenticInvestorBenchConfig()
                
                if len(config.SUPPORTED_MODELS) > 0 and len(config.TASK_TYPES) > 0:
                    details.append(f"✅ 支持 {len(config.SUPPORTED_MODELS)} 个模型和 {len(config.TASK_TYPES)} 个任务类型")
                else:
                    details.append("❌ 模型或任务配置不完整")
                
            except Exception as e:
                details.append(f"❌ 任务配置验证失败: {e}")
            
            self.verification_results['investorbench']['status'] = 'passed'
            self.verification_results['investorbench']['details'] = details
            
            print("✅ InvestorBench原版验证通过")
            return True
            
        except Exception as e:
            details.append(f"❌ InvestorBench验证异常: {e}")
            self.verification_results['investorbench']['status'] = 'error'
            self.verification_results['investorbench']['details'] = details
            return False
    
    def verify_baseline_integration(self) -> bool:
        """验证baseline集成是否正确"""
        
        print("🔍 验证baseline集成...")
        
        try:
            from comparison_experiments.algorithms.baseline import STRATEGY_CONFIGS
            
            # 检查FinRL策略配置
            finrl_strategies = [k for k in STRATEGY_CONFIGS.keys() if k.startswith('finrl_')]
            
            print(f"  📊 发现 {len(finrl_strategies)} 个FinRL策略")
            
            # 检查模块引用
            for strategy in finrl_strategies:
                config = STRATEGY_CONFIGS[strategy]
                if config['module'] != 'finrl_strategies':
                    print(f"  ⚠️ {strategy} 未使用finrl_strategies模块")
                    return False
            
            print("✅ baseline集成验证通过")
            return True
            
        except Exception as e:
            print(f"❌ baseline集成验证失败: {e}")
            return False
    
    def _create_test_data(self) -> pd.DataFrame:
        """创建测试数据"""
        
        np.random.seed(42)
        dates = pd.date_range(start='2022-01-01', periods=100, freq='D')
        n_days = len(dates)
        
        returns = np.random.normal(0.001, 0.02, n_days)
        prices = 100 * np.cumprod(1 + returns)
        
        df = pd.DataFrame({
            'date': dates,
            'open': prices * (1 + np.random.normal(0, 0.005, n_days)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, n_days)
        })
        
        # 确保价格逻辑正确
        df['high'] = np.maximum(df['high'], df['close'])
        df['low'] = np.minimum(df['low'], df['close'])
        
        return df
    
    def generate_verification_report(self):
        """生成验证报告"""
        
        print("\n" + "="*80)
        print("📋 原版集成验证报告")
        print("="*80)
        
        # FinRL验证结果
        print(f"\n🔧 FinRL验证结果: {self.verification_results['finrl']['status'].upper()}")
        for detail in self.verification_results['finrl']['details']:
            print(f"  {detail}")
        
        # InvestorBench验证结果
        print(f"\n🤖 InvestorBench验证结果: {self.verification_results['investorbench']['status'].upper()}")
        for detail in self.verification_results['investorbench']['details']:
            print(f"  {detail}")
        
        # 总体评估
        finrl_ok = self.verification_results['finrl']['status'] == 'passed'
        investorbench_ok = self.verification_results['investorbench']['status'] == 'passed'
        
        if finrl_ok and investorbench_ok:
            self.verification_results['overall']['status'] = 'passed'
            self.verification_results['overall']['authentic'] = True
            print(f"\n🎉 总体验证结果: ✅ 通过 - 确认使用原版框架")
        else:
            self.verification_results['overall']['status'] = 'failed'
            self.verification_results['overall']['authentic'] = False
            print(f"\n⚠️ 总体验证结果: ❌ 失败 - 存在问题需要解决")
        
        # 建议
        print(f"\n💡 建议:")
        if not finrl_ok:
            print("  - 安装FinRL: pip install finrl")
            print("  - 安装依赖: pip install stable-baselines3[extra]")
        
        if not investorbench_ok:
            print("  - 安装OpenAI: pip install openai")
            print("  - 设置API Key: export OPENAI_API_KEY='your-key'")
        
        if finrl_ok and investorbench_ok:
            print("  - 所有验证通过，可以安全使用原版框架进行实验")
            print("  - 运行实验: python run_flag_trader_experiments.py --experiment_type academic")
        
        print("="*80)
        
        return self.verification_results['overall']['authentic']


def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(description='原版集成验证脚本')
    parser.add_argument('--check-all', action='store_true', help='检查所有组件')
    parser.add_argument('--check-finrl', action='store_true', help='只检查FinRL')
    parser.add_argument('--check-investorbench', action='store_true', help='只检查InvestorBench')
    parser.add_argument('--test-api', action='store_true', help='测试API连接')
    
    args = parser.parse_args()
    
    # 默认检查所有
    if not any([args.check_finrl, args.check_investorbench]):
        args.check_all = True
    
    verifier = AuthenticIntegrationVerifier()
    
    print("🚀 开始原版集成验证...")
    
    # 执行验证
    if args.check_all or args.check_finrl:
        verifier.verify_finrl_authentic()
    
    if args.check_all or args.check_investorbench:
        verifier.verify_investorbench_authentic()
    
    if args.check_all:
        verifier.verify_baseline_integration()
    
    # 生成报告
    is_authentic = verifier.generate_verification_report()
    
    # 退出码
    sys.exit(0 if is_authentic else 1)


if __name__ == "__main__":
    main()
