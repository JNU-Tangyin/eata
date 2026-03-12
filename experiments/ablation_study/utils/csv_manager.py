"""
CSV数据管理器
处理消融实验结果的CSV导出、读取和管理
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Any, Optional

class CSVManager:
    """
    CSV数据管理器，负责消融实验结果的CSV操作
    """
    
    def __init__(self, csv_results_dir: Path):
        """
        初始化CSV管理器
        
        Args:
            csv_results_dir: CSV结果存储目录
        """
        self.csv_results_dir = Path(csv_results_dir)
        self.csv_results_dir.mkdir(parents=True, exist_ok=True)
        
    def export_performance_summary(self, results: List[Dict[str, Any]], experiment_id: str) -> Path:
        """
        导出性能汇总CSV
        
        Args:
            results: 实验结果列表
            experiment_id: 实验ID
            
        Returns:
            Path: 保存的CSV文件路径
        """
        summary_data = []
        
        for result in results:
            if 'error' not in result and 'metrics' in result:
                metrics = result['metrics']
                summary_data.append({
                    'experiment_id': experiment_id,
                    'variant': result['variant'],
                    'ticker': result.get('ticker', 'N/A'),
                    'annual_return': metrics.get('annual_return', 0.0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
                    'max_drawdown': metrics.get('max_drawdown', 0.0),
                    'win_rate': metrics.get('win_rate', 0.0),
                    'volatility': metrics.get('volatility', 0.0),
                    'rl_reward': result.get('rl_reward', 0.0),
                    'timestamp': result.get('timestamp', datetime.now().isoformat())
                })
        
        # 创建DataFrame并保存
        df = pd.DataFrame(summary_data)
        csv_file = self.csv_results_dir / f"performance_summary_{experiment_id}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8', float_format='%.6f')
        
        return csv_file
    
    def export_detailed_metrics(self, results: List[Dict[str, Any]], experiment_id: str) -> Path:
        """
        导出详细指标CSV
        """
        detailed_data = []
        
        for result in results:
            if 'error' not in result:
                base_info = {
                    'experiment_id': experiment_id,
                    'variant': result['variant'],
                    'ticker': result.get('ticker', 'N/A'),
                    'timestamp': result.get('timestamp', datetime.now().isoformat())
                }
                
                # 添加所有指标
                if 'metrics' in result:
                    base_info.update(result['metrics'])
                
                # 添加变体信息
                if 'variant_info' in result:
                    variant_info = result['variant_info']
                    base_info.update({
                        'description': variant_info.get('description', ''),
                        'hypothesis': variant_info.get('hypothesis', ''),
                        'modifications': json.dumps(variant_info.get('modifications', {}))
                    })
                
                detailed_data.append(base_info)
        
        df = pd.DataFrame(detailed_data)
        csv_file = self.csv_results_dir / f"detailed_metrics_{experiment_id}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8', float_format='%.6f')
        
        return csv_file
    
    def export_variant_comparison(self, results: List[Dict[str, Any]], experiment_id: str) -> Path:
        """
        导出变体对比CSV
        """
        # 找到基准结果
        baseline_result = None
        for result in results:
            if result.get('variant') == 'EATA-Full' and 'error' not in result:
                baseline_result = result
                break
        
        if not baseline_result:
            print("⚠️ 未找到基准结果，无法生成对比表")
            return None
        
        baseline_metrics = baseline_result.get('metrics', {})
        comparison_data = []
        
        for result in results:
            if 'error' not in result and 'metrics' in result:
                metrics = result['metrics']
                
                # 计算相对于基准的变化
                comparison_row = {
                    'experiment_id': experiment_id,
                    'variant': result['variant'],
                    'annual_return': metrics.get('annual_return', 0.0),
                    'annual_return_change': self._calculate_percentage_change(
                        baseline_metrics.get('annual_return', 0.0),
                        metrics.get('annual_return', 0.0)
                    ),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
                    'sharpe_ratio_change': self._calculate_percentage_change(
                        baseline_metrics.get('sharpe_ratio', 0.0),
                        metrics.get('sharpe_ratio', 0.0)
                    ),
                    'max_drawdown': metrics.get('max_drawdown', 0.0),
                    'max_drawdown_change': self._calculate_percentage_change(
                        baseline_metrics.get('max_drawdown', 0.0),
                        metrics.get('max_drawdown', 0.0)
                    ),
                    'win_rate': metrics.get('win_rate', 0.0),
                    'win_rate_change': self._calculate_percentage_change(
                        baseline_metrics.get('win_rate', 0.0),
                        metrics.get('win_rate', 0.0)
                    )
                }
                
                comparison_data.append(comparison_row)
        
        df = pd.DataFrame(comparison_data)
        csv_file = self.csv_results_dir / f"variant_comparison_{experiment_id}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8', float_format='%.6f')
        
        return csv_file
    
    def export_time_series_results(self, results: List[Dict[str, Any]], experiment_id: str) -> Path:
        """
        导出时序结果CSV
        """
        time_series_data = []
        
        for result in results:
            if 'error' not in result and 'returns' in result:
                returns = result['returns']
                if isinstance(returns, np.ndarray) and len(returns) > 0:
                    # 计算累积收益
                    cumulative_returns = np.cumprod(1 + returns)
                    
                    for i, (ret, cum_ret) in enumerate(zip(returns, cumulative_returns)):
                        time_series_data.append({
                            'experiment_id': experiment_id,
                            'variant': result['variant'],
                            'ticker': result.get('ticker', 'N/A'),
                            'period': i + 1,
                            'return': ret,
                            'cumulative_return': cum_ret,
                            'timestamp': result.get('timestamp', datetime.now().isoformat())
                        })
        
        if time_series_data:
            df = pd.DataFrame(time_series_data)
            csv_file = self.csv_results_dir / f"time_series_results_{experiment_id}.csv"
            df.to_csv(csv_file, index=False, encoding='utf-8', float_format='%.6f')
            return csv_file
        
        return None
    
    def load_experiment_results(self, experiment_id: str) -> Optional[Dict[str, pd.DataFrame]]:
        """
        加载指定实验的所有CSV结果
        
        Args:
            experiment_id: 实验ID
            
        Returns:
            Dict[str, pd.DataFrame]: 包含所有CSV数据的字典
        """
        csv_files = {
            'performance_summary': f"performance_summary_{experiment_id}.csv",
            'detailed_metrics': f"detailed_metrics_{experiment_id}.csv",
            'variant_comparison': f"variant_comparison_{experiment_id}.csv",
            'time_series_results': f"time_series_results_{experiment_id}.csv"
        }
        
        loaded_data = {}
        
        for data_type, filename in csv_files.items():
            csv_path = self.csv_results_dir / filename
            if csv_path.exists():
                try:
                    loaded_data[data_type] = pd.read_csv(csv_path, encoding='utf-8')
                    print(f"✅ 加载 {data_type}: {csv_path}")
                except Exception as e:
                    print(f"❌ 加载 {data_type} 失败: {e}")
            else:
                print(f"⚠️ 文件不存在: {csv_path}")
        
        return loaded_data if loaded_data else None
    
    def merge_experiments(self, experiment_ids: List[str], output_filename: str) -> Path:
        """
        合并多个实验的结果
        
        Args:
            experiment_ids: 实验ID列表
            output_filename: 输出文件名
            
        Returns:
            Path: 合并后的CSV文件路径
        """
        all_data = []
        
        for exp_id in experiment_ids:
            exp_data = self.load_experiment_results(exp_id)
            if exp_data and 'performance_summary' in exp_data:
                all_data.append(exp_data['performance_summary'])
        
        if all_data:
            merged_df = pd.concat(all_data, ignore_index=True)
            output_path = self.csv_results_dir / output_filename
            merged_df.to_csv(output_path, index=False, encoding='utf-8', float_format='%.6f')
            return output_path
        
        return None
    
    def _calculate_percentage_change(self, baseline: float, current: float) -> float:
        """
        计算百分比变化
        """
        if baseline == 0:
            return 0.0 if current == 0 else float('inf')
        return ((current - baseline) / abs(baseline)) * 100
    
    def get_experiment_summary(self, experiment_id: str) -> Dict[str, Any]:
        """
        获取实验汇总信息
        """
        data = self.load_experiment_results(experiment_id)
        if not data or 'performance_summary' not in data:
            return {}
        
        df = data['performance_summary']
        
        summary = {
            'experiment_id': experiment_id,
            'total_variants': len(df),
            'successful_variants': len(df[df['annual_return'].notna()]),
            'best_variant': df.loc[df['sharpe_ratio'].idxmax(), 'variant'] if not df.empty else None,
            'worst_variant': df.loc[df['sharpe_ratio'].idxmin(), 'variant'] if not df.empty else None,
            'average_annual_return': df['annual_return'].mean(),
            'average_sharpe_ratio': df['sharpe_ratio'].mean(),
            'performance_range': {
                'annual_return': {
                    'min': df['annual_return'].min(),
                    'max': df['annual_return'].max()
                },
                'sharpe_ratio': {
                    'min': df['sharpe_ratio'].min(),
                    'max': df['sharpe_ratio'].max()
                }
            }
        }
        
        return summary
