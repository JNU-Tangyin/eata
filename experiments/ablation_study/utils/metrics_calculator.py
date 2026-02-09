"""
指标计算器
计算消融实验的各种性能指标
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from scipy import stats
import warnings

# 隐藏警告信息
warnings.filterwarnings('ignore')
np.seterr(all='ignore')  # 隐藏numpy警告

class MetricsCalculator:
    """
    指标计算器，负责计算各种交易和统计指标
    """
    
    def __init__(self):
        """
        初始化指标计算器
        """
        self.trading_days_per_year = 252
        
    def calculate_basic_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """
        计算基础性能指标
        
        Args:
            returns: 收益率序列
            
        Returns:
            Dict[str, float]: 基础指标字典
        """
        if len(returns) == 0:
            return self._get_empty_metrics()
        
        # 处理无效值
        returns = np.array(returns)
        returns = returns[~np.isnan(returns)]
        
        if len(returns) == 0:
            return self._get_empty_metrics()
        
        # 基础统计
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        # 年化指标
        annual_return = mean_return * self.trading_days_per_year
        annual_volatility = std_return * np.sqrt(self.trading_days_per_year)
        
        # 夏普比率
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0.0
        
        # 最大回撤
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # 胜率
        win_rate = np.sum(returns > 0) / len(returns)
        
        # VaR和CVaR (95%置信水平)
        var_95 = np.percentile(returns, 5)
        cvar_95 = np.mean(returns[returns <= var_95]) if np.any(returns <= var_95) else var_95
        
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'total_return': (cumulative_returns[-1] - 1) if len(cumulative_returns) > 0 else 0.0,
            'num_trades': len(returns)
        }
    
    def calculate_advanced_metrics(self, returns: np.ndarray, benchmark_returns: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        计算高级性能指标
        
        Args:
            returns: 策略收益率序列
            benchmark_returns: 基准收益率序列
            
        Returns:
            Dict[str, float]: 高级指标字典
        """
        if len(returns) == 0:
            return {}
        
        returns = np.array(returns)
        returns = returns[~np.isnan(returns)]
        
        if len(returns) == 0:
            return {}
        
        metrics = {}
        
        # Calmar比率
        annual_return = np.mean(returns) * self.trading_days_per_year
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(np.min(drawdowns))
        
        metrics['calmar_ratio'] = annual_return / max_drawdown if max_drawdown > 0 else 0.0
        
        # Sortino比率
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 0 else 0.0
        downside_volatility = downside_std * np.sqrt(self.trading_days_per_year)
        metrics['sortino_ratio'] = annual_return / downside_volatility if downside_volatility > 0 else 0.0
        
        # 偏度和峰度
        metrics['skewness'] = stats.skew(returns)
        metrics['kurtosis'] = stats.kurtosis(returns)
        
        # 如果有基准数据，计算相对指标
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            benchmark_returns = np.array(benchmark_returns)
            benchmark_returns = benchmark_returns[~np.isnan(benchmark_returns)]
            
            if len(benchmark_returns) == len(returns):
                # 信息比率
                excess_returns = returns - benchmark_returns
                tracking_error = np.std(excess_returns, ddof=1) * np.sqrt(self.trading_days_per_year)
                metrics['information_ratio'] = (np.mean(excess_returns) * self.trading_days_per_year) / tracking_error if tracking_error > 0 else 0.0
                
                # Beta
                if np.var(benchmark_returns) > 0:
                    metrics['beta'] = np.cov(returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
                else:
                    metrics['beta'] = 0.0
                
                # Alpha
                benchmark_annual_return = np.mean(benchmark_returns) * self.trading_days_per_year
                metrics['alpha'] = annual_return - metrics['beta'] * benchmark_annual_return
        
        return metrics
    
    def calculate_drawdown_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """
        计算回撤相关指标
        
        Args:
            returns: 收益率序列
            
        Returns:
            Dict[str, float]: 回撤指标字典
        """
        if len(returns) == 0:
            return {}
        
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        
        # 最大回撤
        max_drawdown = np.min(drawdowns)
        
        # 回撤持续时间
        in_drawdown = drawdowns < 0
        drawdown_periods = []
        current_period = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                    current_period = 0
        
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        # 平均回撤持续时间
        avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0.0
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0.0
        
        # 回撤频率
        num_drawdown_periods = len(drawdown_periods)
        drawdown_frequency = num_drawdown_periods / len(returns) if len(returns) > 0 else 0.0
        
        return {
            'max_drawdown': max_drawdown,
            'avg_drawdown_duration': avg_drawdown_duration,
            'max_drawdown_duration': max_drawdown_duration,
            'drawdown_frequency': drawdown_frequency,
            'num_drawdown_periods': num_drawdown_periods
        }
    
    def calculate_risk_metrics(self, returns: np.ndarray, confidence_levels: List[float] = [0.95, 0.99]) -> Dict[str, float]:
        """
        计算风险指标
        
        Args:
            returns: 收益率序列
            confidence_levels: 置信水平列表
            
        Returns:
            Dict[str, float]: 风险指标字典
        """
        if len(returns) == 0:
            return {}
        
        returns = np.array(returns)
        returns = returns[~np.isnan(returns)]
        
        risk_metrics = {}
        
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            percentile = alpha * 100
            
            # VaR
            var = np.percentile(returns, percentile)
            risk_metrics[f'var_{int(conf_level*100)}'] = var
            
            # CVaR (Expected Shortfall)
            cvar = np.mean(returns[returns <= var]) if np.any(returns <= var) else var
            risk_metrics[f'cvar_{int(conf_level*100)}'] = cvar
        
        # 下行风险
        downside_returns = returns[returns < 0]
        risk_metrics['downside_risk'] = np.std(downside_returns, ddof=1) if len(downside_returns) > 0 else 0.0
        
        # 上行潜力
        upside_returns = returns[returns > 0]
        risk_metrics['upside_potential'] = np.std(upside_returns, ddof=1) if len(upside_returns) > 0 else 0.0
        
        return risk_metrics
    
    def calculate_statistical_significance(self, returns1: np.ndarray, returns2: np.ndarray) -> Dict[str, Any]:
        """
        计算两组收益率的统计显著性
        
        Args:
            returns1: 第一组收益率
            returns2: 第二组收益率
            
        Returns:
            Dict[str, Any]: 统计检验结果
        """
        if len(returns1) == 0 or len(returns2) == 0:
            return {}
        
        results = {}
        
        # 配对t检验
        try:
            t_stat, t_pvalue = stats.ttest_rel(returns1, returns2)
            results['paired_t_test'] = {
                'statistic': t_stat,
                'p_value': t_pvalue,
                'significant_05': t_pvalue < 0.05,
                'significant_01': t_pvalue < 0.01
            }
        except Exception as e:
            results['paired_t_test'] = {'error': str(e)}
        
        # Wilcoxon符号秩检验
        try:
            w_stat, w_pvalue = stats.wilcoxon(returns1, returns2)
            results['wilcoxon_test'] = {
                'statistic': w_stat,
                'p_value': w_pvalue,
                'significant_05': w_pvalue < 0.05,
                'significant_01': w_pvalue < 0.01
            }
        except Exception as e:
            results['wilcoxon_test'] = {'error': str(e)}
        
        # 基础统计
        results['descriptive_stats'] = {
            'mean_diff': np.mean(returns1) - np.mean(returns2),
            'std_diff': np.std(returns1, ddof=1) - np.std(returns2, ddof=1),
            'correlation': np.corrcoef(returns1, returns2)[0, 1] if len(returns1) == len(returns2) else np.nan
        }
        
        return results
    
    def calculate_all_metrics(self, returns: np.ndarray, benchmark_returns: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        计算所有指标
        
        Args:
            returns: 策略收益率序列
            benchmark_returns: 基准收益率序列
            
        Returns:
            Dict[str, Any]: 所有指标字典
        """
        all_metrics = {}
        
        # 基础指标
        all_metrics.update(self.calculate_basic_metrics(returns))
        
        # 高级指标
        all_metrics.update(self.calculate_advanced_metrics(returns, benchmark_returns))
        
        # 回撤指标
        all_metrics.update(self.calculate_drawdown_metrics(returns))
        
        # 风险指标
        all_metrics.update(self.calculate_risk_metrics(returns))
        
        return all_metrics
    
    def _get_empty_metrics(self) -> Dict[str, float]:
        """
        返回空指标字典
        """
        return {
            'annual_return': 0.0,
            'annual_volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'var_95': 0.0,
            'cvar_95': 0.0,
            'total_return': 0.0,
            'num_trades': 0
        }
