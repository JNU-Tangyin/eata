"""
数据加载器
处理消融实验的数据加载和预处理
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import warnings

# 隐藏警告信息
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

class DataLoader:
    """
    数据加载器，负责加载和预处理实验数据
    """
    
    def __init__(self, data_dir: Path):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        
    def load_stock_data(self, filename: str) -> pd.DataFrame:
        """
        加载股票数据
        
        Args:
            filename: 数据文件名
            
        Returns:
            pd.DataFrame: 股票数据
        """
        file_path = self.data_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            print(f"✅ 成功加载数据文件: {file_path}")
            return self._preprocess_data(df)
        except Exception as e:
            raise Exception(f"加载数据文件失败: {e}")
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        预处理数据
        """
        # 确保必要的列存在
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"数据缺少必要列: {missing_columns}")
        
        # 转换日期列
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # 排序
        df = df.sort_values('date').reset_index(drop=True)
        
        # 处理缺失值
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # 数据验证
        if len(df) < 100:
            raise ValueError(f"数据量过少，仅有 {len(df)} 条记录，至少需要100条")
        
        return df
    
    def split_data(self, df: pd.DataFrame, 
                   train_ratio: float = 0.7, 
                   val_ratio: float = 0.15,
                   test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        分割数据为训练、验证、测试集
        
        Args:
            df: 原始数据
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 训练集、验证集、测试集
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            print("⚠️ 数据分割比例之和不等于1，自动调整")
            total = train_ratio + val_ratio + test_ratio
            train_ratio /= total
            val_ratio /= total
            test_ratio /= total
        
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        print(f"📊 数据分割完成:")
        print(f"   训练集: {len(train_df)} 条 ({len(train_df)/n*100:.1f}%)")
        print(f"   验证集: {len(val_df)} 条 ({len(val_df)/n*100:.1f}%)")
        print(f"   测试集: {len(test_df)} 条 ({len(test_df)/n*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def load_multiple_stocks(self, stock_list: list) -> Dict[str, pd.DataFrame]:
        """
        加载多只股票数据
        
        Args:
            stock_list: 股票代码列表
            
        Returns:
            Dict[str, pd.DataFrame]: 股票数据字典
        """
        stock_data = {}
        
        for stock in stock_list:
            filename = f"{stock}.csv"
            df = self.load_stock_data(filename)
            if not df.empty:
                stock_data[stock] = df
            else:
                print(f"⚠️ 跳过股票 {stock}，数据加载失败")
        
        print(f"✅ 成功加载 {len(stock_data)} 只股票数据")
        return stock_data
    
    def get_data_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        获取数据统计信息
        
        Args:
            df: 数据DataFrame
            
        Returns:
            Dict[str, Any]: 统计信息
        """
        if df.empty:
            return {}
        
        stats = {
            'total_records': len(df),
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m-%d') if 'date' in df.columns else 'N/A',
                'end': df['date'].max().strftime('%Y-%m-%d') if 'date' in df.columns else 'N/A'
            },
            'price_statistics': {
                'close_mean': df['close'].mean(),
                'close_std': df['close'].std(),
                'close_min': df['close'].min(),
                'close_max': df['close'].max()
            },
            'volume_statistics': {
                'volume_mean': df['volume'].mean(),
                'volume_std': df['volume'].std(),
                'volume_min': df['volume'].min(),
                'volume_max': df['volume'].max()
            } if 'volume' in df.columns else {},
            'missing_values': df.isnull().sum().to_dict(),
            'data_quality': {
                'has_duplicates': df.duplicated().any(),
                'has_missing_values': df.isnull().any().any(),
                'price_consistency': self._check_price_consistency(df)
            }
        }
        
        return stats
    
    def _check_price_consistency(self, df: pd.DataFrame) -> bool:
        """
        检查价格数据一致性
        """
        try:
            # 检查 high >= max(open, close) 和 low <= min(open, close)
            high_check = (df['high'] >= np.maximum(df['open'], df['close'])).all()
            low_check = (df['low'] <= np.minimum(df['open'], df['close'])).all()
            
            # 检查价格为正数
            positive_check = (df[['open', 'high', 'low', 'close']] > 0).all().all()
            
            return high_check and low_check and positive_check
        except Exception:
            return False
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> Path:
        """
        保存处理后的数据
        
        Args:
            df: 数据DataFrame
            filename: 保存文件名
            
        Returns:
            Path: 保存路径
        """
        processed_dir = self.data_dir / "processed_data"
        processed_dir.mkdir(exist_ok=True)
        
        file_path = processed_dir / filename
        df.to_csv(file_path, index=False, encoding='utf-8')
        
        print(f"💾 保存处理后数据: {file_path}")
        return file_path
