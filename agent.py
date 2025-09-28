

import numpy as np
import pandas as pd
from datetime import datetime
from data import BaostockDataWorker
from preprocess import Preprocessor

# 导入NEMoTS
try:
    from nemots_adapter import NEMoTSPredictor
    NEMOTS_AVAILABLE = True
except ImportError:
    print("⚠️ NEMoTS不可用，使用简化逻辑")
    NEMOTS_AVAILABLE = False

class Agent():
    def __init__(self, df: pd.DataFrame):
        """
        NEMoTS Agent - 完全替换传统技术指标
        @param df: 股票列表 columns=['code', 'name', 'weight', 'sector']
        """
        self.stock_list = df
        
        # 数据准备
        self.dataworker = BaostockDataWorker()
        self.preprcessor = Preprocessor()
        self.window_size = 20
        
        try:
            self.stocks_datum = self._prepare_data(self.stock_list.code, ktype='d')
            self.stock_list['market'] = 'sh.000001'  # 简化大盘指数
        except Exception as e:
            print(f"⚠️ 数据准备失败: {e}")
            self.stocks_datum = []
        
        # 初始化NEMoTS预测器（在数据准备后）
        if NEMOTS_AVAILABLE:
            self.nemots_predictor = NEMoTSPredictor(lookback=20)
            self.__name__ = 'NEMoTS_Agent'
            print("🤖 初始化NEMoTS Agent")
            
            # 尝试用历史数据训练NEMoTS
            try:
                if len(self.stocks_datum) > 0 and len(self.stocks_datum[0]) > 20:
                    print("🧠 开始训练NEMoTS...")
                    self.nemots_predictor.fit(self.stocks_datum[0])
                    print("✅ NEMoTS训练完成")
            except Exception as e:
                print(f"⚠️ NEMoTS训练失败: {e}")
        else:
            self.nemots_predictor = None
            self.__name__ = 'Fallback_Agent'
            print("⚠️ 使用简化Agent")
    
    def _prepare_data(self, codes, ktype='d'):
        """简化数据准备"""
        try:
            d1 = [self.dataworker.latest(c, ktype=ktype, days=self.window_size * 3) for c in codes]
            d2 = [self.preprcessor.load(s).bundle_process() for s in d1]
            return d2
        except:
            return []
    
    def get_market(self, ticker: str) -> str:
        """获取大盘指数代码"""
        return "sh.000001"  # 简化为上证指数
    
    @staticmethod
    def criteria(d: pd.DataFrame) -> int:
        """
        NEMoTS智能信号生成 - 替换所有传统指标逻辑
        @input d: window_size的df
        @output: 交易信号 1(买入)/-1(卖出)/0(持有)
        """
        if NEMOTS_AVAILABLE:
            try:
                # 创建并训练临时NEMoTS预测器
                predictor = NEMoTSPredictor(lookback=min(10, len(d)-1), use_full_nemots=False)
                if len(d) > 10:  # 只有足够数据时才训练
                    # 确保数据包含必要字段
                    d_copy = d.copy()
                    if 'amount' not in d_copy.columns and 'volume' in d_copy.columns and 'close' in d_copy.columns:
                        d_copy['amount'] = d_copy['volume'] * d_copy['close']
                    predictor.fit(d_copy)
                return predictor.predict_action(d)
            except Exception as e:
                print(f"⚠️ NEMoTS预测失败: {e}")
        
        # 简化回退逻辑
        try:
            if len(d) > 0:
                recent_close = d['close'].iloc[-5:].mean() if len(d) >= 5 else d['close'].iloc[-1]
                prev_close = d['close'].iloc[-10:-5].mean() if len(d) >= 10 else d['close'].iloc[0]
                return 1 if recent_close > prev_close else -1
        except:
            pass
        return 0
    
    @classmethod
    def choose_action(cls, s: tuple) -> int:
        """
        NEMoTS智能动作选择 - RL兼容
        @input s: (s0, s1, s2, s3) 分别为5分钟线、股票日线、板块日线、大盘日线
        @output: 交易动作 1/-1/0
        """
        try:
            s0, s1, s2, s3 = s
            return cls.criteria(s1)  # 使用NEMoTS对股票日线数据做决策
        except Exception as e:
            print(f"⚠️ 动作选择失败: {e}")
            return 0
    
    def vote(self) -> int:
        """使用NEMoTS计算ETF总体信号"""
        if NEMOTS_AVAILABLE and self.nemots_predictor and len(self.stocks_datum) > 0:
            try:
                # 使用NEMoTS对每只股票生成信号
                signals = []
                for stock_data in self.stocks_datum:
                    if len(stock_data) > 0:
                        signal = self.nemots_predictor.predict_action(stock_data)
                        signals.append(signal)
                    else:
                        signals.append(0)
                
                # 按权重加权平均
                if len(signals) > 0:
                    weighted_signal = np.average(signals, weights=self.stock_list.weight)
                    return int(np.sign(weighted_signal) * 50)  # 转换为类似原来的范围
            except Exception as e:
                print(f"⚠️ NEMoTS投票失败: {e}")
        
        # 简化回退逻辑
        return 50  # 中性信号
    
    def etf_action(self, score) -> int:
        """ETF动作决策"""
        if score > 80:
            return 1
        elif score < 50:
            return -1
        return 0
    
    def stock_momentum(self):
        """股票动量计算（简化版）"""
        try:
            sig21 = lambda x: 2/(1 + np.exp(-x)) - 1
            
            def criteria(d):
                if len(d) > 1:
                    return d['close'].diff(1).iloc[-1]
                return 0
            
            self.stock_list['stock_momentum'] = [sig21(criteria(s)) for s in self.stocks_datum]
            return self.stock_list['stock_momentum']
        except:
            self.stock_list['stock_momentum'] = [0] * len(self.stock_list)
            return self.stock_list['stock_momentum']
    
    def strength(self, w1: float, w2: float, w3: float, w4: float) -> pd.Series:
        """
        使用NEMoTS计算股票强度
        """
        try:
            # 使用NEMoTS生成各项强度分数
            self.stock_list['stock_strength'] = [self.criteria(d) for d in self.stocks_datum]
            self.stock_list['sector_strength'] = [50] * len(self.stock_list)  # 简化
            self.stock_list['market_strength'] = [50] * len(self.stock_list)  # 简化
            self.stock_momentum()
            
            # 计算总强度
            self.stock_list['strength'] = (
                self.stock_list['stock_strength'] * w1 +
                self.stock_list['sector_strength'] * w2 +
                self.stock_list['market_strength'] * w3 +
                self.stock_list['stock_momentum'] * w4
            )
            
            return self.stock_list['strength']
        except Exception as e:
            print(f"⚠️ 强度计算失败: {e}")
            self.stock_list['strength'] = [50] * len(self.stock_list)
            return self.stock_list['strength']
