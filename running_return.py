#!/usr/bin/env python
#coding=utf-8
# @date 2023.01.26

import pandas as pd
from backtest import test, inference
from agent import Agent
import stockstats

class RunningReturn(Agent):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        self.__name__ = 'RunningReturn'
        # if not isinstance(df, (pd.DataFrame,stockstats.StockDataFrame)):
            # raise TypeError("must be pd.DataFrame or stockstats.StockDataFrame")
        # if 'kdjj' not in df.columns or 'kdjd' not in df.columns:
            # raise ValueError("The DataFrame must contain the 'kdjj' and 'kdjd' columns.")
        if df.empty:
            raise ValueError("The DataFrame must not be empty.")
    
    @staticmethod
    def criteria(d:pd.DataFrame)->int:
        '''
        @input d: a dataframe of window_size
        @output : the score
        '''
        instance_of = [pd.DataFrame, stockstats.StockDataFrame]
        if not any(isinstance(d,i) for i in instance_of):   # 必须是其中之一
            raise TypeError(f"Type must be either of {instance_of}")
        required = ['kdjj','kdjd', 'pctChg']
        if not all(f in d.columns for f in required): # 所有字段都必须存在
            raise ValueError(f"The DataFrame must contain columns:{required}")
        if d.empty:
            raise ValueError("The DataFrame must not be empty.")
        
        if (d.pctChg.mean()/100)>0 and (d.kdjj - d.kdjd).iloc[-1] > 0:
            a = 1
        elif (d.pctChg.mean()/100)<0 and (d.kdjj - d.kdjd).iloc[-1] > 0:
            a = -1
        else:
            a = 0
        return a


    @classmethod
    def choose_action(cls, s: (pd.DataFrame)) -> int:
        ''' action for a single stock, RL compatible'''
        s0, s1, s2, s3 = s  # 将s解析为5分钟线、股票日线、板块日线、大盘日线
        # cls.criteria(s0)  # 分钟线用于判断stock_momentum?
        # score = cls.criteria(s1) * 0.7 + cls.criteria(s2) * 0.2 + cls.criteria(s3) * 0.1
        return cls.criteria(s1) 
        
if __name__ == "__main__":
    df = pd.read_excel("000016(full).xls", dtype={'code':'str'}, header = 0)
    df = df.sample(10)
    # df.loc[len(df)] = ['sz.510050', '上证50ETF', 0, 'sz.510050']   # code	name	weight	sector,  sz50etf, baostock 没有指数数据
    test(RunningReturn, df)
    inference(RunningReturn, df)