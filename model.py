#!/usr/bin/env python
# coding=utf-8
import numpy as np
import pandas as pd
from data_work import DataWorker

bao = DataWorker()
def minute5(ticker:str, days:int)-> pd.DataFrame:
    '''输入股票代码和交易日数量，以最新日期为end，返回end-days个交易日的5分钟线'''
    df = bao.get(ticker, days)
    return df

def sector_of(ticker):
    pass

def minute5sector(ticker:str, days):
    '''输入股票代码和交易日数量，以最新日期为end，返回end-days个交易日的该股票所在板块的5分钟线'''
    sector = sector_of(ticker)
    df = bao.get(sector, days)
    return df

def strength(ticker:str)->int:
    ''' 输入一个股票代码，计算其强弱分数
    具体做法是：
    - 根据ticker今天、昨天以致更远的分钟线进行判定；
    - 根据ticker所在板块的强弱势进行判断；
    - 上述两者加权
    '''
    df1 = minute5(ticker, 5)
    df2 = minute5sector(ticker,5)
    # 计算从这里开始
    # score = 
    return score 

def vote(tickers, weights)->int:
    '''输入多个股票代码以及各自的权重，计算etf总的强弱势'''
    s = [strength(t) for t in tickers]
    return np.dot(s,weights)

if __name__ == 'main':
    df = pd.read_excel("000016closeweight.xls",header = 0)
    # sz50 = query_sz50_stocks() # 没有权重
    v = vote(df.code, df.weights)

