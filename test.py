''' following a RL paradigm, where an agent interacts with a env by .action() and .step()
1. `collect` 50 stocks(sz50 constituent) or 300 stocks(hs300 constituent) in the market, 
  and `label` each stock's sector(index code). note: bs.query_stock_basics() doesn't have the sector index code.
  we have to do it by hand.
2. `construct` a RL environment, especially .step(), .reward() methods
3. `predict` each respectively bandwagon.choose_action()
3. `save` the predicted result as .csv, columns =  ['ticker', 'date', 'close', 'score',  'action', 'pct_chg']
4. `evaluate` the result in evaluate.py, calculating the asset_change, reward (in evaluate.py)
5. `visualize` the evaluated result (in visualize.py)
'''
import datetime
import pandas as pd
import numpy as np
from data import DataStorage, BaostockDataWorker
from preprocess import Preprocessor
from datetime import datetime
import gym

from bandwagon import Bandwagon

class BandwagonEnv(gym.Env):

    def __init__(self, row, days = 2000, window_size=20):
        ''' set of the stocks, algo, and days to trace back
            the file_name must be a xls file and has fields like .code, .sector, .weight
        '''
        super(BandwagonEnv, self).__init__()
        self.dataworker = BaostockDataWorker()
        self.window_size = 20

        def prepare(r, days = 2000, window_size=20):
            stock = self.dataworker.latest(r.code, ktype="5", days = days) # 一年交易日约244天
            stock = Preprocessor(stock).bundle_process() 
            sector = self.dataworker.latest(r.sector, ktype="d", days = days)
            sector = Preprocessor(sector).bundle_process() 
            market_code = self.bw.get_market(r.ticker)
            market = self.dataworker.latest(market_code, ktype="d", days = days)
            market = Preprocessor(market).bundle_process() 
            self.stock_matrices = [x.values for x in stock.resample("20D").rolling(window_size)][window_size-1:] # 获得rolling的矩阵
            self.sector_matrices = [x.values for x in sector.rolling(window_size)][window_size-1:] # 获得rolling的矩阵
            self.market_matrices = [x.values for x in market.rolling(window_size)][window_size-1:] # 获得rolling的矩阵
            # any alignment for three dataframes or matrices here?

        prepare(row)
        self.df = self.add_reward_(self.df)

    def add_reward_(self, df:pd.DataFrame = None)->pd.DataFrame:
        '''attach reward to the df once for all, no more calculation on the fly
        requirement: df has a `landmark` column
        '''
        d = df if df else self.df.copy()

        dd = d[(d.landmark.isin(["v","^"]))] 
        d1, d2 = dd.iloc[:-1].index, dd.iloc[1:].index  # 获取索引并配对拼接

        def reward_by_length(length, direction):
            '''
            @direction 1:# from bottom to top ; -1: # from top to bottom
            @return: tuple of np.array
            '''
            assert direction in [1,-1], "direction must be either 1 or -1"
            hold_reward = np.sin(np.linspace(-0.5*np.pi,1.5*np.pi,length))                       # 底部和顶部hold，均为-1分，中间为1分
            buy_reward = np.sin(np.linspace(direction*0.5*np.pi,-direction*0.5*np.pi,length))    # 底部买入 1分，顶部买入 -1分
            sell_reward = np.sin(np.linspace(-direction*0.5*np.pi,direction*0.5*np.pi,length))   # 底部卖出 -1分，顶部卖出1分

            return buy_reward, hold_reward, sell_reward

        def attach_rewards(a,b):
            piece = d.loc[a:b]     # df.loc 闭区间；df.iloc开区间
            buy_reward, hold_reward, sell_reward =  reward_by_length(len(piece), 1 if piece.iloc[0].landmark == "v" else -1 )
            # df.loc[a:b,['buy_reward','hold_reward','sell_reward']] = buy_reward, hold_reward, sell_reward 
            d.loc[a:b,'buy_reward'] = buy_reward
            d.loc[a:b,'hold_reward'] = hold_reward
            d.loc[a:b,'sell_reward'] = sell_reward 

        d[['buy_reward','hold_reward','sell_reward']] = 0
        [attach_rewards(x1,x2) for x1,x2 in zip(d1,d2)]

        self.df = d if df is None else self.df

        return d

    def _reward(self, a:int):
        assert a in [-1, 0, 1], "invalid action " + a
        column = {1:"buy_reward",0:"hold_reward", -1:"sell_reward"}[a]
        return self.df.iloc[self.iter][column]

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self):
        ''' re-start from the first day
        '''
        self.iter = self.window_size -1
        return self.step()

    def step(self, a:int):
        self.iter = self.iter if a else self.window_size - 1    # if no action, go back to the first row
        s_ = self.df.iloc[self.iter +1 - self.window_size:self.iter+1] # 从当前行倒推window_size
        r = self._reward(a)
        done = False if self.iter < len(self.df) else True
        info = {'price':s_[-1].close, 'date':s_[-1].date, 'ticker':s_[-1].ticker} # 将ticker,价格和交易日期通过info传递
        return s_, r, done, info
    
    # a generator version of step()
    # def _step(self, a:int):
    #     yield 
    

def run(row, days) -> pd.DataFrame:
    env = BandwagonEnv(row, days)
    s = env.reset()
    result = pd.DataFrame(columns=['ticker','date','close','action','reward'])
    for _ in days: # for each row of a stock data, days copied from data days
        a = Bandwagon.choose_action(s)  # a class method of class Bandwagon
        s_, r, done, info = env.step(a)
        result.append(**info, s_.close.iloc[-1], a, r) # close price at the last row of s.
        s = s_
        if done:
            break

    return result
    

if __name__ == "__main__":
    today = datetime.today().strftime("%Y-%m-%d")   # make sure it does not go beyond to the next day
    df = pd.read_excel("000016closeweight(5).xls", dtype={'code':'str'}, header = 0)
    for _, row in df.iterrows(): # for each SZ50 constituent
        result = run(row, days= 2000)
        result.to_csv(f"output{today}/{row.code}.csv")