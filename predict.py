'''
run preditor will return all the predicted action for tickers in watchlist 
'''
import datetime
import pandas as pd
from data import MODEL_PATH, DataStorage
from preprocess import Preprocessor 
from retrying import retry
import pysnooper
import globals

MODEL_PATH = ""

from .bandwagon import Bandwagon

class Predictor:

    def __init__(self,file_name:str):
        ...
        # self.agent = DQN(state_space, action_space, **config)   # .target_net
        # self.agent.load(model_path)
        # self.watchlist = watch_list if watch_list else self.load_watchlist() 
        self.ds = DataStorage()
        # self.dw = DataWorker()
        # self.end_time = datetime.datetime.now().strftime('%Y-%m-%d')
        # self.days_back = 1000   # 某些股票最近没有数据，导致这项过少使获取的股票序列过短，没有拐点而报错
        df = self.load_watchlist(file_name)
        self.bw = Bandwagon(df)

    def load_watchlist(self, filename:str):
        '''
        return self.ds.load_watchlist() # later
        '''
        self.df = pd.read_excel(filename, header=0)
        return self.df

    def predict(self, state):
        action = 1 if self.bw.vote() > 40 else -1 # 总分>40 买入，<40 卖出
        self.ds.save_action()   # 保存今日的action，以备后查
        return action 

    def latest_actions(self,watchlist=WatchList)->list[tuple]:
        ''' pretty much the same as 'watch(·)'
            w.r.t. each ticker in watchlist, get the trend(t). latest action is the last row of the dataframe
            this func can also be replaced by:
                result = [(self.end_time, t, t.iloc[-1].action) for t in self.trends(WatchList)]
                df = pd.DataFrame(result,columns=['date','ticker','action'],dtype=int)
        '''
        latest_action = lambda t: self.trend(t).iloc[-1].action
        result = [(self.end_time, t,latest_action(t)) for t in watchlist]
        df = pd.DataFrame(result,columns=['date','ticker','action'],dtype=int)
        self.ds.save_predicted(df[df.action.isin([-1,1])], if_exists = 'append') # save only action in [-1,1]
        return result # or, df as 'st.table(df)' in visualize.py
        
if __name__ == "__main__":
    predictor = Predictor(MODEL_PATH, WatchList)
    print(predictor.latest_actions(WatchList))
    print(predictor.trends(WatchList))