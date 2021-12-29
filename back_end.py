import pandas as pd
import numpy as np
import yfinance as yf
from yahoo_fin import stock_info as si
import datetime
from datetime import date
import concurrent.futures
import schedule
import time
import sklearn
from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV
import pickle

def get_tickers():
    sp500 = set(si.tickers_sp500())
    nasdaq = set(si.tickers_nasdaq())
    dow = set(si.tickers_dow())
    other = set(si.tickers_other())

    total = set.union(sp500,nasdaq,dow,other)
    len(total)

    filter_lst = ['W','R','P','Q','U','T','L','Z']
    bad_tickers = set()
    for i in total:
        if (len(i) >= 5 and i[-1] in filter_lst) or '$' in i or len(i) == 0:
            bad_tickers.add(i)
            
    good_tickers = total - bad_tickers
    return list(good_tickers)

def timing(ticker,today):
    try:
        end_date = ticker.history(start = today -  datetime.timedelta(10),end = today,interval = '1d')['Close'].index[-1]
        end_end = today
        mid_date = end_date -  datetime.timedelta(90)
        mid_end = mid_date +  datetime.timedelta(10)
        start_date = mid_date -  datetime.timedelta(90)
        start_end = start_date +  datetime.timedelta(10)
        return str(start_date)[:10],str(start_end)[:10],str(mid_date)[:10],str(mid_end)[:10],str(end_date)[:10],str(end_end)[:10]
    except:
        pass
    
def fill(symbol):
    today = date.today()
    ticker = yf.Ticker(symbol)
    try:
        start_date,start_end,mid_date,mid_end,end_date,end_end = timing(ticker,today)
        end_price = float(ticker.history(start = end_date,end = end_end,interval = '1d')['Close'][0])
        start_price = float(ticker.history(start = start_date,end = start_end,interval = '1d')['Close'][0])
        mid_price = float(ticker.history(start = mid_date,end = mid_end,interval = '1d')['Close'][0])
        dividendYield = ticker.info['dividendYield']
        averageVolume = ticker.info['averageVolume']
        fiftyDayAverage = ticker.info['fiftyDayAverage']
        #currentPrice = ticker.info['currentPrice']
        
        return [symbol,averageVolume,fiftyDayAverage,dividendYield,start_price,mid_price,end_price]
    except:
        pass


def collect_data(df, symbols):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = executor.map(fill,symbols)
    for i in result:
        try:
            df[i[0]] = [i[1],i[2],i[3],i[4],i[5],i[6]]
        except:
            pass
    return df

def get_features(features):
    features = features.T
    features.columns = ['averageVolume','fiftyDayAverage','dividendYield','start_price','mid_price','end_price']
    features['dividendYield']=features['dividendYield'].fillna(0)
    features['dividend']=features['dividendYield']*features['mid_price']/365*90
    features = features.dropna()
    features['shares'] = 100000/features['start_price']
    features['returns'] = (features['shares']*(features['mid_price']-features['start_price']))/100000*100
    features['target_shares'] = 100000/features['mid_price']
    features['target_returns'] = (features['target_shares']*(features['end_price']-features['mid_price']))/100000*100
    features['diff'] = features['mid_price'] - features['fiftyDayAverage']
    target = (features['target_returns']>0) 
    target = target.values.tolist()
    mapping = {True: 1, False: 0}
    replace = mapping.get
    target = [replace(n, n) for n in target]
    features['target'] = target
    features.reset_index(drop = True,inplace=True)
    features = features.drop(['start_price','end_price','dividendYield','fiftyDayAverage','shares','target_shares','target_returns'],axis=1)
    return features


def get_x_y(data):
    features = data.drop(['target'],axis = 1)
    mins = features.apply(lambda x: x.min(axis=0))
    maxs = features.apply(lambda x: x.max(axis=0))
    mins = mins.values.tolist()
    maxs = maxs.values.tolist()
    features = features.apply(lambda x: (x - x.min(axis=0))/ (x.max(axis=0) - x.min(axis= 0 )))
    y = np.array(data.target)
    x = np.array(list(zip(features['averageVolume'].tolist(),features['mid_price'].tolist(),features['returns'].tolist(),features['diff'].tolist(),features['dividend'].tolist())))
    return x,y, [mins,maxs]


def main():
    features = pd.DataFrame()
    symbols = get_tickers()
    features = collect_data(features, symbols)
    features = get_features(features)
    
    features.to_csv(r'C:\Users\sunny\Desktop\side\static\my_data.csv',index=False)
    data = pd.read_csv(r'C:\Users\sunny\Desktop\side\static\my_data.csv')
    
    x,y,mins_and_maxs = get_x_y(data)
    x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size = 0.1)
    
    
    param_grid = [{
    'kernel' :['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma':['auto', 1, 0.1, 0.01, 0.001, 0.0001],
    'C': [0.1,1, 10, 100,500],
    'degree':[2,3,4,5,6,7,8]}]

    model = GridSearchCV(svm.SVC(), param_grid = param_grid, refit=True,verbose=2)
    best_model= model.fit(x_train,y_train)
    best_acc = 0
    
    for i in range(1000):
        x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size = 0.1)
        model = best_model.best_estimator_
        model.fit(x_train,y_train)
        y_predict =model.predict(x_test)
        acc = metrics.accuracy_score(y_test,y_predict)
        if acc>best_acc:
            best_acc = acc
            with open(r"C:\Users\sunny\Desktop\side\static\model.pickle",'wb') as f:
                pickle.dump(model,f)
    with open(r"C:\Users\sunny\Desktop\side\static\acc.pickle",'wb') as f:
        pickle.dump(best_acc,f)
    with open(r"C:\Users\sunny\Desktop\side\static\mins_and_maxs.pickle",'wb') as f:
        pickle.dump(mins_and_maxs,f)
    print('done!',best_acc)

schedule.every().wednesday.at('12:13').do(main)
while 1:
    schedule.run_pending()
    time.sleep(1)

