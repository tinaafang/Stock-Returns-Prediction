from skfeature.function.similarity_based import fisher_score
import pandas as pd
import numpy as np
from yahoo_fin import stock_info as si
import datetime
from sklearn import svm, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from datetime import date
import concurrent.futures
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
gc.enable()


def get_tickers():
    '''Get a list of tickers in US market'''
    # Get lists of tickers and put them in a set
    sp500 = set(si.tickers_sp500())
    nasdaq = set(si.tickers_nasdaq())
    dow = set(si.tickers_dow())
    other = set(si.tickers_other())
    total = set.union(sp500, nasdaq, dow, other)

    # Find out the undesired tickers(such as tickers with warrants or rights)
    filter_lst = ['W', 'R', 'P', 'Q', 'U', 'T', 'L', 'Z']
    bad_tickers = set()
    for i in total:
        if (len(i) >= 5 and i[-1] in filter_lst) or '$' in i or len(i) == 0:
            bad_tickers.add(i)

    # Subtract the undesirable tickers from the ticker set
    good_tickers = total - bad_tickers
    # Return the remaining tickers in a list
    return list(good_tickers)


def get_RSI(prices):
    '''calculate the RSI using the given prices'''
    # calculate returns
    returns = prices.pct_change().iloc[1:]*100
    # calculate average gain and average loss
    avgGain = returns[returns > 0].sum()/len(returns)
    avgLoss = returns[returns < 0].sum()/len(returns)*-1
    # calculate RSI and return the result
    RSI = 100 - (100/(1+(avgGain/avgLoss)))
    return RSI


def get_features(ticker):
    '''collect features for technical analysis of one ticker'''

    # imagine we're back to 180 days ago
    # then 180 days ago becomes "now", and 360 days ago becomes "past".
    # we'll use the "now" and "past" data to obtain features
    # the current date now becomes 180 days in the future, we'll use the "future" data to obtain target

    future = date.today()
    now = future - datetime.timedelta(180)
    past = now - datetime.timedelta(180)

    # get stock price for features and targets respectively
    try:
        stockDataFeature = si.get_data(ticker, start_date=past, end_date=now)
        stockDataTarget = si.get_data(ticker, start_date=now, end_date=future)
    # if failed, return an array of NAs
    except:
        return np.repeat(pd.NA, 12)

    pricesFeatures = stockDataFeature['adjclose']
    pricesTarget = stockDataTarget['adjclose']

    # extract closing price, 180 days max and 180 days min for the "now" data
    currentClose = pricesFeatures[-1]
    highest = pricesFeatures.max()
    lowest = pricesFeatures.min()

    # calculate featuers
    SMA = pricesFeatures.mean()
    EMA = pricesFeatures.ewm(span=14, adjust=False).mean()[0]
    MOM = currentClose - pricesFeatures[0]
    STCK = (currentClose - lowest)/(highest - lowest)*100
    MACD = SMA - pricesFeatures.iloc[::-1][0:14].mean()
    RSI = get_RSI(pricesFeatures)
    williamsR = (highest - currentClose)/(highest - lowest)
    ADI = (currentClose - lowest)-(highest - currentClose) / (highest - lowest)
    avgVolume = stockDataFeature['volume'].mean()
    diff = (currentClose - SMA)/SMA
    returns = (currentClose - pricesFeatures[0])/pricesFeatures[0]*100

    # calculate targets
    target = (pricesTarget[-1]-pricesTarget[0])/pricesTarget[0]*100

    # return features and target
    return [ticker, currentClose, SMA, EMA, MOM, STCK, MACD, RSI, williamsR, ADI, avgVolume, diff, returns, target]


def collect_data(goodTickers):
    '''Use multithreading to get information from the prepare_features() function'''

    # run get_features for each of the goodTickers
    with concurrent.futures.ThreadPoolExecutor() as executor:
        data = executor.map(get_features, goodTickers)

    df = pd.DataFrame()

    # store the results in df
    for i in data:
        try:
            df[i[0]] = i[1:14]
        except:
            pass

    return df


# obtain data for each of the goodTickers, and rename the data index
goodTickers = get_tickers()
data = collect_data(goodTickers)
data.index = ['ClosingPrice', 'SMA', 'EMA', 'MOM', 'STCK', 'MACD',
              'RSI', 'williamsR', 'ADI', 'avgVolume', 'diff', 'returns', 'target']

# since we only have a few NAs, we can just drop any stock tickers with NA
data = data.dropna(axis=1, thresh=11)

# transpose the data frame
data = data.T

# convert the continuous values in target into discrete values
data['target'] = np.multiply(data['target'] > 10, 1)

# split data into features and target
features = data.iloc[:, :12]
target = data.iloc[:, 12]

# standarlize features
scaler = MinMaxScaler()
scaler.fit(features)
features.iloc[:, :] = scaler.transform(features)
max = scaler.data_max_
min = scaler.data_min_
with open("mins_and_maxs.pickle", 'wb') as f:
    pickle.dump([min, max], f)

ranks = fisher_score.fisher_score(
    np.array(features), np.array(target), mode='rank')
feature_importance = pd.Series(ranks, features.columns)
data = data.drop(columns=feature_importance.sort_values().index.values[0:2])


# correlation analysis
corr = data.corr()
# convert the upper triangle of the heatmap into a data frame
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))

# for features with a correlation grater than 0.8, remove the feature that has lower correlation with the target
toRemove = []
for i in upper.index:
    for j in upper.columns:
        if abs(upper.loc[i, j]) > 0.8:
            print(i, upper.loc[i, "target"], j, upper.loc[j, "target"])
            if abs(upper.loc[i, "target"]) < abs(upper.loc[j, "target"]):
                toRemove.append(i)
            else:
                toRemove.append(j)
print(toRemove)
features = features.drop(columns=np.unique(toRemove))

# split the data into train and test set
x_train, x_test, y_train, y_test = train_test_split(
    np.array(features), np.array(target), test_size=.2)

param_grid = [{
    'kernel': ['linear', 'rbf', 'sigmoid'],
    'gamma':['auto', 1, 0.1, 0.01, 0.001],
    'C': [0.1, 1, 10, 20]}]
model = GridSearchCV(svm.SVC(), param_grid=param_grid,
                     scoring='accuracy', verbose=2, cv=5)
best_model = model.fit(x_train, y_train)

goodModel = best_model.best_estimator_
x_train, x_test, y_train, y_test = train_test_split(
    np.array(features), np.array(target), test_size=.2)
goodModel.fit(x_train, y_train)
y_predict = goodModel.predict(x_test)
acc = metrics.accuracy_score(y_test, y_predict)
print(acc)

with open("model.pickle", 'wb') as f:
    pickle.dump(goodModel, f)
