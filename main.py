from flask import Flask,render_template,request,flash,redirect,url_for,session
import yfinance as yf
import numpy as np
import datetime
from datetime import datetime,date, timedelta
import pickle
import matplotlib.pyplot as plt
import pandas as pd

# Create an app and set up the secret key for sessions
app = Flask(__name__)
app.secret_key= 'hello'

def get_ticker_info(ticker):
    '''Collect the information needed to predict the ticker(collect the features)'''
    try:
        # Call the ticker
        ticker = yf.Ticker(ticker)
        # Get the average volume and mid price from the info dictionary
        averageVolume = ticker.info['averageVolume']
        current_price = ticker.info['currentPrice']
        average = ticker.info['fiftyDayAverage']
        diff = current_price - average
        # Get the closing price 90 days ago
        today= date.today()
        start_date = str(today -  timedelta(90))[:10]
        end_date = str(today -  timedelta(85))[:10]
        start_price = float(ticker.history(start = start_date, end = end_date, interval = '1d' )['Close'][0])
        # Calculate the dividend, if the output is None, dividend is 0
        try:
            dividend = current_price*ticker.info['dividendYield']/365*90
        except:
            dividend = 0
        # Calculate the returns: capital gains + dividend
        returns= (current_price-start_price+dividend)
        # Return the features in list in array
        return np.array([[averageVolume,current_price,dividend,returns,diff]])
    except:
        # If any error occured, do nothing
        pass

def normalize(lst):
    '''Normalize the features values between 0 to 1'''
    # Get the mins and maxs of the features columns
    minmax_in = open(r"C:\Users\sunny\Desktop\side\static\mins_and_maxs.pickle",'rb')
    mins_and_maxs = pickle.load(minmax_in)
    mins = mins_and_maxs[0]
    maxs = mins_and_maxs[1]
    # Using the mins and maxs to normalize the features
    for i in range(len(lst)):
        lst[i] = (lst[i]-mins[i])/ (maxs[i] - mins[i])
    return lst

def predict_stock(ticker_info):
    '''Given the features, predict the stock returns'''
    # Get the model and accuracy we stored
    model_in = open(r"C:\Users\sunny\Desktop\side\static\model.pickle",'rb')
    acc_in = open(r"C:\Users\sunny\Desktop\side\static\acc.pickle",'rb')
    acc = pickle.load(acc_in)
    model = pickle.load(model_in)
    # Predict the returns
    returns_prediction = model.predict(ticker_info)
    # Return the prediction and the accuracy
    result_lst = [returns_prediction, acc]
    return result_lst

def get_dates():
    '''Get valid dates for extracting the 1 day and the 1 year closing price'''
    '''Note that the stock market opens at 9:30 am, close at 4:00 pm, also closes on weekends  '''
    # Get today's date
    today = date.today() 
    # Store the opening hour of the stock market
    morning = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
    # If today is Saturday or Sunday, use Friday's data
    if today.weekday() == 5:
        start_date1d = today - timedelta(1)
    elif today.weekday() == 6:
        start_date1d = today - timedelta(2)
    else: 
        # Otherwise, use the data on the current date
        start_date1d = today
        # If it is before 9:30 am, use the data from yesterday
        if datetime.now()<morning:
            start_date1d -= timedelta(1)
            # If it is Monday and is before 9:30 am, use data from the last Friday
            if today.weekday() == 0:
                start_date1d -= timedelta(2)

    # Get the date 365 days ago for extracting the 1 year closing price
    start_date1y = today -  timedelta(365)
    # Get tomorrow's date
    tomorrow = today +  timedelta(1)
    # Get the dates in 'yyyy-mm-dd' format
    start_date1y = str(start_date1y)[:10]
    tomorrow = str(tomorrow)[:10]
    return start_date1y,start_date1d, tomorrow


def additional_info(symbol):
    '''Get extra information about the stock, including '''
    '''current price,currency,52 week high,52 week low,P/E ratio,dividend yield,marketCap,open,high,low'''
    # Call the ticker
    ticker = yf.Ticker(symbol)
    # Get valid dates from the get_dates function
    start_date1y,start_date1d, tomorrow = get_dates()
    # Get the 1 year closing price and the average 1 year closing price
    prices1y = ticker.history(start = start_date1y, end = tomorrow, interval = '1d')['Close']
    avg = prices1y.mean()
    # Try to get the 1 day price in 5-min interval and 1d interval
    try:
        start_date1d_str = str(start_date1d)[:10]
        prices5m = ticker.history(start = start_date1d_str, end = tomorrow, interval = '5m')['Open']
        prices1d = ticker.history(start = start_date1d_str, end = tomorrow, interval = '1d')
    # If any error occurs(e.g. the stock market is in holiday), try again using the previous date
    except:
        start_date1d_str = str(start_date1d - timedelta(1))[:10]
        prices5m = ticker.history(start = start_date1d_str, end = tomorrow, interval = '5m')['Open']
        prices1d = ticker.history(start = start_date1d_str, end = tomorrow, interval = '1d')
    
    # Convert the index of prices5m in string 'hh:mm'
    time = prices5m.index.tolist()
    time = [str(i)[11:16] for i in time]
    prices5m.index = time

    # Get the open,.high, low
    open = round(prices1d['Open'][0],2)
    high = round(prices1d['High'][0],2)
    low = round(prices1d['Low'][0],2)
    # 
    prev_close = prices1y[-2]
    current = ticker.info['currentPrice']
    currency = ticker.info['currency']
    high52 = ticker.info['fiftyTwoWeekHigh']
    low52 = ticker.info['fiftyTwoWeekLow']
    marketCap = ticker.info['marketCap']
    unit = ''
    if marketCap > 1000000000000:
        marketCap = marketCap/1000000000000
        unit = 'T'
    elif marketCap > 1000000000:
        marketCap = marketCap/1000000000
        unit = 'B'
    elif marketCap > 1000000000:
        marketCap = marketCap/1000000
        unit = 'M'
    marketCap = str(round(marketCap,2))+unit
    
    try:
        pe = round(ticker.info['trailingPE'],2)
    except:
        pe = '-'
    
    try:
        dividendYield = ticker.info['dividendYield']*100
        if dividendYield is None or dividendYield == 'None':
            dividendYield = '-'
    except:
        dividendYield = 0

    fig = plt.figure(figsize = (6,4))
    ax = fig.add_subplot()
    ax.plot(prices1y.index, prices1y,label = 'Closing Prices')
    ax.axhline(y = avg, color = 'black',linestyle = 'dashed',label = 'Average: '+str(round(avg,2)))
    ax.set_title(symbol+": Historical Closing Price 1Y")
    ax.legend(loc = 'best')
    fig.savefig(r'C:\Users\sunny\Desktop\side\static\price1y.png')
    fig.clear()
    
    fig1 = plt.figure(figsize = (6,4))
    ax1 = fig1.add_subplot()
    ax1.plot(prices5m.index, prices5m,label = 'Closing Prices')
    ax1.axhline(y = prev_close, color = 'black',linestyle = 'dashed',label = 'Previous Close: '+str(round(prev_close,2)))
    ax1.set_title(symbol+": Closing Prices on "+start_date1d_str)
    ax1.legend(loc = 'best')
    ax1.margins(x = 0.05)
    ax1.set_xticks(ax1.get_xticks()[::12])
    fig1.savefig(r'C:\Users\sunny\Desktop\side\static\price1d.png')
    fig1.clear()

    return current,currency,high52,low52,pe,dividendYield,marketCap,open,high,low



@app.route('/',methods = ['POST','GET'])        
@app.route('/prediction',methods = ['POST','GET'])
def prediction():
    if request.method == 'POST':
        ticker = request.form['tkr']
        ticker = ticker.upper()
        if ticker:
            return redirect(url_for("result", ticker = ticker))
        else:
            flash("Please enter a proper ticker.",'warning')
            return redirect(url_for("prediction"))
    else:
        return render_template("prediction.html")

@app.route('/<ticker>',methods = ['POST','GET'])
def result(ticker):
    if request.method == 'POST':
        return redirect(url_for("prediction"))
    if ticker in session:
        result_lst = session[ticker]
        info_lst = additional_info(ticker)
        return render_template("result.html",info_lst = info_lst,ticker = result_lst[0], final_result = result_lst[1], acc = result_lst[2]*100)
    else:
        ticker_info = get_ticker_info(ticker)
        test = np.sum(ticker_info)
        if test is not None:
            ticker_info = normalize(ticker_info)
            result_lst = predict_stock(ticker_info)
            final_result = result_lst[0]
            acc = round(result_lst[1],2)
            if final_result == 1:
                final_result = 'positive'
            else:
                final_result = 'negative'
            info_lst = additional_info(ticker)
            session[ticker] = [ticker,final_result,acc]
            result_lst = session[ticker]
            return render_template("result.html",info_lst = info_lst, ticker = result_lst[0], final_result = result_lst[1], acc = result_lst[2]*100)
        else:
            flash(f"Can't get info for {ticker}, please try another ticker. ",'warning')
            return redirect(url_for("prediction"))





if __name__ == '__main__':
    app.run(debug=True)