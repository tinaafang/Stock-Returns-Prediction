from flask import Flask,render_template,request,flash,redirect,url_for,session
import yfinance as yf
import numpy as np
import datetime
from datetime import date
import pickle

app = Flask(__name__)
app.secret_key= 'hello'

def get_ticker_info(ticker):
    try:
        ticker = yf.Ticker(ticker)
        averageVolume = ticker.info['averageVolume']
        mid_price = ticker.info['currentPrice']
        today= date.today()
        start_date = str(today -  datetime.timedelta(90))[:10]
        end_date = str(today -  datetime.timedelta(85))[:10]
        start_price = float(ticker.history(start = start_date, end = end_date, interval = '1d' )['Close'][0])
        try:
            dividend = mid_price*ticker.info['dividendYield']/365*90
        except:
            dividend = 0
        average = ticker.info['fiftyDayAverage']
        diff = mid_price - average
        shares = 100000 / mid_price
        returns= shares*(mid_price-start_price+dividend)/100000*100
        
        return np.array([[averageVolume,mid_price,dividend,returns,diff]])
    except:
        pass

def normalize(lst):
    minmax_in = open(r"C:\Users\sunny\Desktop\side\static\mm.pickle",'rb')
    mins_and_maxs = pickle.load(minmax_in)
    mins = mins_and_maxs[0]
    maxs = mins_and_maxs[1]
    for i in range(len(lst)):
        lst[i] = (lst[i]-mins[i])/ (maxs[i] - mins[i])
    return lst

def predict_stock(ticker_info):
    model_in = open(r"C:\Users\sunny\Desktop\side\static\model.pickle",'rb')
    acc_in = open(r"C:\Users\sunny\Desktop\side\static\acc.pickle",'rb')
    acc = pickle.load(acc_in)
    model = pickle.load(model_in)
    returns_prediction = model.predict(ticker_info)
    print(returns_prediction)
    result_lst = [returns_prediction, acc]
    return result_lst


@app.route('/')
@app.route('/home')
def home():
    return render_template("index.html")

@app.route('/prediction',methods = ['POST','GET'])
def prediction():
    if request.method == 'POST':
        ticker = request.form['tkr']
        ticker = ticker.upper()
        if ticker:
            return redirect(url_for("result", ticker = ticker))
        else:
            flash("Please enter a proper ticker.",'info')
            return redirect(url_for("prediction"))
    else:
        return render_template("prediction.html")

@app.route('/<ticker>',methods = ['POST','GET'])
def result(ticker):
    if request.method == 'POST':
        return redirect(url_for("prediction"))
    if ticker in session:
        result_lst = session[ticker]
        return render_template("result.html",ticker = result_lst[0], final_result = result_lst[1], acc = result_lst[2]*100)
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
            session[ticker] = [ticker,final_result,acc]
            result_lst = session[ticker]
            return render_template("result.html",ticker = result_lst[0], final_result = result_lst[1], acc = result_lst[2]*100)
        else:
            flash(f"Can't get info for {ticker}, please try another ticker. ",'info')
            return redirect(url_for("prediction"))

    


if __name__ == '__main__':
    app.run(debug=True)