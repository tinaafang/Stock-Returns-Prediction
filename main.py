from flask import Flask, render_template, request, flash, redirect, url_for, session
import yfinance as yf
import numpy as np
import datetime
from datetime import datetime, date, timedelta
import pickle
import matplotlib.pyplot as plt
from yahoo_fin import stock_info as si

# Create an app and set up the secret key for sessions
app = Flask(__name__)
app.secret_key = 'hello'


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


def get_ticker_features(symbol):
    '''Collect the features(information needed to predict the ticker)'''
    try:
        # set up dates for current and past
        current = date.today()
        past = current - timedelta(180)
        # prepare stock prices
        stockData = si.get_data(symbol, start_date=past, end_date=current)
        prices = stockData['adjclose']

        # extract closing price, 180 days max and 180 days min for the "now" data
        currentClose = prices[-1]
        highest = prices.max()
        lowest = prices.min()

        # calculate featuers
        SMA = prices.mean()
        EMA = prices.ewm(span=14, adjust=False).mean()[0]
        MOM = currentClose - prices[0]
        STCK = (currentClose - lowest)/(highest - lowest)*100
        MACD = SMA - prices.iloc[::-1][0:14].mean()
        RSI = get_RSI(prices)
        williamsR = (highest - currentClose)/(highest - lowest)
        ADI = (currentClose - lowest) - \
            (highest - currentClose) / (highest - lowest)
        avgVolume = stockData['volume'].mean()
        diff = (currentClose - SMA)/SMA
        returns = (currentClose - prices[0])/prices[0]*100

        # return features
        # return [currentClose, SMA, EMA, MOM, STCK, MACD, RSI, williamsR, ADI, avgVolume, diff, returns]
        return [currentClose, EMA, STCK, MACD, RSI, ADI, avgVolume, diff, returns]
    except:
        # If any error occured, do nothing
        return("Something went wrong....")


def normalize(features):
    '''Normalize the features values between 0 to 1'''
    # Get the mins and maxs of the features columns
    minmax_in = open("mins_and_maxs.pickle", 'rb')
    mins_and_maxs = pickle.load(minmax_in)
    mins = mins_and_maxs[0]
    maxs = mins_and_maxs[1]
    # Using the mins and maxs to normalize the features
    for i in range(len(features)):
        features[i] = (features[i]-mins[i]) / (maxs[i] - mins[i])
    return features


def predict_stock(ticker_info):
    '''Given the features, predict the stock returns'''
    # Get the model
    model_in = open("model.pickle", 'rb')
    model = pickle.load(model_in)
    # Predict the returns
    prediction = model.predict([ticker_info])
    # Return the prediction
    return prediction


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
        if datetime.now() < morning:
            start_date1d -= timedelta(1)
            # If it is Monday and is before 9:30 am, use data from the last Friday
            if today.weekday() == 0:
                start_date1d -= timedelta(2)

    # Get the date 365 days ago for extracting the 1 year closing price
    start_date1y = today - timedelta(365)
    # Get tomorrow's date
    tomorrow = today + timedelta(1)
    # Get the dates in 'yyyy-mm-dd' format
    start_date1y = str(start_date1y)[:10]
    tomorrow = str(tomorrow)[:10]
    # Return the dates
    return start_date1y, start_date1d, tomorrow


def get_plots(prices5m, prices1y, prev_close, avg, symbol, start_date1d_str):
    '''Save the plots for price change in 1 day and 1 year'''
    # Create a figure and a subplot
    fig1 = plt.figure(figsize=(6, 4))
    ax1 = fig1.add_subplot()
    # Plot the price change in 1 day with the interval of 5 minutes
    ax1.plot(prices5m.index, prices5m, label='Closing Prices')
    # Draw a horizontal line at the closing price of the previous day with the proper label
    ax1.axhline(y=prev_close, color='black', linestyle='dashed',
                label='Previous Close: '+str(prev_close))
    # Format the plot: add the title, legend, margins
    ax1.set_title(symbol+": Closing Prices on "+start_date1d_str)
    ax1.legend(loc='best')
    ax1.margins(x=0.05)
    # Set the xticks in hours as after 9:30 am
    ax1.set_xticks(ax1.get_xticks()[::6])
    ax1.tick_params(labelrotation=45)
    # Save the plot and clear the figure
    fig1.savefig("static/price1d.png")
    fig1.clear()

    # Create a figure and a subplot
    fig2 = plt.figure(figsize=(6, 4))
    ax2 = fig2.add_subplot()
    # Plot the closing price change in 1 year
    ax2.plot(prices1y.index, prices1y, label='Closing Prices')
    # Draw a horizontal line inficating the 1 year average with a proper label
    ax2.axhline(y=avg, color='black', linestyle='dashed',
                label='Average: '+str(round(avg, 2)))
    # Add title and legend
    ax2.set_title(symbol+": Historical Closing Price 1Y")
    ax2.legend(loc='best')
    # Save the plot and clear the figure
    fig2.savefig('static/price1y.png')
    fig2.clear()


def additional_info(symbol):
    '''Get extra information about the stock, including: 
       current price,currency,52 week high,52 week low,P/E ratio,dividend yield,marketCap,open,high,low'''
    # Call the ticker
    ticker = yf.Ticker(symbol)
    # Get valid dates from the get_dates function
    start_date1y, start_date1d, tomorrow = get_dates()
    # Get the 1 year closing price and the average 1 year closing price
    prices1y = ticker.history(
        start=start_date1y, end=tomorrow, interval='1d')['Close']
    avg = prices1y.mean()
    # Try to get the 1 day price in 5-min interval and 1d interval, as well as the closing price on the previous day
    try:
        start_date1d_str = str(start_date1d)[:10]
        previous_date_str = str(start_date1d - timedelta(1))[:10]
        prices5m = ticker.history(
            start=start_date1d_str, end=tomorrow, interval='5m')['Open']
        prices1d = ticker.history(
            start=start_date1d_str, end=tomorrow, interval='1d')
        prev_close = round(ticker.history(
            start=previous_date_str, end=tomorrow, interval='1d')['Close'][0], 2)
    # If any error occurs(e.g. the stock market is in holiday), try again using the previous date
    except:
        start_date1d_str = str(start_date1d - timedelta(1))[:10]
        previous_date_str = str(start_date1d - timedelta(2))[:10]
        prices5m = ticker.history(
            start=start_date1d_str, end=tomorrow, interval='5m')['Open']
        prices1d = ticker.history(
            start=start_date1d_str, end=tomorrow, interval='1d')
        prev_close = round(ticker.history(
            start=previous_date_str, end=tomorrow, interval='1d')['Close'][0], 2)

    # Convert the index of prices5m in string 'hh:mm'
    time = prices5m.index.tolist()
    time = [str(i)[11:16] for i in time]
    prices5m.index = time
    # Plot the prices and save as png
    get_plots(prices5m, prices1y, prev_close, avg, symbol, start_date1d_str)
    # Get the open,high, low
    open = round(prices1d['Open'][0], 2)
    high = round(prices1d['High'][0], 2)
    low = round(prices1d['Low'][0], 2)
    # Get the current price, currency, 52 week high and low, marketCap
    current = ticker.info['currentPrice']
    currency = si.get_quote_data(symbol)['currency']
    high52 = si.get_quote_data(symbol)['fiftyTwoWeekHigh']
    low52 = si.get_quote_data(symbol)['fiftyTwoWeekLow']
    marketCap = si.get_quote_data(symbol)['marketCap']
    # Assign a proper unit(trillion, billion or million) for marketCap
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
    marketCap = str(round(marketCap, 2))+unit
    # Get P/E ratio
    try:
        pe = round(si.get_quote_data(symbol)['trailingPE'], 2)
    except:
        pe = '-'
    # Get dividend yield
    try:
        dividendYield = str(round(si.get_quote_data(
            symbol)['trailingAnnualDividendYield']*100, 2))+"%"
        if dividendYield is None or dividendYield == 'None':
            dividendYield = '-'
    except:
        dividendYield = '-'

    # Return all the info
    return [current, currency, high52, low52, pe, dividendYield, marketCap, open, high, low]


# Set the route for the home page
@app.route('/', methods=['POST', 'GET'])
@app.route('/home', methods=['POST', 'GET'])
def home():
    '''The home page receives the ticker through html form and redirect the user to the result page'''
    # If the submit button id requested
    if request.method == 'POST':
        # Get the ticker from the html form, then make it capitalized
        ticker = request.form['tkr']
        ticker = ticker.upper()
        # If ticker is not empty, redirect user to the result page with the ticker
        if ticker:
            return redirect(url_for("result", ticker=ticker))
        # Otherwise, stay in the same page and prompt user to enter a proper ticker
        else:
            flash("Please enter a proper ticker.", 'warning')
            return redirect(url_for("home"))
    else:
        return render_template("prediction.html")

# Set up the route(which is the ticker name) for the result page


@app.route('/<ticker>', methods=['POST', 'GET'])
def result(ticker):
    '''Predict the returns for the ticker and display some additional info about the stock'''
    # If the 'predict another' ticker button is clicked, redirect the user to the prediction page
    if request.method == 'POST':
        return redirect(url_for("home"))
    # If ticker is already in the session
    if ticker in session:
        # Get the prediction result
        result_lst = session[ticker]
        # Get the additional info and store the price change plot
        info_lst = additional_info(ticker)
        # Display the html page with prediction, additional info and the plots
        return render_template("result.html", info_lst=info_lst, ticker=result_lst[0], final_result=result_lst[1])
    # If the ticker is not in session
    else:
        # First get the features columns needed for the prediction
        ticker_features = get_ticker_features(ticker)
        # check would be None if any of the features is None
        check = np.sum(ticker_features)
        # If check is not None, e.g. if every feature value is valid, we can start the prediction
        if check is not None:
            # First normalize the features into values between 0 and 1
            ticker_features = normalize(ticker_features)
            # Then get the prediction result and the accuracy in the result lst
            final_result = predict_stock(ticker_features)
            # Convert 1 and 0 into positive and negative
            if final_result == 1:
                final_result = 'positive'
            else:
                final_result = 'negative'
            # Get the additional info and store the price change plot
            info_lst = additional_info(ticker)
            # Store the ticker and prediction result into the session
            session[ticker] = [ticker, final_result]
            return render_template("result.html", info_lst=info_lst, ticker=ticker, final_result=final_result)
        # If any error occurs, send a error message to the user, then return to the prediction page
        else:
            flash(
                f"Can't get info for {ticker}, please try another ticker. ", 'warning')
            return redirect(url_for("home"))


# Run the app
if __name__ == '__main__':
    app.run()
