import yfinance as yf

def get_stock_data(symbol, period='1y'):
    stock = yf.Ticker(symbol)
    data = stock.history(period=period)
    return data