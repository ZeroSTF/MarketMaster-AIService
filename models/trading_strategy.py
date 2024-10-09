import pandas as pd
import numpy as np
from utils.data_fetcher import get_stock_data

def calculate_rsi(data, periods=14):
    close_delta = data['Close'].diff()
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    return rsi

def optimize(symbol, strategy_type):
    data = get_stock_data(symbol)
    
    if strategy_type == 'moving_average':
        short_window = 50
        long_window = 200
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0
        signals['short_mavg'] = data['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
        signals['long_mavg'] = data['Close'].rolling(window=long_window, min_periods=1, center=False).mean()
        signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] 
                                                    > signals['long_mavg'][short_window:], 1.0, 0.0)
        signals['positions'] = signals['signal'].diff()
        
        return {
            'symbol': symbol,
            'strategy': 'moving_average',
            'signals': signals['positions'].tolist(),
            'last_signal': 'Buy' if signals['signal'].iloc[-1] == 1 else 'Sell'
        }
    
    elif strategy_type == 'rsi':
        rsi_period = 14
        overbought = 70
        oversold = 30
        
        signals = pd.DataFrame(index=data.index)
        signals['rsi'] = calculate_rsi(data, rsi_period)
        signals['signal'] = 0.0
        signals['signal'] = np.where(signals['rsi'] < oversold, 1.0, 0.0)
        signals['signal'] = np.where(signals['rsi'] > overbought, -1.0, signals['signal'])
        signals['positions'] = signals['signal'].diff()
        
        return {
            'symbol': symbol,
            'strategy': 'rsi',
            'signals': signals['positions'].tolist(),
            'last_signal': 'Buy' if signals['signal'].iloc[-1] == 1 else ('Sell' if signals['signal'].iloc[-1] == -1 else 'Hold')
        }
    
    return {'error': 'Strategy not implemented'}