def calculate_asset_metrics(data, window=14):
    import pandas as pd
    import numpy as np
    
    try:
        # Extract historical data
        historical_data = data['data']['data']
        df = pd.DataFrame(historical_data)
        
        # Date conversion and sorting
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values('date', inplace=True)
        df.set_index('date', inplace=True)
        
        # Calculate CAGR
        start_price = df['close'].iloc[0]
        end_price = df['close'].iloc[-1]
        num_years = (df.index[-1] - df.index[0]).days / 365.0
        cagr = ((end_price / start_price) ** (1 / num_years)) - 1
        
        # Calculate ATR
        df['tr'] = np.maximum.reduce([
            df['high'] - df['low'], 
            np.abs(df['high'] - df['close'].shift(1)),
            np.abs(df['low'] - df['close'].shift(1))
        ])
        atr = df['tr'].rolling(window=window).mean().iloc[-1]
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.clip(lower=0)).rolling(window=window).mean()
        loss = (-delta.clip(upper=0)).abs().rolling(window=window).mean()
        rs = gain / loss
        rsi = 100.0 - (100.0 / (1.0 + rs)).iloc[-1]
        
        # Gaps and Breakouts
        df['gap'] = df['open'] - df['close'].shift(1)
        df['rolling_high'] = df['high'].rolling(window=20).max()
        df['rolling_low'] = df['low'].rolling(window=20).min()
        
        # Fibonacci Retracement Levels
        high = df['high'].max()
        low = df['low'].min()
        fibonacci_levels = {
            "0%": low,
            "23.6%": low + 0.236 * (high - low),
            "38.2%": low + 0.382 * (high - low),
            "50%": (high + low) / 2,
            "61.8%": low + 0.618 * (high - low),
            "100%": high
        }
        
        # Prepare DataFrames for output
        gaps_df = df[df['gap'].abs() > df['tr'].mean()][['gap']].reset_index()
        breakouts_df = df[
            (df['close'] > df['rolling_high']) | 
            (df['close'] < df['rolling_low'])
        ][['close']].reset_index()
        
        # Return structured result
        return {
            'metrics': {
                'CAGR': float(cagr),
                'ATR': float(atr),
                'RSI': float(rsi),
            },
            'fibonacci': fibonacci_levels,
            'gaps': gaps_df.to_dict(orient='records'),
            'breakouts': breakouts_df.to_dict(orient='records')
        }
    
    except Exception as e:
        return {"error": str(e)}
def generate_recommendations(metrics):
    rsi = metrics.get("RSI")
    if rsi < 30:
        return {"action": "buy", "reason": f"RSI={rsi} indicates oversold conditions"}
    elif rsi > 70:
        return {"action": "sell", "reason": f"RSI={rsi} indicates overbought conditions"}
    else:
        return {"action": "hold", "reason": f"RSI={rsi} indicates neutral conditions"}
