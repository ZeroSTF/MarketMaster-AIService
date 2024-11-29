def calculate_asset_metrics(data):
    import pandas as pd
    import numpy as np
    
    try:
        # Extract the nested data
        historical_data = data['data']['data']
        
        # Create DataFrame
        df = pd.DataFrame(historical_data)
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values('date', inplace=True)
        df.set_index('date', inplace=True)
        
        # Compound Annual Growth Rate (CAGR)
        start_price = df['close'].iloc[0]
        end_price = df['close'].iloc[-1]
        num_years = (df.index[-1] - df.index[0]).days / 365.0
        cagr = ((end_price / start_price) ** (1 / num_years)) - 1
        
        # Average True Range (ATR)
        df['tr'] = np.maximum.reduce([
            df['high'] - df['low'], 
            np.abs(df['high'] - df['close'].shift(1)),
            np.abs(df['low'] - df['close'].shift(1))
        ])
        atr = df['tr'].rolling(window=14).mean().iloc[-1]
        
        # Relative Strength Index (RSI)
        delta = df['close'].diff()
        gain = (delta.clip(lower=0)).rolling(window=14).mean()
        loss = (-delta.clip(upper=0)).abs().rolling(window=14).mean()
        rs = gain / loss
        rsi = 100.0 - (100.0 / (1.0 + rs)).iloc[-1]
        
        # Gaps and Breakouts
        df['gap'] = df['open'] - df['close'].shift(1)
        df['rolling_high'] = df['high'].rolling(window=20).max()
        df['rolling_low'] = df['low'].rolling(window=20).min()
        
        # Create separate DataFrames for each metric
        gaps_df = df[df['gap'].abs() > df['tr'].mean()][['gap']].reset_index()
        breakouts_df = df[
            (df['close'] > df['rolling_high']) | 
            (df['close'] < df['rolling_low'])
        ][['close']].reset_index()
        scatter_df = df[['volume', 'close']].reset_index()
        
        # Convert to list of dicts
        return {
            'CAGR': float(cagr),
            'ATR': float(atr),
            'RSI': float(rsi),
            'Gaps': gaps_df.to_dict(orient='records'),
            'Breakouts': breakouts_df.to_dict(orient='records'),
            'ScatterData': scatter_df.to_dict(orient='records')
        }
    
    except Exception as e:
        print(f"Error in calculate_asset_metrics: {e}")
        raise