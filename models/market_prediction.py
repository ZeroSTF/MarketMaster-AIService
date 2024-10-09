import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from utils.data_fetcher import get_stock_data

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

def predict(symbol, days):
    data = get_stock_data(symbol)
    
    df = data[['Close']].copy()
    df['Returns'] = df['Close'].pct_change()
    df = df.dropna()
    
    X = df[['Close', 'Returns']]
    y = df['Close']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    TIME_STEPS = 60
    X_train_lstm, y_train_lstm = create_dataset(X_train, y_train, TIME_STEPS)
    X_test_lstm, y_test_lstm = create_dataset(X_test, y_test, TIME_STEPS)
    
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
        Dropout(0.2),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    model.fit(X_train_lstm, y_train_lstm, epochs=100, batch_size=32, validation_split=0.1, verbose=0)
    
    last_sequence = X_test_scaled[-TIME_STEPS:].reshape((1, TIME_STEPS, X_test_scaled.shape[1]))
    predictions = []
    for _ in range(days):
        next_pred = model.predict(last_sequence)
        predictions.append(next_pred[0, 0])
        last_sequence = np.roll(last_sequence, -1, axis=1)
        last_sequence[0, -1, 0] = next_pred
        last_sequence[0, -1, 1] = (next_pred - last_sequence[0, -2, 0]) / last_sequence[0, -2, 0]
    
    return {'symbol': symbol, 'predictions': predictions}