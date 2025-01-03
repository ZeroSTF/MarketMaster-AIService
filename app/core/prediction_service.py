import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import os
from datetime import datetime
import yfinance as yf
from ..utils.logger import logger

class PredictionService:
    def __init__(self, model_directory="models", max_retries=3, time_steps=60):
        self.model_directory = model_directory
        self.time_steps = time_steps
        self.models = {}
        self.scalers = {}
        
        # Ensure the model directory exists
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)

    def get_model_path(self, symbol):
        return os.path.join(self.model_directory, f"{symbol}_model_artifacts.joblib")

    def load_or_create_model(self, symbol):
        """Loads an existing model or creates a new one if it doesn't exist."""
        model_path = self.get_model_path(symbol)
        
        if os.path.exists(model_path):
            try:
                artifacts = joblib.load(model_path)
                return artifacts['model'], artifacts['feature_scaler']
            except Exception as e:
                logger.error(f"Error loading model for {symbol}: {str(e)}")
                return self._create_new_model()
        else:
            return self._create_new_model()
    
    def _create_new_model(self):
        """Creates a new LSTM model."""
        scaler = MinMaxScaler()
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(self.time_steps, 9)),
            Dropout(0.2),
            LSTM(100, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model, scaler
    
    def fetch_and_prepare_data(self, symbol, period="2y"):
        """Fetches historical data, adds technical indicators, and scales data for model input."""
        try:
            # Fetch historical data from yfinance
            ticker = yf.Ticker(symbol)
            hist_data = ticker.history(period=period)
            
            if hist_data.empty:
                raise ValueError(f"No historical data found for {symbol}")
            
            # Prepare DataFrame to match expected structure
            df = pd.DataFrame({
                'DATE': hist_data.index,
                'CLOTURE': hist_data['Close'],
                'VOLUME': hist_data['Volume'],
                'HIGH': hist_data['High'],
                'LOW': hist_data['Low'],
                'OPEN': hist_data['Open']
            }).reset_index(drop=True)
            
            # Add technical indicators
            df = self.add_technical_indicators(df)
            df.dropna(inplace=True)  # Remove NaN values from indicator calculations
            
            # Scale data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df[['OPEN', 'HIGH', 'LOW', 'CLOTURE', 'VOLUME', 'MA20', 'MA50', 'MA200', 'RSI', 'MACD']].values)
            
            # Prepare sequences for LSTM model
            X, y = [], []
            for i in range(self.time_steps, len(scaled_data)):
                X.append(scaled_data[i - self.time_steps:i])
                y.append(scaled_data[i, 3])  # Closing price is target
            
            X, y = np.array(X), np.array(y)
            return X, y, scaler
        
        except Exception as e:
            logger.error(f"Error fetching and preparing data for {symbol}: {str(e)}")
            raise

    def add_technical_indicators(self, df):
        """Adds technical indicators to the DataFrame."""
        # Moving averages
        df['MA20'] = df['CLOTURE'].rolling(window=20).mean()
        df['MA50'] = df['CLOTURE'].rolling(window=50).mean()
        df['MA200'] = df['CLOTURE'].rolling(window=200).mean()
        
        # RSI
        delta = df['CLOTURE'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['CLOTURE'].ewm(span=12, adjust=False).mean()
        exp2 = df['CLOTURE'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        
        return df

    def train_model(self, symbol, period="2y"):
        """Trains a model on historical data for a specific symbol."""
        X, y, scaler = self.fetch_and_prepare_data(symbol, period)
        model, _ = self.load_or_create_model(symbol)
        
        model.fit(X, y, epochs=10, batch_size=32, validation_split=0.1)
        
        # Save model and scaler
        artifacts = {'model': model, 'feature_scaler': scaler}
        joblib.dump(artifacts, self.get_model_path(symbol))
        
        logger.info(f"Model trained and saved for {symbol}")

    def predict(self, symbol):
        """Predicts future prices based on the trained model."""
        model, scaler = self.load_or_create_model(symbol)
        
        # Fetch recent data for prediction
        X, _, _ = self.fetch_and_prepare_data(symbol, period="60d")
        
        # Predict future prices
        predictions = model.predict(X)
        
        # Inverse transform to get actual prices
        predictions = scaler.inverse_transform(predictions)
        
        return predictions[-1][0]  # Return the latest prediction
