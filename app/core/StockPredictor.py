import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import logging
from datetime import datetime, timedelta
import os
import json

class StockPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.models = {}
    def load_model(self, symbol):
        """
        Load a pre-trained model for a given symbol
        
        :param symbol: Stock symbol
        :return: Loaded Keras model
        """
        model_path = os.path.join('models', f'{symbol}_model.h5')
        if not os.path.exists(model_path):
            return None
        
        return load_model(model_path)
    def fetch_data(self, symbol, days=365):
        """
        Fetch historical stock data for a given symbol
        
        :param symbol: Stock symbol to fetch data for
        :param days: Number of historical days to fetch
        :return: DataFrame with stock data or None
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            data = yf.download(symbol, start=start_date, end=end_date)
            
            if len(data) < 60:
                raise ValueError(f"Not enough data points for {symbol}. Found: {len(data)}")
            
            return data
        except Exception as e:
            logging.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def prepare_data(self, data):
        """
        Prepare data for LSTM model
        
        :param data: DataFrame with stock data
        :return: Prepared training data
        """
        # Use only the 'Close' prices for prediction
        closing_prices = data['Close'].values
        reshaped_data = closing_prices.reshape(-1, 1)

        # Normalize the data
        scaled_data = self.scaler.fit_transform(reshaped_data)

        # Prepare training data
        training_data_len = int(np.ceil(0.8 * len(scaled_data)))
        train_data = scaled_data[0:int(training_data_len), :]
        x_train, y_train = [], []

        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        return x_train, y_train

    def create_model(self, x_train):
        """
        Create LSTM model architecture
        
        :param x_train: Training input data
        :return: Compiled Keras model
        """
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train_and_save_model(self, symbol):
        """
        Train and save model for a specific stock symbol
        
        :param symbol: Stock symbol to train model for
        :return: Dictionary with prediction results
        """
        try:
            # Fetch and prepare data
            data = self.fetch_data(symbol)
            if data is None or len(data) < 60:
                return {'error': 'Insufficient data points. Need at least 60 points.'}
            
            # Prepare training data
            x_train, y_train = self.prepare_data(data)
            
            # Create and train model
            model = self.create_model(x_train)
            model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=0)
            
            # Save the model
            model_dir = 'models'
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f'{symbol}_model.h5')
            model.save(model_path)
            
            # Make predictions
            predictions = self.make_predictions(data, model)
            predictions['model_path'] = model_path
            
            return predictions
        
        except Exception as e:
            logging.error(f"Error training model for {symbol}: {str(e)}")
            return {'error': str(e)}

    def make_predictions(self, data, model=None, symbol=None, future_days=14):
        """
        Make multi-day price predictions with corresponding dates.
        
        :param data: Historical stock data
        :param model: Keras model (optional)
        :param symbol: Stock symbol (optional)
        :param future_days: Number of days to predict into the future
        :return: Dictionary with prediction results
        """
        try:
            # Load model if not provided
            if model is None and symbol:
                model_path = os.path.join('models', f'{symbol}_model.h5')
                if not os.path.exists(model_path):
                    raise ValueError(f"No model found for {symbol}")
                model = load_model(model_path)
            elif model is None:
                raise ValueError("No model provided")
            
            # Prepare the test data
            last_60_days = data['Close'][-60:].values
            last_60_days = last_60_days.reshape(-1, 1)
            scaled_last_60_days = self.scaler.transform(last_60_days)

            # Predict prices for the future_days
            predictions = []
            dates = []
            input_sequence = scaled_last_60_days

            # Start predicting from the next day
            last_date = data.index[-1]
            for i in range(future_days):
                # Prepare input for prediction
                x_test = np.array([input_sequence])
                x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

                # Predict price
                predicted_price = model.predict(x_test)
                predicted_price_unscaled = self.scaler.inverse_transform(predicted_price)
                predictions.append(float(predicted_price_unscaled[0, 0]))

                # Update the sequence with the predicted value
                new_input = np.append(input_sequence[1:], predicted_price, axis=0)
                input_sequence = new_input

                # Add corresponding date
                next_date = last_date + timedelta(days=i + 1)
                dates.append(next_date.strftime('%Y-%m-%d'))

            # Get current price
            current_price = data['Close'].values[-1]

            return {
                'current_price': float(current_price),
                'predicted_prices': [{'date': dates[i], 'price': predictions[i]} for i in range(future_days)],
                'predicted_change': float(((predictions[-1] - current_price) / current_price) * 100)
            }

        except Exception as e:
            logging.error(f"Error making predictions: {str(e)}")
            return {'error': str(e)}

