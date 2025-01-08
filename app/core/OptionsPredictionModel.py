import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from datetime import datetime, timedelta
from functools import lru_cache
import logging
import joblib
import os

logger = logging.getLogger(__name__)

class OptionsPredictionError(Exception):
    """Custom exception for options prediction errors"""
    pass

class OptionsPredictionModel:
    def __init__(self, asset_manager, actuarial_calculator, model_path='model.pkl', scaler_path='scaler.pkl'):
        self.asset_manager = asset_manager
        self.actuarial_calculator = actuarial_calculator
        self.model_path = model_path
        self.scaler_path = scaler_path

        # Load pre-trained model and scaler if they exist
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            logger.info("Loaded pre-trained model from disk.")
        else:
            self.model = RandomForestClassifier(random_state=42, class_weight='balanced')
            logger.info("Initialized a new model.")

        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            logger.info("Loaded fitted scaler from disk.")
        else:
            self.scaler = StandardScaler()
            logger.info("Initialized a new scaler.")

        self._feature_cache = {}

    def _validate_data(self, df):
        if df is None or df.empty or df['close'].isna().any():
            raise OptionsPredictionError("Invalid or missing data")
        return True

    @lru_cache(maxsize=100)
    def _get_actuarial_metrics(self, symbol):
        try:
            return {
                'volatility': self.actuarial_calculator.calculate_annualized_volatility(symbol),
                'sharpe': self.actuarial_calculator.calculate_sharpe_ratio(symbol),
                'var': self.actuarial_calculator.calculate_var(symbol),
                'cvar': self.actuarial_calculator.calculate_cvar(symbol)
            }
        except Exception as e:
            logger.error(f"Error calculating metrics for {symbol}: {e}")
            raise OptionsPredictionError(f"Metrics calculation failed: {str(e)}")

    def prepare_features(self, symbol, lookback_days=30):
        cache_key = f"{symbol}_{lookback_days}"
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]

        hist_data = self.asset_manager.get_historical_data(symbol, timeframe='1D', period='1mo')
        if not hist_data:
            raise OptionsPredictionError(f"No historical data for {symbol}")

        df = pd.DataFrame(hist_data)
        self._validate_data(df)

        # Technical indicators
        df['RSI'] = self._calculate_rsi(df['close'])
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['Volatility'] = df['close'].rolling(window=20).std()
        df['ROC'] = df['close'].pct_change(periods=5)
        df['MOM'] = df['close'] - df['close'].shift(5)

        # Actuarial metrics
        metrics = self._get_actuarial_metrics(symbol)
        df = df.assign(**metrics)

        # Volume analysis
        if 'volume' in df.columns:
            df['Volume_MA'] = df['volume'].rolling(window=10).mean()
            df['Volume_Ratio'] = df['volume'] / df['Volume_MA'].replace(0, 1)

        result = df.dropna()
        self._feature_cache[cache_key] = result
        return result

    def generate_option_signals(self, df, threshold=0.02):  # Reduced threshold
        price_changes = df['close'].pct_change()
        price_trend = df['close'] > df['MA20']
        high_volatility = df['Volatility'] > df['Volatility'].rolling(window=20).mean()
        risk_appetite = df['sharpe'] > 0
        volume_confirmation = df['Volume_Ratio'] > 1.0 if 'Volume_Ratio' in df.columns else pd.Series(True, index=df.index)

        signals = pd.Series(0, index=df.index)
        signals[(price_changes > threshold) & price_trend & high_volatility & risk_appetite & volume_confirmation] = 1  # Call
        signals[(price_changes < -threshold) & ~price_trend & high_volatility & ~risk_appetite & volume_confirmation] = -1  # Put
        return signals
            
            

    def train(self, symbol):
        df = self.prepare_features(symbol)
        if len(df) < 30:
            raise OptionsPredictionError("Insufficient data for training")

        y = self.generate_option_signals(df)
        features = ['RSI', 'MA20', 'Volatility', 'volatility', 'sharpe', 'var', 'cvar', 'ROC', 'MOM']
        if 'Volume_Ratio' in df.columns:
            features.append('Volume_Ratio')

        X = df[features]
        mask = y != 0
        X = X[mask]
        y=y[mask]
        print(y.value_counts())
        train_size = int(len(df) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Fit the scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        logger.info("Fitted the scaler on training data.")

        # Hyperparameter tuning
        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
        grid_search = GridSearchCV(self.model, param_grid, cv=TimeSeriesSplit(n_splits=3), scoring='accuracy')
        grid_search.fit(X_train_scaled, y_train)

        self.model = grid_search.best_estimator_
        joblib.dump(self.model, self.model_path)  # Save trained model
        joblib.dump(self.scaler, self.scaler_path)  # Save fitted scaler
        logger.info("Saved the trained model and scaler to disk.")

        # Evaluate the model
        X_test_scaled = self.scaler.transform(X_test)
        score = self.model.score(X_test_scaled, y_test)
        logger.info(f"Model trained for {symbol} with accuracy: {score:.2f}")
        return score

    def _calculate_optimal_duration(self, confidence, volatility, sharpe_ratio):
        """
        Calculate optimal option duration based on market conditions
        Returns duration in days
        """
        # Base duration range (in days)
        min_duration = 7  # Minimum 1 week
        max_duration = 90  # Maximum 3 months
        
        # Adjust base duration based on confidence
        # Higher confidence = shorter duration
        confidence_factor = 1 - confidence  # Invert so higher confidence reduces duration
        
        # Adjust for volatility (normalized to 0-1 range assuming typical volatility range)
        volatility_factor = min(volatility / 50, 1)  # Cap at 1
        
        # Consider Sharpe ratio for risk adjustment
        # Negative Sharpe = longer duration to allow for recovery
        sharpe_factor = 0.5 if sharpe_ratio <= 0 else min(1.5, 1 + sharpe_ratio)
        
        # Calculate duration
        base_duration = min_duration + (max_duration - min_duration) * confidence_factor
        adjusted_duration = base_duration * (1 + volatility_factor) / sharpe_factor
        
        # Round to nearest week and ensure within bounds
        weeks = round(adjusted_duration / 7)
        final_duration = min(max(weeks * 7, min_duration), max_duration)
        
        return int(final_duration)

    def predict(self, symbol):
        if not hasattr(self.model, 'classes_') or not hasattr(self.scaler, 'mean_'):
            logger.warning("Model or scaler is not fitted. Training the model first...")
            self.train(symbol)

        df = self.prepare_features(symbol)
        features = ['RSI', 'MA20', 'Volatility', 'volatility', 'sharpe', 'var', 'cvar', 'ROC', 'MOM']
        if 'Volume_Ratio' in df.columns:
            features.append('Volume_Ratio')

        X = df[features].iloc[-1:]
        X_scaled = self.scaler.transform(X)

        probabilities = self.model.predict_proba(X_scaled)[0]
        max_prob_idx = np.argmax(probabilities)
        confidence = float(probabilities[max_prob_idx])
        signal_mapping = {0: -1, 1: 0, 2: 1}
        signal = signal_mapping[max_prob_idx]

        if confidence >= 0.4:
            current_price = float(df['close'].iloc[-1])
            volatility = float(df['Volatility'].iloc[-1])
            sharpe = float(df['sharpe'].iloc[-1])
            
            # Calculate dynamic duration
            duration_days = self._calculate_optimal_duration(
                confidence=confidence,
                volatility=volatility,
                sharpe_ratio=sharpe
            )
            expiration_date = datetime.now() + timedelta(days=duration_days)
            
            # Calculate strike price with volatility adjustment
            volatility_adjustment = min(1 + (volatility * 0.1), 1.2)  # Cap at 20% adjustment
            
            # Adjust OTM percentage based on duration
            # Longer duration = larger OTM percentage
            base_otm_pct = 0.03 + (duration_days / 365) * 0.10  # 3% + up to 10% for longer durations
            
            if signal == 1:  # Call option
                strike_price = current_price * (1 + base_otm_pct) * volatility_adjustment
            elif signal == -1:  # Put option
                strike_price = current_price * (1 - base_otm_pct) * volatility_adjustment
            else:
                strike_price = current_price
            
            if signal == 0:
                option_type = 'hold'
            elif signal == 1:
                option_type = 'call'
            else:
                option_type = 'put'


            return {
                'signal': signal,
                'probability': confidence,
                'strike_price': round(strike_price, 2),
                'current_price': current_price,
                'expiration_date': expiration_date,
                'duration_days': duration_days,
                'option_type': option_type,
                'metrics': {
                    'volatility': round(volatility, 2),
                    'sharpe': round(sharpe, 2),
                    'var': round(float(df['var'].iloc[-1]), 2),
                    'otm_percentage': round(base_otm_pct * 100, 1)
                }
            }

        return {
            'signal': 0, 
            'probability': confidence, 
            'current_price': float(df['close'].iloc[-1]),
            'message': 'No option signal at this time'
        }

    def get_recommendation(self, prediction):
        confidence = prediction.get('probability', 0)
        signal = prediction.get('signal', 0)
        action = 'CALL' if signal == 1 else 'PUT' if signal == -1 else 'HOLD'
        description = {
            'CALL': "Buy a call option",
            'PUT': "Buy a put option",
            'HOLD': "No trading signal at this time"
        }.get(action, 'Unable to provide a recommendation')

        return {
            'action': action,
            'description': description,
            'strike_price': prediction.get('strike_price'),
            'expiration': prediction.get('expiration_date'),
            'confidence': confidence,
            'risk_level': 'High' if signal == -1 else 'Medium' if signal == 1 else 'Low'
        }

    def _calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, 1)
        return 100 - (100 / (1 + rs))