import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from datetime import datetime, timedelta
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

class OptionsPredictionError(Exception):
    """Custom exception for options prediction errors"""
    pass

class OptionsPredictionModel:
    def __init__(self, asset_manager, actuarial_calculator):
        self.asset_manager = asset_manager
        self.actuarial_calculator = actuarial_calculator
        self.model = RandomForestClassifier(random_state=42, class_weight='balanced')
        self.scaler = StandardScaler()
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
            df['Volume_Ratio'] = df['volume'] / df['Volume_MA'].replace(0, 1)  # Avoid division by zero

        result = df.dropna()
        self._feature_cache[cache_key] = result
        return result

    def generate_option_signals(self, df, threshold=0.02):
        signals = pd.Series(0, index=df.index)
        price_changes = df['close'].pct_change()
        price_trend = df['close'] > df['MA20']
        high_volatility = df['Volatility'] > df['Volatility'].rolling(window=20).mean()
        risk_appetite = df['sharpe'] > 0
        volume_confirmation = df['Volume_Ratio'] > 1.0 if 'Volume_Ratio' in df.columns else pd.Series(True, index=df.index)

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
        train_size = int(len(df) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Hyperparameter tuning
        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
        grid_search = GridSearchCV(self.model, param_grid, cv=TimeSeriesSplit(n_splits=3), scoring='accuracy')
        grid_search.fit(self.scaler.fit_transform(X_train), y_train)

        self.model = grid_search.best_estimator_
        score = self.model.score(self.scaler.transform(X_test), y_test)
        logger.info(f"Model trained for {symbol} with accuracy: {score:.2f}")
        return score

    def predict(self, symbol):
        if not hasattr(self.model, 'n_features_in_'):
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

        if confidence >= 0.4:  # Configurable threshold
            current_price = float(df['close'].iloc[-1])
            expiration_date = datetime.now() + timedelta(days=30)
            volatility_factor = float(df['Volatility'].iloc[-1] / df['Volatility'].mean())
            strike_price = current_price * (1.05 if signal == 1 else 0.95) * volatility_factor
            option_type = 'call' if signal == 1 else 'put'

            return {
                'signal': signal,
                'probability': confidence,
                'strike_price': strike_price,
                'expiration_date': expiration_date,
                'option_type': option_type,
                'metrics': {
                    'volatility': float(df['Volatility'].iloc[-1]),
                    'sharpe': float(df['sharpe'].iloc[-1]),
                    'var': float(df['var'].iloc[-1])
                }
            }

        return {'signal': 0, 'probability': confidence, 'message': 'No option signal at this time'}

    def _calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, 1)  # Avoid division by zero
        return 100 - (100 / (1 + rs))

    def get_recommendation(self, prediction):
        confidence = prediction.get('probability', 0)
        signal = prediction.get('signal', 0)
        action = 'CALL' if signal == 1 else 'PUT' if signal == -1 else 'ATTENDRE'
        description = {
            'CALL': "Acheter une option d'achat (CALL)",
            'PUT': "Acheter une option de vente (PUT)",
            'ATTENDRE': "Pas de signal d'achat pour le moment"
        }.get(action, 'Unable to provide a recommendation')

        return {
            'action': action,
            'description': description,
            'strike_price': prediction.get('strike_price'),
            'premium': prediction.get('premium', 'N/A'),  # Add premium calculation if needed
            'expiration': prediction.get('expiration_date'),
            'confidence': confidence,
            'risk_level': 'High' if signal == -1 else 'Medium' if signal == 1 else 'Low'
        }