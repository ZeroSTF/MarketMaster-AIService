import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from datetime import datetime, timedelta
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

class OptionsPredictionError(Exception):
    """Custom exception for options prediction errors"""
    pass

class OptionsPredictionModel:
    def __init__(self, asset_manager, actuarial_calculator):
        """
        Initialize the model with asset manager and actuarial calculator
        """
        self.asset_manager = asset_manager
        self.actuarial_calculator = actuarial_calculator
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'  # Handle imbalanced classes
        )
        self.scaler = StandardScaler()
        self._feature_cache = {}
        
    def _validate_data(self, df):
        """Validate the input data"""
        if df is None or df.empty:
            raise OptionsPredictionError("No data available for analysis")
        if df['close'].isna().any():
            raise OptionsPredictionError("Missing closing prices in data")
        return True

    @lru_cache(maxsize=100)
    def _get_actuarial_metrics(self, symbol):
        """Cache actuarial metrics calculations"""
        try:
            return {
                'volatility': self.actuarial_calculator.calculate_annualized_volatility(symbol),
                'sharpe': self.actuarial_calculator.calculate_sharpe_ratio(symbol),
                'var': self.actuarial_calculator.calculate_var(symbol),
                'cvar': self.actuarial_calculator.calculate_cvar(symbol)
            }
        except Exception as e:
            logger.error(f"Error calculating actuarial metrics for {symbol}: {e}")
            raise OptionsPredictionError(f"Failed to calculate risk metrics: {str(e)}")

    def prepare_features(self, symbol, lookback_days=30):
        """
        Prepare features with improved error handling and caching
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_{lookback_days}"
            if cache_key in self._feature_cache:
                return self._feature_cache[cache_key]

            # Get historical data
            hist_data = self.asset_manager.get_historical_data(symbol, timeframe='1D', period='1mo')
            if not hist_data:
                raise OptionsPredictionError(f"No historical data available for {symbol}")

            df = pd.DataFrame(hist_data)
            self._validate_data(df)
            
            # Technical indicators
            df['RSI'] = self._calculate_rsi(df['close'])
            df['MA20'] = df['close'].rolling(window=20).mean()
            df['Volatility'] = df['close'].rolling(window=20).std()
            
            # Add momentum indicators
            df['ROC'] = df['close'].pct_change(periods=5)  # Rate of Change
            df['MOM'] = df['close'] - df['close'].shift(5)  # Momentum
            
            # Actuarial metrics
            metrics = self._get_actuarial_metrics(symbol)
            for key, value in metrics.items():
                df[key] = value
            
            # Volume analysis
            if 'volume' in df.columns:
                df['Volume_MA'] = df['volume'].rolling(window=10).mean()
                df['Volume_Ratio'] = df['volume'] / df['Volume_MA']

            result = df.dropna()
            self._feature_cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Error preparing features for {symbol}: {e}")
            raise OptionsPredictionError(f"Feature preparation failed: {str(e)}")

    def generate_option_signals(self, df, threshold=0.02):
        """
        Generate option signals with improved logic
        """
        try:
            signals = pd.Series(0, index=df.index)
            
            # Price momentum
            price_changes = df['close'].pct_change()
            price_trend = df['close'] > df['MA20']
            
            # Risk metrics
            high_volatility = df['Volatility'] > df['Volatility'].rolling(window=20).mean()
            risk_appetite = df['sharpe'] > 0
            
            # Volume confirmation (if available)
            volume_confirmation = pd.Series(True, index=df.index)
            if 'Volume_Ratio' in df.columns:
                volume_confirmation = df['Volume_Ratio'] > 1.0

            # Generate signals
            bullish_condition = (
                (price_changes > threshold) &
                price_trend &
                high_volatility &
                risk_appetite &
                volume_confirmation
            )
            
            bearish_condition = (
                (price_changes < -threshold) &
                ~price_trend &
                high_volatility &
                ~risk_appetite &
                volume_confirmation
            )
            
            signals[bullish_condition] = 1  # Call signals
            signals[bearish_condition] = -1  # Put signals
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            raise OptionsPredictionError(f"Signal generation failed: {str(e)}")

    def train(self, symbol):
        """
        Train the model with improved validation
        """
        try:
            df = self.prepare_features(symbol)
            if len(df) < 30:  # Minimum required data points
                raise OptionsPredictionError("Insufficient data for training")
                
            y = self.generate_option_signals(df)
            
            features = ['RSI', 'MA20', 'Volatility', 'volatility', 
                       'sharpe', 'var', 'cvar', 'ROC', 'MOM']
            
            if 'Volume_Ratio' in df.columns:
                features.append('Volume_Ratio')
                
            X = df[features]
            
            # Split data chronologically
            train_size = int(len(df) * 0.8)
            X_train = X[:train_size]
            y_train = y[:train_size]
            
            X_scaled = self.scaler.fit_transform(X_train)
            self.model.fit(X_scaled, y_train)
            
            # Validate model performance
            X_test = X[train_size:]
            y_test = y[train_size:]
            X_test_scaled = self.scaler.transform(X_test)
            score = self.model.score(X_test_scaled, y_test)
            
            logger.info(f"Model trained for {symbol} with accuracy: {score:.2f}")
            return score
            
        except Exception as e:
            logger.error(f"Training error for {symbol}: {e}")
            raise OptionsPredictionError(f"Model training failed: {str(e)}")

    def predict(self, symbol):
        try:
            if not hasattr(self.model, 'n_features_in_'):
                self.train(symbol)
                
            df = self.prepare_features(symbol)
            features = ['RSI', 'MA20', 'Volatility', 'volatility', 
                    'sharpe', 'var', 'cvar', 'ROC', 'MOM']
                    
            # Only add Volume_Ratio if it exists in the dataframe
            if 'Volume_Ratio' in df.columns:
                features.append('Volume_Ratio')
            else:
                # If model was trained with Volume_Ratio but current data doesn't have it
                if 'Volume_Ratio' in self.model.feature_names_in_:
                    df['Volume_Ratio'] = 1.0  # Default value
                    features.append('Volume_Ratio')

            X = df[features].iloc[-1:]
            X_scaled = self.scaler.transform(X)

            # Get probabilities first
            probabilities = self.model.predict_proba(X_scaled)[0]
            max_prob_idx = np.argmax(probabilities)
            confidence = float(probabilities[max_prob_idx])
            
            # Map probability index to signal (-1, 0, 1)
            signal_mapping = {0: -1, 1: 0, 2: 1}  # Adjust based on your classes
            signal = signal_mapping[max_prob_idx]

            # Lower threshold for testing
            confidence_threshold = 0.4  # Reduced from 0.65
            
            if confidence >= confidence_threshold:
                current_price = float(df['close'].iloc[-1])
                expiration_date = datetime.now() + timedelta(days=30)
                
                # Rest of your signal logic
                if signal != 0:
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

            return {
                'signal': 0,
                'probability': confidence,
                'message': 'No option signal at this time'
            }

        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {e}")
            raise OptionsPredictionError(f"Prediction failed: {str(e)}")
    def _calculate_rsi(self, prices, period=14):
        """
        Calculate RSI with error handling
        """
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logger.error(f"RSI calculation error: {e}")
            raise OptionsPredictionError(f"RSI calculation failed: {str(e)}")
    def get_recommendation(self, prediction):
        try:
            confidence = prediction.get('probability', 0)
            signal = prediction.get('signal', 0)
            
            if signal == 1:
                return {
                    'action': 'CALL',
                    'description': "Acheter une option d'achat (CALL)",
                    'strike_price': prediction.get('strike_price'),
                    'premium': prediction.get('suggested_premium'),
                    'expiration': prediction.get('expiration_date'),
                    'confidence': confidence
                }
            elif signal == -1:
                return {
                    'action': 'PUT',
                    'description': "Acheter une option de vente (PUT)",
                    'strike_price': prediction.get('strike_price'),
                    'premium': prediction.get('suggested_premium'),
                    'expiration': prediction.get('expiration_date'),
                    'confidence': confidence
                }
            else:
                return {
                    'action': 'ATTENDRE',
                    'description': "Pas de signal d'achat pour le moment",
                    'confidence': confidence
                }
        except Exception as e:
            return {
                'action': 'UNKNOWN',
                'description': 'Unable to provide a recommendation',
                'confidence': 0
            }