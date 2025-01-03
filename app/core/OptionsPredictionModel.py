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
        """
        Predict with confidence thresholds and risk management.
        """
        try:
            if not hasattr(self.model, 'n_features_in_'):  # Check if model is trained
             self.train(symbol)
            # Prepare data for prediction
            df = self.prepare_features(symbol)
            if df is None or len(df) < 2:
                raise OptionsPredictionError("Insufficient data for prediction")

            # Define features for prediction
            features = ['RSI', 'MA20', 'Volatility', 'volatility', 
                        'sharpe', 'var', 'cvar', 'ROC', 'MOM']
            
            if 'Volume_Ratio' in df.columns:
                features.append('Volume_Ratio')
            
            # Validate required features
            missing_features = [feature for feature in features if feature not in df.columns]
            if missing_features:
                raise OptionsPredictionError(f"Missing features: {missing_features}")

            # Extract and scale features
            X = df[features].iloc[-1:]
            X_scaled = self.scaler.transform(X)

            # Make predictions
            signal = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            print(f"probilities :{probabilities}")
            if not hasattr(probabilities, '__iter__'):
             raise OptionsPredictionError("Model probabilities are not iterable.")
            # Get the maximum probability and its corresponding class
            confidence = float(np.max(probabilities))  # Convert to float to avoid numpy.float64 issues
            print(f"confidence :{confidence}")
            # Confidence threshold
            confidence_threshold = 0.65
            if confidence < confidence_threshold:
                return {
                    'signal': 0,
                    'probability': confidence,
                    'message': 'Insufficient confidence to generate signal'
                }

            # Generate signal if confidence is sufficient
            if signal != 0:
                current_price = float(df['close'].iloc[-1])  # Convert to float
                expiration_date = datetime.now() + timedelta(days=30)

                # Adjust strike price based on volatility
                volatility_factor = float(df['Volatility'].iloc[-1] / df['Volatility'].mean())  # Convert to float
                if signal == 1:  # Call option
                    strike_modifier = 1.05 * volatility_factor
                    strike_price = current_price * strike_modifier
                    option_type = 'call'
                else:  # Put option
                    strike_modifier = 0.95 / volatility_factor
                    strike_price = current_price * strike_modifier
                    option_type = 'put'

                # Calculate option premium
                premium = self.actuarial_calculator.calculate_option_premium(
                    symbol, strike_price, expiration_date, option_type
                )

                return {
                    'signal': signal,
                    'probability': confidence,
                    'suggested_premium': premium,
                    'strike_price': strike_price,
                    'expiration_date': expiration_date,
                    'option_type': option_type,
                    'metrics': {
                        'volatility': float(df['Volatility'].iloc[-1]),  # Convert to float
                        'sharpe': float(df['sharpe'].iloc[-1]),         # Convert to float
                        'var': float(df['var'].iloc[-1]),               # Convert to float
                    }
                }

            # No signal generated
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
        """Generate a recommendation based on the prediction."""
        try:
            
            probabilities = prediction.get('probability', [])
            if not probabilities or not hasattr(probabilities, '__iter__'):
                return {
                    'action': 'UNKNOWN',
                    'description': 'Unable to provide a recommendation',
                    'confidence': 0
                }

            max_confidence = max(probabilities) * 100
            if prediction['signal'] == 1:
                return {
                    'action': 'ACHETER_CALL',
                    'description': "Acheter une option d'achat (CALL)",
                    'strike_price': prediction.get('prix_strike'),
                    'premium': prediction.get('prime'),
                    'expiration': prediction.get('date_expiration'),
                    'confidence': max_confidence
                }
            elif prediction['signal'] == -1:
                return {
                    'action': 'ACHETER_PUT',
                    'description': "Acheter une option de vente (PUT)",
                    'strike_price': prediction.get('prix_strike'),
                    'premium': prediction.get('prime'),
                    'expiration': prediction.get('date_expiration'),
                    'confidence': max_confidence
                }
            else:
                return {
                    'action': 'ATTENDRE',
                    'description': "Pas de signal d'achat pour le moment",
                    'confidence': max_confidence
                }    
        except Exception as e:
            raise Exception(f"Error getting recommendation for {symbol}: {str(e)}")
        