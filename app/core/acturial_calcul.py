import numpy as np
import pandas as pd
from scipy.stats import norm
import threading
from .asset_manager import AssetManager
from ..utils.logger import logger

class ActuarialCalculator:
    def __init__(self, asset_manager):
        """
        Initialize the calculator with an AssetManager instance.
        """
        self.asset_manager = asset_manager
        self._lock = threading.Lock()

    def calculate_annualized_volatility(self, symbol):
        """Calculate the annualized volatility of the stock's price for a given symbol."""
        try:
            # Fetch historical data
            data = self.asset_manager.get_historical_data(symbol, timeframe='1D', period='1mo')
            logger.debug(f"Fetched data for {symbol}: {data}")
            
            # Extract closing prices
            historical_prices = [entry['close'] for entry in data]
            logger.debug(f"Close prices for {symbol}: {historical_prices}")
            
            # Check if we have sufficient data
            if len(historical_prices) < 2:
                logger.error(f"Insufficient historical prices for {symbol}")
                return None

            # Calculate daily logarithmic returns
            returns = np.log(np.array(historical_prices[1:]) / np.array(historical_prices[:-1]))
            logger.debug(f"Daily logarithmic returns for {symbol}: {returns}")

            # Calculate annualized volatility
            annualized_volatility = np.sqrt(252) * np.std(returns)
            logger.debug(f"Annualized Volatility for {symbol}: {annualized_volatility}")
            
            return annualized_volatility
        except Exception as e:
            logger.error(f"Error calculating volatility for {symbol}: {str(e)}")
            return None

    def calculate_sharpe_ratio(self, symbol, risk_free_rate=0.02):
        """Calculate the Sharpe ratio of the stock for a given symbol."""
        try:
            data = self.asset_manager.get_asset(symbol)
            if not data:
                logger.error(f"No data found for {symbol} in AssetManager")
                return None

            if 'priceChangePercent' not in data:
                logger.error(f"No price change percent found for {symbol}")
                return None

            annualized_return = data['priceChangePercent'] / 100
            annualized_volatility = self.calculate_annualized_volatility(symbol)
            if annualized_volatility is None:
                return None

            sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
            logger.debug(f"Sharpe Ratio for {symbol}: {sharpe_ratio}")
            return sharpe_ratio
        except Exception as e:
            logger.error(f"Error calculating Sharpe Ratio for {symbol}: {str(e)}")
            return None

    def calculate_var(self, symbol, confidence_level=0.95):
        """Calculate the Value at Risk (VaR) of the stock for a given symbol."""
        try:
            data = self.asset_manager.get_asset(symbol)
            if not data:
                logger.error(f"No data found for {symbol} in AssetManager")
                return None

            annualized_volatility = self.calculate_annualized_volatility(symbol)
            if annualized_volatility is None or 'currentPrice' not in data:
                logger.error(f"Missing data for VaR calculation for {symbol}")
                return None

            current_price = data['currentPrice']
            z_score = norm.ppf(1 - confidence_level)
            var = current_price * (1 - np.exp(z_score * annualized_volatility * np.sqrt(1 / 252)))
            logger.debug(f"VaR for {symbol}: {var}")
            return var
        except Exception as e:
            logger.error(f"Error calculating VaR for {symbol}: {str(e)}")
            return None

    def calculate_cvar(self, symbol, confidence_level=0.95):
        """Calculate the Conditional Value at Risk (CVaR) of the stock for a given symbol."""
        try:
            var = self.calculate_var(symbol, confidence_level)
            if var is None:
                return None

            annualized_volatility = self.calculate_annualized_volatility(symbol)
            data = self.asset_manager.get_asset(symbol)
            if not data or 'currentPrice' not in data:
                logger.error(f"Missing current price for CVaR calculation for {symbol}")
                return None

            current_price = data['currentPrice']
            z_score = norm.ppf(1 - confidence_level)
            cvar = (
                current_price
                * (
                    1
                    - np.exp(z_score * annualized_volatility * np.sqrt(1 / 252))
                    - (1 - confidence_level) * z_score * annualized_volatility * np.sqrt(1 / 252)
                )
            )
            logger.debug(f"CVaR for {symbol}: {cvar}")
            return cvar
        except Exception as e:
            logger.error(f"Error calculating CVaR for {symbol}: {str(e)}")
            return None

    def calculate_beta_adjusted_return(self, symbol, market_return):
        """Calculate the beta-adjusted return of the stock for a given symbol."""
        try:
            data = self.asset_manager.get_asset(symbol)
            if not data:
                logger.error(f"No data found for {symbol} in AssetManager")
                return None

            if 'beta' not in data:
                logger.error(f"No beta value found for {symbol}")
                return None

            beta = data['beta']
            price_change_percent = data['priceChangePercent'] / 100
            return price_change_percent - beta * (market_return - 0.02)
        except Exception as e:
            logger.error(f"Error calculating beta-adjusted return for {symbol}: {str(e)}")
            return None


    def calculate_suggested_premiums(self, symbols):
        """
        Calculate suggested premiums for a list of symbols based on their risk metrics.

        Parameters:
            symbols (list): List of stock symbols to calculate premiums for.

        Returns:
            dict: Dictionary of symbols with their suggested insurance premiums.
        """
        premiums = {}
        for symbol in symbols:
            try:
                # Fetch the asset data for the symbol
                data = self.asset_manager.get_asset(symbol)
                logger.debug(f"Fetched asset data for {symbol}: {data}")
                
                if not data:
                    logger.error(f"No data found for {symbol} in AssetManager")
                    premiums[symbol] = None
                    continue

                # Fetch historical price data for the symbol
                historical_prices = self.asset_manager.get_historical_data(symbol, timeframe='1D', period='1mo')
                logger.debug(f"Fetched historical data for {symbol}: {historical_prices}")
                
                if historical_prices:
                    # Extract closing prices from the historical data
                    closing_prices = [price['close'] for price in historical_prices]
                    logger.debug(f"Closing prices for {symbol}: {closing_prices}")
                    
                    # Calculate returns and annualized volatility
                    returns = np.log(np.array(closing_prices[1:]) / np.array(closing_prices[:-1]))
                    annualized_volatility = np.sqrt(252) * np.std(returns)
                    logger.debug(f"Annualized volatility for {symbol}: {annualized_volatility}")
                else:
                    logger.error(f"Failed to fetch historical data for {symbol}")
                    premiums[symbol] = None
                    continue

                current_price = data['currentPrice']
                logger.debug(f"Current price for {symbol}: {current_price}")
                
                # Calculate Value at Risk (VaR) for the symbol (falling back to 0 if no price)
                var = self.calculate_var(symbol) if current_price else 0
                logger.debug(f"Calculated VaR for {symbol}: {var}")

                # Premium calculation: based on VaR and volatility
                premium = max(current_price * annualized_volatility * 0.02, var * 0.01)
                premiums[symbol] = premium
                logger.debug(f"Suggested premium for {symbol}: {premium}")
            
            except Exception as e:
                logger.error(f"Error calculating premium for {symbol}: {str(e)}")
                premiums[symbol] = None  # None if data could not be fetched

        return premiums
