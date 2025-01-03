import numpy as np
import pandas as pd
from scipy.stats import norm
import threading
from .asset_manager import AssetManager
from ..utils.logger import logger
from datetime import datetime
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
        Calculate suggested premiums for a list of symbols based on enhanced risk metrics.
        
        Parameters:
            symbols (list): List of stock symbols to calculate premiums for.

        Returns:
            dict: Dictionary of symbols with their suggested insurance premiums.
        """
        premiums = {}
        risk_margin = 0.03  # Add a fixed risk margin (3% of calculated risk)
        admin_fee = 50      # Fixed administrative fee in the same currency
        profit_margin = 0.1 # 10% profit margin on the total premium

        for symbol in symbols:
            try:
                # Fetch asset data and validate
                data = self.asset_manager.get_asset(symbol)
                if not data:
                    logger.error(f"No data found for {symbol}")
                    premiums[symbol] = None
                    continue
                
                # Fetch historical price data
                historical_prices = self.asset_manager.get_historical_data(symbol, timeframe='1D', period='1mo')
                if not historical_prices:
                    logger.error(f"No historical data found for {symbol}")
                    premiums[symbol] = None
                    continue

                # Calculate annualized volatility
                closing_prices = [price['close'] for price in historical_prices]
                if len(closing_prices) < 2:
                    logger.warning(f"Not enough data to calculate volatility for {symbol}")
                    premiums[symbol] = None
                    continue

                returns = np.log(np.array(closing_prices[1:]) / np.array(closing_prices[:-1]))
                annualized_volatility = np.sqrt(252) * np.std(returns)

                # Current price and Value at Risk (VaR)
                current_price = data['currentPrice']
                var = self.calculate_var(symbol) if current_price else 0

                # Calculate base premium using volatility and VaR
                base_premium = max(current_price * annualized_volatility * 0.02, var * 0.01)
                logger.debug(f"Base premium for {symbol}: {base_premium}")

                # Enhance premium with risk margin, admin fees, and profit margin
                total_premium = base_premium * (1 + risk_margin) + admin_fee
                total_premium *= (1 + profit_margin)

                premiums[symbol] = round(total_premium, 2)
                logger.info(f"Suggested premium for {symbol}: {premiums[symbol]}")

            except Exception as e:
                logger.error(f"Error calculating premium for {symbol}: {str(e)}")
                premiums[symbol] = None  # Handle failure gracefully

        return premiums
    def calculate_option_premium(self, symbol, strike_price, expiration_date, option_type):
        """
        Calculate suggested option premium based on Black-Scholes model.

        Parameters:
            symbol (str): Stock symbol
            strike_price (float): Strike price of the option
            expiration_date (datetime): Expiration date of the option
            option_type (str): 'call' or 'put'

        Returns:
            float: Suggested option premium
        """
        try:
            print(f"Calculating premium for symbol: {symbol}, strike_price: {strike_price}, expiration_date: {expiration_date}, option_type: {option_type}")

            # Validate inputs
            if option_type.lower() not in ['call', 'put']:
                logger.error(f"Invalid option type: {option_type}. Must be 'call' or 'put'.")
                return None

            # Fetch asset data
            data = self.asset_manager.get_asset(symbol)
            logger.debug(f"Data for {symbol}: {data}")
            if not data:
                self.asset_manager.update_assets()
                data = self.asset_manager.get_asset(symbol)
                if data is None:
                    logger.error(f"No data found for {symbol} after update attempt")
                    return None
            # Calculate time to expiration in years
            current_date = datetime.now()
            time_to_expiration = (expiration_date - current_date).days / 365.25

            # Key asset parameters
            current_price = data.get('currentPrice')
            if current_price is None:
                logger.error(f"No current price found for {symbol}")
                return None

            # Calculate volatility
            historical_prices = self.asset_manager.get_historical_data(symbol, timeframe='1D', period='1mo')
            if not historical_prices or len(historical_prices) < 2:
                logger.error(f"Insufficient historical data for {symbol}")
                return None

            closing_prices = [price['close'] for price in historical_prices]
            returns = np.log(np.array(closing_prices[1:]) / np.array(closing_prices[:-1]))
            annualized_volatility = np.sqrt(252) * np.std(returns)

            # Risk-free rate (assumption - can be updated with actual market rates)
            risk_free_rate = 0.02

            # Black-Scholes-like premium calculation
            d1 = (np.log(current_price / strike_price) + (risk_free_rate + 0.5 * annualized_volatility**2) * time_to_expiration) / (annualized_volatility * np.sqrt(time_to_expiration))
            d2 = d1 - annualized_volatility * np.sqrt(time_to_expiration)

            # Option type specific calculations
            if option_type.lower() == 'call':
                base_premium = (
                    current_price * norm.cdf(d1) - 
                    strike_price * np.exp(-risk_free_rate * time_to_expiration) * norm.cdf(d2)
                )
            else:  # Put option
                base_premium = (
                    strike_price * np.exp(-risk_free_rate * time_to_expiration) * norm.cdf(-d2) - 
                    current_price * norm.cdf(-d1)
                )

            # Enhance premium with administrative fee and profit margin
            admin_fee = 10  # Fixed administrative fee
            profit_margin = 0.15  # 15% profit margin

            # Total premium calculation
            final_premium = base_premium * (1 + profit_margin) + admin_fee
            total_premium_for_100_contracts = final_premium * 100
            # Log and return
            logger.info(f"Option Premium for {symbol} {option_type} (Strike: {strike_price}): {total_premium_for_100_contracts:.2f}")
            return round(total_premium_for_100_contracts, 2)

        except Exception as e:
            logger.error(f"Error calculating option premium for {symbol}: {str(e)}")
            return None
