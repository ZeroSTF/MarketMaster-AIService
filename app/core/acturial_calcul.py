import numpy as np
import pandas as pd
from scipy.stats import norm
import threading
from .data_fetcher import YFinanceDataFetcher
from ..utils.logger import logger

class ActuarialCalculator:
    def __init__(self):
        """Initialise le calculateur sans symbole spécifique."""
        self.data_fetcher = YFinanceDataFetcher()
        self._lock = threading.Lock()

    def calculate_annualized_volatility(self, symbol):
        """Calculate the annualized volatility of the stock's price for a given symbol."""
        data = self.data_fetcher.fetch_data(symbol)
        historical_prices = data['historicalPrices']
        returns = np.log(np.array(historical_prices[1:]) / np.array(historical_prices[:-1]))
        annualized_volatility = np.sqrt(252) * np.std(returns)
        return annualized_volatility

    def calculate_sharpe_ratio(self, symbol, risk_free_rate=0.02):
        """Calculate the Sharpe ratio of the stock for a given symbol."""
        data = self.data_fetcher.fetch_data(symbol)
        annualized_return = data['priceChangePercent'] / 100
        annualized_volatility = self.calculate_annualized_volatility(symbol)
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
        return sharpe_ratio

    def calculate_var(self, symbol, confidence_level=0.95):
        """Calculate the Value at Risk (VaR) of the stock for a given symbol."""
        data = self.data_fetcher.fetch_data(symbol)
        annualized_volatility = self.calculate_annualized_volatility(symbol)
        current_price = data['currentPrice']
        z_score = norm.ppf(1 - confidence_level)
        var = current_price * (1 - np.exp(z_score * annualized_volatility * np.sqrt(1/252)))
        return var

    def calculate_cvar(self, symbol, confidence_level=0.95):
        """Calculate the Conditional Value at Risk (CVaR) of the stock for a given symbol."""
        var = self.calculate_var(symbol, confidence_level)
        annualized_volatility = self.calculate_annualized_volatility(symbol)
        data = self.data_fetcher.fetch_data(symbol)
        current_price = data['currentPrice']
        z_score = norm.ppf(1 - confidence_level)
        cvar = current_price * (1 - np.exp(z_score * annualized_volatility * np.sqrt(1/252)) - (1 - confidence_level) * z_score * annualized_volatility * np.sqrt(1/252))
        return cvar

    def calculate_beta_adjusted_return(self, symbol, market_return):
        """Calculate the beta-adjusted return of the stock for a given symbol."""
        data = self.data_fetcher.fetch_data(symbol)
        beta = data['beta']
        return data['priceChangePercent'] / 100 - beta * (market_return - 0.02)

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
                data = self.data_fetcher.fetch_data(symbol)
                historical_prices = data['historicalPrices']
                current_price = data['currentPrice']
                
                # Calculate volatility for premium estimation
                returns = np.log(np.array(historical_prices[1:]) / np.array(historical_prices[:-1]))
                annualized_volatility = np.sqrt(252) * np.std(returns)
                
                # Premium calculation: based on VaR and volatility
                var = self.calculate_var(symbol) if current_price else 0  # Fall back if no current price
                premium = max(current_price * annualized_volatility * 0.02, var * 0.01)  # Hypothetical formula
                
                premiums[symbol] = premium
            except Exception as e:
                logger.error(f"Error calculating premium for {symbol}: {str(e)}")
                premiums[symbol] = None  # None if data could not be fetched
        
        return premiums
