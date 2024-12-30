import requests
from ..utils.logger import logger
from ..config.settings import Config

class FinnhubNewsFetcher:
    def __init__(self, api_key):
        """Initialize the Finnhub News Fetcher."""
        self._api_key = api_key
        self.base_url = 'https://finnhub.io/api/v1/company-news'

    def fetch_news(self, symbol, start_date, end_date):
        """Fetch news for a specific stock symbol within a given date range."""
        params = {
            'symbol': symbol,
            'from': start_date,
            'to': end_date,
            'token': self._api_key
        }

        try:
            response = requests.get(self.base_url, params=params)
            
            if response.status_code == 200:
                return response.json()
            
            logger.error(f"Failed to fetch news for {symbol}: {response.status_code}")
            return None

        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return None

    def fetch_multiple_symbol_news(self, symbols, start_date, end_date):
        """Fetch news for multiple stock symbols."""
        news_data = {}
        
        for symbol in symbols:
            news = self.fetch_news(symbol, start_date, end_date)
            if news:
                news_data[symbol] = news
        
        return news_data