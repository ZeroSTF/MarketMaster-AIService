import yfinance as yf
from datetime import datetime
import time
from ..utils.logger import logger
from ..config.settings import Config

class YFinanceDataFetcher:
    def __init__(self, max_retries=Config.MAX_RETRIES):
        self.max_retries = max_retries

    def fetch_data(self, symbol):
        """Fetch and process data for a given symbol."""
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Fetching data for {symbol}, attempt {attempt + 1}")
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period='1d')
                
                if not info:
                    raise ValueError("No data returned from yfinance")
                    
                return self._process_data(symbol, info, hist)
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(Config.RETRY_DELAY)

    def fetch_historical_data(self, symbol, timeframe='1D', period='1mo'):
        """Fetch historical OHLCV data"""
        try:
            ticker = yf.Ticker(symbol)
            # Convert timeframe to yfinance interval format
            interval_map = {
                '1': '1m',
                '5': '5m',
                '15': '15m',
                '30': '30m',
                '60': '1h',
                '240': '4h',
                'D': '1d',
                'W': '1wk',
                'M': '1mo'
            }
            interval = interval_map.get(timeframe, '1d')
            
            # Adjust period based on interval to get enough data
            period_map = {
                '1m': '7d',
                '5m': '7d',
                '15m': '7d',
                '30m': '7d',
                '1h': '1mo',
                '4h': '1mo',
                '1d': '3mo',
                '1wk': '1y',
                '1mo': '2y'
            }
            hist_period = period_map.get(interval, '1mo')
            
            hist = ticker.history(period=hist_period, interval=interval)
            
            return [{
                'time': int(date.timestamp()),
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close']),
                'volume': float(row['Volume'])
            } for date, row in hist.iterrows()]
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            raise

    def _process_data(self, symbol, info, hist):
        """Process raw YFinance data into standardized format."""
        try:
            # Extract basic price data
            open_price = info.get('open')
            day_high = info.get('dayHigh')
            day_low = info.get('dayLow')
            current_price = info.get('currentPrice')
            volume = info.get('volume')
            previous_close = info.get('previousClose')
            
            # Calculate price changes from historical data if available
            if not hist.empty:
                price_change = hist['Close'].iloc[-1] - hist['Open'].iloc[0]
                price_change_percent = ((hist['Close'].iloc[-1] / hist['Open'].iloc[0]) - 1) * 100
            else:
                price_change = 0.0
                price_change_percent = 0.0

            # Extract additional market data
            market_cap = info.get('marketCap')
            pe_ratio = info.get('trailingPE')
            dividend_yield = info.get('dividendYield')
            dividend_yield_percent = (dividend_yield * 100) if dividend_yield is not None else None
            beta = info.get('beta')
            year_high = info.get('fiftyTwoWeekHigh')
            year_low = info.get('fiftyTwoWeekLow')
            average_volume = info.get('averageVolume')
            sector = info.get('sector')

            # Fallback to historical data if current price is missing
            if current_price is None and not hist.empty:
                current_price = hist['Close'].iloc[-1]
                logger.warning(f"Using historical data for current price of {symbol}")

            if current_price is None:
                raise ValueError(f"No price data available for {symbol}")

            # Construct response dictionary
            return {
                'symbol': symbol,
                'openPrice': float(open_price) if open_price is not None else None,
                'dayHigh': float(day_high) if day_high is not None else None,
                'dayLow': float(day_low) if day_low is not None else None,
                'currentPrice': float(current_price),
                'volume': int(volume) if volume is not None else None,
                'previousClose': float(previous_close) if previous_close is not None else None,
                'priceChange': float(price_change),
                'priceChangePercent': float(price_change_percent),
                'marketCap': int(market_cap) if market_cap is not None else None,
                'peRatio': float(pe_ratio) if pe_ratio is not None else None,
                'dividendYieldPercent': float(dividend_yield_percent) if dividend_yield_percent is not None else None,
                'beta': float(beta) if beta is not None else None,
                'yearHigh': float(year_high) if year_high is not None else None,
                'yearLow': float(year_low) if year_low is not None else None,
                'averageVolume': int(average_volume) if average_volume is not None else None,
                'sector': sector,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error processing data for {symbol}: {str(e)}")
            raise ValueError(f"Failed to process data for {symbol}: {str(e)}")