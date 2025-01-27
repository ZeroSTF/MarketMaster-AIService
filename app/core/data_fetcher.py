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

    def fetch_historical_data(self, symbol, timeframe='D', period='1d', include_statistics=False):
        """
        Fetch historical OHLCV data with optional extended statistics.
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Time interval ('1', '5', '15', '30', '60', '240', 'D', 'W', 'M')
            period (str): Time period to fetch data for
            include_statistics (bool): Whether to include additional statistics in the response
            
        Returns:
            Union[list, dict]: List of OHLCV data points or dictionary with extended statistics
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Convert timeframe to yfinance interval format
            interval_map = {
                '1': '1m',
                '2': '2m',
                '5': '5m',
                '15': '15m',
                '30': '30m',
                '60': '1h',
                '90': '90m',
                '240': '4h',
                'D': '1d',
                'W': '1wk',
                'M': '1mo'
            }
            interval = interval_map.get(timeframe, '1d')
            
            # Adjust period based on interval to get enough data
            period_map = {
                '1m': '5d',
                '5m': '5d',
                '15m': '5d',
                '30m': '5d',
                '1h': '2y',
                '4h': 'max',
                '1d': 'max',
                '1wk': 'max',
                '1mo': 'max'
            }
            hist_period = period_map.get(interval, '1d')

            if include_statistics:
                hist_period = period_map.get('1h')

            hist = ticker.history(period=hist_period, interval=interval)
            
            if hist.empty:
                raise ValueError(f"No historical data found for {symbol}")
            
            if not include_statistics:
                # Return simple OHLCV format with 'time' field
                return [{
                    'time': int(date.timestamp()),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': float(row['Volume'])
                } for date, row in hist.iterrows()]
            else:
                # Create detailed format with 'date' field
                historical_data = [{
                    'date': date.strftime('%Y-%m-%d %H:%M:%S'),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': float(row['Volume']),
                    'dividends': float(row['Dividends']),
                    'stock_splits': float(row['Stock Splits'])
                } for date, row in hist.iterrows()]

                # Calculate changes for each point
                for i in range(len(historical_data)):
                    if i > 0:
                        prev_close = historical_data[i-1]['close']
                        historical_data[i]['change'] = float(historical_data[i]['close'] - prev_close)
                        historical_data[i]['change_percent'] = float((historical_data[i]['close'] / prev_close - 1) * 100)
                    else:
                        historical_data[i]['change'] = 0.0
                        historical_data[i]['change_percent'] = 0.0

                return {
                    'symbol': symbol,
                    'period': period,
                    'interval': interval,
                    'data': historical_data,
                    'statistics': {
                        'total_points': len(historical_data),
                        'first_date': historical_data[0]['date'],
                        'last_date': historical_data[-1]['date'],
                        'highest_price': max(point['high'] for point in historical_data),
                        'lowest_price': min(point['low'] for point in historical_data),
                        'total_volume': sum(point['volume'] for point in historical_data),
                        'avg_daily_volume': sum(point['volume'] for point in historical_data) / len(historical_data),
                        'price_change': historical_data[-1]['close'] - historical_data[0]['open'],
                        'price_change_percent': ((historical_data[-1]['close'] / historical_data[0]['open']) - 1) * 100
                    },
                    'timestamp': datetime.now().isoformat()
                }
            
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

    def get_game_data(self, symbols, start_date, end_date):
            """Fetch historical data for a list of symbols."""
            historical_data = {}
            logger.info(f"Fetching historical data for symbols: {symbols}")
            for symbol in symbols:
                try:
                    data = self.fetch_data_between(symbol, start_date, end_date)
                    if data:
                        historical_data[symbol] = data
                    else:
                        logger.warning(f"No data for symbol {symbol}")
                except Exception as e:
                    logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            logger.info(f"Historical data fetched: {historical_data}")
            return historical_data

    def fetch_data_between(self, symbol, start_date, end_date, interval="15m"):
       ticker = yf.Ticker(symbol)
       logger.info(f"Fetching data for {symbol} from {start_date} to {end_date} with {interval} interval")  # Debugging step
    # Fetch data with specified interval
       hist = ticker.history(start=start_date, end=end_date, interval=interval)
       if hist.empty:
           logger.warning(f"No data found for {symbol} in date range {start_date} to {end_date}")
           return []
       else:
        # Convert to desired format, adding timestamp as key
        formatted_data = []
        for timestamp, row in hist.iterrows():
            record = {
                "Date": timestamp.isoformat(),
                "Open": row["Open"],
                "High": row["High"],
                "Low": row["Low"],
                "Close": row["Close"],
                "Volume": row["Volume"]
            }
            formatted_data.append(record)
        logger.info(f"Data fetched for {symbol}: {formatted_data}")  # Debugging step
        return formatted_data
