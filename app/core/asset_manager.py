import threading
from ..utils.logger import logger
from datetime import datetime

class AssetManager:
    def __init__(self, data_fetcher, socketio):
        self.data_fetcher = data_fetcher
        self.socketio = socketio
        self.asset_cache = {}
        self._lock = threading.Lock()

    def register_assets(self, symbols):
        """Register new assets for tracking."""
        with self._lock:
            result = {
                'successful': [],
                'failed': [],
                'failure_reasons': {},
                'timestamp': datetime.now().isoformat()
            }
            
            if not symbols:
                result['failure_reasons']['general'] = "No symbols provided"
                return result

            for symbol in symbols:
                try:
                    if symbol not in self.asset_cache:
                        data = self.data_fetcher.fetch_data(symbol)
                        self.asset_cache[symbol] = data
                    result['successful'].append(symbol)
                    logger.info(f"Successfully registered {symbol}")
                except Exception as e:
                    result['failed'].append(symbol)
                    result['failure_reasons'][symbol] = str(e)
                    logger.error(f"Failed to register {symbol}: {str(e)}")
            
            return result

    def update_assets(self):
        """Update all registered assets."""
        with self._lock:
            for symbol in list(self.asset_cache.keys()):
                try:
                    data = self.data_fetcher.fetch_data(symbol)
                    self.asset_cache[symbol] = data
                    self.socketio.emit('asset_update', data)
                    logger.debug(f"Updated {symbol}")
                except Exception as e:
                    logger.error(f"Error updating {symbol}: {str(e)}")

    def get_all_assets(self):
        """Get current data for all registered assets."""
        with self._lock:
            return list(self.asset_cache.values())

    def get_asset(self, symbol):
        """Get current data for a specific asset."""
        with self._lock:
            return self.asset_cache.get(symbol)

    def remove_asset(self, symbol):
        """Remove an asset from tracking."""
        with self._lock:
            if symbol in self.asset_cache:
                del self.asset_cache[symbol]
                return True
            return False