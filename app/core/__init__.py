"""
Core package initialization.
"""
from .asset_manager import AssetManager
from .data_fetcher import YFinanceDataFetcher

__all__ = ['AssetManager', 'YFinanceDataFetcher']