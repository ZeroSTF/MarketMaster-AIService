"""
Core package initialization.
"""
from .asset_manager import AssetManager
from .data_fetcher import YFinanceDataFetcher
from .acturial_calcul import ActuarialCalculator
__all__ = ['AssetManager', 'YFinanceDataFetcher', 'ActuarialCalculator']