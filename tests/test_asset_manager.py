import unittest
from unittest.mock import Mock
from app.core.asset_manager import AssetManager

class TestAssetManager(unittest.TestCase):
    def setUp(self):
        self.mock_socketio = Mock()
        self.mock_data_fetcher = Mock()
        self.asset_manager = AssetManager(self.mock_data_fetcher, self.mock_socketio)

    def test_register_assets(self):
        # Mock the fetch_data response
        self.mock_data_fetcher.fetch_data.return_value = {
            'symbol': 'AAPL',
            'currentPrice': 150.0
        }

        result = self.asset_manager.register_assets(['AAPL'])
        self.assertIn('AAPL', result['successful'])
        self.assertEqual(len(result['failed']), 0)

    def test_get_all_assets(self):
        # Add some test data
        self.mock_data_fetcher.fetch_data.return_value = {
            'symbol': 'AAPL',
            'currentPrice': 150.0
        }
        self.asset_manager.register_assets(['AAPL'])
        
        assets = self.asset_manager.get_all_assets()
        self.assertEqual(len(assets), 1)
        self.assertEqual(assets[0]['symbol'], 'AAPL')