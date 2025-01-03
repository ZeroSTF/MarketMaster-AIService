import unittest
from app.core.data_fetcher import YFinanceDataFetcher

class TestDataFetcher(unittest.TestCase):
    def setUp(self):
        self.fetcher = YFinanceDataFetcher()

    def test_fetch_valid_symbol(self):
        data = self.fetcher.fetch_data("AAPL")
        self.assertIsNotNone(data)
        self.assertEqual(data['symbol'], "AAPL")
        self.assertIsNotNone(data['currentPrice'])

    def test_fetch_invalid_symbol(self):
        with self.assertRaises(Exception):
            self.fetcher.fetch_data("INVALID_SYMBOL_XXX")