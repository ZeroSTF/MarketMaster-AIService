import unittest
from models import market_prediction, portfolio_optimization, sentiment_analysis, trading_strategy

class TestModels(unittest.TestCase):
    
    def test_market_prediction(self):
        result = market_prediction.predict('AAPL', 5)
        self.assertIn('symbol', result)
        self.assertIn('predictions', result)
        self.assertEqual(len(result['predictions']), 5)
    
    def test_portfolio_optimization(self):
        result = portfolio_optimization.optimize(['AAPL', 'GOOGL', 'MSFT'])
        self.assertIn('weights', result)
        self.assertIn('expected_return', result)
    
    def test_sentiment_analysis(self):
        result = sentiment_analysis.analyze("The stock market is performing well today.")
        self.assertIn('compound', result)
        self.assertIn('neg', result)
        self.assertIn('neu', result)
        self.assertIn('pos', result)
    
    def test_trading_strategy(self):
        result = trading_strategy.optimize('AAPL', 'moving_average')
        self.assertIn('symbol', result)
        self.assertIn('strategy', result)
        self.assertIn('signals', result)

if __name__ == '__main__':
    unittest.main()