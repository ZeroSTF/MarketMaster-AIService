from flask import Flask, request, jsonify
from models import market_prediction, portfolio_optimization, sentiment_analysis, trading_strategy
import os

app = Flask(__name__)

@app.route('/predict_market', methods=['POST'])
def predict_market():
    data = request.json
    prediction = market_prediction.predict(data['symbol'], data['days'])
    return jsonify(prediction)

@app.route('/optimize_portfolio', methods=['POST'])
def optimize_portfolio():
    data = request.json
    risk_free_rate = data.get('risk_free_rate', 0.01)
    optimized = portfolio_optimization.optimize(data['portfolio'], risk_free_rate)
    return jsonify(optimized)

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    data = request.json
    sentiment = sentiment_analysis.analyze(data['text'])
    return jsonify(sentiment)

@app.route('/optimize_strategy', methods=['POST'])
def optimize_strategy():
    data = request.json
    strategy = trading_strategy.optimize(data['symbol'], data['strategy_type'])
    return jsonify(strategy)

if __name__ == '__main__':
    app.run(debug=app.config['DEBUG'])