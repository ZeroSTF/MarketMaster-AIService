from flask import jsonify, request
from ..utils.logger import logger
from datetime import datetime

def register_routes(app, socketio, asset_manager,data_fetcher):
    @app.route('/api/assets/register', methods=['POST'])
    def register_assets():
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
                
            symbols = data.get('symbols', [])
            return jsonify(asset_manager.register_assets(symbols))
        except Exception as e:
            logger.error(f"Error in register_assets: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/assets/data', methods=['GET'])
    def get_assets_data():
        try:
            return jsonify(asset_manager.get_all_assets())
        except Exception as e:
            logger.error(f"Error in get_assets_data: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/assets/<symbol>', methods=['GET'])
    def get_asset(symbol):
        try:
            data = asset_manager.get_asset(symbol)
            if data is None:
                return jsonify({'error': 'Asset not found'}), 404
            return jsonify(data)
        except Exception as e:
            logger.error(f"Error in get_asset: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/assets/<symbol>', methods=['DELETE'])
    def remove_asset(symbol):
        try:
            if asset_manager.remove_asset(symbol):
                return jsonify({'message': f'Asset {symbol} removed successfully'})
            return jsonify({'error': 'Asset not found'}), 404
        except Exception as e:
            logger.error(f"Error in remove_asset: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/test/fetch/<symbol>', methods=['GET'])
    def test_fetch(symbol):
        try:
            data = asset_manager.data_fetcher.fetch_data(symbol)
            logger.debug(f"Raw fetch data for {symbol}: {data}")
            return jsonify({
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'data': data
            })
        except Exception as e:
            logger.error(f"Error in test_fetch for {symbol}: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/assets/history', methods=['GET'])
    def get_market_data_between_dates():
      try:
        start = request.args.get('start')
        end = request.args.get('end')
        symbols = request.args.getlist('symbols')  # Capture list of symbols

        if not start or not end or not symbols:
            return jsonify({'error': 'Start date, end date, and at least one symbol are required'}), 400

        # Convert start and end date to datetime objects
        start_date = datetime.fromisoformat(start)
        end_date = datetime.fromisoformat(end)

        # Debugging step to log inputs
        logger.info(f"Fetching historical data for symbols: {symbols} from {start_date} to {end_date}")

        # Use data_fetcher to get historical data
        data = data_fetcher.get_historical_data(start_date=start_date, end_date=end_date, symbols=symbols)
        logger.info(f"Returning historical data: {data}")
        return jsonify(data), 200
      except Exception as e:
        logger.error(f"Error in get_market_data_between_dates: {str(e)}")
        return jsonify({'error': str(e)}), 500

        
    @app.route('/api/assets/news', methods=['GET'])
    def get_news_between_dates():
      try:
        # Get query parameters
        start = request.args.get('start')
        end = request.args.get('end')
        symbols = request.args.getlist('symbols')

        # Check if parameters are missing
        if not start or not end or not symbols:
            return jsonify({'error': 'Start date, end date, and at least one symbol are required'}), 400

        # Convert the start and end date strings to datetime objects
        start_date = datetime.fromisoformat(start).date()
        end_date = datetime.fromisoformat(end).date()

        # Dictionary to store news data for each symbol
        news_data = {}

        # Fetch news for each symbol
        for symbol in symbols:
            news = data_fetcher.fetch_finnhub_news(symbol, start_date, end_date)
            if news:
                news_data[symbol] = news
            else:
                news_data[symbol] = []

        # Return the news data as JSON
        return jsonify(news_data), 200

      except Exception as e:
        logger.error(f"Error in get_news_between_dates: {str(e)}")
        return jsonify({'error': str(e)}), 500



    @app.route('/api/assets/<symbol>/history', methods=['GET'])
    def get_asset_history(symbol):
        try:
            timeframe = request.args.get('timeframe', 'D')
            data = asset_manager.data_fetcher.fetch_historical_data(symbol, timeframe)
            return jsonify({
                'symbol': symbol,
                'timeframe': timeframe,
                'data': data
            })
        except Exception as e:
            logger.error(f"Error in get_asset_history: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @socketio.on('connect')
    def handle_connect():
        logger.info("Client connected")

    @socketio.on('disconnect')
    def handle_disconnect():
        logger.info("Client disconnected")