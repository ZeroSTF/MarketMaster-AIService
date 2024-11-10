from app.core.acturial_calcul import ActuarialCalculator
from flask import jsonify, request
from ..utils.logger import logger
from datetime import datetime

def register_routes(app, socketio, asset_manager):
      
    actuarial_calculator = ActuarialCalculator()

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
    
    @app.route('/api/assets/premiums', methods=['POST'])
    def get_suggested_premiums():
        try:
            data = request.get_json()
            if not data or 'symbols' not in data:
                return jsonify({'error': 'No symbols provided'}), 400
            
            symbols = data['symbols']
            premiums = actuarial_calculator.calculate_suggested_premiums(symbols)
            return jsonify(premiums)
        except Exception as e:
            logger.error(f"Error in get_suggested_premiums: {str(e)}")
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