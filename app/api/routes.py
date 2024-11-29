from flask import jsonify, request
from ..utils.logger import logger
from datetime import datetime
from ..core.prediction_service import PredictionService
from ..core.StockPredictor import StockPredictor
from ..core.calculate_asset_metrics import calculate_asset_metrics
def register_routes(app, socketio, asset_manager):
    # Initialize PredictionService
    prediction_service = PredictionService()
    stock_predictor = StockPredictor()
    # Existing routes
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

    # New prediction routes
    @app.route('/api/predict/<symbol>', methods=['POST'])
    def predict_prices(symbol):
        historical_data = None  # Initialize historical_data to avoid UnboundLocalError
        try:
        # Parse JSON body from request
            data = request.get_json()
            train = data.get('train', False)

        # Check if 'train' is True, perform training
            if train:
                result = stock_predictor.train_and_save_model(symbol)
            else:
                # Ensure historical data is fetched and handled correctly
                historical_data = stock_predictor.fetch_data(symbol)

            # If data is missing or empty, return 404 with an error message
                if historical_data is None or historical_data.empty:
                    return jsonify({'error': 'Insufficient or unavailable stock data'}), 404

            # Make predictions based on the fetched historical data
                result = stock_predictor.make_predictions(historical_data, symbol=symbol)

        # If result contains an 'error' key, return that as a 500 error
            if 'error' in result:
                return jsonify(result), 500

        # Return the result of the prediction as a successful response
            return jsonify(result), 200

        except Exception as e:
        # Log the full exception trace for debugging purposes
            logger.error(f"Error in predict_prices for {symbol}: {str(e)}")
            logger.error(f"Exception Traceback: {traceback.format_exc()}")  # Log the full traceback for deeper debugging
        
        # If historical_data exists, log it for context
            if historical_data is not None:
                logger.error(f"Historical data: {historical_data}")

        # Return the error message with a 500 status code
            return jsonify({'error': str(e)}), 500
    @app.route('/api/predict/models/<symbol>', methods=['DELETE'])
    def delete_prediction_model(symbol):
        try:
            model_path = prediction_service.get_model_path(symbol)
            if prediction_service.delete_model(symbol):
                return jsonify({'message': f'Prediction model for {symbol} deleted successfully'})
            return jsonify({'error': 'Model not found'}), 404
        except Exception as e:
            logger.error(f"Error in delete_prediction_model: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/predict/status/<symbol>', methods=['GET'])
    def get_prediction_status(symbol):
        try:
            status = prediction_service.get_model_status(symbol)
            return jsonify(status)
        except Exception as e:
            logger.error(f"Error in get_prediction_status: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/assets/<symbol>/metrics', methods=['GET'])
    def get_asset_metrics(symbol):
        try:
        # Call the existing function to fetch historical data
            response = get_asset_history(symbol)
            data = response.get_json()

            if 'error' in data:
                return jsonify({'error': data['error']}), 404

        # Calculate metrics
            metrics = calculate_asset_metrics(data)

            return jsonify({
                'symbol': symbol,
                'metrics': metrics
            })
        except Exception as e:
            logger.error(f"Error in get_asset_metrics: {str(e)}")
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

    @socketio.on('request_prediction')
    def handle_prediction_request(data):
        try:
            symbol = data.get('symbol')
            train = data.get('train', False)

            if not symbol:
                return {'error': 'Symbol is required'}

            if train:
                result = stock_predictor.train_and_save_model(symbol)
            else:
                historical_data = stock_predictor.fetch_data(symbol)
                if historical_data.empty:
                    return {'error': 'No historical data'}
                result = stock_predictor.make_predictions(historical_data, symbol=symbol)
        
            socketio.emit('prediction_update', {'symbol': symbol, 'result': result})
        except Exception as e:
            logger.error(f"Error handling WebSocket prediction: {str(e)}")


    return app