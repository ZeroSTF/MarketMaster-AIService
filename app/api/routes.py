from flask import jsonify, request
from ..utils.logger import logger
import logging
import traceback
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from ..core.prediction_service import PredictionService
from ..core.StockPredictor import StockPredictor
from ..core.calculate_asset_metrics import calculate_asset_metrics
from ..core.calculate_asset_metrics import generate_recommendations
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
        try:
            import time
            start_time = time.time()

            # Parse JSON body
            data = request.get_json()
            train = data.get('train', False)
            future_days = data.get('future_days', 14)  # Default to 14 days if not provided

            # Fetch historical data
            fetch_start = time.time()
            historical_data = stock_predictor.fetch_data(symbol)
            print(f"Data fetching took {time.time() - fetch_start} seconds")

            if historical_data is None or historical_data.empty:
                return jsonify({'error': 'Insufficient or unavailable stock data'}), 404

            # Train model if 'train' is True
            if train:
                train_start = time.time()
                result = stock_predictor.train_and_save_model(symbol)
                print(f"Model training took {time.time() - train_start} seconds")

                if isinstance(result, dict) and 'error' in result:
                    return jsonify(result), 500

                print(f"Total time: {time.time() - start_time} seconds")
                return jsonify(result), 200
            else:
                # Load the model
                load_start = time.time()
                model = stock_predictor.load_model(symbol)
                print(f"Model loading took {time.time() - load_start} seconds")

                if not model:
                    return jsonify({'error': f"No trained model found for {symbol}. Please train the model first."}), 404

                # Make predictions
                pred_start = time.time()
                result = stock_predictor.make_predictions(historical_data, model=model, future_days=future_days)
                print(f"Prediction took {time.time() - pred_start} seconds")

                if 'error' in result:
                    return jsonify(result), 500

                print(f"Total time: {time.time() - start_time} seconds")
                return jsonify(result), 200

        except Exception as e:
            logger.error(f"Error in predict_prices for {symbol}: {str(e)}")
            logger.error(f"Exception Traceback: {traceback.format_exc()}")
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
    
<<<<<<< HEAD
=======
    @app.route('/api/assets/<symbol>/history', methods=['GET'])
    def get_asset_history(symbol):
        try:
        # Map timeframe to yfinance interval
            timeframe = request.args.get('timeframe', 'D')
            timeframe_mapping = {
            'D': '1d',
            'W': '1wk',
            'M': '1mo'
        }
            interval = timeframe_mapping.get(timeframe, '1d')
        
        # Fetch historical data
            data = asset_manager.data_fetcher.fetch_historical_data(symbol, interval=interval)
        
            if not data or 'data' not in data or len(data['data']) == 0:
                raise ValueError(f"No historical data found for {symbol}")

        # Return response
            return jsonify({
            'symbol': symbol,
            'timeframe': timeframe,
            'data': data
            })
        except ValueError as ve:
            logger.warning(f"ValueError in get_asset_history: {str(ve)}")
            return jsonify({'error': str(ve)}), 404
        except Exception as e:
            logger.error(f"Error in get_asset_history: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/assets/<symbol>/close_dates', methods=['GET'])
    def get_close_dates(symbol):
        try:
            # Map timeframe to yfinance interval
            timeframe = request.args.get('timeframe', 'D')
            timeframe_mapping = {
                'D': '1d',
                'W': '1wk',
                'M': '1mo'
            }
            interval = timeframe_mapping.get(timeframe, '1d')

            # Fetch historical data
            data = asset_manager.data_fetcher.fetch_historical_data(symbol, interval=interval)

            if not data or 'data' not in data or len(data['data']) == 0:
                raise ValueError(f"No historical data found for {symbol}")

            # Extract close and date
            close_date_list = [
                {"date": entry["date"], "close": entry["close"]} 
                for entry in data["data"]
            ]

            # Return response
            return jsonify({
                "symbol": symbol,
                "timeframe": timeframe,
                "close_date_list": close_date_list
            })

        except ValueError as ve:
            logger.warning(f"ValueError in get_close_dates: {str(ve)}")
            return jsonify({'error': str(ve)}), 404
        except Exception as e:
            logger.error(f"Error in get_close_dates: {str(e)}")
            return jsonify({'error': str(e)}), 500
>>>>>>> f736985 (AI)
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
<<<<<<< HEAD
    
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
=======
    @app.route('/api/performance', methods=['POST'])
    def calculate_performance():
        try:
            data = request.get_json()
            symbols = data.get("symbols", [])

            if not symbols:
                return jsonify({"error": "No symbols provided"}), 400

            results = []
            for symbol in symbols:
                historical_data = stock_predictor.fetch_data(symbol)
                if historical_data is None or len(historical_data) < 60:
                    results.append({"symbol": symbol, "currentPrice": 0, "performance": None})
                    continue

                current_price = historical_data['Close'].iloc[-1]  # Fetch the most recent closing price
                start_price = historical_data['Close'].iloc[0]
                performance = ((current_price - start_price) / start_price) * 100

                results.append({
                    "symbol": symbol,
                    "currentPrice": float(current_price),  # Ensure JSON-serializable
                    "performance": float(performance)  # Ensure JSON-serializable
                })

            return jsonify(results), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/correlation', methods=['POST'])
    def calculate_correlation():
        try:
            data = request.get_json()
            symbols = data.get("symbols", [])
            if not symbols:
                return jsonify({"error": "No symbols provided"}), 400
            
            historical_data = {}
            
            for symbol in symbols:
                # Fetch historical data
                response = get_asset_history(symbol)
                response_data = response.json
                
                if "error" in response_data:
                    print(f"No data found for symbol: {symbol}")
                    continue
                
                # Extract and format close prices
                historical_prices = response_data.get('data', {}).get('data', [])
                if not historical_prices:
                    print(f"No price data for symbol: {symbol}")
                    continue
                
                # Create DataFrame for this symbol
                df = pd.DataFrame(historical_prices)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
                # Ensure only the 'close' column is used for correlation
                historical_data[symbol] = df['close']
            
            if not historical_data:
                return jsonify({"error": "No valid data for any symbol"}), 400
            
            # Combine all Series into a DataFrame
            combined_df = pd.DataFrame(historical_data)
            
            # Drop rows with NaN values
            combined_df.dropna(inplace=True)
            
            # Compute the correlation matrix
            correlation_matrix = combined_df.corr().to_dict()
            
            return jsonify(correlation_matrix), 200
        
        except Exception as e:
            print(f"Error in /api/correlation: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/assets/<symbol>/recommendations', methods=['GET'])
    def asset_recommendations(symbol):
        try:
            # Fetch historical data for the asset
            historical_data_response = get_asset_history(symbol)
            historical_data = historical_data_response.get_json()  # Extract the JSON payload
            
            if 'error' in historical_data:
                return jsonify({"error": historical_data['error']}), 400
            
            # Directly calculate metrics
            metrics_result = calculate_asset_metrics(historical_data)
            
            # Validate metrics
            metrics = metrics_result.get("metrics")
            if not metrics:
                return jsonify({"error": "Metrics calculation failed"}), 400
            
            # Ensure RSI is available
            rsi = metrics.get("RSI")
            if rsi is None:
                return jsonify({"error": "RSI not available"}), 400
            
            # Generate recommendations
            recommendation = generate_recommendations(metrics)
            
            # Return recommendations
            return jsonify({
                "symbol": symbol,
                "recommendation": recommendation
            })
        
        except Exception as e:
            return jsonify({"error": str(e)}), 500


>>>>>>> f736985 (AI)

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