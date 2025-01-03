from app.core import OptionsPredictionModel
from app.core.acturial_calcul import ActuarialCalculator
from flask import jsonify, request
from ..utils.logger import logger
import traceback
import os
import json
import pandas as pd
from datetime import datetime
from ..core.prediction_service import PredictionService
from ..core.StockPredictor import StockPredictor
from ..core.calculate_asset_metrics import calculate_asset_metrics
from ..core.calculate_asset_metrics import generate_recommendations

def register_routes(app, socketio, asset_manager, news_fetcher, prediction_service, stock_predictor):
    actuarial_calculator = ActuarialCalculator(asset_manager=asset_manager)
    predictor = OptionsPredictionModel(asset_manager, actuarial_calculator)
    
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
    
    @app.route('/api/assets/premiums', methods=['GET'])
    def get_suggested_premiums():
        try:
            symbols = request.args.get('symbols', '').split(',')
            symbols = [s.strip() for s in symbols if s.strip()]
            
            if not symbols:
                return jsonify({'error': 'No symbols provided'}), 400
                
            premiums = actuarial_calculator.calculate_suggested_premiums(symbols)
            return jsonify(premiums)
        except Exception as e:
            logger.error(f"Error calculating premiums: {str(e)}")
            return jsonify({'error': str(e)}), 500
    @app.route('/api/assets/calculate_option_premium', methods=['POST'])
    def calculate_option_premium():
        try:
            # Extraire les données de la requête
            data = request.get_json()
            symbol = data.get('symbol')
            strike_price = data.get('strike_price')
            expiration_date_str = data.get('expiration_date')
            option_type = data.get('option_type')

            # Valider les paramètres
            if not symbol or not strike_price or not expiration_date_str or not option_type:
                return jsonify({'error': 'Missing required parameters'}), 400

            # Convertir la date d'expiration en datetime
            try:
                expiration_date = datetime.strptime(expiration_date_str, '%Y-%m-%d')
            except ValueError:
                return jsonify({'error': 'Invalid expiration date format. Use YYYY-MM-DD.'}), 400
            try:
                strike_price = float(strike_price)  # Explicitly cast to float
            except ValueError:
                return jsonify({'error': 'Invalid strike price format. Must be a number.'}), 400   
            # Calculer la prime d'option
            try:
                premium = actuarial_calculator.calculate_option_premium(symbol, strike_price, expiration_date, option_type)
            except Exception as e:
                return jsonify({'error': f'Calculation failed: {str(e)}'}), 500
        
            if premium is None:
                return jsonify({'error': 'Could not calculate the option premium'}), 500

            # Retourner la prime calculée
            return jsonify({'symbol': symbol, 'option_type': option_type, 'premium': premium}), 200
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
        data = asset_manager.data_fetcher.get_game_data(start_date=start_date, end_date=end_date, symbols=symbols)
        logger.info(f"Returning historical data: {data}")
        return jsonify(data), 200
      except Exception as e:
        logger.error(f"Error in get_market_data_between_dates: {str(e)}")
        return jsonify({'error': str(e)}), 500

        
    @app.route('/api/assets/news', methods=['GET'])
    def get_news_between_dates():
        try:
            start = request.args.get('start')
            end = request.args.get('end')
            symbols = request.args.getlist('symbols')
            
            if not start or not end or not symbols:
                return jsonify({'error': 'Start date, end date, and at least one symbol are required'}), 400
            
            start_date = datetime.fromisoformat(start).date().isoformat()
            end_date = datetime.fromisoformat(end).date().isoformat()
            
            news_data = news_fetcher.fetch_multiple_symbol_news(symbols, start_date, end_date)
            
            return jsonify(news_data), 200
        
        except Exception as e:
            logger.error(f"Error in get_news_between_dates: {str(e)}")
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
    

        except Exception as e:
            return jsonify({'error': f'Unexpected error: {str(e)}'}), 500
    
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

    # HELPER FUNCTION FOR CHOHDI
    def get_asset_history_chohdi(symbol):
        try:
        # Map timeframe to yfinance interval
            timeframe = request.args.get('timeframe', 'D')

            data = asset_manager.data_fetcher.fetch_historical_data(symbol, timeframe, '2y', True)
        
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
    ##############################################################################################################################

    @app.route('/api/assets/<symbol>/close_dates', methods=['GET'])
    def get_close_dates(symbol):
        try:
            # Map timeframe to yfinance interval
            timeframe = request.args.get('timeframe', 'D')

            # Fetch historical data
            data = asset_manager.data_fetcher.fetch_historical_data(symbol, timeframe, '2y', True)

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

    @app.route('/api/assets/<symbol>/metrics', methods=['GET'])
    def get_asset_metrics(symbol):
        try:
        # Call the existing function to fetch historical data
            response = get_asset_history_chohdi(symbol)
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
                response = get_asset_history_chohdi(symbol)
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
            historical_data_response = get_asset_history_chohdi(symbol)
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



    @socketio.on('connect')
    def handle_connect():
        logger.info("Client connected")

    @socketio.on('disconnect')
    def handle_disconnect():
        logger.info("Client disconnected")
    @app.route('/api/predict/train/<symbol>', methods=['POST'])
    def train_prediction_model(symbol):
        """Route pour entraîner le modèle pour un symbole spécifique"""
        try:
            success = predictor.train(symbol)
            if success:
                return jsonify({
                    'status': 'success',
                    'message': f'Model trainning succes {symbol}'
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': f'Echec de l\'entraînement pour {symbol}'
                }), 500
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement du modèle pour {symbol}: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    @app.route('/api/predict/signal/<symbol>', methods=['GET'])
    def predict_option_signal(symbol):
        """Route pour obtenir une prédiction pour un symbole"""
        try:
            # Vérifier si l'actif existe
            asset_data = asset_manager.get_asset(symbol)
            if not asset_data:
                return jsonify({
                    'status': 'error',
                    'message': f'Actif {symbol} non trouvé'
                }), 404

            # Obtenir la prédiction
            prediction = predictor.predict(symbol)
            recommendation = predictor.get_recommendation(prediction)
            if prediction is None:
                return jsonify({
                    'status': 'error',
                    'message': 'Impossible de générer une prédiction'
                }), 500

            # Enrichir la réponse avec des données supplémentaires
            response = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'prediction': prediction,
                'current_price': asset_data.get('currentPrice'),
                'recommendation': recommendation
            }

            return jsonify(response)

        except Exception as e:
            logger.error(f"Erreur lors de la prédiction pour {symbol}: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    @app.route('/api/predict/batch', methods=['POST'])
    def predict_batch_signals():
        """Route pour obtenir des prédictions pour plusieurs symboles"""
        try:
            data = request.get_json()
            if not data or 'symbols' not in data:
                return jsonify({
                    'status': 'error',
                    'message': 'Liste de symboles requise'
                }), 400

            symbols = data['symbols']
            predictions = {}

            for symbol in symbols:
                prediction = predictor.predict_option_signal(symbol)
                if prediction is not None:
                    predictions[symbol] = {
                        'prediction': prediction,
                        'timestamp': datetime.now().isoformat()
                    }

            return jsonify({
                'status': 'success',
                'predictions': predictions
            })

        except Exception as e:
            logger.error(f"Erreur lors des prédictions batch: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500



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
