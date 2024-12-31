from app.core import OptionsPredictionModel
from app.core.acturial_calcul import ActuarialCalculator
from flask import jsonify, request
from ..utils.logger import logger
from datetime import datetime

def register_routes(app, socketio, asset_manager):
      
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
                'recommendation': get_recommendation(prediction)
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

def get_recommendation(prediction):
    """Génère une recommandation basée sur la prédiction"""
    if prediction['signal'] == 1:
        return {
            'action': 'ACHETER_CALL',
            'description': 'Acheter une option d\'achat (CALL)',
            'strike_price': prediction.get('prix_strike'),
            'premium': prediction.get('prime'),
            'expiration': prediction.get('date_expiration'),
            'confidence': max(prediction['probabilite']) * 100
        }
    elif prediction['signal'] == -1:
        return {
            'action': 'ACHETER_PUT',
            'description': 'Acheter une option de vente (PUT)',
            'strike_price': prediction.get('prix_strike'),
            'premium': prediction.get('prime'),
            'expiration': prediction.get('date_expiration'),
            'confidence': max(prediction['probabilite']) * 100
        }
    else:
        return {
            'action': 'ATTENDRE',
            'description': 'Pas de signal d\'achat pour le moment',
            'confidence': max(prediction['probabilite']) * 100
        }