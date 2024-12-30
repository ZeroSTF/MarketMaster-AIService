from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
from .api.routes import register_routes
from .utils.scheduler import init_scheduler
from .utils.logger import logger
from .core.data_fetcher import YFinanceDataFetcher
from .core.asset_manager import AssetManager
from .core.news_fetcher import FinnhubNewsFetcher
from .config.settings import Config
from .core.prediction_service import PredictionService
from .core.StockPredictor import StockPredictor

def create_app():
    try:
        app = Flask(__name__)
        CORS(app)
        socketio = SocketIO(app, cors_allowed_origins="*")
        
        # Initialize core components
        data_fetcher = YFinanceDataFetcher()
        news_fetcher = FinnhubNewsFetcher(Config.FINNHUB_API_KEY)
        asset_manager = AssetManager(data_fetcher, socketio)
        prediction_service = PredictionService()
        stock_predictor = StockPredictor()
        
        # Register routes and start scheduler
        register_routes(app, socketio, asset_manager, news_fetcher, prediction_service, stock_predictor)
        init_scheduler(app, asset_manager)
        
        logger.info("Application initialized successfully")
        return app, socketio
    except Exception as e:
        logger.error(f"Failed to create application: {str(e)}")
        raise