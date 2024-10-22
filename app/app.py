from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
from .api.routes import register_routes
from .utils.scheduler import init_scheduler
from .utils.logger import logger
from .core.data_fetcher import YFinanceDataFetcher
from .core.asset_manager import AssetManager
from .config.settings import Config

def create_app():
    try:
        app = Flask(__name__)
        CORS(app)
        socketio = SocketIO(app, cors_allowed_origins="*")
        
        # Initialize core components
        data_fetcher = YFinanceDataFetcher()
        asset_manager = AssetManager(data_fetcher, socketio)
        
        # Register routes and start scheduler
        register_routes(app, socketio, asset_manager)
        init_scheduler(app, asset_manager)
        
        logger.info("Application initialized successfully")
        return app, socketio
    except Exception as e:
        logger.error(f"Failed to create application: {str(e)}")
        raise