from app.app import create_app
from app.config.settings import Config
from app.utils.logger import logger

app, socketio = create_app()

if __name__ == '__main__':
    try:
        logger.info(f"Starting server on {Config.HOST}:{Config.PORT}")
        socketio.run(app, 
                    host=Config.HOST, 
                    port=Config.PORT, 
                    debug=Config.DEBUG)
    except Exception as e:
        logger.error(f"Server failed to start: {str(e)}")