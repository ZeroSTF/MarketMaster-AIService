from apscheduler.schedulers.background import BackgroundScheduler
from ..config.settings import Config
from ..utils.logger import logger

def init_scheduler(app, asset_manager):
    try:
        scheduler = BackgroundScheduler()
        scheduler.add_job(
            asset_manager.update_assets,
            'interval',
            seconds=Config.UPDATE_INTERVAL
        )
        scheduler.start()
        logger.info("Background scheduler started successfully")
    except Exception as e:
        logger.error(f"Failed to start scheduler: {str(e)}")
        raise