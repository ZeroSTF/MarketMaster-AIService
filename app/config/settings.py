class Config:
    # Server settings
    HOST = '0.0.0.0'
    PORT = 5000
    DEBUG = True

    # YFinance settings
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds

    # Scheduler settings
    UPDATE_INTERVAL = 5  # seconds

    # Logging settings
    LOG_LEVEL = 'DEBUG'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'