class Config:
    # Server settings
    HOST = '0.0.0.0'
    PORT = 5000
    DEBUG = False

    # YFinance settings
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds

    # Scheduler settings
    UPDATE_INTERVAL = 5  # seconds

    # Logging settings
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Finnhub settings
    FINNHUB_API_KEY = 'ct4875pr01qo7vqaigpgct4875pr01qo7vqaigq0'