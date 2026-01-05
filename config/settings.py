"""
Application configuration settings.
Optimized for personal use - localhost only, secure by default.
"""
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Setup logger for config warnings
logger = logging.getLogger(__name__)


class Config:
    """Base configuration - Optimized for personal use."""

    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    FLASK_APP = os.getenv('FLASK_APP', 'src.web.app')

    # Database
    SQLALCHEMY_DATABASE_URI = os.getenv(
        'DATABASE_URL',
        f'sqlite:///{BASE_DIR / "keiba.db"}'
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = False

    # Scraping
    SCRAPING_DELAY = int(os.getenv('SCRAPING_DELAY', '3'))
    USER_AGENT = os.getenv(
        'USER_AGENT',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    )
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3

    # Machine Learning
    MODEL_PATH = BASE_DIR / os.getenv('MODEL_PATH', 'data/models')
    TRAIN_TEST_SPLIT = float(os.getenv('TRAIN_TEST_SPLIT', '0.2'))
    RANDOM_STATE = int(os.getenv('RANDOM_STATE', '42'))

    # Data paths
    DATA_DIR = BASE_DIR / 'data'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
    MODEL_DIR = DATA_DIR / 'models'

    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = BASE_DIR / os.getenv('LOG_FILE', 'logs/app.log')

    # Caching
    CACHE_TYPE = os.getenv('CACHE_TYPE', 'SimpleCache')
    CACHE_DEFAULT_TIMEOUT = int(os.getenv('CACHE_DEFAULT_TIMEOUT', 300))  # 5 minutes
    CACHE_REDIS_URL = os.getenv('CACHE_REDIS_URL', 'redis://localhost:6379/0')

    # Scheduler
    ENABLE_SCHEDULER = os.getenv('ENABLE_SCHEDULER', 'false').lower() == 'true'
    SCRAPE_SCHEDULE_HOUR = int(os.getenv('SCRAPE_SCHEDULE_HOUR', '1'))

    @classmethod
    def init_app(cls, app):
        """Initialize application with this config."""
        # Create necessary directories
        cls.MODEL_PATH.mkdir(parents=True, exist_ok=True)
        cls.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


class DevelopmentConfig(Config):
    """Development configuration - Optimized for personal use."""
    DEBUG = True
    SQLALCHEMY_ECHO = True

    @classmethod
    def init_app(cls, app):
        Config.init_app(app)

        # Warn if using default SECRET_KEY
        if cls.SECRET_KEY == 'dev-secret-key-change-in-production':
            logger.warning(
                "\n" + "=" * 80 + "\n"
                "⚠️  WARNING: Using default SECRET_KEY!\n"
                "   For security, generate a new one with:\n"
                "   python -c \"import secrets; print(secrets.token_hex(32))\"\n"
                "   Then add it to your .env file as SECRET_KEY=<generated-key>\n"
                + "=" * 80
            )
            # Also print to console for visibility
            print(
                "\n" + "=" * 80 + "\n"
                "⚠️  WARNING: Using default SECRET_KEY!\n"
                "   For security, generate a new one with:\n"
                "   python -c \"import secrets; print(secrets.token_hex(32))\"\n"
                "   Then add it to your .env file as SECRET_KEY=<generated-key>\n"
                + "=" * 80 + "\n"
            )


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    SQLALCHEMY_ECHO = False

    @classmethod
    def init_app(cls, app):
        Config.init_app(app)

        # Production-specific initialization
        # e.g., send errors to admin email


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(env=None):
    """Get configuration for specified environment."""
    if env is None:
        env = os.getenv('FLASK_ENV', 'development')
    return config.get(env, config['default'])
