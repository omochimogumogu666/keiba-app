"""
Application configuration settings.

このモジュールはアプリケーションの環境設定を管理します。
- 開発環境、本番環境、テスト環境の設定を分離
- 環境変数からの設定読み込み
- セキュリティ検証（本番環境のSECRET_KEY必須化）
- データディレクトリの自動作成

Optimized for personal use - localhost only, secure by default.
"""
import os
import logging
from pathlib import Path
from typing import ClassVar
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
    SECRET_KEY: ClassVar[str] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    FLASK_APP: ClassVar[str] = os.getenv('FLASK_APP', 'src.web.app')

    # Database
    SQLALCHEMY_DATABASE_URI: ClassVar[str] = os.getenv(
        'DATABASE_URL',
        f'sqlite:///{BASE_DIR / "keiba.db"}'
    )
    SQLALCHEMY_TRACK_MODIFICATIONS: ClassVar[bool] = False
    SQLALCHEMY_ECHO: ClassVar[bool] = False

    # Scraping
    SCRAPING_DELAY: ClassVar[int] = int(os.getenv('SCRAPING_DELAY', '3'))
    USER_AGENT: ClassVar[str] = os.getenv(
        'USER_AGENT',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    )
    REQUEST_TIMEOUT: ClassVar[int] = 30
    MAX_RETRIES: ClassVar[int] = 3

    # Machine Learning
    MODEL_PATH: ClassVar[Path] = BASE_DIR / os.getenv('MODEL_PATH', 'data/models')
    TRAIN_TEST_SPLIT: ClassVar[float] = float(os.getenv('TRAIN_TEST_SPLIT', '0.2'))
    RANDOM_STATE: ClassVar[int] = int(os.getenv('RANDOM_STATE', '42'))

    # Data paths
    DATA_DIR: ClassVar[Path] = BASE_DIR / 'data'
    RAW_DATA_DIR: ClassVar[Path] = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR: ClassVar[Path] = DATA_DIR / 'processed'
    MODEL_DIR: ClassVar[Path] = DATA_DIR / 'models'

    # Logging
    LOG_LEVEL: ClassVar[str] = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE: ClassVar[Path] = BASE_DIR / os.getenv('LOG_FILE', 'logs/app.log')

    # Caching
    CACHE_TYPE: ClassVar[str] = os.getenv('CACHE_TYPE', 'SimpleCache')
    CACHE_DEFAULT_TIMEOUT: ClassVar[int] = int(os.getenv('CACHE_DEFAULT_TIMEOUT', 300))  # 5 minutes
    CACHE_REDIS_URL: ClassVar[str] = os.getenv('CACHE_REDIS_URL', 'redis://localhost:6379/0')

    # Scheduler
    ENABLE_SCHEDULER: ClassVar[bool] = os.getenv('ENABLE_SCHEDULER', 'false').lower() == 'true'
    SCRAPE_SCHEDULE_HOUR: ClassVar[int] = int(os.getenv('SCRAPE_SCHEDULE_HOUR', '1'))

    @classmethod
    def init_app(cls, app):
        """Initialize application with this config."""
        # Create necessary directories
        cls.MODEL_PATH.mkdir(parents=True, exist_ok=True)
        cls.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Validate configuration
        cls.validate()

    @classmethod
    def validate(cls):
        """
        設定の妥当性を検証します。

        Raises:
            ValueError: 設定値が不正な場合
        """
        if cls.SCRAPING_DELAY < 2:
            raise ValueError(
                f"SCRAPING_DELAY must be at least 2 seconds (current: {cls.SCRAPING_DELAY}). "
                "This is required to avoid overloading the server."
            )

        if cls.MAX_RETRIES < 1 or cls.MAX_RETRIES > 10:
            raise ValueError(
                f"MAX_RETRIES must be between 1 and 10 (current: {cls.MAX_RETRIES})"
            )


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
    DEBUG: ClassVar[bool] = False
    SQLALCHEMY_ECHO: ClassVar[bool] = False

    @classmethod
    def init_app(cls, app):
        Config.init_app(app)

        # Production-specific security validation
        if cls.SECRET_KEY == 'dev-secret-key-change-in-production':
            raise ValueError(
                "❌ PRODUCTION ERROR: Default SECRET_KEY is not allowed in production!\n"
                "   Generate a secure key with:\n"
                "   python -c \"import secrets; print(secrets.token_hex(32))\"\n"
                "   Then set it as an environment variable:\n"
                "   export SECRET_KEY=<your-generated-key>"
            )

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
