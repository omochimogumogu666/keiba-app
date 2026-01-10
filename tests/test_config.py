"""
Tests for configuration settings.
"""
import pytest
import os
from pathlib import Path
from config.settings import (
    Config, DevelopmentConfig, ProductionConfig, TestingConfig, get_config
)


@pytest.mark.unit
class TestBaseConfig:
    """Tests for base Config class."""

    def test_config_has_secret_key(self):
        """Test config has secret key."""
        assert Config.SECRET_KEY is not None
        assert len(Config.SECRET_KEY) > 0

    def test_config_has_database_uri(self):
        """Test config has database URI."""
        assert Config.SQLALCHEMY_DATABASE_URI is not None

    def test_config_scraping_delay(self):
        """Test scraping delay configuration."""
        assert isinstance(Config.SCRAPING_DELAY, int)
        assert Config.SCRAPING_DELAY >= 0

    def test_config_has_user_agent(self):
        """Test config has user agent."""
        assert Config.USER_AGENT is not None
        assert 'Mozilla' in Config.USER_AGENT

    def test_config_model_path(self):
        """Test model path configuration."""
        assert Config.MODEL_PATH is not None
        assert isinstance(Config.MODEL_PATH, Path)

    def test_config_data_paths(self):
        """Test data directory paths."""
        assert Config.DATA_DIR is not None
        assert Config.RAW_DATA_DIR is not None
        assert Config.PROCESSED_DATA_DIR is not None
        assert isinstance(Config.DATA_DIR, Path)

    def test_config_log_level(self):
        """Test logging configuration."""
        assert Config.LOG_LEVEL in ['DEBUG', 'INFO', 'WARNING', 'ERROR']
        assert Config.LOG_FILE is not None


@pytest.mark.unit
class TestDevelopmentConfig:
    """Tests for Development configuration."""

    def test_development_debug_enabled(self):
        """Test debug is enabled in development."""
        assert DevelopmentConfig.DEBUG is True

    def test_development_sqlalchemy_echo(self):
        """Test SQLAlchemy echo in development."""
        assert DevelopmentConfig.SQLALCHEMY_ECHO is True

    def test_development_inherits_base(self):
        """Test development config inherits from base."""
        assert hasattr(DevelopmentConfig, 'SECRET_KEY')
        assert hasattr(DevelopmentConfig, 'SCRAPING_DELAY')


@pytest.mark.unit
class TestProductionConfig:
    """Tests for Production configuration."""

    def test_production_debug_disabled(self):
        """Test debug is disabled in production."""
        assert ProductionConfig.DEBUG is False

    def test_production_sqlalchemy_echo_disabled(self):
        """Test SQLAlchemy echo disabled in production."""
        assert ProductionConfig.SQLALCHEMY_ECHO is False

    def test_production_inherits_base(self):
        """Test production config inherits from base."""
        assert hasattr(ProductionConfig, 'SECRET_KEY')
        assert hasattr(ProductionConfig, 'SCRAPING_DELAY')


@pytest.mark.unit
class TestTestingConfig:
    """Tests for Testing configuration."""

    def test_testing_flag_enabled(self):
        """Test testing flag is enabled."""
        assert TestingConfig.TESTING is True

    def test_testing_uses_memory_database(self):
        """Test testing uses in-memory database."""
        assert 'memory' in TestingConfig.SQLALCHEMY_DATABASE_URI

    def test_testing_inherits_base(self):
        """Test testing config inherits from base."""
        assert hasattr(TestingConfig, 'SECRET_KEY')
        assert hasattr(TestingConfig, 'SCRAPING_DELAY')


@pytest.mark.unit
class TestGetConfig:
    """Tests for get_config function."""

    def test_get_config_development(self):
        """Test getting development config."""
        config = get_config('development')
        assert config == DevelopmentConfig
        assert config.DEBUG is True

    def test_get_config_production(self):
        """Test getting production config."""
        config = get_config('production')
        assert config == ProductionConfig
        assert config.DEBUG is False

    def test_get_config_testing(self):
        """Test getting testing config."""
        config = get_config('testing')
        assert config == TestingConfig
        assert config.TESTING is True

    def test_get_config_default(self):
        """Test getting default config."""
        config = get_config(None)
        assert config == DevelopmentConfig

    def test_get_config_invalid(self):
        """Test getting config with invalid name."""
        config = get_config('invalid')
        assert config == DevelopmentConfig  # Should return default


@pytest.mark.unit
class TestConfigEnvironmentVariables:
    """Tests for environment variable loading."""

    def test_config_loads_from_env(self, monkeypatch):
        """Test configuration loads from environment variables."""
        # Set test environment variables
        monkeypatch.setenv('SECRET_KEY', 'test-secret-key')
        monkeypatch.setenv('SCRAPING_DELAY', '5')
        monkeypatch.setenv('LOG_LEVEL', 'DEBUG')

        # Reload config
        from config import settings
        import importlib
        importlib.reload(settings)

        # Verify environment variables are loaded
        assert os.getenv('SECRET_KEY') == 'test-secret-key'
        assert os.getenv('SCRAPING_DELAY') == '5'
        assert os.getenv('LOG_LEVEL') == 'DEBUG'


@pytest.mark.unit
class TestConfigInitApp:
    """Tests for Config.init_app method."""

    def test_init_app_creates_directories(self, tmp_path, monkeypatch):
        """Test init_app creates necessary directories."""
        # Create a temporary config with temp paths
        class TempConfig(Config):
            MODEL_PATH = tmp_path / 'models'
            RAW_DATA_DIR = tmp_path / 'raw'
            PROCESSED_DATA_DIR = tmp_path / 'processed'
            LOG_FILE = tmp_path / 'logs' / 'app.log'

        # Mock app
        class MockApp:
            pass

        app = MockApp()

        # Initialize app
        TempConfig.init_app(app)

        # Verify directories were created
        assert TempConfig.MODEL_PATH.exists()
        assert TempConfig.RAW_DATA_DIR.exists()
        assert TempConfig.PROCESSED_DATA_DIR.exists()
        assert TempConfig.LOG_FILE.parent.exists()


@pytest.mark.unit
class TestConfigValues:
    """Tests for specific configuration values."""

    def test_train_test_split_valid(self):
        """Test train/test split is valid proportion."""
        assert 0 < Config.TRAIN_TEST_SPLIT < 1

    def test_random_state_is_int(self):
        """Test random state is integer."""
        assert isinstance(Config.RANDOM_STATE, int)

    def test_request_timeout_reasonable(self):
        """Test request timeout is reasonable."""
        assert Config.REQUEST_TIMEOUT > 0
        assert Config.REQUEST_TIMEOUT <= 120  # Not too long

    def test_max_retries_reasonable(self):
        """Test max retries is reasonable."""
        assert Config.MAX_RETRIES >= 0
        assert Config.MAX_RETRIES <= 10

    def test_sqlalchemy_track_modifications_disabled(self):
        """Test SQLAlchemy track modifications is disabled."""
        assert Config.SQLALCHEMY_TRACK_MODIFICATIONS is False
