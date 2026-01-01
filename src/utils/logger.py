"""
Logger utility for application-wide logging.
"""
from config.logging_config import get_logger


def get_app_logger(name):
    """
    Get application logger.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return get_logger(name)
