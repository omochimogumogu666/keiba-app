"""
Caching configuration for Flask application.

This module provides Flask-Caching setup with Redis backend support
and fallback to simple cache for development.
"""
from flask_caching import Cache

# Initialize cache instance
cache = Cache()


def init_cache(app):
    """
    Initialize cache with Flask app.

    Args:
        app: Flask application instance
    """
    cache_config = {
        'CACHE_TYPE': app.config.get('CACHE_TYPE', 'SimpleCache'),
        'CACHE_DEFAULT_TIMEOUT': app.config.get('CACHE_DEFAULT_TIMEOUT', 300),
    }

    # Redis configuration (if available)
    if cache_config['CACHE_TYPE'] == 'RedisCache':
        cache_config['CACHE_REDIS_URL'] = app.config.get(
            'CACHE_REDIS_URL',
            'redis://localhost:6379/0'
        )

    cache.init_app(app, config=cache_config)

    app.logger.info(f"Cache initialized with type: {cache_config['CACHE_TYPE']}")

    return cache
