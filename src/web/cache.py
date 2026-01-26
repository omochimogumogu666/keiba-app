"""
Caching configuration for Flask application.

キャッシュ層:
- Flask-Cache: 汎用HTTPレスポンスキャッシュ（Redis/SimpleCache）
- StatisticsCache: ML統計専用2層キャッシュ（L1: リクエスト内, L2: Flask-Cache）

This module provides Flask-Caching setup with Redis backend support
and fallback to simple cache for development.
"""
from flask_caching import Cache

from src.ml.stats_cache import init_stats_cache

# Initialize cache instance
cache = Cache()


def init_cache(app):
    """
    Initialize cache with Flask app.

    Flask-Cacheの初期化に加え、ML統計キャッシュも初期化する。

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

    # ML統計キャッシュの初期化（Flask-Cacheをバックエンドとして使用）
    init_stats_cache(cache)

    app.logger.info(f"Cache initialized with type: {cache_config['CACHE_TYPE']}")

    return cache
