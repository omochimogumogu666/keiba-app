"""
統計キャッシュ層 - ML予測の高速化のための2層キャッシュ。

このモジュールは馬・騎手・調教師の統計データをキャッシュして、
予測時のDBクエリを最小化します。

キャッシュ戦略:
- L1: リクエスト内キャッシュ（インメモリ、同一リクエスト内での再利用）
- L2: Flask-Cache/Redis（リクエスト間での再利用、1時間TTL）

TTL=1時間の理由:
- 統計はレース終了時にのみ変化する
- JRAのレースは通常2-3時間間隔
- 1時間で十分な鮮度を維持しつつキャッシュヒット率を最大化
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import hashlib
import json

from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)

# キャッシュTTL（秒）
STATS_CACHE_TTL = 3600  # 1時間


class StatisticsCache:
    """
    2層キャッシュによる統計データの高速アクセス。

    L1キャッシュ: リクエスト内キャッシュ（dict）
    L2キャッシュ: Flask-Cache（Redis/SimpleCache）

    使用例:
        cache = StatisticsCache(flask_cache)
        stats = cache.get_horse_stats(horse_ids, cutoff_date, session)
    """

    def __init__(self, flask_cache=None):
        """
        Args:
            flask_cache: Flask-Caching instance (optional, L2キャッシュ用)
        """
        self.flask_cache = flask_cache
        self._l1_cache: Dict[str, Any] = {}  # リクエスト内キャッシュ

    def clear_l1_cache(self):
        """L1キャッシュをクリア（リクエスト終了時に呼び出し）"""
        self._l1_cache.clear()

    def _make_cache_key(self, prefix: str, ids: List[int], cutoff_date: datetime) -> str:
        """
        キャッシュキーを生成する。

        Args:
            prefix: キータイプ（'horse', 'jockey', 'trainer'）
            ids: エンティティIDリスト
            cutoff_date: 統計計算の基準日

        Returns:
            ハッシュ化されたキャッシュキー
        """
        # IDリストをソートして一貫性を確保
        sorted_ids = sorted(ids)
        # 日付は日単位で丸める（時間は無視）
        date_str = cutoff_date.strftime('%Y-%m-%d')

        key_data = f"{prefix}:{','.join(map(str, sorted_ids))}:{date_str}"
        # 長いキーはハッシュ化
        if len(key_data) > 200:
            key_hash = hashlib.md5(key_data.encode()).hexdigest()
            return f"stats:{prefix}:{key_hash}"
        return f"stats:{key_data}"

    def get(self, key: str) -> Optional[Any]:
        """
        キャッシュから値を取得（L1 → L2の順で検索）。

        Args:
            key: キャッシュキー

        Returns:
            キャッシュされた値、またはNone
        """
        # L1キャッシュをチェック
        if key in self._l1_cache:
            logger.debug(f"L1 cache hit: {key[:50]}...")
            return self._l1_cache[key]

        # L2キャッシュをチェック
        if self.flask_cache:
            value = self.flask_cache.get(key)
            if value is not None:
                logger.debug(f"L2 cache hit: {key[:50]}...")
                # L1キャッシュにも保存
                self._l1_cache[key] = value
                return value

        return None

    def set(self, key: str, value: Any, ttl: int = STATS_CACHE_TTL):
        """
        キャッシュに値を設定（L1とL2の両方）。

        Args:
            key: キャッシュキー
            value: キャッシュする値
            ttl: Time-to-Live（秒）
        """
        # L1キャッシュに保存
        self._l1_cache[key] = value

        # L2キャッシュに保存
        if self.flask_cache:
            try:
                self.flask_cache.set(key, value, timeout=ttl)
                logger.debug(f"Cached to L2: {key[:50]}... (TTL={ttl}s)")
            except Exception as e:
                logger.warning(f"Failed to cache to L2: {e}")

    def get_or_compute(
        self,
        key: str,
        compute_fn,
        ttl: int = STATS_CACHE_TTL
    ) -> Any:
        """
        キャッシュから取得、なければ計算してキャッシュ。

        Args:
            key: キャッシュキー
            compute_fn: キャッシュミス時に呼び出す計算関数
            ttl: Time-to-Live（秒）

        Returns:
            キャッシュされた値または計算結果
        """
        cached = self.get(key)
        if cached is not None:
            return cached

        # キャッシュミス - 計算して保存
        logger.debug(f"Cache miss, computing: {key[:50]}...")
        value = compute_fn()
        self.set(key, value, ttl)
        return value

    def invalidate_for_race(self, race_id: int):
        """
        特定レースに関連するキャッシュを無効化。

        レース結果が確定した時に呼び出す。
        L2キャッシュのみ無効化（L1はリクエスト終了で自動クリア）

        Args:
            race_id: レースID
        """
        if self.flask_cache:
            # パターンベースの無効化はRedisでのみ効率的
            # SimpleCacheの場合はTTL expireに依存
            logger.info(f"Cache invalidation requested for race_id={race_id}")
            # Note: Flask-Cachingはdelete_manyをサポートしていないため、
            # 実際の実装ではRedis SCAN + DELを使うか、TTL expireに依存

    def get_stats_with_cache(
        self,
        entity_type: str,
        entity_ids: List[int],
        cutoff_date: datetime,
        compute_fn
    ) -> Dict[int, Dict]:
        """
        エンティティ統計をキャッシュ付きで取得。

        Args:
            entity_type: 'horse', 'jockey', または 'trainer'
            entity_ids: エンティティIDリスト
            cutoff_date: 統計計算の基準日
            compute_fn: キャッシュミス時の計算関数 (ids: List[int]) -> Dict[int, Dict]

        Returns:
            {entity_id: {stats...}} の辞書
        """
        if not entity_ids:
            return {}

        cache_key = self._make_cache_key(entity_type, entity_ids, cutoff_date)

        def compute():
            return compute_fn(entity_ids)

        return self.get_or_compute(cache_key, compute)


# グローバルキャッシュインスタンス（Flask app初期化時に設定）
_stats_cache: Optional[StatisticsCache] = None


def init_stats_cache(flask_cache=None) -> StatisticsCache:
    """
    統計キャッシュを初期化する。

    Args:
        flask_cache: Flask-Caching instance

    Returns:
        StatisticsCache instance
    """
    global _stats_cache
    _stats_cache = StatisticsCache(flask_cache)
    logger.info("Statistics cache initialized")
    return _stats_cache


def get_stats_cache() -> StatisticsCache:
    """
    統計キャッシュインスタンスを取得する。

    Returns:
        StatisticsCache instance
    """
    global _stats_cache
    if _stats_cache is None:
        # フォールバック: L1キャッシュのみ
        _stats_cache = StatisticsCache()
        logger.warning("Statistics cache not initialized, using L1-only mode")
    return _stats_cache
