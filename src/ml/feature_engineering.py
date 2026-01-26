"""
Feature engineering for horse racing prediction.

このモジュールは競馬予想のための特徴量を抽出します。
- 馬の過去成績統計（距離別・馬場別・競馬場別）
- 騎手・調教師の統計
- レース固有の特徴（距離、馬場、天候など）
- 最近のパフォーマンストレンド

This module extracts features from race entries and historical data
for machine learning model training and prediction.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sqlalchemy import func, case
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.orm.exc import DetachedInstanceError

from src.data.models import (
    Horse, Jockey, Trainer, Race, RaceEntry, RaceResult, Track, db
)
from src.ml.constants import (
    LOOKBACK_DAYS_DEFAULT,
    RECENT_RACES_COUNT,
    MIN_SAMPLE_SIZE_FOR_STATS,
    DISTANCE_BUCKETS,
    RACE_CLASS_HIERARCHY
)
from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)


class FeatureExtractor:
    """
    Extract features from race entries for ML prediction.

    Features include:
    - Horse historical statistics (win rate, place rate by distance/surface/track)
    - Jockey and trainer statistics
    - Race-specific features (distance, surface, track condition, weather)
    - Horse weight and jockey weight
    - Recent performance trends
    """

    def __init__(self, session: Session, lookback_days: int = LOOKBACK_DAYS_DEFAULT):
        """
        Initialize feature extractor.

        Args:
            session: SQLAlchemy database session
            lookback_days: Number of days to look back for historical stats
                          (default: LOOKBACK_DAYS_DEFAULT = 730 days / 2 years)
        """
        self.session = session
        self.lookback_days = lookback_days
        self._stats_cache = {}  # リクエスト内キャッシュ

    def _get_distance_bucket(self, distance: Optional[int]) -> str:
        """
        距離を厳密なバケットに分類する。

        Args:
            distance: 距離（メートル）

        Returns:
            バケット名（'sprint', 'short', 'mile', 'intermediate', 'long', 'extended', 'other'）
        """
        if distance is None:
            return 'other'

        for bucket_name, distances in DISTANCE_BUCKETS.items():
            # 各バケット内の距離と±50m以内ならマッチ
            if any(abs(distance - d) <= 50 for d in distances):
                return bucket_name
        return 'other'

    def _prefetch_all_stats(
        self,
        horse_ids: List[int],
        jockey_ids: List[int],
        trainer_ids: List[int],
        cutoff_date: datetime
    ) -> Dict[str, Dict]:
        """
        全エンティティの統計を一括取得する（N+1クエリ問題の解決）。

        3クエリで馬・騎手・調教師の全統計を取得。
        従来のO(N)クエリからO(1)クエリに最適化。

        Args:
            horse_ids: 馬IDリスト
            jockey_ids: 騎手IDリスト
            trainer_ids: 調教師IDリスト
            cutoff_date: 統計計算の基準日

        Returns:
            {
                'horse': {horse_id: {'total_races': N, 'wins': M, ...}},
                'jockey': {jockey_id: {...}},
                'trainer': {trainer_id: {...}}
            }
        """
        lookback_date = cutoff_date - timedelta(days=self.lookback_days)

        stats = {
            'horse': {},
            'jockey': {},
            'trainer': {}
        }

        # 馬の統計を一括取得
        if horse_ids:
            horse_stats_query = self.session.query(
                RaceEntry.horse_id,
                func.count(RaceEntry.id).label('total_races'),
                func.sum(case((RaceResult.finish_position == 1, 1), else_=0)).label('wins'),
                func.sum(case((RaceResult.finish_position <= 3, 1), else_=0)).label('places'),
                func.avg(RaceResult.finish_position).label('avg_position')
            ).join(Race).join(RaceResult).filter(
                RaceEntry.horse_id.in_(horse_ids),
                Race.race_date < cutoff_date.date(),
                Race.race_date >= lookback_date.date(),
                Race.status == 'completed'
            ).group_by(RaceEntry.horse_id).all()

            for row in horse_stats_query:
                total = row.total_races or 0
                stats['horse'][row.horse_id] = {
                    'total_races': total,
                    'wins': row.wins or 0,
                    'places': row.places or 0,
                    'avg_position': float(row.avg_position) if row.avg_position else 0.0,
                    'win_rate': (row.wins or 0) / total if total > 0 else 0.0,
                    'place_rate': (row.places or 0) / total if total > 0 else 0.0
                }

        # 騎手の統計を一括取得
        if jockey_ids:
            jockey_stats_query = self.session.query(
                RaceEntry.jockey_id,
                func.count(RaceEntry.id).label('total_races'),
                func.sum(case((RaceResult.finish_position == 1, 1), else_=0)).label('wins'),
                func.sum(case((RaceResult.finish_position <= 3, 1), else_=0)).label('places')
            ).join(Race).join(RaceResult).filter(
                RaceEntry.jockey_id.in_(jockey_ids),
                Race.race_date < cutoff_date.date(),
                Race.race_date >= lookback_date.date(),
                Race.status == 'completed'
            ).group_by(RaceEntry.jockey_id).all()

            for row in jockey_stats_query:
                total = row.total_races or 0
                stats['jockey'][row.jockey_id] = {
                    'total_races': total,
                    'wins': row.wins or 0,
                    'places': row.places or 0,
                    'win_rate': (row.wins or 0) / total if total > 0 else 0.0,
                    'place_rate': (row.places or 0) / total if total > 0 else 0.0
                }

        # 調教師の統計を一括取得
        if trainer_ids:
            trainer_stats_query = self.session.query(
                Horse.trainer_id,
                func.count(RaceEntry.id).label('total_races'),
                func.sum(case((RaceResult.finish_position == 1, 1), else_=0)).label('wins'),
                func.sum(case((RaceResult.finish_position <= 3, 1), else_=0)).label('places')
            ).join(RaceEntry, RaceEntry.horse_id == Horse.id).join(
                Race, RaceEntry.race_id == Race.id
            ).join(RaceResult).filter(
                Horse.trainer_id.in_(trainer_ids),
                Race.race_date < cutoff_date.date(),
                Race.race_date >= lookback_date.date(),
                Race.status == 'completed'
            ).group_by(Horse.trainer_id).all()

            for row in trainer_stats_query:
                total = row.total_races or 0
                stats['trainer'][row.trainer_id] = {
                    'total_races': total,
                    'wins': row.wins or 0,
                    'places': row.places or 0,
                    'win_rate': (row.wins or 0) / total if total > 0 else 0.0,
                    'place_rate': (row.places or 0) / total if total > 0 else 0.0
                }

        logger.debug(f"Prefetched stats: {len(stats['horse'])} horses, {len(stats['jockey'])} jockeys, {len(stats['trainer'])} trainers")
        return stats

    def extract_features_for_race(self, race_id: int) -> pd.DataFrame:
        """
        Extract features for all entries in a race.

        最適化: 全エンティティの統計を3クエリで一括取得（N+1問題の解決）

        Args:
            race_id: Database race ID

        Returns:
            DataFrame with features for each horse in the race
        """
        logger.info(f"Extracting features for race_id={race_id}")

        # Get race with eager loading of track relationship
        race = self.session.query(Race).options(
            joinedload(Race.track)
        ).filter_by(id=race_id).first()

        if not race:
            raise ValueError(f"Race with id={race_id} not found")

        # Get entries with eager loading of related objects
        entries = self.session.query(RaceEntry).options(
            joinedload(RaceEntry.horse).joinedload(Horse.trainer),
            joinedload(RaceEntry.jockey),
            joinedload(RaceEntry.result)
        ).filter_by(race_id=race_id).all()

        if not entries:
            logger.warning(f"No entries found for race_id={race_id}")
            return pd.DataFrame()

        # 全エンティティのIDを収集
        horse_ids = [e.horse_id for e in entries if e.horse_id]
        jockey_ids = [e.jockey_id for e in entries if e.jockey_id]
        trainer_ids = [e.horse.trainer_id for e in entries if e.horse and e.horse.trainer_id]

        # 統計を一括取得（3クエリのみ）
        cutoff_date = datetime.utcnow()
        prefetched_stats = self._prefetch_all_stats(
            horse_ids, jockey_ids, trainer_ids, cutoff_date
        )

        # ペア統計を一括取得（Phase 3追加）
        pairs = [(e.horse_id, e.jockey_id) for e in entries if e.horse_id and e.jockey_id]
        prefetched_pair_stats = self._prefetch_pair_stats(pairs, cutoff_date)

        # Extract features for each entry with odds-based features
        features_list = []
        entries_with_odds = []

        for entry in entries:
            features = self._extract_entry_features(
                entry, race,
                prefetched_stats=prefetched_stats
            )

            # ペア統計を追加（Phase 3追加）
            pair_key = (entry.horse_id, entry.jockey_id)
            if pair_key in prefetched_pair_stats:
                pair_stats = prefetched_pair_stats[pair_key]
                horse_stats = prefetched_stats.get('horse', {}).get(entry.horse_id, {})
                jockey_stats = prefetched_stats.get('jockey', {}).get(entry.jockey_id, {})

                features['pair_total_races'] = pair_stats['total_races']
                features['pair_win_rate'] = pair_stats['win_rate']
                features['pair_place_rate'] = pair_stats['place_rate']
                features['pair_is_first_ride'] = 1 if pair_stats['total_races'] == 0 else 0

                # シナジー計算
                horse_win_rate = horse_stats.get('win_rate', 0.0)
                jockey_win_rate = jockey_stats.get('win_rate', 0.0)
                expected = (horse_win_rate + jockey_win_rate) / 2 if (horse_win_rate + jockey_win_rate) > 0 else 0.0
                features['pair_synergy'] = pair_stats['win_rate'] - expected
            else:
                features['pair_total_races'] = 0
                features['pair_win_rate'] = 0.0
                features['pair_place_rate'] = 0.0
                features['pair_synergy'] = 0.0
                features['pair_is_first_ride'] = 1

            # For prediction, use morning_odds as proxy for final_odds
            odds = entry.morning_odds if entry.morning_odds else 0.0
            features['final_odds'] = float(odds)

            features_list.append(features)
            entries_with_odds.append({
                'features': features,
                'odds': odds or 999.0  # High value for missing
            })

        # Calculate odds_rank within race (1 = lowest odds = most favored)
        sorted_entries = sorted(entries_with_odds, key=lambda x: x['odds'])
        for rank, entry_data in enumerate(sorted_entries, 1):
            entry_data['features']['odds_rank'] = float(rank)

        df = pd.DataFrame([e['features'] for e in entries_with_odds])
        logger.info(f"Extracted {len(df)} feature rows with {len(df.columns)} columns")

        return df

    def extract_features_for_training(
        self,
        min_date: Optional[datetime] = None,
        max_date: Optional[datetime] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract features and labels for all completed races in date range.

        Args:
            min_date: Minimum race date (inclusive)
            max_date: Maximum race date (inclusive)

        Returns:
            Tuple of (features DataFrame, labels Series with finish positions)
        """
        logger.info("Extracting features for training")

        # Query completed races with results and eager load track
        query = self.session.query(Race).options(
            joinedload(Race.track)
        ).filter(Race.status == 'completed')

        if min_date:
            query = query.filter(Race.race_date >= min_date)
        if max_date:
            query = query.filter(Race.race_date <= max_date)

        races = query.order_by(Race.race_date).all()
        logger.info(f"Found {len(races)} completed races for training")

        all_features = []
        all_labels = []

        for race in races:
            try:
                # Get entries with results and eager load relationships
                entries = self.session.query(RaceEntry).options(
                    joinedload(RaceEntry.horse).joinedload(Horse.trainer),
                    joinedload(RaceEntry.jockey),
                    joinedload(RaceEntry.result)
                ).filter_by(race_id=race.id).all()

                # Collect race-level data for odds ranking
                race_entries_with_odds = []
                for entry in entries:
                    # Skip if no result
                    if not entry.result:
                        continue

                    # Extract features (without using future data)
                    # Convert date to datetime for as_of_date parameter
                    as_of_datetime = datetime.combine(race.race_date, datetime.min.time())
                    features = self._extract_entry_features(
                        entry, race, as_of_date=as_of_datetime
                    )

                    # Add final_odds from race result
                    final_odds = entry.result.final_odds
                    features['final_odds'] = float(final_odds) if final_odds else 0.0

                    # Label is finish position
                    label = entry.result.finish_position

                    if features and label:
                        race_entries_with_odds.append({
                            'features': features,
                            'label': label,
                            'final_odds': final_odds or 999.0  # Use high value for missing odds
                        })

                # Calculate odds_rank within race (1 = lowest odds = most favored)
                if race_entries_with_odds:
                    # Sort by odds to determine rank
                    sorted_entries = sorted(race_entries_with_odds, key=lambda x: x['final_odds'])
                    for rank, entry_data in enumerate(sorted_entries, 1):
                        entry_data['features']['odds_rank'] = float(rank)

                    # Add to all features
                    for entry_data in race_entries_with_odds:
                        all_features.append(entry_data['features'])
                        all_labels.append(entry_data['label'])

            except Exception as e:
                logger.error(f"Error extracting features for race {race.id}: {e}")
                continue

        if not all_features:
            logger.warning("No training features extracted")
            return pd.DataFrame(), pd.Series()

        df_features = pd.DataFrame(all_features)
        series_labels = pd.Series(all_labels, name='finish_position')

        logger.info(f"Extracted {len(df_features)} training samples with {len(df_features.columns)} features")

        return df_features, series_labels

    def _extract_entry_features(
        self,
        entry: RaceEntry,
        race: Race,
        as_of_date: Optional[datetime] = None,
        prefetched_stats: Optional[Dict] = None
    ) -> Dict:
        """
        Extract features for a single race entry.

        Args:
            entry: RaceEntry instance
            race: Race instance
            as_of_date: Calculate historical stats as of this date (for training)
                       If None, uses current date (for prediction)
            prefetched_stats: プリフェッチされた統計データ（N+1最適化用）

        Returns:
            Dictionary of features
        """
        features = {}

        # Basic identifiers
        features['race_id'] = race.id
        features['horse_id'] = entry.horse_id
        features['race_entry_id'] = entry.id

        # Race date as string for splitting (will be excluded from training features)
        features['race_date'] = str(race.race_date)

        # Race features
        features.update(self._extract_race_features(race))

        # Entry-specific features (explicitly convert to float)
        features['post_position'] = float(entry.post_position) if entry.post_position is not None else 0.0
        features['horse_number'] = float(entry.horse_number) if entry.horse_number is not None else 0.0
        features['weight'] = float(entry.weight) if entry.weight is not None else 0.0  # 斤量
        features['horse_weight'] = float(entry.horse_weight) if entry.horse_weight is not None else 0.0  # 馬体重
        features['horse_weight_change'] = float(entry.horse_weight_change) if entry.horse_weight_change is not None else 0.0  # 馬体重変化
        features['morning_odds'] = float(entry.morning_odds) if entry.morning_odds is not None else 0.0

        # Historical features (avoid data leakage in training)
        cutoff_date = as_of_date if as_of_date else datetime.utcnow()

        # Horse historical stats（プリフェッチ使用時は基本統計のみ取得）
        features.update(self._extract_horse_stats(
            entry.horse, race, cutoff_date,
            prefetched_stats=prefetched_stats
        ))

        # Jockey historical stats
        features.update(self._extract_jockey_stats(
            entry.jockey, race, cutoff_date,
            prefetched_stats=prefetched_stats
        ))

        # Trainer historical stats
        if entry.horse.trainer:
            features.update(self._extract_trainer_stats(
                entry.horse.trainer, race, cutoff_date,
                prefetched_stats=prefetched_stats
            ))

        # Recent performance trends（常に個別クエリ - 最近3レースのみ）
        features.update(self._extract_recent_performance(entry.horse, race, cutoff_date))

        # Class-specific stats（レースクラス別統計 - Phase 2追加）
        features.update(self._extract_class_stats(entry.horse, race, cutoff_date))

        # Jockey×Horse pair stats（騎手×馬ペア統計 - Phase 3追加）
        # Note: バッチモード使用時はextract_features_batchで別途処理
        if not prefetched_stats:
            # 非バッチモード時のみ個別計算
            horse_win_rate = features.get('horse_win_rate', 0.0)
            jockey_win_rate = features.get('jockey_win_rate', 0.0)
            features.update(self._extract_pair_stats(
                entry.horse, entry.jockey, cutoff_date,
                horse_win_rate=horse_win_rate,
                jockey_win_rate=jockey_win_rate
            ))

        return features

    def _extract_race_features(self, race: Race) -> Dict:
        """Extract features from race information."""
        features = {}

        features['distance'] = race.distance
        features['prize_money'] = race.prize_money or 0

        # Encode surface
        features['surface_turf'] = 1 if race.surface == 'turf' else 0
        features['surface_dirt'] = 1 if race.surface == 'dirt' else 0

        # Encode track condition
        track_conditions = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
        features['track_condition'] = track_conditions.get(race.track_condition, 0)

        # Encode weather
        weather_map = {'晴': 0, '曇': 1, '雨': 2, '雪': 3}
        features['weather'] = weather_map.get(race.weather, 0)

        # Encode race class
        class_map = {'G1': 5, 'G2': 4, 'G3': 3, 'OP': 2, '1600万': 1, '1000万': 0, '500万': -1, '未勝利': -2, '新馬': -3}
        features['race_class'] = class_map.get(race.race_class, 0)

        # Track encoding (one-hot for major tracks)
        track_name = race.track.name if race.track else 'unknown'
        major_tracks = ['東京', '中山', '京都', '阪神', '中京', '新潟', '福島', '小倉', '札幌', '函館']
        for track in major_tracks:
            features[f'track_{track}'] = 1 if track_name == track else 0

        return features

    def _extract_horse_stats(
        self,
        horse: Horse,
        race: Race,
        cutoff_date: datetime,
        prefetched_stats: Optional[Dict] = None
    ) -> Dict:
        """
        Extract historical statistics for a horse.

        Includes:
        - Overall win/place rate
        - Distance-specific stats (厳密なバケット分類)
        - Surface-specific stats
        - Track-specific stats
        - Total races run

        Args:
            horse: Horse instance
            race: Race instance
            cutoff_date: 統計計算の基準日
            prefetched_stats: プリフェッチされた基本統計（N+1最適化用）
        """
        features = {}

        # プリフェッチデータから基本統計を取得（N+1最適化）
        if prefetched_stats and horse.id in prefetched_stats.get('horse', {}):
            stats = prefetched_stats['horse'][horse.id]
            features['horse_total_races'] = stats['total_races']
            features['horse_win_rate'] = stats['win_rate']
            features['horse_place_rate'] = stats['place_rate']
            features['horse_avg_finish_position'] = stats['avg_position']

            if stats['total_races'] == 0:
                # No historical data - use defaults for specific stats
                features['horse_distance_win_rate'] = 0.0
                features['horse_distance_races'] = 0
                features['horse_surface_win_rate'] = 0.0
                features['horse_surface_races'] = 0
                features['horse_track_win_rate'] = 0.0
                features['horse_track_races'] = 0
                return features
        else:
            # フォールバック: 従来のクエリ方式
            lookback_date = cutoff_date - timedelta(days=self.lookback_days)
            past_entries = self.session.query(RaceEntry).join(Race).join(RaceResult).filter(
                RaceEntry.horse_id == horse.id,
                Race.race_date < cutoff_date.date(),
                Race.race_date >= lookback_date.date(),
                Race.status == 'completed'
            ).all()

            total_races = len(past_entries)
            features['horse_total_races'] = total_races

            if total_races == 0:
                features['horse_win_rate'] = 0.0
                features['horse_place_rate'] = 0.0
                features['horse_avg_finish_position'] = 0.0
                features['horse_distance_win_rate'] = 0.0
                features['horse_distance_races'] = 0
                features['horse_surface_win_rate'] = 0.0
                features['horse_surface_races'] = 0
                features['horse_track_win_rate'] = 0.0
                features['horse_track_races'] = 0
                return features

            wins = sum(1 for e in past_entries if e.result and e.result.finish_position == 1)
            places = sum(1 for e in past_entries if e.result and e.result.finish_position is not None and e.result.finish_position <= 3)
            avg_position = np.mean([e.result.finish_position for e in past_entries if e.result and e.result.finish_position is not None])

            features['horse_win_rate'] = wins / total_races
            features['horse_place_rate'] = places / total_races
            features['horse_avg_finish_position'] = avg_position if not np.isnan(avg_position) else 0.0

        # 距離別・馬場別・競馬場別の統計は追加クエリで取得
        # （プリフェッチでは基本統計のみ取得するため）
        lookback_date = cutoff_date - timedelta(days=self.lookback_days)
        past_entries = self.session.query(RaceEntry).join(Race).join(RaceResult).filter(
            RaceEntry.horse_id == horse.id,
            Race.race_date < cutoff_date.date(),
            Race.race_date >= lookback_date.date(),
            Race.status == 'completed'
        ).all()

        # Distance-specific stats（厳密なバケット分類を使用）
        current_bucket = self._get_distance_bucket(race.distance)
        distance_entries = [
            e for e in past_entries
            if self._get_distance_bucket(e.race.distance) == current_bucket
        ]
        if distance_entries:
            distance_wins = sum(1 for e in distance_entries if e.result and e.result.finish_position == 1)
            features['horse_distance_win_rate'] = distance_wins / len(distance_entries)
            features['horse_distance_races'] = len(distance_entries)
        else:
            features['horse_distance_win_rate'] = 0.0
            features['horse_distance_races'] = 0

        # Surface-specific stats
        surface_entries = [e for e in past_entries if e.race.surface == race.surface]
        if surface_entries:
            surface_wins = sum(1 for e in surface_entries if e.result and e.result.finish_position == 1)
            features['horse_surface_win_rate'] = surface_wins / len(surface_entries)
            features['horse_surface_races'] = len(surface_entries)
        else:
            features['horse_surface_win_rate'] = 0.0
            features['horse_surface_races'] = 0

        # Track-specific stats
        track_entries = [e for e in past_entries if e.race.track_id == race.track_id]
        if track_entries:
            track_wins = sum(1 for e in track_entries if e.result and e.result.finish_position == 1)
            features['horse_track_win_rate'] = track_wins / len(track_entries)
            features['horse_track_races'] = len(track_entries)
        else:
            features['horse_track_win_rate'] = 0.0
            features['horse_track_races'] = 0

        return features

    def _extract_jockey_stats(
        self,
        jockey: Jockey,
        race: Race,
        cutoff_date: datetime,
        prefetched_stats: Optional[Dict] = None
    ) -> Dict:
        """
        Extract historical statistics for a jockey.

        Args:
            jockey: Jockey instance
            race: Race instance
            cutoff_date: 統計計算の基準日
            prefetched_stats: プリフェッチされた統計（N+1最適化用）
        """
        features = {}

        # プリフェッチデータから統計を取得（N+1最適化）
        if prefetched_stats and jockey.id in prefetched_stats.get('jockey', {}):
            stats = prefetched_stats['jockey'][jockey.id]
            features['jockey_total_races'] = stats['total_races']
            features['jockey_win_rate'] = stats['win_rate']
            features['jockey_place_rate'] = stats['place_rate']
            return features

        # フォールバック: 従来のクエリ方式
        lookback_date = cutoff_date - timedelta(days=self.lookback_days)

        past_entries = self.session.query(RaceEntry).join(Race).join(RaceResult).filter(
            RaceEntry.jockey_id == jockey.id,
            Race.race_date < cutoff_date.date(),
            Race.race_date >= lookback_date.date(),
            Race.status == 'completed'
        ).all()

        total_races = len(past_entries)
        features['jockey_total_races'] = total_races

        if total_races == 0:
            features['jockey_win_rate'] = 0.0
            features['jockey_place_rate'] = 0.0
            return features

        wins = sum(1 for e in past_entries if e.result and e.result.finish_position == 1)
        places = sum(1 for e in past_entries if e.result and e.result.finish_position is not None and e.result.finish_position <= 3)

        features['jockey_win_rate'] = wins / total_races
        features['jockey_place_rate'] = places / total_races

        return features

    def _extract_trainer_stats(
        self,
        trainer: Trainer,
        race: Race,
        cutoff_date: datetime,
        prefetched_stats: Optional[Dict] = None
    ) -> Dict:
        """
        Extract historical statistics for a trainer.

        Args:
            trainer: Trainer instance
            race: Race instance
            cutoff_date: 統計計算の基準日
            prefetched_stats: プリフェッチされた統計（N+1最適化用）
        """
        features = {}

        # プリフェッチデータから統計を取得（N+1最適化）
        if prefetched_stats and trainer.id in prefetched_stats.get('trainer', {}):
            stats = prefetched_stats['trainer'][trainer.id]
            features['trainer_total_races'] = stats['total_races']
            features['trainer_win_rate'] = stats['win_rate']
            features['trainer_place_rate'] = stats['place_rate']
            return features

        # フォールバック: 従来のクエリ方式
        lookback_date = cutoff_date - timedelta(days=self.lookback_days)

        # Query horses trained by this trainer
        past_entries = self.session.query(RaceEntry).join(Horse).join(Race).join(RaceResult).filter(
            Horse.trainer_id == trainer.id,
            Race.race_date < cutoff_date.date(),
            Race.race_date >= lookback_date.date(),
            Race.status == 'completed'
        ).all()

        total_races = len(past_entries)
        features['trainer_total_races'] = total_races

        if total_races == 0:
            features['trainer_win_rate'] = 0.0
            features['trainer_place_rate'] = 0.0
            return features

        wins = sum(1 for e in past_entries if e.result and e.result.finish_position == 1)
        places = sum(1 for e in past_entries if e.result and e.result.finish_position is not None and e.result.finish_position <= 3)

        features['trainer_win_rate'] = wins / total_races
        features['trainer_place_rate'] = places / total_races

        return features

    def _extract_recent_performance(self, horse: Horse, race: Race, cutoff_date: datetime) -> Dict:
        """
        Extract recent performance trends (last 3 races).

        Features include:
        - Average finish position in last N races
        - Best finish in last N races
        - Days since last race
        - Last race position
        - Average odds in recent races
        """
        features = {}

        # Get last N races before cutoff
        recent_races = 3
        past_entries = self.session.query(RaceEntry).join(Race).join(RaceResult).filter(
            RaceEntry.horse_id == horse.id,
            Race.race_date < cutoff_date.date(),
            Race.status == 'completed'
        ).order_by(Race.race_date.desc()).limit(recent_races).all()

        if not past_entries:
            features['recent_avg_position'] = 0.0
            features['recent_best_position'] = 0
            features['days_since_last_race'] = 999
            features['last_race_position'] = 0
            features['recent_avg_odds'] = 0.0
            return features

        positions = [e.result.finish_position for e in past_entries if e.result and e.result.finish_position is not None]
        odds = [e.result.final_odds for e in past_entries if e.result and e.result.final_odds is not None]

        if positions:
            features['recent_avg_position'] = np.mean(positions)
            features['recent_best_position'] = min(positions)
            features['last_race_position'] = positions[0]  # Most recent race
        else:
            features['recent_avg_position'] = 0.0
            features['recent_best_position'] = 0
            features['last_race_position'] = 0

        # Average odds in recent races (market sentiment)
        if odds:
            features['recent_avg_odds'] = np.mean(odds)
        else:
            features['recent_avg_odds'] = 0.0

        # Days since last race
        last_race_date = past_entries[0].race.race_date
        days_diff = (cutoff_date.date() - last_race_date).days
        features['days_since_last_race'] = days_diff

        return features

    def _get_race_class_level(self, race_class: Optional[str]) -> int:
        """
        レースクラスを数値レベルに変換する。

        Args:
            race_class: レースクラス文字列

        Returns:
            クラスレベル（G1=6, 新馬=-2）
        """
        if race_class is None:
            return 0
        return RACE_CLASS_HIERARCHY.get(race_class, 0)

    def _extract_class_stats(self, horse: Horse, race: Race, cutoff_date: datetime) -> Dict:
        """
        レースクラス別の統計を抽出する（Phase 2追加）。

        同クラス・上位クラスでのパフォーマンスを計算し、
        クラス昇格/降格時の予測精度を向上させる。

        Features:
        - horse_same_class_win_rate: 同クラスでの勝率
        - horse_same_class_races: 同クラス出走数
        - horse_higher_class_place_rate: 上位クラスでの連対率
        - horse_class_step: 前走からのクラス変動（正=昇格、負=降格）

        Args:
            horse: Horse instance
            race: Race instance（現在のレース）
            cutoff_date: 統計計算の基準日

        Returns:
            クラス別統計の辞書
        """
        features = {}
        current_class_level = self._get_race_class_level(race.race_class)

        # 過去のレース履歴を取得
        lookback_date = cutoff_date - timedelta(days=self.lookback_days)
        past_entries = self.session.query(RaceEntry).join(Race).join(RaceResult).filter(
            RaceEntry.horse_id == horse.id,
            Race.race_date < cutoff_date.date(),
            Race.race_date >= lookback_date.date(),
            Race.status == 'completed'
        ).order_by(Race.race_date.desc()).all()

        if not past_entries:
            features['horse_same_class_win_rate'] = 0.0
            features['horse_same_class_races'] = 0
            features['horse_higher_class_place_rate'] = 0.0
            features['horse_class_step'] = 0
            return features

        # 同クラスでの成績
        same_class_entries = [
            e for e in past_entries
            if self._get_race_class_level(e.race.race_class) == current_class_level
        ]
        if same_class_entries:
            same_class_wins = sum(
                1 for e in same_class_entries
                if e.result and e.result.finish_position == 1
            )
            features['horse_same_class_win_rate'] = same_class_wins / len(same_class_entries)
            features['horse_same_class_races'] = len(same_class_entries)
        else:
            features['horse_same_class_win_rate'] = 0.0
            features['horse_same_class_races'] = 0

        # 上位クラスでの成績（現在のクラスより上のレース）
        higher_class_entries = [
            e for e in past_entries
            if self._get_race_class_level(e.race.race_class) > current_class_level
        ]
        if higher_class_entries:
            higher_class_places = sum(
                1 for e in higher_class_entries
                if e.result and e.result.finish_position is not None and e.result.finish_position <= 3
            )
            features['horse_higher_class_place_rate'] = higher_class_places / len(higher_class_entries)
        else:
            features['horse_higher_class_place_rate'] = 0.0

        # 前走からのクラス変動
        last_race = past_entries[0]  # 最新のレース
        last_class_level = self._get_race_class_level(last_race.race.race_class)
        features['horse_class_step'] = current_class_level - last_class_level

        return features

    def _extract_pair_stats(
        self,
        horse: Horse,
        jockey: Jockey,
        cutoff_date: datetime,
        horse_win_rate: float = 0.0,
        jockey_win_rate: float = 0.0
    ) -> Dict:
        """
        騎手×馬ペアの統計を抽出する（Phase 3追加）。

        同じ騎手と馬の組み合わせでの過去成績を計算し、
        相性（シナジー）を評価する。

        Features:
        - pair_total_races: ペアでの出走数
        - pair_win_rate: ペアでの勝率
        - pair_place_rate: ペアでの連対率
        - pair_synergy: 期待値との乖離（相性指標）
        - pair_is_first_ride: 初騎乗フラグ

        Args:
            horse: Horse instance
            jockey: Jockey instance
            cutoff_date: 統計計算の基準日
            horse_win_rate: 馬の勝率（シナジー計算用）
            jockey_win_rate: 騎手の勝率（シナジー計算用）

        Returns:
            ペア統計の辞書
        """
        features = {}

        if not horse or not jockey:
            features['pair_total_races'] = 0
            features['pair_win_rate'] = 0.0
            features['pair_place_rate'] = 0.0
            features['pair_synergy'] = 0.0
            features['pair_is_first_ride'] = 1
            return features

        # ペアでの過去レースを取得
        lookback_date = cutoff_date - timedelta(days=self.lookback_days)
        pair_entries = self.session.query(RaceEntry).join(Race).join(RaceResult).filter(
            RaceEntry.horse_id == horse.id,
            RaceEntry.jockey_id == jockey.id,
            Race.race_date < cutoff_date.date(),
            Race.race_date >= lookback_date.date(),
            Race.status == 'completed'
        ).all()

        total_pair_races = len(pair_entries)
        features['pair_total_races'] = total_pair_races

        if total_pair_races == 0:
            # 初騎乗
            features['pair_win_rate'] = 0.0
            features['pair_place_rate'] = 0.0
            features['pair_synergy'] = 0.0
            features['pair_is_first_ride'] = 1
            return features

        # ペアでの成績
        pair_wins = sum(
            1 for e in pair_entries
            if e.result and e.result.finish_position == 1
        )
        pair_places = sum(
            1 for e in pair_entries
            if e.result and e.result.finish_position is not None and e.result.finish_position <= 3
        )

        pair_win_rate = pair_wins / total_pair_races
        pair_place_rate = pair_places / total_pair_races

        features['pair_win_rate'] = pair_win_rate
        features['pair_place_rate'] = pair_place_rate
        features['pair_is_first_ride'] = 0

        # シナジー計算: 実際の勝率 - 期待勝率
        # 期待勝率 = 馬と騎手の個別勝率の幾何平均（独立と仮定した場合）
        # シンプルに平均を使用
        expected_win_rate = (horse_win_rate + jockey_win_rate) / 2 if (horse_win_rate + jockey_win_rate) > 0 else 0.0
        features['pair_synergy'] = pair_win_rate - expected_win_rate

        return features

    def _prefetch_pair_stats(
        self,
        horse_jockey_pairs: List[Tuple[int, int]],
        cutoff_date: datetime
    ) -> Dict[Tuple[int, int], Dict]:
        """
        複数の騎手×馬ペアの統計を一括取得する（バッチ最適化用）。

        Args:
            horse_jockey_pairs: (horse_id, jockey_id) のタプルリスト
            cutoff_date: 統計計算の基準日

        Returns:
            {(horse_id, jockey_id): {'total_races': N, 'wins': M, ...}}
        """
        if not horse_jockey_pairs:
            return {}

        lookback_date = cutoff_date - timedelta(days=self.lookback_days)
        pair_stats = {}

        # SQLAlchemyでペア条件を構築
        # Note: 大量のペアがある場合はバッチ処理が必要かもしれない
        for horse_id, jockey_id in horse_jockey_pairs:
            cache_key = (horse_id, jockey_id)

            # 個別クエリ（ペア数が多くなければ問題ない）
            pair_query = self.session.query(
                func.count(RaceEntry.id).label('total_races'),
                func.sum(case((RaceResult.finish_position == 1, 1), else_=0)).label('wins'),
                func.sum(case((RaceResult.finish_position <= 3, 1), else_=0)).label('places')
            ).join(Race).join(RaceResult).filter(
                RaceEntry.horse_id == horse_id,
                RaceEntry.jockey_id == jockey_id,
                Race.race_date < cutoff_date.date(),
                Race.race_date >= lookback_date.date(),
                Race.status == 'completed'
            ).first()

            if pair_query:
                total = pair_query.total_races or 0
                pair_stats[cache_key] = {
                    'total_races': total,
                    'wins': pair_query.wins or 0,
                    'places': pair_query.places or 0,
                    'win_rate': (pair_query.wins or 0) / total if total > 0 else 0.0,
                    'place_rate': (pair_query.places or 0) / total if total > 0 else 0.0
                }
            else:
                pair_stats[cache_key] = {
                    'total_races': 0, 'wins': 0, 'places': 0,
                    'win_rate': 0.0, 'place_rate': 0.0
                }

        logger.debug(f"Prefetched pair stats for {len(pair_stats)} pairs")
        return pair_stats

    def extract_features_batch(self, race_ids: List[int]) -> Dict[int, pd.DataFrame]:
        """
        複数レースの特徴量を一括抽出する（Phase 3追加）。

        全レースのエンティティIDを収集し、統計を一括取得することで
        レース間で共有されるデータのクエリを最小化。

        Args:
            race_ids: レースIDのリスト

        Returns:
            {race_id: DataFrame} の辞書
        """
        logger.info(f"Batch extracting features for {len(race_ids)} races")

        if not race_ids:
            return {}

        # 全レースを取得
        races = self.session.query(Race).options(
            joinedload(Race.track)
        ).filter(Race.id.in_(race_ids)).all()
        race_map = {r.id: r for r in races}

        # 全エントリを取得
        all_entries = self.session.query(RaceEntry).options(
            joinedload(RaceEntry.horse).joinedload(Horse.trainer),
            joinedload(RaceEntry.jockey),
            joinedload(RaceEntry.result)
        ).filter(RaceEntry.race_id.in_(race_ids)).all()

        # レースごとにエントリをグループ化
        entries_by_race: Dict[int, List[RaceEntry]] = {}
        for entry in all_entries:
            if entry.race_id not in entries_by_race:
                entries_by_race[entry.race_id] = []
            entries_by_race[entry.race_id].append(entry)

        # 全エンティティのIDを収集
        all_horse_ids = set()
        all_jockey_ids = set()
        all_trainer_ids = set()
        all_pairs = set()

        for entry in all_entries:
            if entry.horse_id:
                all_horse_ids.add(entry.horse_id)
            if entry.jockey_id:
                all_jockey_ids.add(entry.jockey_id)
            if entry.horse and entry.horse.trainer_id:
                all_trainer_ids.add(entry.horse.trainer_id)
            if entry.horse_id and entry.jockey_id:
                all_pairs.add((entry.horse_id, entry.jockey_id))

        # 統計を一括取得（1回のprefetch）
        cutoff_date = datetime.utcnow()
        prefetched_stats = self._prefetch_all_stats(
            list(all_horse_ids),
            list(all_jockey_ids),
            list(all_trainer_ids),
            cutoff_date
        )

        # ペア統計を一括取得
        prefetched_pair_stats = self._prefetch_pair_stats(list(all_pairs), cutoff_date)

        logger.debug(f"Batch prefetched: {len(all_horse_ids)} horses, {len(all_jockey_ids)} jockeys, {len(all_pairs)} pairs")

        # 各レースの特徴量を抽出
        results = {}
        for race_id in race_ids:
            race = race_map.get(race_id)
            entries = entries_by_race.get(race_id, [])

            if not race or not entries:
                logger.warning(f"No data for race_id={race_id}")
                results[race_id] = pd.DataFrame()
                continue

            features_list = []
            entries_with_odds = []

            for entry in entries:
                features = self._extract_entry_features(
                    entry, race,
                    prefetched_stats=prefetched_stats
                )

                # ペア統計を追加
                pair_key = (entry.horse_id, entry.jockey_id)
                if pair_key in prefetched_pair_stats:
                    pair_stats = prefetched_pair_stats[pair_key]
                    horse_stats = prefetched_stats.get('horse', {}).get(entry.horse_id, {})
                    jockey_stats = prefetched_stats.get('jockey', {}).get(entry.jockey_id, {})

                    features['pair_total_races'] = pair_stats['total_races']
                    features['pair_win_rate'] = pair_stats['win_rate']
                    features['pair_place_rate'] = pair_stats['place_rate']
                    features['pair_is_first_ride'] = 1 if pair_stats['total_races'] == 0 else 0

                    # シナジー計算
                    horse_win_rate = horse_stats.get('win_rate', 0.0)
                    jockey_win_rate = jockey_stats.get('win_rate', 0.0)
                    expected = (horse_win_rate + jockey_win_rate) / 2 if (horse_win_rate + jockey_win_rate) > 0 else 0.0
                    features['pair_synergy'] = pair_stats['win_rate'] - expected
                else:
                    features['pair_total_races'] = 0
                    features['pair_win_rate'] = 0.0
                    features['pair_place_rate'] = 0.0
                    features['pair_synergy'] = 0.0
                    features['pair_is_first_ride'] = 1

                # オッズ処理
                odds = entry.morning_odds if entry.morning_odds else 0.0
                features['final_odds'] = float(odds)

                features_list.append(features)
                entries_with_odds.append({
                    'features': features,
                    'odds': odds or 999.0
                })

            # オッズランク計算
            sorted_entries = sorted(entries_with_odds, key=lambda x: x['odds'])
            for rank, entry_data in enumerate(sorted_entries, 1):
                entry_data['features']['odds_rank'] = float(rank)

            df = pd.DataFrame([e['features'] for e in entries_with_odds])
            results[race_id] = df

        logger.info(f"Batch extraction complete: {sum(len(df) for df in results.values())} total entries")
        return results


def save_features_to_csv(df: pd.DataFrame, filepath: str) -> None:
    """
    Save extracted features to CSV file.

    Args:
        df: DataFrame with features
        filepath: Output CSV file path
    """
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    logger.info(f"Saved {len(df)} feature rows to {filepath}")


def load_features_from_csv(filepath: str) -> pd.DataFrame:
    """
    Load features from CSV file.

    Args:
        filepath: Input CSV file path

    Returns:
        DataFrame with features
    """
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    logger.info(f"Loaded {len(df)} feature rows from {filepath}")
    return df
