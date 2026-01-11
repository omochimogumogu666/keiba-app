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
from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.orm.exc import DetachedInstanceError

from src.data.models import (
    Horse, Jockey, Trainer, Race, RaceEntry, RaceResult, Track, db
)
from src.ml.constants import (
    LOOKBACK_DAYS_DEFAULT,
    RECENT_RACES_COUNT,
    MIN_SAMPLE_SIZE_FOR_STATS
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

    def extract_features_for_race(self, race_id: int) -> pd.DataFrame:
        """
        Extract features for all entries in a race.

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

        # Extract features for each entry
        features_list = []
        for entry in entries:
            features = self._extract_entry_features(entry, race)
            features_list.append(features)

        df = pd.DataFrame(features_list)
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

                    # Label is finish position
                    label = entry.result.finish_position

                    if features and label:
                        all_features.append(features)
                        all_labels.append(label)

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
        as_of_date: Optional[datetime] = None
    ) -> Dict:
        """
        Extract features for a single race entry.

        Args:
            entry: RaceEntry instance
            race: Race instance
            as_of_date: Calculate historical stats as of this date (for training)
                       If None, uses current date (for prediction)

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

        # Horse historical stats
        features.update(self._extract_horse_stats(entry.horse, race, cutoff_date))

        # Jockey historical stats
        features.update(self._extract_jockey_stats(entry.jockey, race, cutoff_date))

        # Trainer historical stats
        if entry.horse.trainer:
            features.update(self._extract_trainer_stats(entry.horse.trainer, race, cutoff_date))

        # Recent performance trends
        features.update(self._extract_recent_performance(entry.horse, race, cutoff_date))

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

    def _extract_horse_stats(self, horse: Horse, race: Race, cutoff_date: datetime) -> Dict:
        """
        Extract historical statistics for a horse.

        Includes:
        - Overall win/place rate
        - Distance-specific stats
        - Surface-specific stats
        - Track-specific stats
        - Total races run
        """
        features = {}

        # Query historical race entries before cutoff date
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
            # No historical data - use defaults
            features['horse_win_rate'] = 0.0
            features['horse_place_rate'] = 0.0
            features['horse_avg_finish_position'] = 0.0
            features['horse_distance_win_rate'] = 0.0
            features['horse_surface_win_rate'] = 0.0
            features['horse_track_win_rate'] = 0.0
            return features

        # Overall statistics
        wins = sum(1 for e in past_entries if e.result and e.result.finish_position == 1)
        places = sum(1 for e in past_entries if e.result and e.result.finish_position is not None and e.result.finish_position <= 3)
        avg_position = np.mean([e.result.finish_position for e in past_entries if e.result and e.result.finish_position is not None])

        features['horse_win_rate'] = wins / total_races
        features['horse_place_rate'] = places / total_races
        features['horse_avg_finish_position'] = avg_position if not np.isnan(avg_position) else 0.0

        # Distance-specific stats (within ±200m)
        distance_entries = [
            e for e in past_entries
            if e.race.distance is not None and race.distance is not None
            and abs(e.race.distance - race.distance) <= 200
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

    def _extract_jockey_stats(self, jockey: Jockey, race: Race, cutoff_date: datetime) -> Dict:
        """Extract historical statistics for a jockey."""
        features = {}

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

    def _extract_trainer_stats(self, trainer: Trainer, race: Race, cutoff_date: datetime) -> Dict:
        """Extract historical statistics for a trainer."""
        features = {}

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
            return features

        positions = [e.result.finish_position for e in past_entries if e.result and e.result.finish_position is not None]

        if positions:
            features['recent_avg_position'] = np.mean(positions)
            features['recent_best_position'] = min(positions)
        else:
            features['recent_avg_position'] = 0.0
            features['recent_best_position'] = 0

        # Days since last race
        last_race_date = past_entries[0].race.race_date
        days_diff = (cutoff_date.date() - last_race_date).days
        features['days_since_last_race'] = days_diff

        return features


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
