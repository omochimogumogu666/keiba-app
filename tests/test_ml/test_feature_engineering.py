"""
Tests for feature engineering module.
"""
import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.web.app import create_app
from src.data.models import db, Horse, Jockey, Trainer, Track, Race, RaceEntry, RaceResult
from src.data.database import (
    get_or_create_horse,
    get_or_create_jockey,
    get_or_create_trainer,
    get_or_create_track
)
from src.ml.feature_engineering import FeatureExtractor, save_features_to_csv, load_features_from_csv


@pytest.fixture
def app():
    """Create test Flask app with in-memory database."""
    app = create_app('testing')
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


def _create_sample_data():
    """Helper function to create sample race data."""
    # Create track
    track = get_or_create_track('東京')

    # Create horses
    horse1 = get_or_create_horse('TEST001', 'テスト馬1', sex='牡')
    horse2 = get_or_create_horse('TEST002', 'テスト馬2', sex='牝')

    # Create jockey and trainer
    jockey1 = get_or_create_jockey('JOC001', 'テスト騎手1')
    jockey2 = get_or_create_jockey('JOC002', 'テスト騎手2')
    trainer1 = get_or_create_trainer('TRA001', 'テスト調教師1')

    # Update horses with trainer
    horse1.trainer_id = trainer1.id
    horse2.trainer_id = trainer1.id
    db.session.commit()

    # Create past race (for historical stats)
    past_race = Race(
        jra_race_id='2024010101',
        track_id=track.id,
        race_date=(datetime.utcnow() - timedelta(days=30)).date(),
        race_number=1,
        race_name='過去レース',
        distance=1600,
        surface='turf',
        track_condition='良',
        weather='晴',
        race_class='1000万',
        status='completed'
    )
    db.session.add(past_race)
    db.session.commit()

    # Create past race entries
    entry1 = RaceEntry(
        race_id=past_race.id,
        horse_id=horse1.id,
        jockey_id=jockey1.id,
        post_position=1,
        horse_number=1,
        weight=55.0,
        horse_weight=480,
        horse_weight_change=0
    )
    entry2 = RaceEntry(
        race_id=past_race.id,
        horse_id=horse2.id,
        jockey_id=jockey2.id,
        post_position=2,
        horse_number=2,
        weight=54.0,
        horse_weight=460,
        horse_weight_change=2
    )
    db.session.add_all([entry1, entry2])
    db.session.commit()

    # Create race results
    result1 = RaceResult(
        race_entry_id=entry1.id,
        finish_position=1,
        finish_time=96.5,
        final_odds=3.2,
        popularity=2
    )
    result2 = RaceResult(
        race_entry_id=entry2.id,
        finish_position=2,
        finish_time=96.8,
        final_odds=5.1,
        popularity=3
    )
    db.session.add_all([result1, result2])
    db.session.commit()

    # Create current race (for prediction)
    current_race = Race(
        jra_race_id='2024020101',
        track_id=track.id,
        race_date=datetime.utcnow().date(),
        race_number=1,
        race_name='現在レース',
        distance=1600,
        surface='turf',
        track_condition='良',
        weather='晴',
        race_class='1000万',
        status='upcoming'
    )
    db.session.add(current_race)
    db.session.commit()

    # Create current race entries
    current_entry1 = RaceEntry(
        race_id=current_race.id,
        horse_id=horse1.id,
        jockey_id=jockey1.id,
        post_position=1,
        horse_number=1,
        weight=55.0,
        horse_weight=482,
        horse_weight_change=2,
        morning_odds=3.5
    )
    current_entry2 = RaceEntry(
        race_id=current_race.id,
        horse_id=horse2.id,
        jockey_id=jockey2.id,
        post_position=2,
        horse_number=2,
        weight=54.0,
        horse_weight=462,
        horse_weight_change=2,
        morning_odds=5.0
    )
    db.session.add_all([current_entry1, current_entry2])
    db.session.commit()

    return {
        'past_race_id': past_race.id,
        'current_race_id': current_race.id,
        'horse1_id': horse1.id,
        'horse2_id': horse2.id,
        'jockey1_id': jockey1.id,
        'jockey2_id': jockey2.id,
        'trainer1_id': trainer1.id,
        'track_id': track.id
    }


@pytest.mark.unit
class TestFeatureExtractor:
    """Test FeatureExtractor class."""

    def test_initialization(self, app):
        """Test FeatureExtractor initialization."""
        with app.app_context():
            extractor = FeatureExtractor(db.session, lookback_days=365)
            assert extractor.session == db.session
            assert extractor.lookback_days == 365

    def test_extract_features_for_race(self, app):
        """Test feature extraction for a race."""
        with app.app_context():
            data_ids = _create_sample_data()
            extractor = FeatureExtractor(db.session)

            df = extractor.extract_features_for_race(data_ids['current_race_id'])

            assert not df.empty
            assert len(df) == 2  # Two horses in the race

            # Check for expected columns
            expected_columns = [
                'race_id', 'horse_id', 'distance', 'surface_turf',
                'horse_win_rate', 'jockey_win_rate', 'trainer_win_rate',
                'horse_weight', 'weight', 'recent_avg_position'
            ]
            for col in expected_columns:
                assert col in df.columns, f"Missing column: {col}"

    def test_extract_features_for_race_invalid_id(self, app):
        """Test feature extraction with invalid race ID."""
        with app.app_context():
            extractor = FeatureExtractor(db.session)

            with pytest.raises(ValueError, match="not found"):
                extractor.extract_features_for_race(999999)

    def test_extract_features_for_training(self, app):
        """Test feature extraction for training."""
        with app.app_context():
            _create_sample_data()
            extractor = FeatureExtractor(db.session)

            X, y = extractor.extract_features_for_training()

            assert isinstance(X, pd.DataFrame)
            assert isinstance(y, pd.Series)

            # Should have at least the past race entries
            assert len(X) >= 2
            assert len(y) == len(X)

            # Check labels are finish positions
            assert all(y > 0)
            assert all(y <= 18)  # Max 18 horses in JRA race

    def test_extract_race_features(self, app):
        """Test race feature extraction."""
        with app.app_context():
            data_ids = _create_sample_data()
            extractor = FeatureExtractor(db.session)

            race = db.session.query(Race).filter_by(id=data_ids['current_race_id']).first()
            features = extractor._extract_race_features(race)

            assert features['distance'] == 1600
            assert features['surface_turf'] == 1
            assert features['surface_dirt'] == 0
            assert 'track_condition' in features
            assert 'weather' in features
            assert 'track_東京' in features
            assert features['track_東京'] == 1

    def test_extract_horse_stats(self, app):
        """Test horse statistics extraction."""
        with app.app_context():
            data_ids = _create_sample_data()
            extractor = FeatureExtractor(db.session)

            horse = db.session.query(Horse).filter_by(id=data_ids['horse1_id']).first()
            race = db.session.query(Race).filter_by(id=data_ids['current_race_id']).first()
            cutoff_date = datetime.utcnow()

            features = extractor._extract_horse_stats(horse, race, cutoff_date)

            assert 'horse_total_races' in features
            assert 'horse_win_rate' in features
            assert 'horse_place_rate' in features
            assert features['horse_total_races'] >= 0

            # Horse1 won the past race, so win rate should be > 0
            if features['horse_total_races'] > 0:
                assert features['horse_win_rate'] > 0

    def test_extract_horse_stats_no_history(self, app):
        """Test horse stats with no historical data."""
        with app.app_context():
            data_ids = _create_sample_data()
            extractor = FeatureExtractor(db.session)

            # Create new horse with no race history
            new_horse = get_or_create_horse('NEWH001', '新馬テスト', sex='牡')
            race = db.session.query(Race).filter_by(id=data_ids['current_race_id']).first()
            cutoff_date = datetime.utcnow()

            features = extractor._extract_horse_stats(new_horse, race, cutoff_date)

            assert features['horse_total_races'] == 0
            assert features['horse_win_rate'] == 0.0
            assert features['horse_place_rate'] == 0.0

    def test_extract_jockey_stats(self, app):
        """Test jockey statistics extraction."""
        with app.app_context():
            data_ids = _create_sample_data()
            extractor = FeatureExtractor(db.session)

            jockey = db.session.query(Jockey).filter_by(id=data_ids['jockey1_id']).first()
            race = db.session.query(Race).filter_by(id=data_ids['current_race_id']).first()
            cutoff_date = datetime.utcnow()

            features = extractor._extract_jockey_stats(jockey, race, cutoff_date)

            assert 'jockey_total_races' in features
            assert 'jockey_win_rate' in features
            assert 'jockey_place_rate' in features

    def test_extract_trainer_stats(self, app):
        """Test trainer statistics extraction."""
        with app.app_context():
            data_ids = _create_sample_data()
            extractor = FeatureExtractor(db.session)

            trainer = db.session.query(Trainer).filter_by(id=data_ids['trainer1_id']).first()
            race = db.session.query(Race).filter_by(id=data_ids['current_race_id']).first()
            cutoff_date = datetime.utcnow()

            features = extractor._extract_trainer_stats(trainer, race, cutoff_date)

            assert 'trainer_total_races' in features
            assert 'trainer_win_rate' in features
            assert 'trainer_place_rate' in features

    def test_extract_recent_performance(self, app):
        """Test recent performance extraction."""
        with app.app_context():
            data_ids = _create_sample_data()
            extractor = FeatureExtractor(db.session)

            horse = db.session.query(Horse).filter_by(id=data_ids['horse1_id']).first()
            race = db.session.query(Race).filter_by(id=data_ids['current_race_id']).first()
            cutoff_date = datetime.utcnow()

            features = extractor._extract_recent_performance(horse, race, cutoff_date)

            assert 'recent_avg_position' in features
            assert 'recent_best_position' in features
            assert 'days_since_last_race' in features

            # Horse has past race, so should have data
            if features['recent_avg_position'] > 0:
                assert features['recent_best_position'] > 0
                assert features['days_since_last_race'] > 0

    def test_avoid_data_leakage(self, app):
        """Test that as_of_date prevents data leakage in training."""
        with app.app_context():
            data_ids = _create_sample_data()
            extractor = FeatureExtractor(db.session)

            horse = db.session.query(Horse).filter_by(id=data_ids['horse1_id']).first()
            past_race = db.session.query(Race).filter_by(id=data_ids['past_race_id']).first()

            # Extract features as of past race date
            past_date = datetime.combine(past_race.race_date, datetime.min.time())

            # Should not include the past race itself when extracting for that race
            entry = db.session.query(RaceEntry).filter_by(
                race_id=past_race.id,
                horse_id=horse.id
            ).first()

            features = extractor._extract_entry_features(entry, past_race, as_of_date=past_date)

            # With no earlier races, should have 0 total races
            assert features['horse_total_races'] == 0


@pytest.mark.unit
def test_save_and_load_features(app, tmp_path):
    """Test saving and loading features to/from CSV."""
    with app.app_context():
        data_ids = _create_sample_data()
        extractor = FeatureExtractor(db.session)

        df = extractor.extract_features_for_race(data_ids['current_race_id'])

        # Save to CSV
        filepath = tmp_path / "test_features.csv"
        save_features_to_csv(df, str(filepath))

        assert filepath.exists()

        # Load from CSV
        df_loaded = load_features_from_csv(str(filepath))

        assert len(df_loaded) == len(df)
        assert list(df_loaded.columns) == list(df.columns)
