"""
Pytest configuration and fixtures for testing.
"""
import pytest
import sys
from pathlib import Path
from datetime import datetime, date

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.web.app import create_app
from src.data.models import (
    db, Horse, Jockey, Trainer, Track, Race, RaceEntry, RaceResult, Prediction
)


@pytest.fixture(scope='session')
def app():
    """Create and configure a test Flask application."""
    app = create_app('testing')

    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()


@pytest.fixture(scope='function')
def test_db(app):
    """Create a fresh database for each test."""
    with app.app_context():
        # Create tables
        db.create_all()

        yield db

        # Cleanup
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app, test_db):
    """Create a test client for the Flask app."""
    return app.test_client()


@pytest.fixture
def runner(app):
    """Create a test CLI runner."""
    return app.test_cli_runner()


# Sample data fixtures

@pytest.fixture
def sample_track(test_db):
    """Create a sample track."""
    track = Track(
        name='東京',
        location='東京都府中市',
        surface_types=['turf', 'dirt']
    )
    test_db.session.add(track)
    test_db.session.commit()
    return track


@pytest.fixture
def sample_jockey(test_db):
    """Create a sample jockey."""
    jockey = Jockey(
        jra_jockey_id='J12345',
        name='テスト騎手',
        weight=52.0
    )
    test_db.session.add(jockey)
    test_db.session.commit()
    return jockey


@pytest.fixture
def sample_trainer(test_db):
    """Create a sample trainer."""
    trainer = Trainer(
        jra_trainer_id='T12345',
        name='テスト調教師',
        stable='美浦'
    )
    test_db.session.add(trainer)
    test_db.session.commit()
    return trainer


@pytest.fixture
def sample_horse(test_db, sample_trainer):
    """Create a sample horse."""
    horse = Horse(
        jra_horse_id='H12345',
        name='テストホース',
        birth_date=date(2020, 3, 15),
        sex='牡',
        trainer_id=sample_trainer.id
    )
    test_db.session.add(horse)
    test_db.session.commit()
    return horse


@pytest.fixture
def sample_race(test_db, sample_track):
    """Create a sample race."""
    race = Race(
        jra_race_id='2024010101',
        track_id=sample_track.id,
        race_date=date(2024, 1, 1),
        race_number=1,
        race_name='テストレース',
        distance=1600,
        surface='turf',
        track_condition='良',
        weather='晴',
        race_class='OP',
        prize_money=10000000,
        status='upcoming'
    )
    test_db.session.add(race)
    test_db.session.commit()
    return race


@pytest.fixture
def sample_race_entry(test_db, sample_race, sample_horse, sample_jockey):
    """Create a sample race entry."""
    entry = RaceEntry(
        race_id=sample_race.id,
        horse_id=sample_horse.id,
        jockey_id=sample_jockey.id,
        post_position=1,
        horse_number=1,
        weight=57.0,
        horse_weight=500,
        horse_weight_change=-2,
        morning_odds=3.5
    )
    test_db.session.add(entry)
    test_db.session.commit()
    return entry


@pytest.fixture
def sample_race_result(test_db, sample_race_entry):
    """Create a sample race result."""
    result = RaceResult(
        race_entry_id=sample_race_entry.id,
        finish_position=1,
        finish_time=96.5,
        margin='クビ',
        final_odds=3.2,
        popularity=2,
        running_positions=[3, 3, 2, 1],
        comment='Good run'
    )
    test_db.session.add(result)
    test_db.session.commit()
    return result


@pytest.fixture
def sample_prediction(test_db, sample_race, sample_horse):
    """Create a sample prediction."""
    prediction = Prediction(
        race_id=sample_race.id,
        horse_id=sample_horse.id,
        predicted_position=1,
        win_probability=0.35,
        confidence_score=0.75,
        model_version='1.0',
        model_name='RandomForest'
    )
    test_db.session.add(prediction)
    test_db.session.commit()
    return prediction


# Utility fixtures

@pytest.fixture
def multiple_races(test_db, sample_track):
    """Create multiple races for testing."""
    races = []
    for i in range(1, 6):
        race = Race(
            jra_race_id=f'202401010{i}',
            track_id=sample_track.id,
            race_date=date(2024, 1, 1),
            race_number=i,
            race_name=f'テストレース{i}',
            distance=1600,
            surface='turf',
            race_class='OP',
            status='upcoming'
        )
        test_db.session.add(race)
        races.append(race)

    test_db.session.commit()
    return races


@pytest.fixture
def auth_headers():
    """Create authentication headers for API testing."""
    return {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
