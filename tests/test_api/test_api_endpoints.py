"""
Tests for API endpoints.
"""
import pytest
from datetime import datetime, date
from src.web.app import create_app
from src.data.models import (
    db, Race, Track, Horse, Jockey, Trainer, RaceEntry, RaceResult, Prediction
)


@pytest.fixture
def app():
    """Create application for testing."""
    app = create_app('testing')
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture
def sample_data(app):
    """Create sample data for testing."""
    with app.app_context():
        # Create track
        track = Track(name='東京', location='東京都', surface_types=['turf', 'dirt'])
        db.session.add(track)
        db.session.flush()

        # Create race
        race = Race(
            jra_race_id='2026010101011',
            race_name='テストレース',
            race_number=1,
            race_date=date(2026, 1, 1),
            track_id=track.id,
            distance=1600,
            surface='芝',
            status='completed'
        )
        db.session.add(race)
        db.session.flush()

        # Create trainer first
        trainer = Trainer(
            jra_trainer_id='01234',
            name='テスト調教師',
            stable='美浦'
        )
        db.session.add(trainer)
        db.session.flush()

        # Create horse
        horse = Horse(
            jra_horse_id='2020000001',
            name='テストホース',
            birth_date=None,
            sex='牡',
            trainer_id=trainer.id
        )
        db.session.add(horse)

        # Create jockey
        jockey = Jockey(
            jra_jockey_id='01234',
            name='テスト騎手'
        )
        db.session.add(jockey)
        db.session.flush()

        # Create race entry
        entry = RaceEntry(
            race_id=race.id,
            horse_id=horse.id,
            jockey_id=jockey.id,
            horse_number=1,
            post_position=1,
            weight=54.0,
            horse_weight=480,
            morning_odds=2.5
        )
        db.session.add(entry)
        db.session.flush()

        # Create race result
        result = RaceResult(
            race_entry_id=entry.id,
            finish_position=1,
            final_odds=2.3,
            popularity=1,
            finish_time=95.2,
            margin='0.0'
        )
        db.session.add(result)

        # Create prediction
        prediction = Prediction(
            race_id=race.id,
            horse_id=horse.id,
            predicted_position=1,
            win_probability=0.45,
            confidence_score=0.82,
            model_name='test_model'
        )
        db.session.add(prediction)

        db.session.commit()

        # Return IDs instead of objects to avoid DetachedInstanceError
        return {
            'track_id': track.id,
            'race_id': race.id,
            'horse_id': horse.id,
            'jockey_id': jockey.id,
            'trainer_id': trainer.id,
            'entry_id': entry.id,
            'result_id': result.id,
            'prediction_id': prediction.id
        }


class TestRaceAPI:
    """Test race API endpoints."""

    def test_get_races(self, client, sample_data):
        """Test GET /api/races."""
        response = client.get('/api/races')
        assert response.status_code == 200

        data = response.get_json()
        assert 'races' in data
        assert 'meta' in data
        assert len(data['races']) > 0
        assert data['races'][0]['race_name'] == 'テストレース'

    def test_get_races_with_filters(self, client, sample_data):
        """Test GET /api/races with filters."""
        response = client.get('/api/races?status=completed&date=2026-01-01')
        assert response.status_code == 200

        data = response.get_json()
        assert len(data['races']) == 1

    def test_get_race_detail(self, client, sample_data):
        """Test GET /api/races/<id>."""
        race_id = sample_data['race_id']
        response = client.get(f'/api/races/{race_id}')
        assert response.status_code == 200

        data = response.get_json()
        assert data['race_name'] == 'テストレース'
        assert data['distance'] == 1600

    def test_get_race_with_entries(self, client, sample_data):
        """Test GET /api/races/<id> with entries."""
        race_id = sample_data['race_id']
        response = client.get(f'/api/races/{race_id}?include_entries=true')
        assert response.status_code == 200

        data = response.get_json()
        assert 'entries' in data
        assert len(data['entries']) == 1
        assert data['entries'][0]['horse_number'] == 1

    def test_get_race_with_results(self, client, sample_data):
        """Test GET /api/races/<id> with results."""
        race_id = sample_data['race_id']
        response = client.get(f'/api/races/{race_id}?include_results=true')
        assert response.status_code == 200

        data = response.get_json()
        assert 'results' in data
        assert len(data['results']) == 1
        assert data['results'][0]['finish_position'] == 1

    def test_get_race_not_found(self, client, sample_data):
        """Test GET /api/races/<id> with non-existent ID."""
        response = client.get('/api/races/99999')
        assert response.status_code == 404


class TestHorseAPI:
    """Test horse API endpoints."""

    def test_get_horses(self, client, sample_data):
        """Test GET /api/horses."""
        response = client.get('/api/horses')
        assert response.status_code == 200

        data = response.get_json()
        assert 'horses' in data
        assert 'meta' in data
        assert len(data['horses']) > 0

    def test_get_horses_with_search(self, client, sample_data):
        """Test GET /api/horses with search."""
        response = client.get('/api/horses?search=テスト')
        assert response.status_code == 200

        data = response.get_json()
        assert len(data['horses']) == 1
        assert data['horses'][0]['name'] == 'テストホース'

    def test_get_horse_detail(self, client, sample_data):
        """Test GET /api/horses/<id>."""
        horse_id = sample_data['horse_id']
        response = client.get(f'/api/horses/{horse_id}')
        assert response.status_code == 200

        data = response.get_json()
        assert data['name'] == 'テストホース'
        assert data['sex'] == '牡'

    def test_get_horse_with_stats(self, client, sample_data):
        """Test GET /api/horses/<id> with statistics."""
        horse_id = sample_data['horse_id']
        response = client.get(f'/api/horses/{horse_id}?include_stats=true')
        assert response.status_code == 200

        data = response.get_json()
        assert 'statistics' in data
        assert data['statistics']['wins'] == 1
        assert data['statistics']['win_rate'] == 1.0

    def test_get_horse_with_races(self, client, sample_data):
        """Test GET /api/horses/<id> with recent races."""
        horse_id = sample_data['horse_id']
        response = client.get(f'/api/horses/{horse_id}?include_races=true')
        assert response.status_code == 200

        data = response.get_json()
        assert 'recent_races' in data
        assert len(data['recent_races']) == 1


class TestJockeyAPI:
    """Test jockey API endpoints."""

    def test_get_jockeys(self, client, sample_data):
        """Test GET /api/jockeys."""
        response = client.get('/api/jockeys')
        assert response.status_code == 200

        data = response.get_json()
        assert 'jockeys' in data
        assert 'meta' in data

    def test_get_jockey_detail(self, client, sample_data):
        """Test GET /api/jockeys/<id>."""
        jockey_id = sample_data['jockey_id']
        response = client.get(f'/api/jockeys/{jockey_id}')
        assert response.status_code == 200

        data = response.get_json()
        assert data['name'] == 'テスト騎手'

    def test_get_jockey_with_stats(self, client, sample_data):
        """Test GET /api/jockeys/<id> with statistics."""
        jockey_id = sample_data['jockey_id']
        response = client.get(f'/api/jockeys/{jockey_id}?include_stats=true')
        assert response.status_code == 200

        data = response.get_json()
        assert 'statistics' in data
        assert data['statistics']['wins'] == 1


class TestTrainerAPI:
    """Test trainer API endpoints."""

    def test_get_trainers(self, client, sample_data):
        """Test GET /api/trainers."""
        response = client.get('/api/trainers')
        assert response.status_code == 200

        data = response.get_json()
        assert 'trainers' in data
        assert 'meta' in data

    def test_get_trainer_detail(self, client, sample_data):
        """Test GET /api/trainers/<id>."""
        trainer_id = sample_data['trainer_id']
        response = client.get(f'/api/trainers/{trainer_id}')
        assert response.status_code == 200

        data = response.get_json()
        assert data['name'] == 'テスト調教師'
        assert data['stable'] == '美浦'

    def test_get_trainer_with_stats(self, client, sample_data):
        """Test GET /api/trainers/<id> with statistics."""
        trainer_id = sample_data['trainer_id']
        response = client.get(f'/api/trainers/{trainer_id}?include_stats=true')
        assert response.status_code == 200

        data = response.get_json()
        assert 'statistics' in data
        assert data['statistics']['wins'] == 1


class TestPredictionAPI:
    """Test prediction API endpoints."""

    def test_get_predictions(self, client, sample_data):
        """Test GET /api/predictions."""
        response = client.get('/api/predictions')
        assert response.status_code == 200

        data = response.get_json()
        assert 'predictions' in data
        assert 'meta' in data
        assert len(data['predictions']) > 0

    def test_get_predictions_by_race(self, client, sample_data):
        """Test GET /api/predictions with race_id filter."""
        race_id = sample_data['race_id']
        response = client.get(f'/api/predictions?race_id={race_id}')
        assert response.status_code == 200

        data = response.get_json()
        assert len(data['predictions']) == 1

    def test_get_race_predictions(self, client, sample_data):
        """Test GET /api/predictions/race/<race_id>."""
        race_id = sample_data['race_id']
        response = client.get(f'/api/predictions/race/{race_id}')
        assert response.status_code == 200

        data = response.get_json()
        assert 'race' in data
        assert 'predictions' in data
        assert len(data['predictions']) == 1
        assert data['predictions'][0]['predicted_position'] == 1


class TestTrackAPI:
    """Test track API endpoints."""

    def test_get_tracks(self, client, sample_data):
        """Test GET /api/tracks."""
        response = client.get('/api/tracks')
        assert response.status_code == 200

        data = response.get_json()
        assert 'tracks' in data
        assert len(data['tracks']) > 0
        assert data['tracks'][0]['name'] == '東京'

    def test_get_track_detail(self, client, sample_data):
        """Test GET /api/tracks/<id>."""
        track_id = sample_data['track_id']
        response = client.get(f'/api/tracks/{track_id}')
        assert response.status_code == 200

        data = response.get_json()
        assert data['name'] == '東京'
        assert data['location'] == '東京都'

    def test_get_track_with_races(self, client, sample_data):
        """Test GET /api/tracks/<id> with races."""
        track_id = sample_data['track_id']
        response = client.get(f'/api/tracks/{track_id}?include_races=true')
        assert response.status_code == 200

        data = response.get_json()
        assert 'recent_races' in data
        assert len(data['recent_races']) == 1


class TestPagination:
    """Test pagination functionality."""

    def test_pagination_default(self, client, sample_data):
        """Test default pagination."""
        response = client.get('/api/horses')
        assert response.status_code == 200

        data = response.get_json()
        assert data['meta']['page'] == 1
        assert data['meta']['per_page'] == 20

    def test_pagination_custom(self, client, sample_data):
        """Test custom pagination."""
        response = client.get('/api/horses?page=1&per_page=10')
        assert response.status_code == 200

        data = response.get_json()
        assert data['meta']['page'] == 1
        assert data['meta']['per_page'] == 10

    def test_pagination_max_limit(self, client, sample_data):
        """Test max items per page limit."""
        response = client.get('/api/horses?per_page=200')
        assert response.status_code == 200

        data = response.get_json()
        assert data['meta']['per_page'] == 100  # Should be capped at 100
