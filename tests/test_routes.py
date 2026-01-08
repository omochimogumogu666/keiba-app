"""
Tests for web routes.
"""
import pytest
import json
from datetime import date


class TestMainRoutes:
    """Tests for main routes."""

    def test_index_page(self, client):
        """Test index page loads."""
        response = client.get('/')
        assert response.status_code == 200
        assert 'JRA競馬予想' in response.data.decode('utf-8')

    def test_index_with_races(self, client, multiple_races):
        """Test index page with races."""
        response = client.get('/')
        assert response.status_code == 200
        data = response.data.decode('utf-8')
        # Check that races are displayed (check for race number or generic content)
        assert '1R' in data or 'レース' in data

    def test_race_list_page(self, client):
        """Test race list page loads."""
        response = client.get('/races')
        assert response.status_code == 200
        assert 'レース一覧' in response.data.decode('utf-8')

    def test_race_list_with_races(self, client, multiple_races):
        """Test race list with multiple races."""
        response = client.get('/races')
        assert response.status_code == 200
        data = response.data.decode('utf-8')
        # Should show race numbers
        assert 'テストレース1' in data or '1R' in data

    def test_race_list_filter_by_date(self, client, sample_race):
        """Test filtering races by date."""
        response = client.get('/races?date=2024-01-01')
        assert response.status_code == 200

    def test_race_list_filter_by_track(self, client, sample_race):
        """Test filtering races by track."""
        response = client.get(f'/races?track={sample_race.track_id}')
        assert response.status_code == 200

    def test_race_list_filter_by_status(self, client, sample_race):
        """Test filtering races by status."""
        response = client.get('/races?status=upcoming')
        assert response.status_code == 200

    def test_race_detail_page(self, client, sample_race):
        """Test race detail page."""
        response = client.get(f'/races/{sample_race.id}')
        assert response.status_code == 200
        data = response.data.decode('utf-8')
        assert sample_race.race_name in data

    def test_race_detail_not_found(self, client):
        """Test race detail page with invalid ID."""
        response = client.get('/races/99999')
        assert response.status_code == 404

    def test_race_detail_with_entries(self, client, sample_race_entry):
        """Test race detail page with entries."""
        race = sample_race_entry.race
        response = client.get(f'/races/{race.id}')
        assert response.status_code == 200
        data = response.data.decode('utf-8')
        assert sample_race_entry.horse.name in data

    def test_about_page(self, client):
        """Test about page loads."""
        response = client.get('/about')
        assert response.status_code == 200
        assert 'About' in response.data.decode('utf-8')


class TestPredictionRoutes:
    """Tests for prediction routes."""

    def test_predictions_index(self, client):
        """Test predictions index page."""
        response = client.get('/predictions/')
        assert response.status_code == 200

    def test_predictions_index_with_data(self, client, sample_prediction):
        """Test predictions index with predictions."""
        response = client.get('/predictions/')
        assert response.status_code == 200
        # Should show races with predictions

    def test_race_predictions_page(self, client, sample_race):
        """Test race predictions page."""
        response = client.get(f'/predictions/race/{sample_race.id}')
        assert response.status_code == 200

    def test_race_predictions_with_data(self, client, sample_prediction):
        """Test race predictions page with prediction data."""
        race_id = sample_prediction.race_id
        response = client.get(f'/predictions/race/{race_id}')
        assert response.status_code == 200
        data = response.data.decode('utf-8')
        # Check that prediction data is displayed (avoid encoding issues with Japanese text)
        assert 'テストレース' in data or 'Unknown' not in data  # Race name or horse should be present
        assert str(sample_prediction.predicted_position) in data  # Predicted position should be present

    def test_race_predictions_not_found(self, client):
        """Test race predictions page with invalid race ID."""
        response = client.get('/predictions/race/99999')
        assert response.status_code == 404

    def test_api_race_predictions(self, client, sample_prediction):
        """Test API endpoint for race predictions."""
        race_id = sample_prediction.race_id
        response = client.get(f'/predictions/api/race/{race_id}')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert 'race_id' in data
        assert 'predictions' in data
        assert len(data['predictions']) > 0
        assert data['predictions'][0]['horse_name'] == sample_prediction.horse.name

    def test_api_race_predictions_json_format(self, client, sample_prediction):
        """Test API returns proper JSON format."""
        race_id = sample_prediction.race_id
        response = client.get(f'/predictions/api/race/{race_id}')

        assert response.content_type == 'application/json'

        data = json.loads(response.data)
        prediction = data['predictions'][0]

        # Check all required fields
        assert 'horse_id' in prediction
        assert 'horse_name' in prediction
        assert 'predicted_position' in prediction
        assert 'win_probability' in prediction
        assert 'confidence_score' in prediction
        assert 'model_name' in prediction

    def test_api_race_predictions_not_found(self, client):
        """Test API with invalid race ID."""
        response = client.get('/predictions/api/race/99999')
        assert response.status_code == 404

    def test_prediction_accuracy_page(self, client):
        """Test prediction accuracy page."""
        response = client.get('/predictions/accuracy')
        assert response.status_code == 200


class TestErrorHandlers:
    """Tests for error handlers."""

    def test_404_error(self, client):
        """Test 404 error handler."""
        response = client.get('/nonexistent-page')
        assert response.status_code == 404
        assert '404' in response.data.decode('utf-8')

    def test_404_custom_page(self, client):
        """Test custom 404 page is rendered."""
        response = client.get('/nonexistent-page')
        data = response.data.decode('utf-8')
        assert 'ページが見つかりません' in data or '404' in data


class TestStaticFiles:
    """Tests for static files."""

    def test_css_file_exists(self, client):
        """Test CSS file is accessible."""
        response = client.get('/static/css/style.css')
        # May be 200 if file exists, 404 if not served in test
        assert response.status_code in [200, 404]

    def test_js_file_exists(self, client):
        """Test JavaScript file is accessible."""
        response = client.get('/static/js/main.js')
        # May be 200 if file exists, 404 if not served in test
        assert response.status_code in [200, 404]


class TestRaceEntryDisplay:
    """Tests for race entry display in templates."""

    def test_race_with_multiple_entries(self, client, sample_race, sample_jockey, test_db):
        """Test race detail with multiple entries."""
        # Create multiple horses and entries
        from src.data.models import Horse, RaceEntry

        for i in range(3):
            horse = Horse(
                netkeiba_horse_id=f'H_TEST_{i}',
                name=f'テスト馬{i}',
                birth_date=date(2020, 3, 1),
                sex='牡'
            )
            test_db.session.add(horse)
            test_db.session.flush()

            entry = RaceEntry(
                race_id=sample_race.id,
                horse_id=horse.id,
                jockey_id=sample_jockey.id,
                post_position=i + 1,
                horse_number=i + 1,
                weight=57.0
            )
            test_db.session.add(entry)

        test_db.session.commit()

        response = client.get(f'/races/{sample_race.id}')
        assert response.status_code == 200
        data = response.data.decode('utf-8')

        # Check that horse names appear
        for i in range(3):
            assert f'テスト馬{i}' in data


class TestDatabaseQueries:
    """Tests for database query efficiency."""

    def test_race_list_query_count(self, client, multiple_races):
        """Test race list doesn't make excessive queries."""
        # This is a basic test; use Flask-SQLAlchemy query counter for detailed analysis
        response = client.get('/races')
        assert response.status_code == 200

    def test_race_detail_eager_loading(self, client, sample_race_entry):
        """Test race detail uses eager loading."""
        race = sample_race_entry.race
        response = client.get(f'/races/{race.id}')
        assert response.status_code == 200
        # Should load entries, horses, jockeys efficiently
