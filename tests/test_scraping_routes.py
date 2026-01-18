"""
Tests for scraping routes and task manager.

This module tests:
- Scraping routes (/scraping/*)
- ScrapingTaskManager class
- SSE progress streaming
"""
import pytest
import json
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.web.scraping_manager import ScrapingTaskManager


class TestScrapingManager:
    """Tests for ScrapingTaskManager class."""

    def test_manager_initialization(self):
        """Test manager initializes with empty state."""
        manager = ScrapingTaskManager()
        assert manager._tasks == {}
        assert manager.get_running_task() is None

    def test_get_progress_not_found(self):
        """Test get_progress returns None for unknown task."""
        manager = ScrapingTaskManager()
        result = manager.get_progress('nonexistent-task-id')
        assert result is None

    def test_cancel_task_not_found(self):
        """Test cancel_task returns False for unknown task."""
        manager = ScrapingTaskManager()
        result = manager.cancel_task('nonexistent-task-id')
        assert result is False

    def test_get_recent_tasks_empty(self):
        """Test get_recent_tasks returns empty list initially."""
        manager = ScrapingTaskManager()
        tasks = manager.get_recent_tasks()
        assert tasks == []

    def test_cleanup_old_tasks_empty(self):
        """Test cleanup with no tasks doesn't error."""
        manager = ScrapingTaskManager()
        manager.cleanup_old_tasks()
        assert len(manager._tasks) == 0


@pytest.mark.integration
class TestScrapingRoutes:
    """Tests for scraping web routes."""

    def test_scraping_index_page(self, client):
        """Test scraping index page loads."""
        response = client.get('/scraping/')
        assert response.status_code == 200
        assert 'スクレイピング' in response.data.decode('utf-8')

    def test_scraping_index_contains_form(self, client):
        """Test scraping index contains configuration form."""
        response = client.get('/scraping/')
        data = response.data.decode('utf-8')
        assert 'form' in data.lower()
        assert 'startDate' in data
        assert 'endDate' in data

    def test_api_start_missing_body(self, client):
        """Test start API with missing body returns error."""
        response = client.post(
            '/scraping/api/start',
            content_type='application/json'
        )
        assert response.status_code == 400

    def test_api_start_missing_dates(self, client):
        """Test start API with missing dates returns error."""
        response = client.post(
            '/scraping/api/start',
            data=json.dumps({}),
            content_type='application/json'
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] is False

    def test_api_start_invalid_date_format(self, client):
        """Test start API with invalid date format returns error."""
        response = client.post(
            '/scraping/api/start',
            data=json.dumps({
                'start_date': 'invalid',
                'end_date': '2025-01-31'
            }),
            content_type='application/json'
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] is False
        assert '日付形式' in data['error']

    def test_api_start_invalid_date_range(self, client):
        """Test start API with end date before start date returns error."""
        response = client.post(
            '/scraping/api/start',
            data=json.dumps({
                'start_date': '2025-01-31',
                'end_date': '2025-01-01'
            }),
            content_type='application/json'
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] is False
        assert '開始日' in data['error']

    def test_api_start_future_date(self, client):
        """Test start API with future end date returns error."""
        future_date = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
        response = client.post(
            '/scraping/api/start',
            data=json.dumps({
                'start_date': '2025-01-01',
                'end_date': future_date
            }),
            content_type='application/json'
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] is False
        assert '未来' in data['error']

    def test_api_status_not_found(self, client):
        """Test status API with unknown task returns 404."""
        response = client.get('/scraping/api/status/nonexistent-task-id')
        assert response.status_code == 404
        data = json.loads(response.data)
        assert data['success'] is False

    def test_api_cancel_not_found(self, client):
        """Test cancel API with unknown task returns 404."""
        response = client.post('/scraping/api/cancel/nonexistent-task-id')
        assert response.status_code == 404
        data = json.loads(response.data)
        assert data['success'] is False

    def test_api_running_no_task(self, client):
        """Test running API returns null when no task is running."""
        response = client.get('/scraping/api/running')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert data['task'] is None

    def test_api_history_empty(self, client):
        """Test history API returns empty list initially."""
        response = client.get('/scraping/api/history')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert isinstance(data['tasks'], list)

    def test_api_history_with_limit(self, client):
        """Test history API respects limit parameter."""
        response = client.get('/scraping/api/history?limit=5')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True


class TestScrapingManagerThreadSafety:
    """Tests for thread safety of ScrapingTaskManager."""

    def test_concurrent_progress_updates(self):
        """Test concurrent updates to progress don't cause errors."""
        manager = ScrapingTaskManager()

        # Manually create a task entry
        task_id = 'test-task'
        manager._tasks[task_id] = {
            'task_id': task_id,
            'status': 'running',
            'started_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'progress': {
                'percent_complete': 0,
                'total_dates': 10,
                'processed_dates': 0,
            },
            'error': None,
            'stats': None,
            'cancelled': False,
        }

        # Update from multiple threads
        def update_progress(n):
            for i in range(10):
                manager._update_progress(task_id, {
                    'event': 'date_progress',
                    'percent_complete': (i + 1) * 10,
                    'stats': {'dates_processed': i + 1}
                })
                time.sleep(0.01)

        threads = [threading.Thread(target=update_progress, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should not have crashed
        progress = manager.get_progress(task_id)
        assert progress is not None
        assert progress['progress']['percent_complete'] >= 0

    def test_concurrent_get_progress(self):
        """Test concurrent reads of progress don't cause errors."""
        manager = ScrapingTaskManager()

        task_id = 'test-task'
        manager._tasks[task_id] = {
            'task_id': task_id,
            'status': 'running',
            'started_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'progress': {'percent_complete': 50},
            'error': None,
            'stats': None,
            'cancelled': False,
        }

        results = []

        def read_progress():
            for _ in range(10):
                result = manager.get_progress(task_id)
                results.append(result)
                time.sleep(0.01)

        threads = [threading.Thread(target=read_progress) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All reads should have succeeded
        assert len(results) == 50
        assert all(r is not None for r in results)


class TestHistoricalScraperModule:
    """Tests for historical_scraper module functions."""

    def test_is_weekend_saturday(self):
        """Test Saturday is detected as weekend."""
        from src.scrapers.historical_scraper import is_weekend
        saturday = datetime(2025, 1, 11)  # Saturday
        assert is_weekend(saturday) is True

    def test_is_weekend_sunday(self):
        """Test Sunday is detected as weekend."""
        from src.scrapers.historical_scraper import is_weekend
        sunday = datetime(2025, 1, 12)  # Sunday
        assert is_weekend(sunday) is True

    def test_is_weekend_weekday(self):
        """Test weekday is not weekend."""
        from src.scrapers.historical_scraper import is_weekend
        monday = datetime(2025, 1, 13)  # Monday
        assert is_weekend(monday) is False

    def test_generate_date_list_weekends_only(self):
        """Test date list generation with weekends only."""
        from src.scrapers.historical_scraper import generate_date_list
        start = datetime(2025, 1, 6)  # Monday
        end = datetime(2025, 1, 12)  # Sunday
        dates = generate_date_list(start, end, weekends_only=True)
        # Should only include Sat (11) and Sun (12)
        assert len(dates) == 2
        assert dates[0].day == 11
        assert dates[1].day == 12

    def test_generate_date_list_all_days(self):
        """Test date list generation with all days."""
        from src.scrapers.historical_scraper import generate_date_list
        start = datetime(2025, 1, 6)
        end = datetime(2025, 1, 12)
        dates = generate_date_list(start, end, weekends_only=False)
        # Should include all 7 days
        assert len(dates) == 7

    def test_should_scrape_race_g1(self):
        """Test G1 race should be scraped."""
        from src.scrapers.historical_scraper import should_scrape_race
        assert should_scrape_race('G1') is True

    def test_should_scrape_race_op(self):
        """Test Open class should be scraped."""
        from src.scrapers.historical_scraper import should_scrape_race
        assert should_scrape_race('オープン') is True

    def test_should_scrape_race_2win(self):
        """Test 2勝クラス should be scraped."""
        from src.scrapers.historical_scraper import should_scrape_race
        assert should_scrape_race('2勝クラス') is True

    def test_should_scrape_race_1win(self):
        """Test 1勝クラス should NOT be scraped."""
        from src.scrapers.historical_scraper import should_scrape_race
        assert should_scrape_race('1勝クラス') is False

    def test_should_scrape_race_maiden(self):
        """Test 未勝利 should NOT be scraped."""
        from src.scrapers.historical_scraper import should_scrape_race
        assert should_scrape_race('未勝利') is False

    def test_should_scrape_race_none(self):
        """Test None race class returns False."""
        from src.scrapers.historical_scraper import should_scrape_race
        assert should_scrape_race(None) is False

    def test_should_scrape_track_tokyo(self):
        """Test Tokyo track should be scraped."""
        from src.scrapers.historical_scraper import should_scrape_track
        assert should_scrape_track('東京') is True

    def test_should_scrape_track_nakayama(self):
        """Test Nakayama track should be scraped."""
        from src.scrapers.historical_scraper import should_scrape_track
        assert should_scrape_track('中山') is True

    def test_should_scrape_track_sapporo(self):
        """Test Sapporo track should NOT be scraped."""
        from src.scrapers.historical_scraper import should_scrape_track
        assert should_scrape_track('札幌') is False

    def test_should_scrape_track_kokura(self):
        """Test Kokura track should NOT be scraped."""
        from src.scrapers.historical_scraper import should_scrape_track
        assert should_scrape_track('小倉') is False
