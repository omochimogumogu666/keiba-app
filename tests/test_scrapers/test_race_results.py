"""
Tests for race result scraping and database saving.
"""
import pytest
from datetime import datetime
from src.scrapers.netkeiba_scraper import NetkeibaScraper
from src.data.database import save_race_to_db, save_race_entries_to_db, save_race_results_to_db
from src.data.models import Race, RaceEntry, RaceResult, Horse, Jockey, Trainer
from src.web.app import create_app


@pytest.fixture
def app():
    """Create test Flask app."""
    app = create_app('testing')
    with app.app_context():
        from src.data.models import db
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def sample_race_data():
    """Sample race data for testing."""
    return {
        'netkeiba_race_id': '202601010101',
        'track': '中山',
        'race_date': datetime(2026, 1, 1),
        'race_number': 11,
        'race_name': 'テストレース',
        'distance': 2000,
        'surface': 'turf',
        'race_class': 'G1',
        'status': 'upcoming'
    }


@pytest.fixture
def sample_entries():
    """Sample race entries for testing."""
    return [
        {
            'netkeiba_horse_id': 'H001',
            'horse_name': 'テストホース1',
            'netkeiba_jockey_id': 'J001',
            'jockey_name': 'テスト騎手1',
            'netkeiba_trainer_id': 'T001',
            'trainer_name': 'テスト調教師1',
            'post_position': 1,
            'horse_number': 1,
            'weight': 57.0,
            'horse_weight': 480,
            'horse_weight_change': 0,
            'morning_odds': 5.5
        },
        {
            'netkeiba_horse_id': 'H002',
            'horse_name': 'テストホース2',
            'netkeiba_jockey_id': 'J002',
            'jockey_name': 'テスト騎手2',
            'netkeiba_trainer_id': 'T002',
            'trainer_name': 'テスト調教師2',
            'post_position': 2,
            'horse_number': 2,
            'weight': 56.0,
            'horse_weight': 475,
            'horse_weight_change': -5,
            'morning_odds': 3.2
        }
    ]


@pytest.fixture
def sample_results():
    """Sample race results for testing."""
    return [
        {
            'horse_number': 1,
            'finish_position': 2,
            'finish_time': 120.5,  # 2:00.5
            'margin': 'クビ',
            'final_odds': 5.8,
            'popularity': 3,
            'running_positions': [2, 2, 2, 2],
            'comment': '直線で伸びたが届かず'
        },
        {
            'horse_number': 2,
            'finish_position': 1,
            'finish_time': 120.4,  # 2:00.4
            'margin': '',
            'final_odds': 3.5,
            'popularity': 1,
            'running_positions': [1, 1, 1, 1],
            'comment': '完勝'
        }
    ]


@pytest.mark.integration
class TestRaceResultSaving:
    """Test race result saving to database."""

    def test_save_race_to_db(self, app, sample_race_data):
        """Test saving race to database."""
        with app.app_context():
            race = save_race_to_db(sample_race_data)

            assert race is not None
            assert race.id is not None
            assert race.netkeiba_race_id == sample_race_data['netkeiba_race_id']
            assert race.race_name == sample_race_data['race_name']
            assert race.distance == sample_race_data['distance']
            assert race.status == 'upcoming'

    def test_save_race_entries_to_db(self, app, sample_race_data, sample_entries):
        """Test saving race entries to database."""
        with app.app_context():
            # First save race
            race = save_race_to_db(sample_race_data)

            # Then save entries
            entries = save_race_entries_to_db(race.id, sample_entries)

            assert len(entries) == 2
            assert entries[0].horse_number == 1
            assert entries[1].horse_number == 2

            # Check that horses were created
            horse1 = Horse.query.filter_by(netkeiba_horse_id='H001').first()
            horse2 = Horse.query.filter_by(netkeiba_horse_id='H002').first()

            assert horse1 is not None
            assert horse2 is not None
            assert horse1.name == 'テストホース1'
            assert horse2.name == 'テストホース2'

    def test_save_race_results_to_db(self, app, sample_race_data, sample_entries, sample_results):
        """Test saving race results to database."""
        with app.app_context():
            # Setup: save race and entries first
            race = save_race_to_db(sample_race_data)
            entries = save_race_entries_to_db(race.id, sample_entries)

            # Save results
            results = save_race_results_to_db(race.id, sample_results)

            assert len(results) == 2

            # Check result details
            result1 = RaceResult.query.filter_by(race_entry_id=entries[0].id).first()
            result2 = RaceResult.query.filter_by(race_entry_id=entries[1].id).first()

            assert result1 is not None
            assert result2 is not None

            # Horse 2 won (finish_position = 1)
            assert result2.finish_position == 1
            assert result2.finish_time == 120.4
            assert result2.final_odds == 3.5

            # Horse 1 came second
            assert result1.finish_position == 2
            assert result1.finish_time == 120.5
            assert result1.margin == 'クビ'

    def test_update_existing_results(self, app, sample_race_data, sample_entries, sample_results):
        """Test updating existing race results."""
        with app.app_context():
            # Setup
            race = save_race_to_db(sample_race_data)
            entries = save_race_entries_to_db(race.id, sample_entries)
            results_v1 = save_race_results_to_db(race.id, sample_results)

            # Update results with new data
            updated_results = [
                {
                    'horse_number': 1,
                    'finish_position': 2,
                    'finish_time': 120.6,  # Updated time
                    'margin': 'アタマ',  # Updated margin
                    'final_odds': 6.0,  # Updated odds
                    'popularity': 3,
                    'running_positions': [2, 2, 2, 2],
                    'comment': '更新されたコメント'
                }
            ]

            results_v2 = save_race_results_to_db(race.id, updated_results)

            # Should still be same result object, just updated
            result = RaceResult.query.filter_by(race_entry_id=entries[0].id).first()

            assert result.finish_time == 120.6
            assert result.margin == 'アタマ'
            assert result.final_odds == 6.0
            assert result.comment == '更新されたコメント'


@pytest.mark.unit
class TestRaceResultParsing:
    """Test race result parsing logic."""

    def test_scrape_race_result_structure(self):
        """Test that scrape_race_result returns expected structure."""
        scraper = NetkeibaScraper(delay=0)

        # Test with mock HTML (unit test)
        # For now, just verify method exists and has correct signature
        import inspect
        sig = inspect.signature(scraper.scrape_race_result)
        params = list(sig.parameters.keys())

        assert 'race_id' in params

    def test_result_data_validation(self, sample_results):
        """Test that result data has required fields."""
        for result in sample_results:
            assert 'horse_number' in result
            assert 'finish_position' in result
            assert isinstance(result['horse_number'], int)
            assert isinstance(result['finish_position'], int)


@pytest.mark.scraper
@pytest.mark.integration
class TestRaceResultScraping:
    """Integration tests for race result scraping."""

    def test_scrape_recent_results(self):
        """
        Integration test: Try to scrape recent race results.

        Note: This test requires internet connection and may fail if
        there are no completed races available.
        """
        from datetime import timedelta

        with NetkeibaScraper(delay=3) as scraper:
            # Try to find races from past week
            for days_ago in range(1, 8):
                target_date = datetime.now() - timedelta(days=days_ago)
                races = scraper.scrape_race_calendar(target_date)

                if races:
                    # Try first race
                    race = races[0]
                    netkeiba_race_id = race.get('netkeiba_race_id')

                    if netkeiba_race_id:
                        result = scraper.scrape_race_result(netkeiba_race_id)

                        if result:
                            # If we got a result, verify structure
                            assert 'results' in result

                            if result['results']:
                                # Check first result entry
                                first_result = result['results'][0]
                                assert 'finish_position' in first_result
                                assert 'horse_number' in first_result
                                print(f"✓ Successfully scraped result for race {netkeiba_race_id}")
                                return  # Test passed

            # If we get here, no results were found
            pytest.skip("No completed races with results found in past week")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
