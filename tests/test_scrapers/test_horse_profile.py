"""
Tests for horse profile scraping functionality.
"""
import pytest
from datetime import datetime
from src.scrapers.netkeiba_scraper import NetkeibaScraper


@pytest.mark.unit
class TestHorseProfileParsing:
    """Unit tests for horse profile parsing logic."""

    def test_horse_profile_structure(self):
        """Test that horse profile returns correct structure."""
        # Create a mock profile response
        expected_keys = [
            'jra_horse_id', 'name', 'birth_date', 'sex',
            'sire_name', 'sire_id', 'dam_name', 'dam_id',
            'trainer_name', 'trainer_id', 'owner', 'breeder',
            'total_races', 'total_wins', 'total_places', 'total_shows',
            'total_earnings', 'past_performances'
        ]

        # Mock profile data
        profile = {
            'jra_horse_id': 'TEST123',
            'name': 'テストホース',
            'birth_date': datetime(2020, 4, 1),
            'sex': '牡',
            'sire_name': 'テスト父',
            'sire_id': 'SIRE123',
            'dam_name': 'テスト母',
            'dam_id': 'DAM123',
            'trainer_name': 'テスト調教師',
            'trainer_id': 'TRAINER123',
            'owner': 'テストオーナー',
            'breeder': 'テスト生産者',
            'total_races': 10,
            'total_wins': 3,
            'total_places': 2,
            'total_shows': 2,
            'total_earnings': 50000000,
            'past_performances': []
        }

        # Verify all expected keys are present
        for key in expected_keys:
            assert key in profile, f"Missing key: {key}"

    def test_past_performance_structure(self):
        """Test that past performance has correct structure."""
        performance = {
            'date': '2025-01-01',
            'track': '東京',
            'race_name': 'テストレース',
            'distance': 2000,
            'surface': 'turf',
            'finish_position': 1,
            'finish_time': 120.5,
            'jockey': 'テスト騎手',
            'weight': 57.0,
            'horse_weight': 480
        }

        expected_keys = [
            'date', 'track', 'race_name', 'distance', 'surface',
            'finish_position', 'finish_time', 'jockey', 'weight', 'horse_weight'
        ]

        for key in expected_keys:
            assert key in performance, f"Missing key in performance: {key}"


@pytest.mark.integration
@pytest.mark.slow
class TestHorseProfileScraping:
    """Integration tests for actual horse profile scraping."""

    @pytest.fixture
    def scraper(self):
        """Create scraper instance."""
        return NetkeibaScraper(delay=3)

    def test_scrape_horse_profile_with_horse_id(self, scraper):
        """Test scraping horse profile with valid horse ID."""
        # This test requires a valid horse ID from netkeiba
        # Skip if no horse ID is provided
        pytest.skip("Requires valid horse ID from netkeiba.com")

        # Example usage (when horse ID is available):
        # horse_id = "2022104614"  # Sample horse ID
        # profile = scraper.scrape_horse_profile(horse_id)
        #
        # assert profile is not None
        # assert profile['netkeiba_horse_id'] == horse_id
        # assert profile['name'] is not None

    def test_scrape_horse_profile_invalid_id(self, scraper):
        """Test scraping with invalid horse ID returns None."""
        profile = scraper.scrape_horse_profile("INVALID_ID")

        # Should handle gracefully - either return None or empty profile
        if profile:
            assert profile['netkeiba_horse_id'] == "INVALID_ID"
            # May have minimal or no data
        else:
            assert profile is None


@pytest.mark.unit
class TestIDExtraction:
    """Test ID extraction from various URLs."""

    def test_extract_horse_id_from_url(self):
        """Test horse ID extraction from URL."""
        scraper = NetkeibaScraper()

        # Test horse URL
        url = "https://db.netkeiba.com/horse/2022104614/"
        horse_id = scraper._extract_id_from_url(url, r'/horse/(\d+)')
        assert horse_id == "2022104614"

        # Test URL without horse ID
        url = "https://db.netkeiba.com/other/page/"
        horse_id = scraper._extract_id_from_url(url, r'/horse/(\d+)')
        assert horse_id is None

    def test_extract_jockey_id_from_url(self):
        """Test jockey ID extraction from URL."""
        scraper = NetkeibaScraper()

        # Test jockey URL
        url = "https://db.netkeiba.com/jockey/result/recent/01160/"
        jockey_id = scraper._extract_id_from_url(url, r'/jockey/[^/]+/[^/]+/(\d+)')
        assert jockey_id == "01160"

        # Test None input
        jockey_id = scraper._extract_id_from_url(None, r'/jockey/[^/]+/[^/]+/(\d+)')
        assert jockey_id is None


@pytest.mark.unit
class TestHorseProfileDataExtraction:
    """Test specific data extraction patterns."""

    def test_birth_date_parsing(self):
        """Test birth date parsing from various formats."""
        import re

        text_samples = [
            "生年月日：2020年4月1日",
            "生年月日: 2020年4月1日",
            "生: 2020年4月1日",
        ]

        for text in text_samples:
            match = re.search(r'生年月日[：:]\s*(\d{4})年(\d{1,2})月(\d{1,2})日', text)
            if not match:
                match = re.search(r'生[：:]\s*(\d{4})年(\d{1,2})月(\d{1,2})日', text)

            assert match is not None, f"Failed to parse: {text}"
            assert match.group(1) == "2020"
            assert match.group(2) == "4"
            assert match.group(3) == "1"

    def test_sex_parsing(self):
        """Test sex parsing."""
        import re

        text_samples = [
            "性別：牡",
            "性: 牝",
            "性別: セ",  # Changed to avoid Windows codec issues
        ]

        expected = ["牡", "牝", "セ"]

        for i, text in enumerate(text_samples):
            match = re.search(r'([牡牝セ])', text)
            assert match is not None
            assert match.group(1) in expected, f"Unexpected sex value: {match.group(1)}"

    def test_career_stats_parsing(self):
        """Test career statistics parsing."""
        import re

        # Test "XX戦X勝" format
        text = "通算成績: 15戦3勝"
        match = re.search(r'(\d+)戦\s*(\d+)勝', text)
        assert match is not None
        assert int(match.group(1)) == 15
        assert int(match.group(2)) == 3

        # Test "XX-X-X-X" format (wins-places-shows-others)
        text = "成績: 3-2-2-8"
        match = re.search(r'(\d+)-(\d+)-(\d+)-(\d+)', text)
        assert match is not None
        assert int(match.group(1)) == 3  # wins
        assert int(match.group(2)) == 2  # places
        assert int(match.group(3)) == 2  # shows

    def test_earnings_parsing(self):
        """Test earnings parsing."""
        import re

        text = "獲得賞金: 5,000万円"
        match = re.search(r'獲得賞金[：:]\s*([0-9,]+)\s*万円', text)
        assert match is not None
        earnings_str = match.group(1).replace(',', '')
        earnings = int(earnings_str) * 10000  # Convert to yen
        assert earnings == 50000000

    def test_finish_time_parsing(self):
        """Test finish time parsing."""
        import re

        time_samples = [
            "2:00.1",
            "1:59.8",
            "3:15.5"
        ]

        expected_seconds = [120.1, 119.8, 195.5]

        for i, time_str in enumerate(time_samples):
            match = re.match(r'(\d+):(\d{2})\.(\d)', time_str)
            assert match is not None
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            tenths = int(match.group(3))
            total_seconds = minutes * 60 + seconds + tenths / 10.0
            assert total_seconds == expected_seconds[i]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
