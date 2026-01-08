"""
Tests for scraper utilities.
"""
import pytest
from src.scrapers.utils import (
    parse_japanese_number,
    parse_distance,
    parse_time,
    clean_text,
    validate_jra_id
)


class TestParseJapaneseNumber:
    """Tests for parse_japanese_number function."""

    def test_parse_arabic_numbers(self):
        """Test parsing Arabic numerals."""
        assert parse_japanese_number('1') == 1
        assert parse_japanese_number('10') == 10
        assert parse_japanese_number('99') == 99

    def test_parse_japanese_kanji(self):
        """Test parsing Japanese kanji numbers."""
        assert parse_japanese_number('一') == 1
        assert parse_japanese_number('二') == 2
        assert parse_japanese_number('三') == 3
        assert parse_japanese_number('五') == 5
        assert parse_japanese_number('十') == 10

    def test_parse_with_suffix(self):
        """Test parsing numbers with suffixes."""
        assert parse_japanese_number('1着') == 1
        assert parse_japanese_number('5番') == 5
        assert parse_japanese_number('3人気') == 3

    def test_parse_empty_string(self):
        """Test parsing empty string."""
        assert parse_japanese_number('') is None
        assert parse_japanese_number(None) is None

    def test_parse_invalid_text(self):
        """Test parsing invalid text."""
        assert parse_japanese_number('abc') is None
        assert parse_japanese_number('???') is None


class TestParseDistance:
    """Tests for parse_distance function."""

    def test_parse_meters(self):
        """Test parsing distance in meters."""
        assert parse_distance('1600m') == 1600
        assert parse_distance('2000m') == 2000
        assert parse_distance('1200m') == 1200

    def test_parse_japanese_meters(self):
        """Test parsing distance with Japanese characters."""
        assert parse_distance('1600メートル') == 1600
        assert parse_distance('2400メートル') == 2400

    def test_parse_number_only(self):
        """Test parsing plain numbers."""
        assert parse_distance('1800') == 1800
        assert parse_distance('3200') == 3200

    def test_parse_empty_string(self):
        """Test parsing empty string."""
        assert parse_distance('') is None
        assert parse_distance(None) is None

    def test_parse_invalid_text(self):
        """Test parsing invalid text."""
        assert parse_distance('abc') is None


class TestParseTime:
    """Tests for parse_time function."""

    def test_parse_minutes_seconds(self):
        """Test parsing time in M:SS.s format."""
        assert parse_time('1:36.5') == 96.5
        assert parse_time('2:00.0') == 120.0
        assert parse_time('1:30.8') == 90.8

    def test_parse_seconds_only(self):
        """Test parsing time in SS.s format."""
        assert parse_time('96.5') == 96.5
        assert parse_time('120.0') == 120.0
        assert parse_time('58.3') == 58.3

    def test_parse_empty_string(self):
        """Test parsing empty string."""
        assert parse_time('') is None
        assert parse_time(None) is None

    def test_parse_invalid_format(self):
        """Test parsing invalid format."""
        assert parse_time('abc') is None
        assert parse_time('1:2:3') is None


class TestCleanText:
    """Tests for clean_text function."""

    def test_remove_extra_whitespace(self):
        """Test removing extra whitespace."""
        assert clean_text('  hello  world  ') == 'hello world'
        assert clean_text('multiple   spaces') == 'multiple spaces'

    def test_remove_newlines(self):
        """Test removing newlines."""
        assert clean_text('line1\nline2') == 'line1 line2'
        assert clean_text('text\r\nmore text') == 'text more text'

    def test_strip_whitespace(self):
        """Test stripping leading/trailing whitespace."""
        assert clean_text('  text  ') == 'text'
        assert clean_text('\ttext\n') == 'text'

    def test_empty_string(self):
        """Test cleaning empty string."""
        assert clean_text('') == ''
        assert clean_text(None) == ''

    def test_normal_text(self):
        """Test cleaning normal text."""
        assert clean_text('normal text') == 'normal text'


class TestValidateJRAId:
    """Tests for validate_jra_id function."""

    def test_validate_race_id(self):
        """Test validating race IDs."""
        assert validate_jra_id('202401010101', 'race') is True
        assert validate_jra_id('2024020201', 'race') is True

    def test_validate_race_id_invalid(self):
        """Test invalid race IDs."""
        assert validate_jra_id('123', 'race') is False
        assert validate_jra_id('abc', 'race') is False
        assert validate_jra_id('', 'race') is False

    def test_validate_horse_id(self):
        """Test validating horse IDs."""
        assert validate_jra_id('H12345', 'horse') is True
        assert validate_jra_id('HORSE123', 'horse') is True

    def test_validate_jockey_id(self):
        """Test validating jockey IDs."""
        assert validate_jra_id('J12345', 'jockey') is True
        assert validate_jra_id('JOCKEY1', 'jockey') is True

    def test_validate_empty_id(self):
        """Test validating empty ID."""
        assert validate_jra_id('', 'race') is False
        assert validate_jra_id(None, 'race') is False


@pytest.mark.skip(reason="JRAScraper is deprecated, use NetkeibaScraper instead")
class TestJRAScraper:
    """Tests for JRAScraper class."""

    @pytest.fixture
    def scraper(self):
        """Create a scraper instance."""
        from src.scrapers.jra_scraper import JRAScraper
        return JRAScraper(delay=0)  # No delay for testing

    def test_scraper_initialization(self, scraper):
        """Test scraper initialization."""
        assert scraper is not None
        assert scraper.session is not None
        assert scraper.delay == 0

    def test_context_manager(self):
        """Test using scraper as context manager."""
        from src.scrapers.jra_scraper import JRAScraper

        with JRAScraper(delay=0) as scraper:
            assert scraper is not None

        # Session should be closed after exiting context
        # We can't directly test this without accessing private attributes

    def test_scrape_race_calendar_returns_list(self, scraper):
        """Test that scrape_race_calendar returns a list."""
        from datetime import datetime

        result = scraper.scrape_race_calendar(datetime.now())
        assert isinstance(result, list)
        # Currently returns empty list as placeholder

    def test_scrape_race_card_returns_none(self, scraper):
        """Test that scrape_race_card returns empty structure for invalid race ID."""
        result = scraper.scrape_race_card('202401010101')
        # Without CNAME, scraper returns empty race card structure, not None
        assert result is not None
        assert 'entries' in result
        assert len(result['entries']) == 0  # No entries for invalid race

    def test_scrape_race_result_returns_none(self, scraper):
        """Test that scrape_race_result returns empty structure for invalid race ID."""
        result = scraper.scrape_race_result('202401010101')
        # Without CNAME, scraper returns empty result structure, not None
        assert result is not None
        assert 'results' in result
        assert len(result['results']) == 0  # No results for invalid race

    def test_get_upcoming_races(self, scraper):
        """Test getting upcoming races."""
        result = scraper.get_upcoming_races(days_ahead=1)
        assert isinstance(result, list)


class TestRetryDecorator:
    """Tests for retry_on_failure decorator."""

    def test_retry_successful_call(self):
        """Test retry with successful call."""
        from src.scrapers.utils import retry_on_failure

        call_count = 0

        @retry_on_failure(max_retries=3, delay=0.01)
        def successful_function():
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_function()
        assert result == "success"
        assert call_count == 1

    def test_retry_eventual_success(self):
        """Test retry with eventual success."""
        from src.scrapers.utils import retry_on_failure

        call_count = 0

        @retry_on_failure(max_retries=3, delay=0.01, backoff=1.5)
        def eventually_successful():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not yet")
            return "success"

        result = eventually_successful()
        assert result == "success"
        assert call_count == 3

    def test_retry_max_retries_exceeded(self):
        """Test retry with max retries exceeded."""
        from src.scrapers.utils import retry_on_failure

        call_count = 0

        @retry_on_failure(max_retries=2, delay=0.01)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError):
            always_fails()

        assert call_count == 3  # Initial + 2 retries
