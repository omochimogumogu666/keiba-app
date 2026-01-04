"""
Test script for NetkeibaScraper.
Tests the updated scraper with actual netkeiba.com data.
"""
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scrapers.netkeiba_scraper import NetkeibaScraper
from src.utils.logger import get_app_logger
from config.logging_config import setup_logging

# Setup logging
setup_logging(log_level='INFO')
logger = get_app_logger(__name__)


def test_race_calendar():
    """Test scraping race calendar."""
    print("=" * 80)
    print("TEST 1: Race Calendar")
    print("=" * 80)

    with NetkeibaScraper(delay=3) as scraper:
        # Test with 2026-01-04 (today - might have races)
        test_date = datetime(2026, 1, 4)
        logger.info(f"Testing race calendar for {test_date.strftime('%Y-%m-%d')}")

        races = scraper.scrape_race_calendar(test_date)

        print(f"\nFound {len(races)} races")

        if races:
            print("\nFirst 3 races:")
            for i, race in enumerate(races[:3]):
                print(f"\nRace {i+1}:")
                for key, value in race.items():
                    print(f"  {key}: {value}")
        else:
            print("No races found for this date (might be non-racing day)")
            print("Trying with a known date: 2025-12-28")

            test_date = datetime(2025, 12, 28)
            races = scraper.scrape_race_calendar(test_date)

            print(f"\nFound {len(races)} races for 2025-12-28")
            if races:
                print("\nFirst race:")
                for key, value in races[0].items():
                    print(f"  {key}: {value}")

        return races


def test_race_card(race_id: str):
    """Test scraping race card."""
    print("\n" + "=" * 80)
    print("TEST 2: Race Card (Shutuba)")
    print("=" * 80)

    with NetkeibaScraper(delay=3) as scraper:
        logger.info(f"Testing race card for race_id={race_id}")

        race_card = scraper.scrape_race_card(race_id)

        if race_card:
            print("\nRace Info:")
            for key, value in race_card['race_info'].items():
                print(f"  {key}: {value}")

            print(f"\nEntries: {len(race_card['entries'])} horses")

            if race_card['entries']:
                print("\nFirst 3 entries:")
                for i, entry in enumerate(race_card['entries'][:3]):
                    print(f"\nHorse {i+1}:")
                    for key, value in entry.items():
                        print(f"  {key}: {value}")
        else:
            print("Failed to scrape race card")

        return race_card


def test_race_result(race_id: str):
    """Test scraping race result."""
    print("\n" + "=" * 80)
    print("TEST 3: Race Result")
    print("=" * 80)

    with NetkeibaScraper(delay=3) as scraper:
        logger.info(f"Testing race result for race_id={race_id}")

        result = scraper.scrape_race_result(race_id)

        if result:
            print(f"\nResults: {len(result.get('results', []))} horses")

            if result.get('results'):
                print("\nTop 3 finishers:")
                for i, res in enumerate(result['results'][:3]):
                    print(f"\nPosition {i+1}:")
                    for key, value in res.items():
                        print(f"  {key}: {value}")

            print(f"\nPayouts: {len(result.get('payouts', {}))} types")
            if result.get('payouts'):
                for bet_type, payouts in result['payouts'].items():
                    print(f"\n{bet_type}:")
                    for payout in payouts[:2]:  # First 2 of each type
                        print(f"  {payout}")
        else:
            print("Failed to scrape race result (race might not be finished yet)")

        return result


def test_horse_profile(horse_id: str):
    """Test scraping horse profile."""
    print("\n" + "=" * 80)
    print("TEST 4: Horse Profile")
    print("=" * 80)

    with NetkeibaScraper(delay=3) as scraper:
        logger.info(f"Testing horse profile for horse_id={horse_id}")

        profile = scraper.scrape_horse_profile(horse_id)

        if profile:
            print("\nHorse Profile:")
            for key, value in profile.items():
                print(f"  {key}: {value}")
        else:
            print("Failed to scrape horse profile")

        return profile


if __name__ == "__main__":
    print("NetkeibaScraper Test Suite")
    print("=" * 80)
    print(f"Time: {datetime.now()}")
    print()

    try:
        # Test 1: Race Calendar
        races = test_race_calendar()

        # Test 2: Race Card (use first race from calendar if available)
        if races:
            race_id = races[0]['netkeiba_race_id']
            race_card = test_race_card(race_id)

            # Test 3: Race Result (might fail if race hasn't happened yet)
            test_race_result(race_id)

            # Test 4: Horse Profile (use first horse from race card if available)
            if race_card and race_card.get('entries'):
                horse_id = race_card['entries'][0].get('netkeiba_horse_id')
                if horse_id:
                    test_horse_profile(horse_id)
        else:
            # Use a known race ID for testing
            print("\nNo races found in calendar, using known race ID for testing")
            race_id = '202506050811'  # Example race ID
            test_race_card(race_id)

        print("\n" + "=" * 80)
        print("All tests completed!")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nError during testing: {e}")
        import traceback
        traceback.print_exc()
