"""
Find a recent completed race for testing.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.logging_config import setup_logging
from src.scrapers.netkeiba_scraper import NetkeibaScraper
from src.utils.logger import get_app_logger

setup_logging(log_level='INFO')
logger = get_app_logger(__name__)


def find_recent_race():
    """Find a recent race (from past week)."""

    with NetkeibaScraper(delay=3) as scraper:
        # Check past 14 days
        for days_ago in range(14):
            target_date = datetime.now() - timedelta(days=days_ago)
            logger.info(f"Checking {target_date.strftime('%Y-%m-%d')} ({days_ago} days ago)")

            races = scraper.scrape_race_calendar(target_date)

            if races:
                logger.info(f"Found {len(races)} races on {target_date.strftime('%Y-%m-%d')}")

                # Show first 3 race IDs
                for i, race in enumerate(races[:3]):
                    logger.info(f"  Race {i+1}: {race['netkeiba_race_id']} at {race['track']} R{race['race_number']}")

                # Return first race
                return races[0]['netkeiba_race_id']

        logger.warning("No races found in past 14 days")
        return None


if __name__ == "__main__":
    race_id = find_recent_race()
    if race_id:
        logger.info(f"\n=== Use this race ID for testing: {race_id} ===")
