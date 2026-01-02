"""
Helper script to get horse CNAME from recent race cards.

This script scrapes recent race cards and extracts horse CNAMEs
for testing horse profile scraping.
"""
import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.scrapers.jra_scraper import JRAScraper
from src.utils.logger import get_app_logger
from config.logging_config import setup_logging

setup_logging(log_level='INFO')
logger = get_app_logger(__name__)


def get_sample_horse_cnames(limit: int = 5):
    """
    Get sample horse CNAMEs from recent race cards.

    Args:
        limit: Maximum number of horse CNAMEs to collect

    Returns:
        List of dictionaries with horse info and CNAME
    """
    horse_cnames = []

    with JRAScraper(delay=3) as scraper:
        # Get today's races
        today = datetime.now()
        races = scraper.scrape_race_calendar(today)

        if not races:
            logger.warning("No races found for today, trying tomorrow...")
            from datetime import timedelta
            tomorrow = today + timedelta(days=1)
            races = scraper.scrape_race_calendar(tomorrow)

        if not races:
            logger.error("No races found")
            return horse_cnames

        # Process first race to get horse CNAMEs
        for race in races[:3]:  # Check first 3 races
            race_id = race.get('jra_race_id')
            cname = race.get('cname')

            if not cname:
                continue

            logger.info(f"Scraping race card for race {race_id}")
            race_card = scraper.scrape_race_card(race_id, cname=cname)

            if not race_card or not race_card.get('entries'):
                continue

            # Extract horse CNAMEs from entries
            for entry in race_card['entries'][:limit]:
                horse_cname = entry.get('jra_horse_id')
                horse_name = entry.get('horse_name')

                if horse_cname and horse_cname not in ['H_UNKNOWN', '']:
                    horse_info = {
                        'horse_name': horse_name,
                        'jra_horse_id': horse_cname,
                        'race_id': race_id,
                        'race_name': race.get('race_name')
                    }
                    horse_cnames.append(horse_info)

                    if len(horse_cnames) >= limit:
                        return horse_cnames

    return horse_cnames


if __name__ == "__main__":
    print("Fetching sample horse CNAMEs from recent race cards...")
    print("=" * 70)

    horse_cnames = get_sample_horse_cnames(limit=5)

    if not horse_cnames:
        print("\nNo horse CNAMEs found.")
        print("This might be because:")
        print("  1. There are no races scheduled for today/tomorrow")
        print("  2. The race card scraping failed")
        print("  3. The CNAME extraction logic needs adjustment")
        sys.exit(1)

    print(f"\nFound {len(horse_cnames)} horse CNAMEs:\n")

    for i, horse_info in enumerate(horse_cnames, 1):
        try:
            print(f"{i}. {horse_info['horse_name']}")
        except UnicodeEncodeError:
            print(f"{i}. [Unicode name]")

        print(f"   CNAME: {horse_info['jra_horse_id']}")
        print(f"   Race: {horse_info['race_id']}")
        print()

    print("=" * 70)
    print("\nTo debug horse profile HTML structure, run:")
    print(f"  python scripts/debug_horse_profile.py {horse_cnames[0]['jra_horse_id']}")
