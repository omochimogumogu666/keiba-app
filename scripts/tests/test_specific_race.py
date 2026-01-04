"""
Test scraping a specific race to check HTML structure.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.logging_config import setup_logging
from src.scrapers.netkeiba_scraper import NetkeibaScraper
from src.utils.logger import get_app_logger

setup_logging(log_level='INFO')
logger = get_app_logger(__name__)


def test_race(race_id):
    """Test scraping specific race."""

    with NetkeibaScraper(delay=3) as scraper:
        # Test race card
        logger.info(f"\n=== Testing Race Card: {race_id} ===")
        race_card = scraper.scrape_race_card(race_id)

        if race_card:
            logger.info("Race Card Data:")
            logger.info(f"  Race Name: {race_card['race_info'].get('race_name')}")
            logger.info(f"  Distance: {race_card['race_info'].get('distance')}")
            logger.info(f"  Surface: {race_card['race_info'].get('surface')}")
            logger.info(f"  Race Class: {race_card['race_info'].get('race_class')}")
            logger.info(f"  Weather: {race_card['race_info'].get('weather')}")
            logger.info(f"  Track Condition: {race_card['race_info'].get('track_condition')}")
            logger.info(f"  Entries: {len(race_card['entries'])}")

            if race_card['entries']:
                entry = race_card['entries'][0]
                logger.info(f"\n  First Entry:")
                logger.info(f"    Horse: {entry.get('horse_name')}")
                logger.info(f"    Jockey: {entry.get('jockey_name')}")
                logger.info(f"    Weight: {entry.get('weight')}")
                logger.info(f"    Horse Weight: {entry.get('horse_weight')}")
        else:
            logger.error("Failed to scrape race card")

        # Test race result
        logger.info(f"\n=== Testing Race Result: {race_id} ===")
        race_result = scraper.scrape_race_result(race_id)

        if race_result:
            logger.info(f"Result Data:")
            logger.info(f"  Results: {len(race_result.get('results', []))}")
            logger.info(f"  Payouts: {len(race_result.get('payouts', []))}")

            if race_result.get('results'):
                result = race_result['results'][0]
                logger.info(f"\n  First Result:")
                logger.info(f"    Horse Number: {result.get('horse_number')}")
                logger.info(f"    Finish Position: {result.get('finish_position')}")
                logger.info(f"    Time: {result.get('finish_time')}")
                logger.info(f"    Final Odds: {result.get('final_odds')}")
        else:
            logger.warning("No race result found (race may not have run yet)")


if __name__ == "__main__":
    # Test with a known completed race from December 2025
    test_race("202506050712")  # 2025年12月28日, 中山(06), 5回, 7日, 12R
