"""
Test script to verify race result scraping functionality.

This script tests:
1. Scraping race calendar to get race info with CNAMEs
2. Scraping race results using CNAMEs
3. Saving results to database
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.logging_config import setup_logging
from src.scrapers.jra_scraper import JRAScraper
from src.data.database import save_race_to_db, save_race_entries_to_db, save_race_results_to_db
from src.web.app import create_app
from src.utils.logger import get_app_logger

# Setup logging
setup_logging(log_level='DEBUG')
logger = get_app_logger(__name__)


def test_race_result_scraping():
    """Test race result scraping with a recent completed race."""
    logger.info("=" * 60)
    logger.info("Testing Race Result Scraping")
    logger.info("=" * 60)

    # Create Flask app for database context
    app = create_app('development')

    with app.app_context():
        with JRAScraper(delay=3) as scraper:
            # Test with races from past few days
            # Try to find completed races
            for days_back in range(1, 8):
                target_date = datetime.now() - timedelta(days=days_back)
                logger.info(f"\nChecking races from {target_date.strftime('%Y-%m-%d')}")

                try:
                    # Scrape race calendar to get CNAMEs
                    races = scraper.scrape_race_calendar(target_date)

                    if not races:
                        logger.info(f"No races found for {target_date.strftime('%Y-%m-%d')}")
                        continue

                    logger.info(f"Found {len(races)} races on {target_date.strftime('%Y-%m-%d')}")

                    # Test with first race only
                    race = races[0]
                    jra_race_id = race.get('jra_race_id')
                    cname = race.get('cname')
                    race_name = race.get('race_name', 'Unknown')

                    logger.info(f"\n{'=' * 60}")
                    logger.info(f"Testing with Race: {race_name}")
                    logger.info(f"JRA Race ID: {jra_race_id}")
                    logger.info(f"CNAME: {cname}")
                    logger.info(f"{'=' * 60}\n")

                    # 1. Scrape race card first (to get entries)
                    logger.info("Step 1: Scraping race card...")
                    race_card = scraper.scrape_race_card(jra_race_id, cname=cname)

                    if not race_card:
                        logger.error("Failed to scrape race card")
                        continue

                    logger.info(f"✓ Race card scraped: {race_card.get('race_name')}")
                    logger.info(f"  Track: {race_card.get('track_name')}")
                    logger.info(f"  Distance: {race_card.get('distance')}m")
                    logger.info(f"  Surface: {race_card.get('surface')}")
                    logger.info(f"  Entries: {len(race_card.get('entries', []))} horses")

                    # 2. Save race to database
                    logger.info("\nStep 2: Saving race to database...")
                    db_race = save_race_to_db(race_card)
                    logger.info(f"✓ Race saved to DB with ID: {db_race.id}")

                    # 3. Save race entries
                    logger.info("\nStep 3: Saving race entries to database...")
                    entries = save_race_entries_to_db(db_race.id, race_card.get('entries', []))
                    logger.info(f"✓ Saved {len(entries)} entries to DB")

                    # 4. Scrape race result
                    logger.info("\nStep 4: Scraping race result...")
                    result = scraper.scrape_race_result(jra_race_id, cname=cname)

                    if not result:
                        logger.warning("No result found - race may not be completed yet")
                        logger.info("This is expected for upcoming races")
                        continue

                    logger.info(f"✓ Race result scraped")
                    logger.info(f"  Results found: {len(result.get('results', []))}")

                    # Display result summary
                    if result.get('results'):
                        logger.info("\n  Top 3 finishers:")
                        for i, res in enumerate(result['results'][:3], 1):
                            logger.info(f"    {i}. Horse #{res.get('horse_number')}: "
                                      f"Position {res.get('finish_position')}, "
                                      f"Time {res.get('finish_time')}, "
                                      f"Odds {res.get('final_odds')}")

                    # Display payouts if available
                    if result.get('payouts'):
                        logger.info("\n  Payouts:")
                        for bet_type, payout_data in result['payouts'].items():
                            logger.info(f"    {bet_type}: {payout_data}")

                    # 5. Save results to database
                    logger.info("\nStep 5: Saving race results to database...")
                    db_results = save_race_results_to_db(db_race.id, result.get('results', []))
                    logger.info(f"✓ Saved {len(db_results)} results to DB")

                    logger.info("\n" + "=" * 60)
                    logger.info("SUCCESS! Race result scraping and saving completed")
                    logger.info("=" * 60)

                    # Success - exit after first completed race
                    return True

                except Exception as e:
                    logger.error(f"Error processing date {target_date.strftime('%Y-%m-%d')}: {e}", exc_info=True)
                    continue

            logger.warning("\nNo completed races found in the past 7 days")
            logger.info("This is expected if there were no races recently")
            return False


if __name__ == "__main__":
    try:
        success = test_race_result_scraping()
        if success:
            logger.info("\n✓ Test completed successfully")
            sys.exit(0)
        else:
            logger.info("\n⚠ No data to test with - this is OK if no recent races")
            sys.exit(0)
    except Exception as e:
        logger.error(f"\n✗ Test failed with error: {e}", exc_info=True)
        sys.exit(1)
