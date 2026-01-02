"""
Script to scrape race results and save to database.

This script demonstrates the complete workflow:
1. Get races from a specific date
2. Scrape race results (for completed races only)
3. Save results to database

Usage:
    python scripts/scrape_and_save_results.py [date]

    date: Optional date in YYYY-MM-DD format (default: yesterday)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from src.scrapers.jra_scraper import JRAScraper
from src.data.database import save_race_to_db, save_race_entries_to_db, save_race_results_to_db
from src.web.app import create_app
from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)


def scrape_and_save_race_results(target_date):
    """
    Scrape and save race results for a specific date.

    Args:
        target_date: Date to scrape results for
    """
    # Create Flask app context for database operations
    app = create_app('development')

    with app.app_context():
        with JRAScraper(delay=3) as scraper:
            print(f"\n{'='*80}")
            print(f"Scraping race results for {target_date.strftime('%Y-%m-%d')}")
            print(f"{'='*80}\n")

            # Step 1: Get race calendar for the date
            races = scraper.scrape_race_calendar(target_date)

            if not races:
                print(f"No races found for {target_date.strftime('%Y-%m-%d')}")
                return

            print(f"Found {len(races)} races\n")

            # Step 2: Process each race
            for i, race_info in enumerate(races):
                print(f"\n[{i+1}/{len(races)}] Processing race...")
                try:
                    print(f"  Race: {race_info.get('race_name', 'Unknown')}")
                except UnicodeEncodeError:
                    print(f"  Race: [Japanese name]")

                print(f"  JRA Race ID: {race_info['jra_race_id']}")
                print(f"  CNAME: {race_info.get('cname', 'N/A')}")

                jra_race_id = race_info['jra_race_id']
                cname = race_info.get('cname')

                if not cname:
                    print("  ✗ Skipping: No CNAME available")
                    continue

                # Step 2a: Save race to database
                try:
                    race = save_race_to_db(race_info)
                    print(f"  ✓ Saved race to database (ID: {race.id})")
                except Exception as e:
                    print(f"  ✗ Error saving race: {e}")
                    continue

                # Step 2b: Scrape race card (entries)
                try:
                    race_card = scraper.scrape_race_card(jra_race_id, cname=cname)

                    if race_card and race_card.get('entries'):
                        entries = save_race_entries_to_db(race.id, race_card['entries'])
                        print(f"  ✓ Saved {len(entries)} entries to database")
                    else:
                        print("  ✗ No entries found in race card")
                        continue

                except Exception as e:
                    print(f"  ✗ Error scraping/saving race card: {e}")
                    logger.exception("Race card error")
                    continue

                # Step 2c: Scrape race result (only if race is completed)
                try:
                    result = scraper.scrape_race_result(jra_race_id, cname=cname)

                    if result and result.get('results'):
                        # Save results to database
                        race_results = save_race_results_to_db(race.id, result['results'])
                        print(f"  ✓ Saved {len(race_results)} race results to database")

                        # Update race status to completed
                        race.status = 'completed'
                        from src.data.models import db
                        db.session.commit()
                        print("  ✓ Updated race status to 'completed'")

                    else:
                        print("  ℹ No results available (race may not be completed yet)")

                except Exception as e:
                    print(f"  ⚠ Could not scrape results (race may not be completed): {e}")
                    logger.debug(f"Result scraping failed: {e}")

            print(f"\n{'='*80}")
            print("Processing complete")
            print(f"{'='*80}\n")


def main():
    """Main function."""
    # Parse command line arguments
    if len(sys.argv) > 1:
        date_str = sys.argv[1]
        try:
            target_date = datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            print(f"Error: Invalid date format '{date_str}'. Use YYYY-MM-DD")
            sys.exit(1)
    else:
        # Default: yesterday (more likely to have completed races)
        target_date = datetime.now() - timedelta(days=1)

    scrape_and_save_race_results(target_date)


if __name__ == "__main__":
    main()
