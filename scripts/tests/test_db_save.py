"""
Test script to verify database saving functionality.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from src.scrapers.netkeiba_scraper import NetkeibaScraper
from src.data.database import save_race_to_db, save_race_entries_to_db, save_race_results_to_db, save_payouts_to_db
from src.web.app import create_app
from src.data.models import db, Race, RaceEntry, RaceResult

def main():
    """Test database saving with a single recent race."""

    print("="*80)
    print("DATABASE SAVE TEST")
    print("="*80)

    # Create Flask app context
    app = create_app('development')

    with app.app_context():
        # Test 1: Check database connection
        print("\n1. Testing database connection...")
        try:
            db.create_all()
            print("   [OK] Database tables created")
            print(f"   [OK] Database location: {app.config['SQLALCHEMY_DATABASE_URI']}")
        except Exception as e:
            print(f"   [ERROR] Database connection failed: {e}")
            return

        # Test 2: Check existing data
        print("\n2. Checking existing data...")
        race_count = Race.query.count()
        entry_count = RaceEntry.query.count()
        result_count = RaceResult.query.count()
        print(f"   - Races: {race_count}")
        print(f"   - Entries: {entry_count}")
        print(f"   - Results: {result_count}")

        # Test 3: Scrape and save a single race from a past date with known races
        print("\n3. Testing scraping and saving...")
        test_date = datetime(2024, 12, 28)  # Saturday - more likely to have races

        with NetkeibaScraper(delay=3) as scraper:
            print(f"\n   Scraping races for {test_date.strftime('%Y-%m-%d')}...")
            races = scraper.scrape_race_calendar(test_date)

            if not races:
                print("   [INFO] No races found for this date")
                print("   (This is normal if there were no races on this date)")
                return

            print(f"   [OK] Found {len(races)} races")

            # Process first race only
            race_info = races[0]
            netkeiba_race_id = race_info.get('netkeiba_race_id')
            print(f"\n   Processing race {netkeiba_race_id}...")

            # Save race
            try:
                race = save_race_to_db(race_info)
                print(f"   [OK] Saved race (DB ID: {race.id})")
            except Exception as e:
                print(f"   [ERROR] Failed to save race: {e}")
                import traceback
                traceback.print_exc()
                return

            # Scrape and save race card
            try:
                race_card = scraper.scrape_race_card(netkeiba_race_id)

                if race_card and race_card.get('entries'):
                    # Update race with additional info
                    if race_card.get('race_info'):
                        for key, value in race_card['race_info'].items():
                            if value and key not in ['netkeiba_race_id']:
                                race_info[key] = value
                        race = save_race_to_db(race_info)

                    entries = save_race_entries_to_db(race.id, race_card['entries'])
                    print(f"   [OK] Saved {len(entries)} entries")
                else:
                    print("   [ERROR] No entries found")
                    return
            except Exception as e:
                print(f"   [ERROR] Failed to save entries: {e}")
                import traceback
                traceback.print_exc()
                return

            # Scrape and save results
            try:
                result = scraper.scrape_race_result(netkeiba_race_id)

                if result and result.get('results'):
                    race_results = save_race_results_to_db(race.id, result['results'])
                    print(f"   [OK] Saved {len(race_results)} results")

                    if result.get('payouts'):
                        payouts = save_payouts_to_db(race.id, result['payouts'])
                        print(f"   [OK] Saved {len(payouts)} payouts")

                    race.status = 'completed'
                    db.session.commit()
                else:
                    print("   [INFO] No results available (race not yet completed)")
            except Exception as e:
                print(f"   [WARN] Could not save results: {e}")

        # Test 4: Verify data was saved
        print("\n4. Verifying saved data...")
        race_count_after = Race.query.count()
        entry_count_after = RaceEntry.query.count()
        result_count_after = RaceResult.query.count()

        print(f"   - Races: {race_count} → {race_count_after} (+" +
              f"{race_count_after - race_count})")
        print(f"   - Entries: {entry_count} → {entry_count_after} (+" +
              f"{entry_count_after - entry_count})")
        print(f"   - Results: {result_count} → {result_count_after} (+" +
              f"{result_count_after - result_count})")

        if race_count_after > race_count:
            print("\n   [SUCCESS] Data successfully saved to database!")
        else:
            print("\n   [WARN] No new data was added (race may already exist)")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
