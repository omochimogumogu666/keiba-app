"""
Test script to verify database saving functionality.
"""
from datetime import datetime
from src.scrapers.jra_scraper import JRAScraper
from src.data.database import save_race_to_db, save_race_entries_to_db
from src.data.models import db, Race, RaceEntry
from src.web.app import create_app
from config.logging_config import setup_logging

setup_logging(log_level='INFO')

def test_save_race_card_to_db():
    """Test scraping and saving race card to database."""
    print("\n=== Testing Database Save ===")

    # Create Flask app and initialize database
    app = create_app()

    with app.app_context():
        # Create tables
        db.create_all()

        # Scrape a race card
        with JRAScraper(delay=1) as scraper:
            cname = "pw01dde0106202601011120260104/6C"
            race_id = "2026010101"

            print(f"\nScraping race: {race_id}")
            race_card = scraper.scrape_race_card(race_id, cname=cname)

            if not race_card:
                print("Failed to scrape race card")
                return

            print(f"Successfully scraped {len(race_card['entries'])} entries")

            # Prepare race data for saving
            race_data = {
                'jra_race_id': race_id,
                'track': '中山',  # Nakayama (from CNAME 01)
                'race_date': datetime(2026, 1, 1),
                'race_number': 11,  # From CNAME
                'race_name': '日経新春杯',  # Would normally scrape this
                'distance': 2000,
                'surface': 'turf',
                'race_class': 'G3',
                'status': 'upcoming'
            }

            # Add race info from scraping
            race_data.update(race_card['race_info'])

            print(f"\nSaving race to database...")
            try:
                print(f"  Race: {race_data['race_name']}")
                print(f"  Track: {race_data['track']}")
            except UnicodeEncodeError:
                print(f"  Race: [Unicode characters]")
                print(f"  Track: [Unicode characters]")
            print(f"  Date: {race_data['race_date']}")
            print(f"  Distance: {race_data['distance']}m")
            print(f"  Prize: {race_data.get('prize_money', 0):,} yen")

            # Save race
            try:
                race = save_race_to_db(race_data)
                print(f"[OK] Race saved with ID: {race.id}")
            except Exception as e:
                print(f"[ERROR] Failed to save race: {e}")
                import traceback
                traceback.print_exc()
                return

            # Save entries
            print(f"\nSaving {len(race_card['entries'])} entries...")
            try:
                entries = save_race_entries_to_db(race.id, race_card['entries'])
                print(f"[OK] Saved {len(entries)} entries")

                # Show first entry details
                if entries:
                    first_entry = entries[0]
                    print(f"\nFirst entry saved:")
                    try:
                        print(f"  Horse: {first_entry.horse.name}")
                        print(f"  Jockey: {first_entry.jockey.name}")
                    except UnicodeEncodeError:
                        print(f"  Horse: [Unicode characters]")
                        print(f"  Jockey: [Unicode characters]")
                    print(f"  Weight: {first_entry.weight}kg")
                    print(f"  Post Position: {first_entry.post_position}")
                    print(f"  Horse Number: {first_entry.horse_number}")

            except Exception as e:
                print(f"[ERROR] Failed to save entries: {e}")
                import traceback
                traceback.print_exc()
                return

            # Query and verify
            print(f"\n=== Verifying Database ===")

            saved_race = Race.query.filter_by(jra_race_id=race_id).first()
            if saved_race:
                print(f"[OK] Race found in database:")
                print(f"  ID: {saved_race.id}")
                try:
                    print(f"  Name: {saved_race.race_name}")
                except UnicodeEncodeError:
                    print(f"  Name: [Unicode characters]")
                print(f"  Entries: {len(saved_race.race_entries)}")
            else:
                print(f"[ERROR] Race not found in database")

            print(f"\n=== Test Complete ===")

if __name__ == "__main__":
    test_save_race_card_to_db()
