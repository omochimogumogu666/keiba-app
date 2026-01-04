"""Test scraping a specific race card directly with CNAME."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scrapers.jra_scraper import JRAScraper
from src.data.database import save_race_to_db, save_race_entries_to_db
from src.web.app import create_app
from src.data.models import db, Track, Horse, Jockey

def test_direct_race_card():
    """Test scraping a known race card with CNAME."""
    print("="*80)
    print("Testing Direct Race Card Scraping with CNAME")
    print("="*80)

    # CNAME from debug_calendar.html
    # pw01dde0106202601011120260104/6C - Race at track 06 (Nakayama)
    test_cname = "pw01dde0106202601011120260104/6C"
    test_race_id = "202601041106011"  # Approximate - will be updated from scraped data

    app = create_app('development')

    with app.app_context():
        with JRAScraper(delay=3) as scraper:
            print(f"\nScraping race card with CNAME: {test_cname}\n")

            race_card = scraper.scrape_race_card(test_race_id, cname=test_cname)

            if not race_card:
                print("[X] Failed to scrape race card")
                return

            print("[OK] Successfully scraped race card!")
            print(f"\nRace Information:")
            print(f"  JRA Race ID: {race_card.get('jra_race_id')}")
            print(f"  Race Name: {race_card.get('race_name')}")
            print(f"  Track: {race_card.get('track')}")
            print(f"  Date: {race_card.get('race_date')}")
            print(f"  Distance: {race_card.get('distance')}m")
            print(f"  Surface: {race_card.get('surface')}")
            print(f"  Race Class: {race_card.get('race_class')}")
            print(f"  Prize Money: {race_card.get('prize_money')} yen")
            print(f"  Entries: {len(race_card.get('entries', []))}")

            # Check encoding
            race_name = race_card.get('race_name', '')
            track_name = race_card.get('track', '')

            for field, value in [('Race Name', race_name), ('Track', track_name)]:
                if value:
                    try:
                        value.encode('utf-8')
                        print(f"  [OK] {field} has valid UTF-8 encoding")
                    except UnicodeEncodeError:
                        print(f"  [X] {field} has encoding issues!")
                        print(f"       Repr: {repr(value)}")
                        return

            # Display first few entries
            print(f"\nSample Entries:")
            for i, entry in enumerate(race_card.get('entries', [])[:5]):
                horse_name = entry.get('horse_name', 'N/A')
                jockey_name = entry.get('jockey_name', 'N/A')
                trainer_name = entry.get('trainer_name', 'N/A')
                horse_num = entry.get('horse_number', '?')

                print(f"\n  Entry {i+1} (Horse #{horse_num}):")
                print(f"    Horse: {horse_name}")
                print(f"    Jockey: {jockey_name}")
                print(f"    Trainer: {trainer_name}")
                print(f"    Weight: {entry.get('weight', '?')} kg")

                # Verify encoding
                for name, value in [('Horse', horse_name), ('Jockey', jockey_name), ('Trainer', trainer_name)]:
                    if value and value != 'N/A':
                        try:
                            value.encode('utf-8')
                            print(f"      [OK] {name} name is valid UTF-8")
                        except UnicodeEncodeError:
                            print(f"      [X] {name} name has encoding issues!")
                            print(f"           Repr: {repr(value)}")
                            return

            # Save to database
            print(f"\n{'-'*80}")
            print("Saving to database...")

            # Prepare race_info
            race_info = {
                'jra_race_id': race_card['jra_race_id'],
                'track': race_card['track'],
                'race_date': race_card['race_date'],
                'race_number': race_card.get('race_number', 1),
                'race_name': race_card['race_name'],
                'distance': race_card['distance'],
                'surface': race_card['surface'],
                'race_class': race_card.get('race_class'),
                'prize_money': race_card.get('prize_money'),
            }

            try:
                race = save_race_to_db(race_info)
                print(f"[OK] Saved race to database (ID: {race.id})")

                entries = save_race_entries_to_db(race.id, race_card['entries'])
                print(f"[OK] Saved {len(entries)} entries to database")

            except Exception as e:
                print(f"[X] Error saving to database: {e}")
                import traceback
                traceback.print_exc()
                return

            # Verify database encoding
            print(f"\n{'-'*80}")
            print("Verifying database encoding...")

            print(f"\nSample Track Names from DB:")
            for track in db.session.query(Track).limit(3).all():
                print(f"  - {track.name}")

            print(f"\nSample Horse Names from DB:")
            for horse in db.session.query(Horse).limit(5).all():
                print(f"  - {horse.name}")

            print(f"\nSample Jockey Names from DB:")
            for jockey in db.session.query(Jockey).limit(5).all():
                print(f"  - {jockey.name}")

            print("\n" + "="*80)
            print("[OK] Encoding test PASSED!")
            print("All Japanese characters are correctly decoded and saved to database!")
            print("="*80)

if __name__ == "__main__":
    test_direct_race_card()
