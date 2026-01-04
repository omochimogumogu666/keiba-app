"""Test scraping this week's races with corrected encoding."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from src.scrapers.jra_scraper import JRAScraper
from src.data.database import save_race_to_db, save_race_entries_to_db
from src.web.app import create_app
from src.data.models import db, Track, Race, Horse, Jockey, Trainer

def test_and_save():
    """Test scraping and saving this week's races."""
    print("="*80)
    print("Testing This Week's Races with Encoding Fix")
    print("="*80)

    app = create_app('development')

    with app.app_context():
        with JRAScraper(delay=3) as scraper:
            # Scrape today's races (or this week's)
            today = datetime.now()
            print(f"\nScraping races for {today.strftime('%Y-%m-%d')}...")

            races = scraper.scrape_race_calendar(today)

            if not races:
                print("[!] No races found for today")
                print("This is expected if there are no races today.")
                print("The scraper only gets 'featured races' from thisweek page.")
                return

            print(f"[OK] Found {len(races)} races\n")

            for i, race_info in enumerate(races):
                print(f"\n[{i+1}/{len(races)}] Processing race...")
                print(f"  JRA Race ID: {race_info.get('jra_race_id')}")
                print(f"  Track: {race_info.get('track')}")
                print(f"  Race Name: {race_info.get('race_name')}")
                print(f"  CNAME: {race_info.get('cname')}")

                # Verify encoding
                track_name = race_info.get('track', '')
                race_name = race_info.get('race_name', '')

                for field, value in [('Track', track_name), ('Race Name', race_name)]:
                    if value:
                        try:
                            value.encode('utf-8')
                            print(f"  [OK] {field} has valid UTF-8 encoding")
                        except UnicodeEncodeError:
                            print(f"  [X] {field} has encoding issues!")
                            return

                # Save race to database
                try:
                    race = save_race_to_db(race_info)
                    print(f"  [OK] Saved race to database (ID: {race.id})")
                except Exception as e:
                    print(f"  [X] Error saving race: {e}")
                    continue

                # Scrape race card
                jra_race_id = race_info['jra_race_id']
                cname = race_info.get('cname')

                if not cname:
                    print("  [!] No CNAME, skipping race card")
                    continue

                try:
                    race_card = scraper.scrape_race_card(jra_race_id, cname=cname)

                    if race_card and race_card.get('entries'):
                        print(f"  [OK] Scraped {len(race_card['entries'])} entries")

                        # Check first entry encoding
                        if race_card['entries']:
                            first_entry = race_card['entries'][0]
                            horse_name = first_entry.get('horse_name', '')
                            jockey_name = first_entry.get('jockey_name', '')

                            print(f"    Sample entry:")
                            print(f"      Horse: {horse_name}")
                            print(f"      Jockey: {jockey_name}")

                            # Verify encoding
                            for field, value in [('Horse', horse_name), ('Jockey', jockey_name)]:
                                if value:
                                    try:
                                        value.encode('utf-8')
                                        print(f"      [OK] {field} name has valid UTF-8")
                                    except UnicodeEncodeError:
                                        print(f"      [X] {field} name has encoding issues!")
                                        return

                        # Save entries to database
                        try:
                            entries = save_race_entries_to_db(race.id, race_card['entries'])
                            print(f"  [OK] Saved {len(entries)} entries to database")
                        except Exception as e:
                            print(f"  [X] Error saving entries: {e}")
                            import traceback
                            traceback.print_exc()

                    else:
                        print("  [!] No entries found in race card")

                except Exception as e:
                    print(f"  [X] Error scraping race card: {e}")
                    import traceback
                    traceback.print_exc()

        # Check database for encoding
        print("\n" + "="*80)
        print("Checking Database for Encoding Issues")
        print("="*80)

        print(f"\nDatabase counts:")
        print(f"  Tracks: {db.session.query(Track).count()}")
        print(f"  Races: {db.session.query(Race).count()}")
        print(f"  Horses: {db.session.query(Horse).count()}")
        print(f"  Jockeys: {db.session.query(Jockey).count()}")

        print(f"\nSample Track Names:")
        for track in db.session.query(Track).limit(5).all():
            print(f"  - {track.name}")

        print(f"\nSample Horse Names:")
        for horse in db.session.query(Horse).limit(5).all():
            print(f"  - {horse.name}")

        print(f"\nSample Jockey Names:")
        for jockey in db.session.query(Jockey).limit(5).all():
            print(f"  - {jockey.name}")

        print("\n" + "="*80)
        print("[OK] Test completed!")
        print("="*80)

if __name__ == "__main__":
    test_and_save()
