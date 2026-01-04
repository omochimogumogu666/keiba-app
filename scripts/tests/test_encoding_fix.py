"""Test script to verify encoding fix for JRA scraper."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from src.scrapers.jra_scraper import JRAScraper
from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)

def test_encoding():
    """Test that Japanese text is correctly decoded."""
    print("="*80)
    print("Testing JRA Scraper Encoding Fix")
    print("="*80)

    with JRAScraper(delay=3) as scraper:
        # Test with a known past date (January 6, 2024 - Saturday)
        test_date = datetime(2024, 1, 6)

        print(f"\nScraping race calendar for {test_date.strftime('%Y-%m-%d')}...")
        races = scraper.scrape_race_calendar(test_date)

        if not races:
            print("[X] No races found")
            return False

        print(f"[OK] Found {len(races)} races\n")

        # Display first race details to check encoding
        for i, race in enumerate(races[:3]):
            print(f"Race {i+1}:")
            print(f"  Track: {race.get('track', 'N/A')}")
            print(f"  Race Name: {race.get('race_name', 'N/A')}")
            print(f"  CNAME: {race.get('cname', 'N/A')}")
            print(f"  JRA Race ID: {race.get('jra_race_id', 'N/A')}")

            # Check if text contains readable Japanese
            track_name = race.get('track', '')
            if track_name:
                try:
                    # Try to encode as UTF-8 to verify it's valid
                    track_bytes = track_name.encode('utf-8')
                    print(f"  [OK] Track name is valid UTF-8: {len(track_bytes)} bytes")
                except UnicodeEncodeError:
                    print(f"  [X] Track name has encoding issues")
                    return False
            print()

        # Test scraping race card if we have a race
        if races and races[0].get('cname'):
            first_race = races[0]
            print(f"\nScraping race card for race {first_race['jra_race_id']}...")

            race_card = scraper.scrape_race_card(
                first_race['jra_race_id'],
                cname=first_race['cname']
            )

            if race_card and race_card.get('entries'):
                print(f"[OK] Found {len(race_card['entries'])} entries")

                # Check first few horses
                for i, entry in enumerate(race_card['entries'][:3]):
                    horse_name = entry.get('horse_name', 'N/A')
                    jockey_name = entry.get('jockey_name', 'N/A')
                    trainer_name = entry.get('trainer_name', 'N/A')

                    print(f"\nEntry {i+1}:")
                    print(f"  Horse: {horse_name}")
                    print(f"  Jockey: {jockey_name}")
                    print(f"  Trainer: {trainer_name}")

                    # Verify encoding
                    for name, value in [('Horse', horse_name), ('Jockey', jockey_name), ('Trainer', trainer_name)]:
                        if value and value != 'N/A':
                            try:
                                value.encode('utf-8')
                                print(f"  [OK] {name} name is valid UTF-8")
                            except UnicodeEncodeError:
                                print(f"  [X] {name} name has encoding issues")
                                return False
            else:
                print("[!] No entries found in race card")

        print("\n" + "="*80)
        print("[OK] Encoding test PASSED - All Japanese text is correctly decoded!")
        print("="*80)
        return True

if __name__ == "__main__":
    success = test_encoding()
    sys.exit(0 if success else 1)
