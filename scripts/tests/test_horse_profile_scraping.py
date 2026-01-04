"""
Integration test script for horse profile scraping.

This script demonstrates the complete flow of:
1. Scraping race cards to get horse CNAMEs
2. Scraping horse profiles using those CNAMEs
3. Saving horse profiles to the database
"""
import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.scrapers.jra_scraper import JRAScraper
from src.data.database import save_horse_profile_to_db
from src.web.app import create_app
from src.utils.logger import get_app_logger
from config.logging_config import setup_logging

setup_logging(log_level='INFO')
logger = get_app_logger(__name__)


def test_horse_profile_scraping(cname: str = None, save_to_db: bool = False):
    """
    Test horse profile scraping functionality.

    Args:
        cname: Optional horse CNAME to test. If not provided, will fetch from race card.
        save_to_db: Whether to save the profile to database
    """
    print("=" * 70)
    print("Horse Profile Scraping Test")
    print("=" * 70)

    with JRAScraper(delay=3) as scraper:
        # If no CNAME provided, get one from a race card
        if not cname:
            print("\nNo CNAME provided. Fetching from race card...")

            today = datetime.now()
            races = scraper.scrape_race_calendar(today)

            if not races:
                from datetime import timedelta
                tomorrow = today + timedelta(days=1)
                races = scraper.scrape_race_calendar(tomorrow)

            if not races:
                print("\nNo races found. Cannot proceed with test.")
                print("Please provide a horse CNAME manually:")
                print("  python scripts/test_horse_profile_scraping.py <cname>")
                return None

            # Get first race with CNAME
            for race in races[:3]:
                race_id = race.get('jra_race_id')
                race_cname = race.get('cname')

                if not race_cname:
                    continue

                print(f"\nScraping race card: {race_id}")
                race_card = scraper.scrape_race_card(race_id, cname=race_cname)

                if race_card and race_card.get('entries'):
                    # Get first horse CNAME
                    for entry in race_card['entries']:
                        horse_cname = entry.get('jra_horse_id')
                        horse_name = entry.get('horse_name')

                        if horse_cname and horse_cname not in ['H_UNKNOWN', '']:
                            cname = horse_cname
                            try:
                                print(f"Found horse: {horse_name} (CNAME: {cname})")
                            except UnicodeEncodeError:
                                print(f"Found horse: [Unicode name] (CNAME: {cname})")
                            break

                    if cname:
                        break

        if not cname:
            print("\nCould not obtain horse CNAME.")
            return None

        # Scrape horse profile
        print(f"\n{'=' * 70}")
        print(f"Scraping horse profile: {cname}")
        print(f"{'=' * 70}\n")

        profile = scraper.scrape_horse_profile(cname)

        if not profile:
            print("Failed to scrape horse profile.")
            return None

        # Display profile information
        print("\n--- Horse Profile ---")
        print(f"JRA Horse ID: {profile['jra_horse_id']}")

        try:
            print(f"Name: {profile.get('name', 'N/A')}")
        except UnicodeEncodeError:
            print("Name: [Unicode characters]")

        print(f"Birth Date: {profile.get('birth_date', 'N/A')}")
        print(f"Sex: {profile.get('sex', 'N/A')}")

        try:
            print(f"Sire: {profile.get('sire_name', 'N/A')} (ID: {profile.get('sire_id', 'N/A')})")
            print(f"Dam: {profile.get('dam_name', 'N/A')} (ID: {profile.get('dam_id', 'N/A')})")
            print(f"Trainer: {profile.get('trainer_name', 'N/A')} (ID: {profile.get('trainer_id', 'N/A')})")
            print(f"Owner: {profile.get('owner', 'N/A')}")
            print(f"Breeder: {profile.get('breeder', 'N/A')}")
        except UnicodeEncodeError:
            print("Pedigree/connections: [Unicode characters]")

        print(f"\n--- Career Statistics ---")
        print(f"Total Races: {profile.get('total_races', 0)}")
        print(f"Record: {profile.get('total_wins', 0)}-{profile.get('total_places', 0)}-{profile.get('total_shows', 0)}")
        print(f"Total Earnings: ¥{profile.get('total_earnings', 0):,}")

        print(f"\n--- Past Performances ---")
        print(f"Found {len(profile.get('past_performances', []))} past races")

        # Display first 5 past performances
        for i, perf in enumerate(profile.get('past_performances', [])[:5], 1):
            print(f"\n{i}. {perf.get('date', 'N/A')}")
            try:
                print(f"   Track: {perf.get('track', 'N/A')}")
                print(f"   Race: {perf.get('race_name', 'N/A')}")
            except UnicodeEncodeError:
                print("   Track/Race: [Unicode characters]")

            print(f"   Distance: {perf.get('distance', 'N/A')}m")
            print(f"   Surface: {perf.get('surface', 'N/A')}")
            print(f"   Finish: {perf.get('finish_position', 'N/A')}")
            print(f"   Time: {perf.get('finish_time', 'N/A')}s")

            try:
                print(f"   Jockey: {perf.get('jockey', 'N/A')}")
            except UnicodeEncodeError:
                print("   Jockey: [Unicode characters]")

            print(f"   Weight: {perf.get('weight', 'N/A')}kg")
            print(f"   Horse Weight: {perf.get('horse_weight', 'N/A')}kg")

        # Save to database if requested
        if save_to_db:
            print(f"\n{'=' * 70}")
            print("Saving to database...")
            print(f"{'=' * 70}\n")

            # Create Flask app context
            app = create_app('development')
            with app.app_context():
                try:
                    horse = save_horse_profile_to_db(profile)
                    print(f"✓ Successfully saved horse profile to database")
                    print(f"  Database ID: {horse.id}")
                    try:
                        print(f"  Name: {horse.name}")
                    except UnicodeEncodeError:
                        print("  Name: [Unicode characters]")
                except Exception as e:
                    print(f"✗ Error saving to database: {e}")
                    import traceback
                    traceback.print_exc()

        print(f"\n{'=' * 70}")
        print("Test completed successfully!")
        print(f"{'=' * 70}\n")

        return profile


if __name__ == "__main__":
    # Parse command line arguments
    cname = None
    save_to_db = False

    if len(sys.argv) > 1:
        cname = sys.argv[1]

    if len(sys.argv) > 2 and sys.argv[2] == '--save':
        save_to_db = True

    # Run test
    test_horse_profile_scraping(cname=cname, save_to_db=save_to_db)

    # Display usage
    if not cname:
        print("\nUsage:")
        print("  python scripts/test_horse_profile_scraping.py [cname] [--save]")
        print("\nExamples:")
        print("  python scripts/test_horse_profile_scraping.py")
        print("  python scripts/test_horse_profile_scraping.py pw01sde3240230405/E3")
        print("  python scripts/test_horse_profile_scraping.py pw01sde3240230405/E3 --save")
