"""Test scraping a single race to see what data we actually get."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from src.scrapers.netkeiba_scraper import NetkeibaScraper
import json

def main():
    """Test single race scraping."""

    print("=" * 80)
    print("SINGLE RACE SCRAPE TEST")
    print("=" * 80)

    test_race_id = "202406050912"  # Known race from 2024-12-28

    with NetkeibaScraper(delay=3) as scraper:
        print(f"\n1. Scraping race card for {test_race_id}...")
        race_card = scraper.scrape_race_card(test_race_id)

        if race_card:
            print("\n[OK] Race card scraped successfully!")
            print("\nRace Info:")
            print(json.dumps(race_card.get('race_info', {}), indent=2, ensure_ascii=False))

            print(f"\nEntries: {len(race_card.get('entries', []))}")
            if race_card.get('entries'):
                print("\nFirst entry sample:")
                print(json.dumps(race_card['entries'][0], indent=2, ensure_ascii=False))
        else:
            print("\n[ERROR] Failed to scrape race card")
            return

        print(f"\n2. Scraping race result for {test_race_id}...")
        race_result = scraper.scrape_race_result(test_race_id)

        if race_result:
            print("\n[OK] Race result scraped successfully!")
            print(f"\nResults: {len(race_result.get('results', []))}")

            if race_result.get('results'):
                print("\nFirst result sample:")
                print(json.dumps(race_result['results'][0], indent=2, ensure_ascii=False))

            print(f"\nPayouts: {len(race_result.get('payouts', {}))}")
            if race_result.get('payouts'):
                print("\nPayouts:")
                print(json.dumps(race_result['payouts'], indent=2, ensure_ascii=False))
        else:
            print("\n[INFO] No race result available (race not completed or error)")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
