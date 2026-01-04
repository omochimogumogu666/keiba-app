"""
Debug script to fetch and save race result HTML for analysis.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scrapers.jra_scraper import JRAScraper
from datetime import datetime

def save_race_result_html(race_id: str, cname: str, output_file: str):
    """Fetch and save race result HTML to file."""
    with JRAScraper(delay=3) as scraper:
        # Construct URL
        result_url = f"{scraper.BASE_URL}/JRADB/accessK.html?CNAME={cname}"

        print(f"Fetching: {result_url}")
        response = scraper._make_request(result_url)

        if response:
            # Save HTML to file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print(f"Saved HTML to: {output_file}")
            print(f"HTML length: {len(response.text)} characters")
        else:
            print("Failed to fetch HTML")

if __name__ == "__main__":
    # First, get a race with CNAME from recent races
    print("Fetching recent races to get CNAMEs...")

    with JRAScraper(delay=3) as scraper:
        # Try to get races from this week
        races = scraper.scrape_race_calendar(datetime.now())

        if races:
            print(f"\nFound {len(races)} races")

            # Use the first race with a CNAME
            for race in races[:3]:  # Check first 3 races
                race_id = race.get('jra_race_id')
                cname = race.get('cname')
                race_name = race.get('race_name', 'Unknown')

                print(f"\nRace: {race_name}")
                print(f"Race ID: {race_id}")
                print(f"CNAME: {cname}")

                if cname:
                    # Save the HTML
                    output_file = f"docs/debug_race_result_{race_id}.html"
                    os.makedirs("docs", exist_ok=True)
                    save_race_result_html(race_id, cname, output_file)
                    break
        else:
            print("No races found. Using test CNAME...")
            # Use a test CNAME from a recent race (may expire)
            test_cname = "pw01dli0106202601011120260104/6C"
            test_race_id = "2026010401011"
            output_file = "docs/debug_race_result_test.html"
            os.makedirs("docs", exist_ok=True)
            save_race_result_html(test_race_id, test_cname, output_file)
