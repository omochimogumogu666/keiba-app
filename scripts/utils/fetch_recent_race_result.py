"""
Fetch a recent race result HTML for analysis.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scrapers.jra_scraper import JRAScraper
from datetime import datetime, timedelta

def main():
    """Fetch recent race results."""
    with JRAScraper(delay=3) as scraper:
        # Try dates from recent past (races usually completed)
        for days_ago in range(1, 14):  # Try up to 2 weeks back
            target_date = datetime.now() - timedelta(days=days_ago)
            print(f"\nTrying date: {target_date.strftime('%Y-%m-%d')}")

            races = scraper.scrape_race_calendar(target_date)

            if races:
                print(f"Found {len(races)} races")

                # Try to scrape result for first race
                for race in races[:1]:  # Just try the first one
                    race_id = race.get('jra_race_id')
                    cname = race.get('cname')
                    race_name = race.get('race_name', 'Unknown')

                    print(f"\nRace: {race_name}")
                    print(f"Race ID: {race_id}")
                    print(f"CNAME: {cname}")

                    if cname:
                        # Try to fetch race result
                        result_url = f"{scraper.BASE_URL}/JRADB/accessS.html?CNAME={cname}"
                        print(f"Result URL: {result_url}")

                        response = scraper._make_request(result_url)

                        if response and 'パラメータエラー' not in response.text:
                            # Save HTML
                            output_file = f"docs/debug_race_result_{target_date.strftime('%Y%m%d')}.html"
                            os.makedirs("docs", exist_ok=True)

                            with open(output_file, 'w', encoding='utf-8') as f:
                                f.write(response.text)

                            print(f"\n✓ Saved HTML to: {output_file}")
                            print(f"HTML length: {len(response.text)} characters")

                            # Check if this is a result page
                            if '着順' in response.text or '結果' in response.text:
                                print("✓ This appears to be a result page")
                                return  # Success!
                            else:
                                print("Note: This might be a race card page, not a result page")
                        else:
                            print("× Failed to fetch or parameter error")
            else:
                print("No races found for this date")

        print("\n× Could not find valid race results in the past 2 weeks")
        print("Note: Race results may only be available for completed races")

if __name__ == "__main__":
    main()
