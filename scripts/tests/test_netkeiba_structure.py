"""
Test script to fetch and analyze actual netkeiba.com HTML structure.
This helps us understand the real DOM structure to fix the scraper.
"""
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time

BASE_URL = "https://race.netkeiba.com"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

def test_race_calendar():
    """Test fetching race calendar page."""
    print("=" * 80)
    print("Testing Race Calendar Page")
    print("=" * 80)

    # Try a recent date (2025-12-28 from the debug file)
    url = f"{BASE_URL}/top/race_list.html"
    params = {'kaisai_date': '20251228'}

    headers = {'User-Agent': USER_AGENT}

    print(f"\nFetching: {url}")
    print(f"Params: {params}")

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()

        print(f"Status: {response.status_code}")
        print(f"Content-Type: {response.headers.get('Content-Type')}")
        print(f"Content-Length: {len(response.content)} bytes")

        # Parse with EUC-JP
        html_text = response.content.decode('euc-jp', errors='replace')
        soup = BeautifulSoup(html_text, 'lxml')

        # Save to file for inspection
        output_file = 'data/netkeiba_calendar_test.html'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(soup.prettify())
        print(f"\nSaved HTML to: {output_file}")

        # Analyze structure
        print("\n" + "=" * 80)
        print("HTML Structure Analysis")
        print("=" * 80)

        # Check for race list containers
        race_list_divs = soup.find_all('div', class_=lambda x: x and 'RaceList' in x if x else False)
        print(f"\nDivs with 'RaceList' in class: {len(race_list_divs)}")
        for div in race_list_divs[:5]:
            print(f"  - {div.get('class')}")

        # Check for race links
        race_links = soup.find_all('a', href=lambda x: x and '/race/' in x if x else False)
        print(f"\nLinks with '/race/' in href: {len(race_links)}")
        for link in race_links[:10]:
            print(f"  - {link.get('href')}")

        # Check for race_id in links
        race_id_links = soup.find_all('a', href=lambda x: x and 'race_id=' in x if x else False)
        print(f"\nLinks with 'race_id=' in href: {len(race_id_links)}")
        for link in race_id_links[:5]:
            href = link.get('href')
            text = link.get_text().strip()[:50]
            print(f"  - {href} | {text}")

        # Check for JavaScript-loaded content indicators
        script_tags = soup.find_all('script')
        print(f"\nScript tags: {len(script_tags)}")

        ajax_scripts = [s for s in script_tags if s.string and ('ajax' in s.string.lower() or 'json' in s.string.lower())]
        print(f"Scripts with 'ajax' or 'json': {len(ajax_scripts)}")

        # Check for common race-related classes
        common_classes = [
            'RaceList_DataItem',
            'RaceList_Item',
            'Race_Item',
            'RaceCard',
            'RaceData',
            'Shutuba',
        ]

        print("\nSearching for common class patterns:")
        for cls in common_classes:
            elements = soup.find_all(class_=lambda x: x and cls in x if x else False)
            if elements:
                print(f"  - '{cls}': {len(elements)} elements")
                for elem in elements[:2]:
                    print(f"    - {elem.name} class={elem.get('class')}")

        return soup

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_race_card():
    """Test fetching a specific race card."""
    print("\n\n" + "=" * 80)
    print("Testing Race Card Page")
    print("=" * 80)

    # Use a known race_id format: YYYYKKRRDDNN
    # Example: 202506050811 (2025, track 06, meeting 05, day 08, race 11)
    race_id = '202506050811'

    url = f"{BASE_URL}/race/shutuba.html"
    params = {'race_id': race_id}

    headers = {'User-Agent': USER_AGENT}

    print(f"\nFetching: {url}")
    print(f"Params: {params}")

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        print(f"Status: {response.status_code}")

        if response.status_code != 200:
            print("  Note: Race may not exist yet (future date)")
            return None

        # Parse
        html_text = response.content.decode('euc-jp', errors='replace')
        soup = BeautifulSoup(html_text, 'lxml')

        # Save to file
        output_file = 'data/netkeiba_shutuba_test.html'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(soup.prettify())
        print(f"Saved HTML to: {output_file}")

        # Analyze structure
        print("\nLooking for shutuba table...")
        tables = soup.find_all('table', class_=lambda x: x and 'Shutuba' in x if x else False)
        print(f"Tables with 'Shutuba' in class: {len(tables)}")

        for table in tables:
            print(f"  - {table.get('class')}")
            rows = table.find_all('tr')
            print(f"    Rows: {len(rows)}")

        # Check for horse list
        horse_lists = soup.find_all(class_=lambda x: x and 'Horse' in x if x else False)
        print(f"\nElements with 'Horse' in class: {len(horse_lists)}")
        unique_classes = set()
        for elem in horse_lists:
            classes = elem.get('class')
            if classes:
                unique_classes.add(' '.join(classes))

        print("Unique Horse-related classes:")
        for cls in sorted(unique_classes)[:10]:
            print(f"  - {cls}")

        return soup

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("NetKeiba Structure Test")
    print("=" * 80)
    print(f"Time: {datetime.now()}")
    print()

    # Test calendar
    calendar_soup = test_race_calendar()

    time.sleep(3)  # Rate limiting

    # Test race card
    shutuba_soup = test_race_card()

    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review saved HTML files:")
    print("   - data/netkeiba_calendar_test.html")
    print("   - data/netkeiba_shutuba_test.html")
    print("2. Identify actual class names and structure")
    print("3. Update NetkeibaScraper selectors accordingly")
