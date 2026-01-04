"""
Test script for JRA scraper.
"""
from datetime import datetime
from config.logging_config import setup_logging
from src.scrapers.jra_scraper import JRAScraper

# Setup logging
setup_logging(log_level='DEBUG')

def test_race_calendar():
    """Test race calendar scraping."""
    print("\n=== Testing Race Calendar Scraping ===")

    with JRAScraper(delay=1) as scraper:
        # Test with today's date
        today = datetime.now()
        print(f"Scraping races for: {today.strftime('%Y-%m-%d')}")

        # First, test without date filter
        print("\n--- Getting all races from this week ---")
        import re
        response = scraper._make_request(f"{scraper.BASE_URL}/keiba/thisweek/")
        if response:
            soup = scraper._parse_html(response.text)
            race_links = scraper._extract_race_links(soup)

            print(f"Total race links extracted: {len(race_links)}")

            # Parse each link
            for i, link_info in enumerate(race_links[:5], 1):  # Show first 5
                cname = link_info['cname']
                link_text = link_info['text']

                # Extract date from CNAME
                date_match = re.search(r'(20\d{6})', cname)
                if date_match:
                    date_str = date_match.group(1)
                    try:
                        race_date = datetime.strptime(date_str, '%Y%m%d')
                        print(f"\nRace {i}:")
                        print(f"  CNAME: {cname}")
                        print(f"  Date: {race_date.strftime('%Y-%m-%d')}")
                        print(f"  Text: {link_text if link_text else '[empty]'}")
                    except ValueError:
                        pass

        # Now test with scrape_race_calendar
        races = scraper.scrape_race_calendar(today)

        print(f"\nFiltered races for {today.strftime('%Y-%m-%d')}: {len(races)}")

        if races:
            print("\nFirst race details:")
            for key, value in races[0].items():
                try:
                    print(f"  {key}: {value}")
                except UnicodeEncodeError:
                    print(f"  {key}: [Unicode characters]")
        else:
            print("\nNo races found for exact date match.")
            print("Tip: Use get_upcoming_races() to get all upcoming races")

        return races

def test_race_card():
    """Test race card scraping with actual CNAME."""
    print("\n=== Testing Race Card Scraping ===")

    with JRAScraper(delay=1) as scraper:
        # Use a CNAME discovered from the calendar
        # Format: pw01dde0106202601011120260104/6C
        # This appears to be for 2026-01-01
        cname = "pw01dde0106202601011120260104/6C"
        race_id = "2026010101"  # Constructed from date + race number

        print(f"Testing with CNAME: {cname}")
        print(f"Race ID: {race_id}")

        race_card = scraper.scrape_race_card(race_id, cname=cname)

        if race_card:
            print("\n--- Race Card Data ---")

            # Print race info
            if 'race_info' in race_card:
                print("\nRace Info:")
                for key, value in race_card['race_info'].items():
                    try:
                        print(f"  {key}: {value}")
                    except UnicodeEncodeError:
                        print(f"  {key}: [Unicode characters]")

            # Print entries
            if 'entries' in race_card:
                print(f"\nTotal Entries: {len(race_card['entries'])}")
                if race_card['entries']:
                    print("\nFirst Entry:")
                    for key, value in race_card['entries'][0].items():
                        try:
                            print(f"  {key}: {value}")
                        except UnicodeEncodeError:
                            print(f"  {key}: [Unicode characters]")
        else:
            print("\nFailed to scrape race card")
            print("This might be expected if:")
            print("  - The race hasn't been published yet")
            print("  - The CNAME format has changed")
            print("  - The page structure is different than expected")

        return race_card

def test_get_page_content():
    """Test basic page fetching."""
    print("\n=== Testing Page Fetching ===")

    with JRAScraper(delay=1) as scraper:
        response = scraper._make_request(f"{scraper.BASE_URL}/keiba/thisweek/")

        if response:
            print(f"Successfully fetched page")
            print(f"Status code: {response.status_code}")
            print(f"Content length: {len(response.text)} chars")

            # Check for common elements
            soup = scraper._parse_html(response.text)
            if soup:
                title = soup.find('title')
                try:
                    title_text = title.get_text() if title else 'Not found'
                    print(f"Page title: {title_text}")
                except UnicodeEncodeError:
                    print("Page title: [Unicode characters - cannot display in console]")

                # Count links
                links = soup.find_all('a', href=True)
                print(f"Total links found: {len(links)}")

                # Look for JRADB links
                jradb_links = [link for link in links if 'JRADB' in link.get('href', '')]
                print(f"JRADB links found: {len(jradb_links)}")

                # Look for CNAME links
                cname_links = [link for link in links if 'CNAME=' in link.get('href', '')]
                print(f"CNAME links found: {len(cname_links)}")

                if jradb_links:
                    print("\nSample JRADB links:")
                    for link in jradb_links[:3]:
                        href = link.get('href', '')
                        print(f"  {href}")
                        try:
                            text = link.get_text(strip=True)[:50]
                            print(f"    Text: {text}")
                        except UnicodeEncodeError:
                            print(f"    Text: [Unicode characters]")

                # Test extracting race links
                race_links = scraper._extract_race_links(soup)
                print(f"\nExtracted race links: {len(race_links)}")
                if race_links:
                    print("\nFirst race link:")
                    for key, value in race_links[0].items():
                        try:
                            print(f"  {key}: {value}")
                        except UnicodeEncodeError:
                            print(f"  {key}: [Unicode characters]")

                    # Return first race link for use in other tests
                    return race_links[0]
        else:
            print("Failed to fetch page")

        return None

if __name__ == "__main__":
    print("JRA Scraper Test Suite")
    print("=" * 50)

    # Test basic page fetching first
    race_link = test_get_page_content()

    # Test race calendar scraping
    races = test_race_calendar()

    # Test race card scraping with discovered CNAME
    race_card = test_race_card()

    print("\n" + "=" * 50)
    print("Test completed")
    print("\nSummary:")
    print(f"  - Race links found: {2 if race_link else 0}")
    print(f"  - Races in calendar: {len(races) if races else 0}")
    print(f"  - Race card scraped: {'Yes' if race_card else 'No'}")
