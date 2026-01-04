"""
Debug script to fetch and save JRA race card HTML for analysis.
"""
from src.scrapers.jra_scraper import JRAScraper
from config.logging_config import setup_logging

setup_logging(log_level='INFO')

def fetch_and_save_race_card_html():
    """Fetch race card HTML and save to file for analysis."""
    with JRAScraper(delay=1) as scraper:
        # Use the CNAME we discovered
        cname = "pw01dde0106202601011120260104/6C"
        url = f"{scraper.BASE_URL}/JRADB/accessD.html?CNAME={cname}"

        print(f"Fetching URL: {url}")

        response = scraper._make_request(url)
        if response:
            # Save HTML to file
            output_file = "debug_race_card.html"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(response.text)

            print(f"HTML saved to {output_file}")
            print(f"File size: {len(response.text)} characters")

            # Parse and show basic structure
            soup = scraper._parse_html(response.text)
            if soup:
                # Find all tables
                tables = soup.find_all('table')
                print(f"\nFound {len(tables)} tables")

                for i, table in enumerate(tables, 1):
                    print(f"\nTable {i}:")
                    # Show table attributes
                    if table.get('class'):
                        print(f"  Classes: {table.get('class')}")
                    if table.get('id'):
                        print(f"  ID: {table.get('id')}")

                    # Count rows
                    rows = table.find_all('tr')
                    print(f"  Rows: {len(rows)}")

                    # Show first row structure if exists
                    if rows:
                        first_row = rows[0]
                        headers = first_row.find_all(['th', 'td'])
                        if headers:
                            print(f"  First row has {len(headers)} cells")
                            for j, header in enumerate(headers[:5], 1):  # Show first 5
                                try:
                                    text = header.get_text(strip=True)[:30]
                                    print(f"    Cell {j}: {text if text else '[empty]'}")
                                except UnicodeEncodeError:
                                    print(f"    Cell {j}: [Unicode characters]")

                # Look for divs with specific classes or IDs
                print("\n--- Looking for race info sections ---")

                # Common JRA selectors to check
                selectors_to_check = [
                    ('div.race_info', 'div'),
                    ('div.race-data', 'div'),
                    ('div.race_head', 'div'),
                    ('table.race_table_01', 'table'),
                    ('div.shutuba_list', 'div'),
                ]

                for selector, tag in selectors_to_check:
                    elements = soup.select(selector)
                    if elements:
                        print(f"  Found {len(elements)} elements matching: {selector}")

        else:
            print("Failed to fetch page")

if __name__ == "__main__":
    fetch_and_save_race_card_html()
