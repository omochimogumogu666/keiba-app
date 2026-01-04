"""
Debug script to analyze JRA horse profile page HTML structure.

This script fetches a sample horse profile page and saves the HTML
for analysis to understand the actual structure.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.scrapers.jra_scraper import JRAScraper
from src.utils.logger import get_app_logger
from config.logging_config import setup_logging

setup_logging(log_level='DEBUG')
logger = get_app_logger(__name__)


def save_html_to_file(html_content: str, filename: str):
    """Save HTML content to file for inspection."""
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'debug')
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"HTML saved to: {filepath}")
    return filepath


def debug_horse_profile(cname: str):
    """
    Debug horse profile scraping by fetching and saving HTML.

    Args:
        cname: Horse CNAME parameter from race card
    """
    with JRAScraper(delay=3) as scraper:
        # Construct horse profile URL using CNAME
        horse_url = f"{scraper.BASE_URL}/JRADB/accessS.html?CNAME={cname}"

        logger.info(f"Fetching horse profile from: {horse_url}")

        response = scraper._make_request(horse_url)

        if not response:
            logger.error("Failed to fetch horse profile")
            return

        # Save HTML for inspection
        filepath = save_html_to_file(response.text, f'horse_profile_{cname}.html')

        # Parse and display key information
        soup = scraper._parse_html(response.text)
        if not soup:
            logger.error("Failed to parse HTML")
            return

        # Display page title
        title = soup.find('title')
        if title:
            print(f"\nPage title: {title.get_text()}")

        # Display all h1 tags
        print("\n--- H1 Tags ---")
        for h1 in soup.find_all('h1'):
            print(f"H1: {h1.get_text(strip=True)}")

        # Display all h2 tags
        print("\n--- H2 Tags ---")
        for h2 in soup.find_all('h2'):
            print(f"H2: {h2.get_text(strip=True)}")

        # Display all tables (first few rows only)
        print("\n--- Tables ---")
        tables = soup.find_all('table')
        print(f"Found {len(tables)} tables")

        for i, table in enumerate(tables[:3]):  # First 3 tables only
            print(f"\nTable {i+1}:")
            # Get table class
            table_class = table.get('class', [])
            print(f"  Class: {table_class}")

            # Get first few rows
            rows = table.find_all('tr')[:5]
            for j, row in enumerate(rows):
                cells = row.find_all(['td', 'th'])
                cell_texts = [cell.get_text(strip=True)[:30] for cell in cells]
                print(f"  Row {j+1}: {cell_texts}")

        # Display all divs with class containing "profile" or "data"
        print("\n--- Profile/Data Divs ---")
        for div in soup.find_all('div', class_=True):
            class_str = ' '.join(div.get('class', []))
            if 'profile' in class_str.lower() or 'data' in class_str.lower():
                print(f"Div class: {class_str}")
                print(f"  Content preview: {div.get_text(strip=True)[:100]}")

        print(f"\nFull HTML saved to: {filepath}")
        print("Please inspect the HTML file to understand the structure.")


if __name__ == "__main__":
    # Example CNAME from a race card
    # This will be replaced with actual CNAME from scraping

    if len(sys.argv) > 1:
        cname = sys.argv[1]
    else:
        # Default: use a sample CNAME (this needs to be obtained from race card first)
        print("Usage: python debug_horse_profile.py <cname>")
        print("\nTo get a CNAME, first scrape a race card:")
        print("  1. Run: python scripts/scrape_data.py")
        print("  2. Find a horse CNAME in the race card data")
        print("  3. Run: python scripts/debug_horse_profile.py <cname>")
        sys.exit(1)

    debug_horse_profile(cname)
