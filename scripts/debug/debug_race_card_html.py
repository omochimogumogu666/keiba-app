"""
Debug script to inspect actual race card HTML structure.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.logging_config import setup_logging
from src.scrapers.jra_scraper import JRAScraper
from src.utils.logger import get_app_logger

# Setup logging
setup_logging(log_level='INFO')
logger = get_app_logger(__name__)


def debug_race_card_html():
    """Fetch and save race card HTML for inspection."""

    with JRAScraper(delay=3) as scraper:
        # Get recent races
        target_date = datetime.now() - timedelta(days=1)

        for days_back in range(7):
            target_date = datetime.now() - timedelta(days=days_back)
            logger.info(f"Checking {target_date.strftime('%Y-%m-%d')}")

            races = scraper.scrape_race_calendar(target_date)

            if races:
                race = races[0]
                jra_race_id = race.get('jra_race_id')
                cname = race.get('cname')

                logger.info(f"Fetching race card for {jra_race_id} with CNAME {cname}")

                # Construct URL
                race_url = f"{scraper.BASE_URL}/JRADB/accessD.html?CNAME={cname}"

                response = scraper._make_request(race_url)
                if response:
                    # Save HTML to file
                    output_file = Path(__file__).parent.parent / 'data' / 'debug_race_card.html'
                    output_file.parent.mkdir(parents=True, exist_ok=True)

                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(response.text)

                    logger.info(f"HTML saved to {output_file}")
                    logger.info(f"Race ID: {jra_race_id}")
                    logger.info(f"CNAME: {cname}")

                    # Parse and print title
                    soup = scraper._parse_html(response.text)
                    if soup:
                        title = soup.find('title')
                        if title:
                            logger.info(f"Title: {title.get_text()}")

                        # Look for race info section
                        h1_tags = soup.find_all('h1')
                        logger.info(f"Found {len(h1_tags)} H1 tags")
                        for i, h1 in enumerate(h1_tags):
                            h1_text = h1.get_text().strip()
                            logger.info(f"H1 #{i}: '{h1_text}' (length: {len(h1_text)})")
                            logger.info(f"H1 #{i} HTML: {str(h1)[:200]}")

                        h2_tags = soup.find_all('h2')
                        for h2 in h2_tags:
                            logger.info(f"H2: {h2.get_text().strip()}")

                        # Look for race details
                        divs_with_data = soup.find_all('div', class_='data')
                        for div in divs_with_data:
                            logger.info(f"Data div: {div.get_text().strip()}")

                    return True

        logger.warning("No races found in the past 7 days")
        return False


if __name__ == "__main__":
    debug_race_card_html()
