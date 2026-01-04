"""
Debug script to inspect actual netkeiba race card HTML structure.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.logging_config import setup_logging
from src.scrapers.netkeiba_scraper import NetkeibaScraper
from src.utils.logger import get_app_logger

# Setup logging
setup_logging(log_level='INFO')
logger = get_app_logger(__name__)


def debug_race_card_html():
    """Fetch and save race card HTML for inspection."""

    with NetkeibaScraper(delay=3) as scraper:
        # Get recent races - try yesterday first
        for days_back in range(7):
            target_date = datetime.now() - timedelta(days=days_back)
            logger.info(f"Checking {target_date.strftime('%Y-%m-%d')}")

            races = scraper.scrape_race_calendar(target_date)

            if races:
                race = races[0]
                race_id = race.get('netkeiba_race_id')

                logger.info(f"Fetching race card for {race_id}")

                # Construct URL
                race_url = f"{scraper.BASE_URL}/race/shutuba.html?race_id={race_id}"

                response = scraper._make_request(race_url)
                if response:
                    # Save HTML to file
                    output_file = Path(__file__).parent.parent / 'data' / 'debug_netkeiba_race_card.html'
                    output_file.parent.mkdir(parents=True, exist_ok=True)

                    # Save with EUC-JP encoding
                    with open(output_file, 'wb') as f:
                        f.write(response.content)

                    logger.info(f"HTML saved to {output_file}")
                    logger.info(f"Race ID: {race_id}")

                    # Parse and print structure
                    soup = scraper._parse_html(response)
                    if soup:
                        title = soup.find('title')
                        if title:
                            logger.info(f"Title: {title.get_text()}")

                        # Look for RaceData01
                        race_data01 = soup.find('div', class_='RaceData01')
                        if race_data01:
                            logger.info(f"RaceData01 found: {race_data01.get_text().strip()}")
                        else:
                            logger.warning("RaceData01 not found!")

                        # Look for RaceData02
                        race_data02 = soup.find('div', class_='RaceData02')
                        if race_data02:
                            logger.info(f"RaceData02 found: {race_data02.get_text().strip()}")
                        else:
                            logger.warning("RaceData02 not found!")

                        # Look for race name
                        race_name_elem = soup.find('div', class_='RaceName')
                        if race_name_elem:
                            logger.info(f"RaceName found: {race_name_elem.get_text().strip()}")
                        else:
                            logger.warning("RaceName not found!")

                        # Look for table
                        shutuba_table = soup.find('table', class_='Shutuba_Table')
                        if shutuba_table:
                            logger.info("Shutuba_Table found!")
                            rows = shutuba_table.find_all('tr', class_=lambda x: x and 'HorseList' in x)
                            logger.info(f"Found {len(rows)} HorseList rows")
                        else:
                            logger.warning("Shutuba_Table not found!")

                    return True

        logger.warning("No races found in the past 7 days")
        return False


def debug_race_result_html():
    """Fetch and save race result HTML for inspection."""

    with NetkeibaScraper(delay=3) as scraper:
        # Get recent races
        for days_back in range(7):
            target_date = datetime.now() - timedelta(days=days_back)
            logger.info(f"Checking results for {target_date.strftime('%Y-%m-%d')}")

            races = scraper.scrape_race_calendar(target_date)

            if races:
                race = races[0]
                race_id = race.get('netkeiba_race_id')

                logger.info(f"Fetching race result for {race_id}")

                # Construct URL
                result_url = f"{scraper.BASE_URL}/race/result.html?race_id={race_id}"

                response = scraper._make_request(result_url)
                if response:
                    # Save HTML to file
                    output_file = Path(__file__).parent.parent / 'data' / 'debug_netkeiba_race_result.html'
                    output_file.parent.mkdir(parents=True, exist_ok=True)

                    with open(output_file, 'wb') as f:
                        f.write(response.content)

                    logger.info(f"HTML saved to {output_file}")
                    logger.info(f"Race ID: {race_id}")

                    # Parse and print structure
                    soup = scraper._parse_html(response)
                    if soup:
                        # Look for result table
                        result_table = soup.find('table', class_=lambda x: x and ('Result_Table' in x or 'ResultRaceShutuba' in x))
                        if result_table:
                            logger.info(f"Result table found with class: {result_table.get('class')}")
                            rows = result_table.find_all('tr', class_=lambda x: x and 'HorseList' in x)
                            logger.info(f"Found {len(rows)} result rows")

                            # Check first row structure
                            if rows:
                                first_row = rows[0]
                                tds = first_row.find_all('td')
                                logger.info(f"First row has {len(tds)} td elements")
                                for i, td in enumerate(tds[:10]):  # First 10 cells
                                    classes = td.get('class', [])
                                    text = td.get_text().strip()[:30]
                                    logger.info(f"  TD {i}: class={classes}, text='{text}'")
                        else:
                            logger.warning("Result table not found!")

                            # Debug: show all tables
                            all_tables = soup.find_all('table')
                            logger.info(f"Found {len(all_tables)} tables total")
                            for i, table in enumerate(all_tables[:5]):
                                classes = table.get('class', [])
                                logger.info(f"  Table {i}: class={classes}")

                    return True

        logger.warning("No races found in the past 7 days")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--type', choices=['card', 'result', 'both'], default='both',
                       help='Which HTML to debug (card/result/both)')
    args = parser.parse_args()

    if args.type in ['card', 'both']:
        logger.info("=== Debugging Race Card HTML ===")
        debug_race_card_html()

    if args.type in ['result', 'both']:
        logger.info("\n=== Debugging Race Result HTML ===")
        debug_race_result_html()
