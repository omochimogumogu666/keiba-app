"""
Script to scrape JRA race data.
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


def scrape_upcoming_races(days_ahead: int = 7):
    """
    Scrape upcoming races.

    Args:
        days_ahead: Number of days to look ahead
    """
    logger.info(f"Scraping upcoming races for next {days_ahead} days")

    with JRAScraper() as scraper:
        races = scraper.get_upcoming_races(days_ahead=days_ahead)

    logger.info(f"Scraped {len(races)} upcoming races")

    # TODO: Save races to database
    # for race in races:
    #     save_race_to_db(race)

    return races


def scrape_race_results(days_back: int = 7):
    """
    Scrape past race results.

    Args:
        days_back: Number of days to look back
    """
    logger.info(f"Scraping race results from past {days_back} days")

    with JRAScraper() as scraper:
        results = []
        today = datetime.now()

        for day_offset in range(days_back):
            target_date = today - timedelta(days=day_offset)
            logger.info(f"Scraping results for {target_date.strftime('%Y-%m-%d')}")

            # Get races for this date
            races = scraper.scrape_race_calendar(target_date)

            # Scrape results for each race
            for race in races:
                race_id = race.get('jra_race_id')
                if race_id:
                    result = scraper.scrape_race_result(race_id)
                    if result:
                        results.append(result)

    logger.info(f"Scraped {len(results)} race results")

    # TODO: Save results to database
    # for result in results:
    #     save_result_to_db(result)

    return results


def scrape_specific_race(race_id: str):
    """
    Scrape a specific race by ID.

    Args:
        race_id: JRA race ID
    """
    logger.info(f"Scraping race {race_id}")

    with JRAScraper() as scraper:
        # Scrape race card
        race_card = scraper.scrape_race_card(race_id)
        if race_card:
            logger.info(f"Scraped race card for {race_id}")
            # TODO: Save to database

        # Scrape race result (if completed)
        result = scraper.scrape_race_result(race_id)
        if result:
            logger.info(f"Scraped result for {race_id}")
            # TODO: Save to database

    return race_card, result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Scrape JRA race data')
    parser.add_argument(
        '--upcoming',
        type=int,
        metavar='DAYS',
        help='Scrape upcoming races for N days ahead'
    )
    parser.add_argument(
        '--results',
        type=int,
        metavar='DAYS',
        help='Scrape race results from N days back'
    )
    parser.add_argument(
        '--race-id',
        type=str,
        metavar='ID',
        help='Scrape specific race by ID'
    )

    args = parser.parse_args()

    if args.upcoming:
        scrape_upcoming_races(days_ahead=args.upcoming)
    elif args.results:
        scrape_race_results(days_back=args.results)
    elif args.race_id:
        scrape_specific_race(args.race_id)
    else:
        # Default: scrape upcoming races for next 3 days
        logger.info("No arguments provided, scraping upcoming races for next 3 days")
        scrape_upcoming_races(days_ahead=3)

    logger.info("Scraping complete")
