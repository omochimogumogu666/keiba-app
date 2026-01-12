"""
Script to scrape historical race results from netkeiba.com and save to database.

This script allows scraping multiple years of historical data with configurable
date ranges and error handling.

Usage:
    python scripts/scrape_historical_data.py [--years YEARS] [--start-date START] [--end-date END]

Options:
    --years YEARS        Number of years to scrape (default: 5)
    --start-date START   Start date in YYYY-MM-DD format (overrides --years)
    --end-date END       End date in YYYY-MM-DD format (default: today)

Examples:
    # Scrape last 5 years (default)
    python scripts/scrape_historical_data.py

    # Scrape last 3 years
    python scripts/scrape_historical_data.py --years 3

    # Scrape specific date range
    python scripts/scrape_historical_data.py --start-date 2021-01-01 --end-date 2023-12-31

Note:
    - Only scrapes weekends (Sat/Sun) as JRA races primarily occur on weekends
    - Uses 3-second delay between requests to be respectful to netkeiba servers
    - Progress is saved incrementally, so you can stop and resume
    - Failed dates are logged and can be retried later
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import datetime, timedelta
from src.scrapers.netkeiba_scraper import NetkeibaScraper
from src.data.database import save_race_to_db, save_race_entries_to_db, save_race_results_to_db, save_payouts_to_db
from src.web.app import create_app
from src.utils.logger import get_app_logger
from src.data.models import db

logger = get_app_logger(__name__)


def is_weekend(date):
    """Check if date is Saturday (5) or Sunday (6)."""
    return date.weekday() in [5, 6]


def generate_date_list(start_date, end_date, weekends_only=True):
    """
    Generate list of dates to scrape.

    Args:
        start_date: Start date (datetime)
        end_date: End date (datetime)
        weekends_only: If True, only include weekends (default: True)

    Returns:
        List of datetime objects
    """
    dates = []
    current = start_date

    while current <= end_date:
        if not weekends_only or is_weekend(current):
            dates.append(current)
        current += timedelta(days=1)

    return dates


def should_scrape_race(race_class):
    """
    Determine if a race should be scraped based on race class.
    Only scrape races of 2勝クラス or higher.

    Args:
        race_class: Race class string (e.g., 'G1', '2勝クラス', '未勝利')

    Returns:
        True if race should be scraped, False otherwise
    """
    # Priority races to scrape (2勝クラス以上)
    target_classes = [
        'G1', 'G2', 'G3',           # Grade races
        'OP', 'オープン',            # Open class
        'Listed', 'L',              # Listed races
        '3勝クラス', '３勝クラス',   # 3-win class
        '2勝クラス', '２勝クラス',   # 2-win class
    ]

    if not race_class:
        return False

    # Check if race class matches target classes
    return any(target in race_class for target in target_classes)


def should_scrape_track(track_name):
    """
    Determine if a race should be scraped based on track.
    Only scrape races from major tracks: 東京、中山、阪神、京都、中京.

    Args:
        track_name: Track name string (e.g., '東京', '札幌')

    Returns:
        True if race should be scraped, False otherwise
    """
    target_tracks = ['東京', '中山', '阪神', '京都', '中京']
    return track_name in target_tracks


def scrape_and_save_date(scraper, target_date, stats):
    """
    Scrape and save race results for a specific date.

    Args:
        scraper: NetkeibaScraper instance
        target_date: Date to scrape results for
        stats: Dictionary to track statistics

    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"\n{'='*80}")
        print(f"Scraping {target_date.strftime('%Y-%m-%d (%a)')}")
        print(f"{'='*80}")

        # Get race calendar for the date
        races = scraper.scrape_race_calendar(target_date)

        if not races:
            print(f"No races found")
            stats['dates_no_races'] += 1
            return True

        # Filter races by track (主要5競馬場)
        filtered_races = [r for r in races if should_scrape_track(r.get('track'))]

        if not filtered_races:
            print(f"Found {len(races)} races, but none from target tracks (東京、中山、阪神、京都、中京)")
            stats['races_filtered_by_track'] += len(races)
            stats['dates_no_races'] += 1
            return True

        print(f"Found {len(races)} races ({len(filtered_races)} from target tracks)")
        stats['total_races_found'] += len(filtered_races)
        stats['races_filtered_by_track'] += len(races) - len(filtered_races)

        # Process each race (only target track races)
        for i, race_info in enumerate(filtered_races):
            netkeiba_race_id = race_info.get('netkeiba_race_id')
            track = race_info.get('track', 'Unknown')
            print(f"\n[{i+1}/{len(filtered_races)}] Processing race {netkeiba_race_id} ({track})...")

            if not netkeiba_race_id:
                print("  ✗ Skipping: No race ID available")
                stats['races_failed'] += 1
                continue

            try:
                # Scrape race card first to get race class
                race_card = scraper.scrape_race_card(netkeiba_race_id)

                if not race_card or not race_card.get('entries'):
                    print("  ✗ No race card data found")
                    stats['races_failed'] += 1
                    continue

                # Merge race_info with race_card race_info
                if race_card.get('race_info'):
                    for key, value in race_card['race_info'].items():
                        if value and key not in ['netkeiba_race_id']:
                            race_info[key] = value

                # Check race class filter (2勝クラス以上)
                race_class = race_info.get('race_class')
                if not should_scrape_race(race_class):
                    print(f"  ⊘ Skipping: Race class '{race_class}' is below 2勝クラス")
                    stats['races_filtered_by_class'] += 1
                    continue

                # Save race to database
                race = save_race_to_db(race_info)
                print(f"  ✓ Saved race (Class: {race_class}, DB ID: {race.id})")

                # Save race entries
                entries = save_race_entries_to_db(race.id, race_card['entries'])
                print(f"  ✓ Saved {len(entries)} entries")
                stats['total_entries_saved'] += len(entries)

                # Scrape and save race result
                try:
                    result = scraper.scrape_race_result(netkeiba_race_id)

                    if result and result.get('results'):
                        race_results = save_race_results_to_db(race.id, result['results'])
                        print(f"  ✓ Saved {len(race_results)} results")
                        stats['total_results_saved'] += len(race_results)

                        # Save payouts if available
                        if result.get('payouts'):
                            payouts = save_payouts_to_db(race.id, result['payouts'])
                            print(f"  ✓ Saved {len(payouts)} payouts")
                            stats['total_payouts_saved'] += len(payouts)

                        # Update race status to completed
                        race.status = 'completed'
                        db.session.commit()

                        stats['races_completed'] += 1
                    else:
                        print("  ℹ No results available (race not completed)")
                        stats['races_no_results'] += 1

                except Exception as e:
                    print(f"  ⚠ Could not scrape results: {e}")
                    logger.debug(f"Result scraping failed: {e}")
                    stats['races_no_results'] += 1

            except Exception as e:
                print(f"  ✗ Error processing race: {e}")
                logger.exception("Race processing error")
                stats['races_failed'] += 1
                continue

        stats['dates_processed'] += 1
        return True

    except Exception as e:
        print(f"✗ Error processing date {target_date.strftime('%Y-%m-%d')}: {e}")
        logger.exception("Date processing error")
        stats['dates_failed'] += 1
        return False


def print_statistics(stats, start_time):
    """Print final statistics."""
    duration = datetime.now() - start_time
    hours = duration.total_seconds() / 3600

    print(f"\n{'='*80}")
    print("SCRAPING STATISTICS")
    print(f"{'='*80}")
    print(f"Execution time: {hours:.2f} hours")
    print(f"\nDates:")
    print(f"  - Total dates attempted: {stats['dates_total']}")
    print(f"  - Dates processed: {stats['dates_processed']}")
    print(f"  - Dates with no races: {stats['dates_no_races']}")
    print(f"  - Dates failed: {stats['dates_failed']}")
    print(f"\nRaces:")
    print(f"  - Total races found (target tracks): {stats['total_races_found']}")
    print(f"  - Races filtered by track: {stats['races_filtered_by_track']}")
    print(f"  - Races filtered by class: {stats['races_filtered_by_class']}")
    print(f"  - Races completed: {stats['races_completed']}")
    print(f"  - Races without results: {stats['races_no_results']}")
    print(f"  - Races failed: {stats['races_failed']}")
    print(f"\nData:")
    print(f"  - Total entries saved: {stats['total_entries_saved']}")
    print(f"  - Total results saved: {stats['total_results_saved']}")
    print(f"  - Total payouts saved: {stats['total_payouts_saved']}")
    print(f"{'='*80}\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Scrape historical JRA race results from netkeiba.com',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--years',
        type=int,
        default=5,
        help='Number of years to scrape (default: 5)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date in YYYY-MM-DD format (overrides --years)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date in YYYY-MM-DD format (default: today)'
    )
    parser.add_argument(
        '--all-days',
        action='store_true',
        help='Scrape all days, not just weekends'
    )

    args = parser.parse_args()

    # Determine date range
    if args.end_date:
        try:
            end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        except ValueError:
            print(f"Error: Invalid end date format '{args.end_date}'. Use YYYY-MM-DD")
            sys.exit(1)
    else:
        end_date = datetime.now()

    if args.start_date:
        try:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        except ValueError:
            print(f"Error: Invalid start date format '{args.start_date}'. Use YYYY-MM-DD")
            sys.exit(1)
    else:
        # Calculate start date based on years
        start_date = end_date - timedelta(days=365 * args.years)

    # Generate date list
    weekends_only = not args.all_days
    dates = generate_date_list(start_date, end_date, weekends_only=weekends_only)

    print(f"\n{'='*80}")
    print("HISTORICAL DATA SCRAPING - OPTIMIZED FOR MAJOR RACES")
    print(f"{'='*80}")
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Total dates to scrape: {len(dates)}")
    print(f"Weekends only: {weekends_only}")
    print(f"\nFiltering criteria:")
    print(f"  - Tracks: 東京、中山、阪神、京都、中京 (5 major tracks)")
    print(f"  - Race class: 2勝クラス以上 (excludes 1勝クラス, 未勝利, 新馬)")
    print(f"\nEstimated time: {len(dates) * 2 / 60:.1f} hours (optimized)")
    print(f"{'='*80}\n")

    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return

    # Initialize statistics
    stats = {
        'dates_total': len(dates),
        'dates_processed': 0,
        'dates_no_races': 0,
        'dates_failed': 0,
        'total_races_found': 0,
        'races_completed': 0,
        'races_no_results': 0,
        'races_failed': 0,
        'races_filtered_by_track': 0,
        'races_filtered_by_class': 0,
        'total_entries_saved': 0,
        'total_results_saved': 0,
        'total_payouts_saved': 0,
    }

    start_time = datetime.now()

    # Create Flask app context
    app = create_app('development')

    with app.app_context():
        with NetkeibaScraper(delay=3) as scraper:
            # Process each date
            for i, target_date in enumerate(dates):
                print(f"\n[{i+1}/{len(dates)}] Progress: {(i/len(dates)*100):.1f}%")

                scrape_and_save_date(scraper, target_date, stats)

                # Print interim statistics every 10 dates
                if (i + 1) % 10 == 0:
                    print(f"\n--- Interim Stats (after {i+1} dates) ---")
                    print(f"Races completed: {stats['races_completed']}")
                    print(f"Entries saved: {stats['total_entries_saved']}")
                    print(f"Results saved: {stats['total_results_saved']}")
                    print(f"Payouts saved: {stats['total_payouts_saved']}")

    # Print final statistics
    print_statistics(stats, start_time)


if __name__ == "__main__":
    main()
