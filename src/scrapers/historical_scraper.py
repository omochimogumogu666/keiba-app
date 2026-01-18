"""
Historical data scraping module with progress callback support.

This module provides core scraping logic extracted from scripts/scrape_historical_data.py
for reuse in web-based scraping with real-time progress updates.
"""
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Any
from src.scrapers.netkeiba_scraper import NetkeibaScraper
from src.data.database import (
    save_race_to_db, save_race_entries_to_db,
    save_race_results_to_db, save_payouts_to_db
)
from src.data.models import db
from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)


def is_weekend(date: datetime) -> bool:
    """Check if date is Saturday (5) or Sunday (6)."""
    return date.weekday() in [5, 6]


def generate_date_list(
    start_date: datetime,
    end_date: datetime,
    weekends_only: bool = True
) -> List[datetime]:
    """
    Generate list of dates to scrape.

    Args:
        start_date: Start date
        end_date: End date
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


# Target classes for scraping (2勝クラス or higher)
TARGET_CLASSES = [
    'G1', 'G2', 'G3',
    'OP', 'オープン',
    'Listed', 'L',
    '3勝クラス', '３勝クラス',
    '2勝クラス', '２勝クラス',
]

# Target tracks (major 5 tracks)
TARGET_TRACKS = ['東京', '中山', '阪神', '京都', '中京']


def should_scrape_race(race_class: Optional[str]) -> bool:
    """
    Determine if a race should be scraped based on race class.
    Only scrape races of 2勝クラス or higher.
    """
    if not race_class:
        return False
    return any(target in race_class for target in TARGET_CLASSES)


def should_scrape_track(track_name: str) -> bool:
    """
    Determine if a race should be scraped based on track.
    Only scrape races from major tracks.
    """
    return track_name in TARGET_TRACKS


def scrape_and_save_date(
    scraper: NetkeibaScraper,
    target_date: datetime,
    stats: Dict[str, int],
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    skip_existing: bool = True
) -> bool:
    """
    Scrape and save race results for a specific date.

    Args:
        scraper: NetkeibaScraper instance
        target_date: Date to scrape results for
        stats: Dictionary to track statistics
        progress_callback: Optional callback for progress updates
        cancel_check: Optional function to check for cancellation
        skip_existing: If True, skip races with existing results (default: True)

    Returns:
        True if successful, False if cancelled or failed
    """
    def notify_progress(event: str, **kwargs):
        if progress_callback:
            progress_callback({
                'event': event,
                'date': target_date.strftime('%Y-%m-%d'),
                'stats': stats.copy(),
                **kwargs
            })

    try:
        # Check for cancellation
        if cancel_check and cancel_check():
            return False

        notify_progress('date_start')

        # Get race calendar for the date
        races = scraper.scrape_race_calendar(target_date)

        if not races:
            stats['dates_no_races'] += 1
            notify_progress('date_end', races_found=0)
            return True

        # Filter races by track
        filtered_races = [r for r in races if should_scrape_track(r.get('track', ''))]

        if not filtered_races:
            stats['races_filtered_by_track'] += len(races)
            stats['dates_no_races'] += 1
            notify_progress('date_end', races_found=0, filtered_by_track=len(races))
            return True

        stats['total_races_found'] += len(filtered_races)
        stats['races_filtered_by_track'] += len(races) - len(filtered_races)

        notify_progress('races_discovered',
                       races_found=len(filtered_races),
                       total_races=len(races))

        # Process each race
        for i, race_info in enumerate(filtered_races):
            if cancel_check and cancel_check():
                return False

            netkeiba_race_id = race_info.get('netkeiba_race_id')
            track = race_info.get('track', 'Unknown')

            notify_progress('race_start',
                           race_index=i + 1,
                           total_races=len(filtered_races),
                           race_id=netkeiba_race_id,
                           track=track)

            if not netkeiba_race_id:
                stats['races_failed'] += 1
                continue

            try:
                # Check if race already exists with results (skip if enabled)
                if skip_existing:
                    from src.data.models import Race, RaceResult, RaceEntry
                    existing_race = Race.query.filter_by(netkeiba_race_id=netkeiba_race_id).first()
                    if existing_race and existing_race.status == 'completed':
                        # Check if race has results
                        has_results = db.session.query(RaceResult).join(
                            RaceEntry, RaceResult.race_entry_id == RaceEntry.id
                        ).filter(RaceEntry.race_id == existing_race.id).count() > 0

                        if has_results:
                            stats['races_skipped_existing'] = stats.get('races_skipped_existing', 0) + 1
                            logger.debug(f"Skipping race {netkeiba_race_id} - results already exist")
                            notify_progress('race_skipped',
                                           race_id=netkeiba_race_id,
                                           reason='already_completed')
                            continue

                # Scrape race card first to get race class
                race_card = scraper.scrape_race_card(netkeiba_race_id)

                if not race_card or not race_card.get('entries'):
                    stats['races_failed'] += 1
                    continue

                # Merge race_info with race_card race_info
                if race_card.get('race_info'):
                    for key, value in race_card['race_info'].items():
                        if value and key not in ['netkeiba_race_id']:
                            race_info[key] = value

                # Check race class filter
                race_class = race_info.get('race_class')
                if not should_scrape_race(race_class):
                    stats['races_filtered_by_class'] += 1
                    notify_progress('race_filtered',
                                   race_id=netkeiba_race_id,
                                   race_class=race_class)
                    continue

                # Save race to database
                race = save_race_to_db(race_info)

                # Save race entries
                entries = save_race_entries_to_db(race.id, race_card['entries'])
                stats['total_entries_saved'] += len(entries)

                # Scrape and save race result
                try:
                    result = scraper.scrape_race_result(netkeiba_race_id)

                    if result and result.get('results'):
                        race_results = save_race_results_to_db(race.id, result['results'])
                        stats['total_results_saved'] += len(race_results)

                        # Save payouts if available
                        if result.get('payouts'):
                            payouts = save_payouts_to_db(race.id, result['payouts'])
                            stats['total_payouts_saved'] += len(payouts)

                        # Update race status to completed
                        race.status = 'completed'
                        db.session.commit()

                        stats['races_completed'] += 1
                        notify_progress('race_complete',
                                       race_id=netkeiba_race_id,
                                       race_index=i + 1,
                                       total_races=len(filtered_races),
                                       entries_count=len(entries),
                                       results_count=len(race_results),
                                       race_class=race_class)
                    else:
                        stats['races_no_results'] += 1

                except Exception as e:
                    logger.debug(f"Result scraping failed: {e}")
                    stats['races_no_results'] += 1

            except Exception as e:
                logger.exception("Race processing error")
                stats['races_failed'] += 1
                continue

        stats['dates_processed'] += 1
        notify_progress('date_end',
                       races_processed=len(filtered_races))
        return True

    except Exception as e:
        logger.exception("Date processing error")
        stats['dates_failed'] += 1
        notify_progress('date_error', error=str(e))
        return False


def scrape_historical_data(
    start_date: datetime,
    end_date: datetime,
    weekends_only: bool = True,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    scraper_delay: int = 3
) -> Dict[str, Any]:
    """
    Core scraping function for historical data.

    Args:
        start_date: Start date for scraping
        end_date: End date for scraping
        weekends_only: If True, only scrape weekends (default: True)
        progress_callback: Optional callback for progress updates
        cancel_check: Optional function to check for cancellation
        scraper_delay: Delay between requests in seconds (default: 3)

    Returns:
        Final statistics dictionary
    """
    # Generate date list
    dates = generate_date_list(start_date, end_date, weekends_only=weekends_only)

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
        'races_skipped_existing': 0,
        'total_entries_saved': 0,
        'total_results_saved': 0,
        'total_payouts_saved': 0,
    }

    def notify_progress(event: str, **kwargs):
        if progress_callback:
            percent = (stats['dates_processed'] / stats['dates_total'] * 100
                      if stats['dates_total'] > 0 else 0)
            progress_callback({
                'event': event,
                'percent_complete': round(percent, 1),
                'stats': stats.copy(),
                **kwargs
            })

    start_time = datetime.now()
    notify_progress('scraping_start',
                   start_date=start_date.strftime('%Y-%m-%d'),
                   end_date=end_date.strftime('%Y-%m-%d'),
                   total_dates=len(dates),
                   weekends_only=weekends_only)

    try:
        with NetkeibaScraper(delay=scraper_delay) as scraper:
            for i, target_date in enumerate(dates):
                # Check for cancellation
                if cancel_check and cancel_check():
                    notify_progress('scraping_cancelled')
                    stats['cancelled'] = True
                    break

                # Update progress
                percent = ((i + 1) / len(dates) * 100)
                notify_progress('date_progress',
                               current_date=target_date.strftime('%Y-%m-%d'),
                               date_index=i + 1,
                               total_dates=len(dates),
                               percent_complete=round(percent, 1))

                # Scrape this date
                success = scrape_and_save_date(
                    scraper, target_date, stats,
                    progress_callback=progress_callback,
                    cancel_check=cancel_check
                )

                if not success and cancel_check and cancel_check():
                    break

    except Exception as e:
        logger.exception("Scraping error")
        stats['error'] = str(e)
        notify_progress('scraping_error', error=str(e))

    # Calculate duration
    duration = datetime.now() - start_time
    stats['duration_seconds'] = duration.total_seconds()
    stats['duration_hours'] = round(duration.total_seconds() / 3600, 2)

    notify_progress('scraping_complete')

    return stats
