"""
Scraping task manager with thread-safe progress tracking.

This module manages scraping tasks executed in background threads,
providing real-time progress updates via SSE.
"""
import threading
import uuid
from datetime import datetime, timedelta
from typing import Callable, Dict, Optional, Any
from flask import Flask
from src.scrapers.historical_scraper import scrape_historical_data
from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)


class ScrapingTaskManager:
    """
    Thread-safe manager for scraping tasks.

    Handles task lifecycle including start, progress tracking, and cancellation.
    """

    # Maximum number of concurrent tasks
    MAX_CONCURRENT_TASKS = 1

    # Task TTL for cleanup (hours)
    TASK_TTL_HOURS = 24

    def __init__(self):
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._threads: Dict[str, threading.Thread] = {}

    def start_task(
        self,
        app: Flask,
        start_date: datetime,
        end_date: datetime,
        weekends_only: bool = True,
        scraper_delay: int = 3
    ) -> tuple[bool, str, Optional[str]]:
        """
        Start a new scraping task.

        Args:
            app: Flask application instance
            start_date: Start date for scraping
            end_date: End date for scraping
            weekends_only: If True, only scrape weekends
            scraper_delay: Delay between requests in seconds

        Returns:
            Tuple of (success, task_id or error_message, error_type)
        """
        with self._lock:
            # Check for running tasks
            running_tasks = [
                tid for tid, task in self._tasks.items()
                if task['status'] == 'running'
            ]

            if len(running_tasks) >= self.MAX_CONCURRENT_TASKS:
                return False, "既にスクレイピングタスクが実行中です", "concurrent_limit"

            # Generate task ID
            task_id = str(uuid.uuid4())

            # Initialize task state
            self._tasks[task_id] = {
                'task_id': task_id,
                'status': 'running',
                'started_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'params': {
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d'),
                    'weekends_only': weekends_only,
                    'scraper_delay': scraper_delay,
                },
                'progress': {
                    'percent_complete': 0,
                    'total_dates': 0,
                    'processed_dates': 0,
                    'current_date': None,
                    'current_race': None,
                    'races_completed': 0,
                    'entries_saved': 0,
                    'results_saved': 0,
                },
                'error': None,
                'stats': None,
                'cancelled': False,
            }

        # Start worker thread
        thread = threading.Thread(
            target=self._scrape_worker,
            args=(app, task_id, start_date, end_date, weekends_only, scraper_delay),
            daemon=True
        )
        self._threads[task_id] = thread
        thread.start()

        logger.info(f"Started scraping task {task_id}")
        return True, task_id, None

    def _scrape_worker(
        self,
        app: Flask,
        task_id: str,
        start_date: datetime,
        end_date: datetime,
        weekends_only: bool,
        scraper_delay: int
    ):
        """Worker function for scraping thread."""
        def progress_callback(data: Dict[str, Any]):
            self._update_progress(task_id, data)

        def cancel_check() -> bool:
            with self._lock:
                task = self._tasks.get(task_id)
                return task.get('cancelled', False) if task else True

        try:
            with app.app_context():
                stats = scrape_historical_data(
                    start_date=start_date,
                    end_date=end_date,
                    weekends_only=weekends_only,
                    progress_callback=progress_callback,
                    cancel_check=cancel_check,
                    scraper_delay=scraper_delay
                )

                with self._lock:
                    if task_id in self._tasks:
                        if self._tasks[task_id].get('cancelled'):
                            self._tasks[task_id]['status'] = 'cancelled'
                        else:
                            self._tasks[task_id]['status'] = 'completed'
                        self._tasks[task_id]['stats'] = stats
                        self._tasks[task_id]['updated_at'] = datetime.now().isoformat()

                logger.info(f"Scraping task {task_id} completed")

        except Exception as e:
            logger.exception(f"Scraping task {task_id} failed")
            with self._lock:
                if task_id in self._tasks:
                    self._tasks[task_id]['status'] = 'failed'
                    self._tasks[task_id]['error'] = str(e)
                    self._tasks[task_id]['updated_at'] = datetime.now().isoformat()

    def _update_progress(self, task_id: str, data: Dict[str, Any]):
        """Update task progress from callback data."""
        with self._lock:
            if task_id not in self._tasks:
                return

            task = self._tasks[task_id]
            task['updated_at'] = datetime.now().isoformat()

            # Update progress based on event type
            event = data.get('event', '')
            stats = data.get('stats', {})

            task['progress']['percent_complete'] = data.get('percent_complete', 0)

            if 'total_dates' in data:
                task['progress']['total_dates'] = data['total_dates']

            if 'current_date' in data:
                task['progress']['current_date'] = data['current_date']

            if 'date_index' in data:
                task['progress']['processed_dates'] = data['date_index']

            if stats:
                task['progress']['races_completed'] = stats.get('races_completed', 0)
                task['progress']['entries_saved'] = stats.get('total_entries_saved', 0)
                task['progress']['results_saved'] = stats.get('total_results_saved', 0)

            # Update current race info
            if event == 'race_start':
                track = data.get('track', '')
                race_index = data.get('race_index', 0)
                total_races = data.get('total_races', 0)
                task['progress']['current_race'] = f"{track} ({race_index}/{total_races})"

            elif event == 'race_complete':
                task['progress']['current_race'] = None

            elif event in ('date_end', 'date_error'):
                task['progress']['current_race'] = None

            # Handle errors
            if event == 'scraping_error':
                task['error'] = data.get('error')

    def get_progress(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current progress for a task.

        Args:
            task_id: Task identifier

        Returns:
            Task state dictionary or None if not found
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                return task.copy()
            return None

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task.

        Args:
            task_id: Task identifier

        Returns:
            True if cancellation was requested, False if task not found
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task['status'] == 'running':
                task['cancelled'] = True
                logger.info(f"Cancellation requested for task {task_id}")
                return True
            return False

    def get_running_task(self) -> Optional[Dict[str, Any]]:
        """Get currently running task if any."""
        with self._lock:
            for task in self._tasks.values():
                if task['status'] == 'running':
                    return task.copy()
            return None

    def get_recent_tasks(self, limit: int = 10) -> list[Dict[str, Any]]:
        """
        Get recent tasks sorted by start time.

        Args:
            limit: Maximum number of tasks to return

        Returns:
            List of task dictionaries
        """
        with self._lock:
            tasks = list(self._tasks.values())
            tasks.sort(key=lambda x: x['started_at'], reverse=True)
            return [t.copy() for t in tasks[:limit]]

    def cleanup_old_tasks(self):
        """Remove tasks older than TTL."""
        cutoff = datetime.now() - timedelta(hours=self.TASK_TTL_HOURS)
        cutoff_str = cutoff.isoformat()

        with self._lock:
            to_remove = [
                tid for tid, task in self._tasks.items()
                if task['status'] != 'running' and task['started_at'] < cutoff_str
            ]

            for tid in to_remove:
                del self._tasks[tid]
                if tid in self._threads:
                    del self._threads[tid]

            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} old tasks")


# Global singleton instance
task_manager = ScrapingTaskManager()
