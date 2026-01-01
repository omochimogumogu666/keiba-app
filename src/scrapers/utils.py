"""
Utility functions for web scraping.
"""
import re
import time
import random
from functools import wraps
from typing import Callable, Optional
from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)


def retry_on_failure(max_retries: int = 3, delay: float = 2.0, backoff: float = 2.0):
    """
    Decorator to retry function on failure with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry

    Usage:
        @retry_on_failure(max_retries=3, delay=2.0)
        def scrape_data():
            # scraping code
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        logger.error(f"{func.__name__} failed after {max_retries} retries: {e}")
                        raise

                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff

            return None
        return wrapper
    return decorator


def random_delay(min_seconds: float = 1.0, max_seconds: float = 3.0):
    """
    Add random delay to avoid detection.

    Args:
        min_seconds: Minimum delay in seconds
        max_seconds: Maximum delay in seconds
    """
    delay = random.uniform(min_seconds, max_seconds)
    logger.debug(f"Waiting {delay:.2f} seconds...")
    time.sleep(delay)


def parse_japanese_number(text: str) -> Optional[int]:
    """
    Parse Japanese number text to integer.

    Args:
        text: Japanese number text (e.g., "一着", "2着")

    Returns:
        Integer value or None
    """
    # Japanese number kanji to digit mapping
    jp_to_digit = {
        '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
        '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
        '壱': 1, '弐': 2, '参': 3
    }

    if not text:
        return None

    # Remove common suffixes
    text = text.replace('着', '').replace('番', '').replace('人気', '').strip()

    # Try direct conversion
    try:
        return int(text)
    except ValueError:
        pass

    # Try Japanese kanji conversion
    for kanji, digit in jp_to_digit.items():
        if kanji in text:
            return digit

    return None


def parse_distance(text: str) -> Optional[int]:
    """
    Parse distance text to meters.

    Args:
        text: Distance text (e.g., "1600m", "2000メートル")

    Returns:
        Distance in meters or None
    """
    if not text:
        return None

    # Remove non-numeric characters except digits
    numbers = re.findall(r'\d+', text)

    if numbers:
        return int(numbers[0])

    return None


def parse_time(text: str) -> Optional[float]:
    """
    Parse race time to seconds.

    Args:
        text: Time text (e.g., "1:36.5", "96.5")

    Returns:
        Time in seconds or None
    """
    if not text:
        return None

    try:
        # Format: "M:SS.s"
        if ':' in text:
            parts = text.split(':')
            if len(parts) != 2:
                return None
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        else:
            # Format: "SS.s"
            return float(text)
    except (ValueError, IndexError):
        return None


def clean_text(text: str) -> str:
    """
    Clean scraped text by removing extra whitespace and special characters.

    Args:
        text: Raw text

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def validate_jra_id(jra_id: str, id_type: str = 'race') -> bool:
    """
    Validate JRA ID format.

    Args:
        jra_id: JRA ID to validate
        id_type: Type of ID ('race', 'horse', 'jockey', 'trainer')

    Returns:
        True if valid, False otherwise
    """
    if not jra_id:
        return False

    # Basic validation - adjust based on actual JRA ID formats
    if id_type == 'race':
        # Example: 12 digits for race ID (YYYYMMDDRRNN)
        return len(jra_id) >= 10 and jra_id.isdigit()
    elif id_type in ['horse', 'jockey', 'trainer']:
        # Example: alphanumeric IDs
        return len(jra_id) > 0

    return True
