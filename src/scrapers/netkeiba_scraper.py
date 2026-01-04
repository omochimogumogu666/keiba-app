"""
Netkeiba.com scraper for JRA horse racing data.

This scraper replaces the JRA scraper with a more stable implementation
using netkeiba.com as the data source.
"""
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse, parse_qs

import requests
from bs4 import BeautifulSoup

from config.settings import Config
from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)


class NetkeibaScraper:
    """
    Scraper for netkeiba.com horse racing data.

    Provides the same API as JRAScraper but uses netkeiba.com as source.
    No CNAME parameters needed - much simpler URL structure!
    """

    BASE_URL = "https://race.netkeiba.com"
    MOBILE_URL = "https://race.sp.netkeiba.com"  # Mobile site for calendar (no JS)
    DB_URL = "https://db.netkeiba.com"
    ENCODING = "euc-jp"  # Netkeiba uses EUC-JP (not Shift_JIS like JRA)

    # Track name mapping: netkeiba Japanese -> standard names
    TRACK_MAPPING = {
        '札幌': '札幌',
        '函館': '函館',
        '福島': '福島',
        '新潟': '新潟',
        '東京': '東京',
        '中山': '中山',
        '中京': '中京',
        '京都': '京都',
        '阪神': '阪神',
        '小倉': '小倉',
    }

    def __init__(self, delay: int = None):
        """
        Initialize scraper with configurable delay.

        Args:
            delay: Seconds to wait between requests (default: from Config)
        """
        self.delay = delay if delay is not None else Config.SCRAPING_DELAY
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': Config.USER_AGENT
        })
        logger.info(f"NetkeibaScraper initialized with {self.delay}s delay")

    def _make_request(self, url: str, params: Dict = None) -> Optional[requests.Response]:
        """
        Make HTTP request with rate limiting and error handling.

        Args:
            url: URL to request
            params: Query parameters

        Returns:
            Response object or None on failure
        """
        try:
            response = self.session.get(
                url,
                params=params,
                timeout=Config.REQUEST_TIMEOUT
            )
            response.raise_for_status()

            # Rate limiting - always wait after successful request
            time.sleep(self.delay)

            logger.debug(f"Request successful: {url}")
            return response

        except requests.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            return None

    def _parse_html(self, response: requests.Response) -> Optional[BeautifulSoup]:
        """
        Parse HTML response with EUC-JP encoding.

        Args:
            response: Response object

        Returns:
            BeautifulSoup object or None on failure
        """
        try:
            # Netkeiba uses EUC-JP encoding
            html_text = response.content.decode(self.ENCODING, errors='replace')
            soup = BeautifulSoup(html_text, 'lxml')
            return soup

        except Exception as e:
            logger.error(f"HTML parsing failed: {e}")
            return None

    def _extract_id_from_url(self, url: str, pattern: str = r'/(\d+)/') -> Optional[str]:
        """
        Extract ID from URL using regex pattern.

        Args:
            url: URL string
            pattern: Regex pattern to match ID

        Returns:
            Extracted ID or None
        """
        if not url:
            return None

        match = re.search(pattern, url)
        if match:
            return match.group(1)
        return None

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        return re.sub(r'\s+', ' ', text).strip()

    def _parse_distance(self, text: str) -> Optional[int]:
        """
        Parse distance from text (e.g., '1600m' -> 1600).

        Args:
            text: Text containing distance

        Returns:
            Distance in meters or None
        """
        if not text:
            return None

        match = re.search(r'(\d+)m?', text)
        if match:
            return int(match.group(1))
        return None

    def _parse_horse_weight(self, text: str) -> tuple:
        """
        Parse horse weight text (e.g., '500(+10)' or '500 (+10)' -> (500, 10)).

        Args:
            text: Weight text

        Returns:
            Tuple of (weight, change) or (None, None)
        """
        if not text:
            return None, None

        # Remove all spaces first
        text = re.sub(r'\s+', '', text)

        # Pattern: 500(+10) or 500(-5) or 500(0)
        match = re.match(r'(\d+)\(([+-]?\d+)\)', text)
        if match:
            weight = int(match.group(1))
            change = int(match.group(2))
            return weight, change

        # Just weight, no change
        match = re.match(r'(\d+)', text)
        if match:
            return int(match.group(1)), None

        return None, None

    def _parse_time_to_seconds(self, time_str: str) -> Optional[float]:
        """
        Parse race time to seconds (e.g., '1:34.5' -> 94.5).

        Args:
            time_str: Time string

        Returns:
            Time in seconds or None
        """
        if not time_str:
            return None

        # Pattern: 1:34.5
        match = re.match(r'(\d+):(\d+)\.(\d+)', time_str.strip())
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            fraction = int(match.group(3))
            return minutes * 60 + seconds + fraction / 10.0

        # Pattern: 34.5 (no minutes)
        match = re.match(r'(\d+)\.(\d+)', time_str.strip())
        if match:
            seconds = int(match.group(1))
            fraction = int(match.group(2))
            return seconds + fraction / 10.0

        return None

    def scrape_race_calendar(self, date: datetime) -> List[Dict]:
        """
        Scrape race calendar for a specific date.

        Uses mobile site because PC version loads race list via JavaScript.

        Args:
            date: Date to scrape races for

        Returns:
            List of race dictionaries with basic info
        """
        # Use mobile site - it has static HTML with race_id links
        url = f"{self.MOBILE_URL}/"
        params = {'pid': 'race_list', 'kaisai_date': date.strftime('%Y%m%d')}

        logger.info(f"Scraping race calendar for {date.strftime('%Y-%m-%d')}")

        response = self._make_request(url, params)
        if not response:
            return []

        soup = self._parse_html(response)
        if not soup:
            return []

        races = []

        # Find all race links with race_id parameter
        race_links = soup.find_all('a', href=re.compile(r'race_id=\d{12}'))

        # Use set to avoid duplicates (mobile site has multiple links per race)
        seen_race_ids = set()

        for race_link in race_links:
            try:
                href = race_link.get('href', '')

                # Extract race_id from URL
                match = re.search(r'race_id=(\d{12})', href)
                if not match:
                    continue

                race_id = match.group(1)

                # Skip duplicates
                if race_id in seen_race_ids:
                    continue
                seen_race_ids.add(race_id)

                # Parse race_id to extract components
                # Format: YYYYKKRRDDNN
                # YYYY = year, KK = kaisai code, RR = meeting, DD = day, NN = race number
                if len(race_id) == 12:
                    year = race_id[0:4]
                    kaisai_code = race_id[4:6]
                    meeting_number = int(race_id[6:8])
                    day_number = int(race_id[8:10])
                    race_number = int(race_id[10:12])
                else:
                    logger.warning(f"Invalid race_id format: {race_id}")
                    continue

                # Track code mapping (kaisai_code to track name)
                # Based on JRA track codes
                track_code_map = {
                    '01': '札幌',
                    '02': '函館',
                    '03': '福島',
                    '04': '新潟',
                    '05': '東京',
                    '06': '中山',
                    '07': '中京',
                    '08': '京都',
                    '09': '阪神',
                    '10': '小倉',
                }

                track_name = track_code_map.get(kaisai_code, None)

                race_info = {
                    'netkeiba_race_id': race_id,
                    'track': track_name,
                    'race_date': date.date(),
                    'race_number': race_number,
                    'kaisai_code': kaisai_code,
                    'meeting_number': meeting_number,
                    'day_number': day_number,
                }

                races.append(race_info)
                logger.debug(f"Found race: {race_id} at {track_name} R{race_number}")

            except Exception as e:
                logger.error(f"Error parsing race link: {e}")
                continue

        logger.info(f"Found {len(races)} races for {date.strftime('%Y-%m-%d')}")
        return races

    def scrape_race_card(self, race_id: str) -> Optional[Dict]:
        """
        Scrape race card (shutuba) for a specific race.

        Args:
            race_id: Netkeiba race ID (12 digits, e.g., '202506050811')

        Returns:
            Dictionary with race info and entries, or None on failure
        """
        url = f"{self.BASE_URL}/race/shutuba.html"
        params = {'race_id': race_id}

        logger.info(f"Scraping race card for race_id={race_id}")

        response = self._make_request(url, params)
        if not response:
            return None

        soup = self._parse_html(response)
        if not soup:
            return None

        try:
            race_data = self._parse_race_card_data(soup, race_id)
            return race_data

        except Exception as e:
            logger.error(f"Error parsing race card for {race_id}: {e}")
            return None

    def _parse_race_card_data(self, soup: BeautifulSoup, race_id: str) -> Dict:
        """Parse race card HTML into structured data."""

        # Extract race name from h1 tag with class RaceName
        race_name = None
        race_name_elem = soup.find('h1', class_='RaceName')
        if race_name_elem:
            race_name = self._clean_text(race_name_elem.get_text())
        else:
            # Fallback: try RaceList_Item02 and extract just the first line
            race_list_elem = soup.find('div', class_='RaceList_Item02')
            if race_list_elem:
                # Get first non-empty line
                lines = [line.strip() for line in race_list_elem.get_text().split('\n') if line.strip()]
                if lines:
                    race_name = lines[0]

        # Extract race metadata from RaceData01
        race_title_elem = soup.find('div', class_='RaceData01')
        distance = None
        surface = None
        weather = None
        track_condition = None

        if race_title_elem:
            title_text = self._clean_text(race_title_elem.get_text())

            # Extract distance (e.g., "芝1600m" or "ダ1200m")
            distance_match = re.search(r'[芝ダ](\d+)m', title_text)
            if distance_match:
                distance = int(distance_match.group(1))

            # Extract surface
            if '芝' in title_text:
                surface = 'turf'
            elif 'ダ' in title_text or 'ダート' in title_text:
                surface = 'dirt'

            # Extract weather from text (e.g., "天候:晴")
            weather_match = re.search(r'天候[:：]\s*([晴曇雨小雨]+)', title_text)
            if weather_match:
                weather = weather_match.group(1)

            # Extract track condition from text (e.g., "馬場:良")
            track_match = re.search(r'馬場[:：]\s*([良稍重不良]+)', title_text)
            if track_match:
                track_condition = track_match.group(1)

        # Extract race class from RaceData02
        race_data02 = soup.find('div', class_='RaceData02')
        race_class = None

        if race_data02:
            data_text = self._clean_text(race_data02.get_text())

            # Extract race class (G1, G2, G3, etc.)
            # Note: Use both half-width (3) and full-width (３) numbers
            if 'G1' in data_text or 'GⅠ' in data_text or 'GI' in data_text:
                race_class = 'G1'
            elif 'G2' in data_text or 'GⅡ' in data_text or 'GII' in data_text:
                race_class = 'G2'
            elif 'G3' in data_text or 'GⅢ' in data_text or 'GIII' in data_text:
                race_class = 'G3'
            elif 'OP' in data_text or 'オープン' in data_text:
                race_class = 'OP'
            elif 'L' in data_text or 'リステッド' in data_text:
                race_class = 'Listed'
            elif '3勝クラス' in data_text or '３勝クラス' in data_text:
                race_class = '3勝クラス'
            elif '2勝クラス' in data_text or '２勝クラス' in data_text:
                race_class = '2勝クラス'
            elif '1勝クラス' in data_text or '１勝クラス' in data_text:
                race_class = '1勝クラス'
            elif '未勝利' in data_text:
                race_class = '未勝利'
            elif '新馬' in data_text:
                race_class = '新馬'

        # Parse entries table
        entries = []
        shutuba_table = soup.find('table', class_='Shutuba_Table')

        if not shutuba_table:
            logger.warning(f"No Shutuba_Table found for race_id={race_id}")
        else:
            entry_rows = shutuba_table.find_all('tr', class_=re.compile(r'HorseList'))

            for row in entry_rows:
                try:
                    entry = self._parse_entry_row(row)
                    if entry:
                        entries.append(entry)
                except Exception as e:
                    logger.error(f"Error parsing entry row: {e}")
                    continue

        logger.info(f"Parsed race card: {len(entries)} entries")

        return {
            'race_info': {
                'netkeiba_race_id': race_id,
                'race_name': race_name,
                'distance': distance,
                'surface': surface,
                'race_class': race_class,
                'weather': weather,
                'track_condition': track_condition,
            },
            'entries': entries
        }

    def _parse_entry_row(self, row: BeautifulSoup) -> Optional[Dict]:
        """Parse a single entry row from shutuba table."""

        entry = {}

        # Horse number (枠番 and 馬番)
        # Actual classes: Waku1, Waku2, etc. - find td with class starting with 'Waku'
        waku_td = row.find('td', class_=re.compile(r'Waku\d+'))
        if waku_td:
            waku_text = self._clean_text(waku_td.get_text())
            try:
                entry['post_position'] = int(waku_text)
            except ValueError:
                entry['post_position'] = None

        # Actual classes: Umaban1, Umaban2, etc. - find td with class starting with 'Umaban'
        umaban_td = row.find('td', class_=re.compile(r'Umaban\d+'))
        if umaban_td:
            umaban_text = self._clean_text(umaban_td.get_text())
            try:
                entry['horse_number'] = int(umaban_text)
            except ValueError:
                entry['horse_number'] = None

        # Horse info
        horse_td = row.find('td', class_='HorseInfo')
        if horse_td:
            horse_link = horse_td.find('a')
            if horse_link:
                entry['horse_name'] = self._clean_text(horse_link.get_text())
                horse_url = horse_link.get('href', '')
                # URL format: https://db.netkeiba.com/horse/2022104614
                entry['netkeiba_horse_id'] = self._extract_id_from_url(horse_url, r'/horse/(\d+)')

        # Jockey
        jockey_td = row.find('td', class_='Jockey')
        if jockey_td:
            jockey_link = jockey_td.find('a')
            if jockey_link:
                entry['jockey_name'] = self._clean_text(jockey_link.get_text())
                jockey_url = jockey_link.get('href', '')
                # URL format: https://db.netkeiba.com/jockey/result/recent/01160/
                entry['netkeiba_jockey_id'] = self._extract_id_from_url(jockey_url, r'/jockey/[^/]+/[^/]+/(\d+)')

        # Trainer
        trainer_td = row.find('td', class_='Trainer')
        if trainer_td:
            trainer_link = trainer_td.find('a')
            if trainer_link:
                entry['trainer_name'] = self._clean_text(trainer_link.get_text())
                trainer_url = trainer_link.get('href', '')
                # URL format: https://db.netkeiba.com/trainer/result/recent/01128/
                entry['netkeiba_trainer_id'] = self._extract_id_from_url(trainer_url, r'/trainer/[^/]+/[^/]+/(\d+)')

        # Weight (斤量) - find td after Barei column
        tds = row.find_all('td')
        for i, td in enumerate(tds):
            # Look for td after Barei (age/sex column)
            if 'Barei' in td.get('class', []):
                if i + 1 < len(tds):
                    weight_td = tds[i + 1]
                    weight_text = self._clean_text(weight_td.get_text())
                    try:
                        entry['weight'] = float(weight_text)
                    except ValueError:
                        entry['weight'] = None
                break

        # Horse weight (馬体重) - class 'Weight'
        weight_td = row.find('td', class_='Weight')
        if weight_td:
            weight_text = self._clean_text(weight_td.get_text())
            horse_weight, weight_change = self._parse_horse_weight(weight_text)
            entry['horse_weight'] = horse_weight
            entry['horse_weight_change'] = weight_change

        # Morning odds - class 'Popular' with specific span
        odds_td = row.find('td', class_=re.compile(r'Popular'))
        if odds_td:
            # Look for span with id starting with 'odds-'
            odds_span = odds_td.find('span', id=re.compile(r'odds-'))
            if odds_span:
                odds_text = self._clean_text(odds_span.get_text())
                # Handle '---.-' for unavailable odds
                if odds_text and odds_text != '---.-':
                    try:
                        entry['morning_odds'] = float(odds_text)
                    except ValueError:
                        entry['morning_odds'] = None
                else:
                    entry['morning_odds'] = None

        return entry if entry else None

    def scrape_race_result(self, race_id: str) -> Optional[Dict]:
        """
        Scrape race result including payouts.

        Args:
            race_id: Netkeiba race ID

        Returns:
            Dictionary with results and payouts, or None on failure
        """
        url = f"{self.BASE_URL}/race/result.html"
        params = {'race_id': race_id}

        logger.info(f"Scraping race result for race_id={race_id}")

        response = self._make_request(url, params)
        if not response:
            return None

        soup = self._parse_html(response)
        if not soup:
            return None

        try:
            result_data = self._parse_race_result_data(soup, race_id)
            return result_data

        except Exception as e:
            logger.error(f"Error parsing race result for {race_id}: {e}")
            return None

    def _parse_race_result_data(self, soup: BeautifulSoup, race_id: str) -> Dict:
        """Parse race result HTML into structured data."""

        results = []
        payouts = {}

        # Find result table - looking for table with RaceTable01 class
        result_table = soup.find('table', class_=re.compile(r'RaceTable01'))

        if not result_table:
            logger.warning(f"No result table found for race_id={race_id}")
        else:
            result_rows = result_table.find_all('tr', class_=re.compile(r'HorseList'))

            for row in result_rows:
                try:
                    result = self._parse_result_row(row)
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error parsing result row: {e}")
                    continue

        # Parse payouts
        payout_tables = soup.find_all('table', class_=re.compile(r'Payout'))
        for table in payout_tables:
            try:
                parsed_payouts = self._parse_payout_table(table)
                payouts.update(parsed_payouts)
            except Exception as e:
                logger.error(f"Error parsing payout table: {e}")
                continue

        logger.info(f"Parsed results: {len(results)} horses, {len(payouts)} payout types")

        return {
            'results': results,
            'payouts': payouts
        }

    def _parse_result_row(self, row: BeautifulSoup) -> Optional[Dict]:
        """Parse a single result row."""

        result = {}

        # Finish position - class 'Result_Num'
        result_num_td = row.find('td', class_='Result_Num')
        if result_num_td:
            position_text = self._clean_text(result_num_td.get_text())
            try:
                result['finish_position'] = int(position_text)
            except ValueError:
                result['finish_position'] = None

        # Horse number - find td with class containing 'Num' and 'Txt_C'
        tds = row.find_all('td')
        for td in tds:
            classes = td.get('class', [])
            if 'Num' in classes and 'Txt_C' in classes:
                horse_num_text = self._clean_text(td.get_text())
                try:
                    result['horse_number'] = int(horse_num_text)
                except ValueError:
                    pass
                break

        # Time - class 'Time'
        time_tds = row.find_all('td', class_='Time')
        if time_tds and len(time_tds) > 0:
            time_text = self._clean_text(time_tds[0].get_text())
            if time_text:  # Only parse if not empty
                result['finish_time'] = self._parse_time_to_seconds(time_text)

        # Final odds - class 'Odds'
        odds_td = row.find('td', class_='Odds')
        if odds_td:
            odds_text = self._clean_text(odds_td.get_text())
            try:
                result['final_odds'] = float(odds_text)
            except ValueError:
                result['final_odds'] = None

        # Popularity - usually in td with BgYellow class
        for td in tds:
            classes = td.get('class', [])
            if 'BgYellow' in classes:
                ninki_text = self._clean_text(td.get_text())
                try:
                    result['popularity'] = int(ninki_text)
                except ValueError:
                    pass
                break

        # Margin (着差) - may be in a specific column, try to find it
        margin_td = row.find('td', class_='Distance')
        if margin_td:
            result['margin'] = self._clean_text(margin_td.get_text())

        # Running positions (通過順位) - class 'PassingRank'
        passing_td = row.find('td', class_='PassingRank')
        if passing_td:
            passing_text = self._clean_text(passing_td.get_text())
            # Format: "11-11-11-11"
            if passing_text and '-' in passing_text:
                positions = [int(p) for p in passing_text.split('-') if p.strip().isdigit()]
                result['running_positions'] = positions if positions else None

        return result if result else None

    def _parse_payout_table(self, table: BeautifulSoup) -> Dict:
        """Parse payout table into structured data."""

        payouts = {}

        rows = table.find_all('tr')
        for row in rows:
            try:
                # Find bet type header
                th = row.find('th')
                if not th:
                    continue

                bet_type_text = self._clean_text(th.get_text())

                # Map Japanese bet types to English
                bet_type_map = {
                    '単勝': 'win',
                    '複勝': 'place',
                    '枠連': 'bracket_quinella',
                    '馬連': 'quinella',
                    '馬単': 'exacta',
                    'ワイド': 'wide',
                    '3連複': 'trio',
                    '3連単': 'trifecta',
                }

                bet_type = None
                for jp, en in bet_type_map.items():
                    if jp in bet_type_text:
                        bet_type = en
                        break

                if not bet_type:
                    continue

                # Find payout cells
                payout_tds = row.find_all('td')

                payout_list = []
                for td in payout_tds:
                    # Extract combination and payout
                    td_text = self._clean_text(td.get_text())

                    # Pattern: "1-2" or "1" followed by payout amount
                    parts = td_text.split()
                    if len(parts) >= 2:
                        combination = parts[0]
                        payout_str = parts[1].replace(',', '').replace('円', '')

                        try:
                            payout_amount = int(payout_str)
                            payout_list.append({
                                'combination': combination,
                                'payout': payout_amount
                            })
                        except ValueError:
                            continue

                if payout_list:
                    payouts[bet_type] = payout_list

            except Exception as e:
                logger.error(f"Error parsing payout row: {e}")
                continue

        return payouts

    def scrape_horse_profile(self, horse_id: str) -> Optional[Dict]:
        """
        Scrape detailed horse profile.

        Args:
            horse_id: Netkeiba horse ID

        Returns:
            Dictionary with horse profile data, or None on failure
        """
        url = f"{self.DB_URL}/horse/{horse_id}/"

        logger.info(f"Scraping horse profile for horse_id={horse_id}")

        response = self._make_request(url)
        if not response:
            return None

        soup = self._parse_html(response)
        if not soup:
            return None

        try:
            profile = self._parse_horse_profile_data(soup, horse_id)
            return profile

        except Exception as e:
            logger.error(f"Error parsing horse profile for {horse_id}: {e}")
            return None

    def _parse_horse_profile_data(self, soup: BeautifulSoup, horse_id: str) -> Dict:
        """Parse horse profile HTML into structured data."""

        profile = {
            'netkeiba_horse_id': horse_id
        }

        # Extract horse name
        horse_title = soup.find('div', class_='horse_title')
        if horse_title:
            name_elem = horse_title.find('h1')
            if name_elem:
                profile['name'] = self._clean_text(name_elem.get_text())

        # Extract profile table data
        profile_table = soup.find('table', class_='db_prof_table')
        if profile_table:
            rows = profile_table.find_all('tr')
            for row in rows:
                th = row.find('th')
                td = row.find('td')

                if not th or not td:
                    continue

                header = self._clean_text(th.get_text())
                value = self._clean_text(td.get_text())

                # Parse specific fields
                if '生年月日' in header or '生' in header:
                    # Parse birth date
                    date_match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', value)
                    if date_match:
                        year = int(date_match.group(1))
                        month = int(date_match.group(2))
                        day = int(date_match.group(3))
                        profile['birth_date'] = datetime(year, month, day).date()

                elif '性' in header:
                    # Sex
                    if '牡' in value:
                        profile['sex'] = '牡'
                    elif '牝' in value:
                        profile['sex'] = '牝'
                    elif 'セ' in value or 'せん' in value:
                        profile['sex'] = 'セン'

                elif '調教師' in header:
                    # Trainer
                    trainer_link = td.find('a')
                    if trainer_link:
                        profile['trainer_name'] = self._clean_text(trainer_link.get_text())
                        trainer_url = trainer_link.get('href', '')
                        profile['netkeiba_trainer_id'] = self._extract_id_from_url(
                            trainer_url, r'/trainer/[^/]+/(\d+)/'
                        )

        # Extract pedigree (父、母)
        blood_table = soup.find('table', class_=re.compile(r'blood_table'))
        if blood_table:
            # Find sire (father)
            sire_link = blood_table.find('a', string=re.compile(r'.+'))  # First link
            if sire_link:
                profile['sire_name'] = self._clean_text(sire_link.get_text())
                sire_url = sire_link.get('href', '')
                profile['sire_netkeiba_id'] = self._extract_id_from_url(sire_url, r'/horse/(\d+)/')

        return profile

    def get_upcoming_races(self, days_ahead: int = 7) -> List[Dict]:
        """
        Get upcoming races for the next N days.

        Args:
            days_ahead: Number of days to look ahead

        Returns:
            List of all races in the date range
        """
        all_races = []
        today = datetime.now()

        for i in range(days_ahead):
            date = today + timedelta(days=i)
            races = self.scrape_race_calendar(date)
            all_races.extend(races)

        logger.info(f"Found {len(all_races)} races in next {days_ahead} days")
        return all_races

    def close(self):
        """Close the session."""
        self.session.close()
        logger.info("NetkeibaScraper session closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
