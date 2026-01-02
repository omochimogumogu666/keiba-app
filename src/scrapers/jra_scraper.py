"""
JRA website scraper for horse racing data.

This module handles scraping race information, horse data, and results
from the JRA website while respecting rate limits and robots.txt.
"""
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from config.settings import Config
from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)


class JRAScraper:
    """JRA website scraper."""

    # JRA URLs - These are placeholder URLs and need to be updated with actual JRA endpoints
    BASE_URL = "https://www.jra.go.jp"
    RACE_CALENDAR_URL = f"{BASE_URL}/JRADB/accessS.html"
    RACE_CARD_URL = f"{BASE_URL}/JRADB/accessS.html"
    RACE_RESULT_URL = f"{BASE_URL}/JRADB/accessK.html"

    def __init__(self, delay: int = None):
        """
        Initialize JRA scraper.

        Args:
            delay: Delay between requests in seconds (default from config)
        """
        self.delay = delay if delay is not None else Config.SCRAPING_DELAY
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': Config.USER_AGENT
        })
        logger.info(f"JRA Scraper initialized with {self.delay}s delay")

    def _make_request(self, url: str, params: Optional[Dict] = None) -> Optional[requests.Response]:
        """
        Make HTTP request with error handling and rate limiting.

        Args:
            url: URL to request
            params: Query parameters

        Returns:
            Response object or None if failed
        """
        try:
            logger.debug(f"Requesting: {url}")
            response = self.session.get(
                url,
                params=params,
                timeout=Config.REQUEST_TIMEOUT
            )
            response.raise_for_status()

            # JRA uses Shift_JIS encoding
            response.encoding = 'shift_jis'

            # Polite scraping: wait before next request
            time.sleep(self.delay)

            return response
        except requests.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            return None

    def _parse_html(self, html: str) -> Optional[BeautifulSoup]:
        """
        Parse HTML content.

        Args:
            html: HTML string

        Returns:
            BeautifulSoup object or None
        """
        try:
            return BeautifulSoup(html, 'lxml')
        except Exception as e:
            logger.error(f"HTML parsing failed: {e}")
            return None

    def _extract_cname_from_url(self, url: str) -> Optional[str]:
        """
        Extract CNAME parameter from JRA URL.

        Args:
            url: JRA URL string

        Returns:
            CNAME value or None
        """
        if '?CNAME=' in url:
            try:
                return url.split('?CNAME=')[1].split('&')[0]
            except IndexError:
                return None
        return None

    def _extract_race_links(self, soup: BeautifulSoup) -> List[Dict]:
        """
        Extract race links and CNAME from soup object.

        Args:
            soup: BeautifulSoup object

        Returns:
            List of dictionaries with race link information
        """
        race_links = []

        # Find all links to JRADB
        for link in soup.find_all('a', href=True):
            href = link.get('href', '')

            # Check if this is a JRADB race link
            if '/JRADB/accessD.html' in href or 'CNAME=' in href:
                cname = self._extract_cname_from_url(href)
                if cname:
                    race_links.append({
                        'url': href if href.startswith('http') else self.BASE_URL + href,
                        'cname': cname,
                        'text': link.get_text(strip=True)
                    })

        return race_links

    def scrape_race_calendar(self, date: datetime) -> List[Dict]:
        """
        Scrape race calendar for a specific date.

        Args:
            date: Date to scrape races for

        Returns:
            List of race information dictionaries
        """
        from src.scrapers.utils import parse_distance, clean_text
        import re

        logger.info(f"Scraping race calendar for {date.strftime('%Y-%m-%d')}")

        races = []

        # Access "This Week's Races" page
        calendar_url = f"{self.BASE_URL}/keiba/thisweek/"
        response = self._make_request(calendar_url)

        if not response:
            logger.error("Failed to fetch race calendar")
            return races

        soup = self._parse_html(response.text)
        if not soup:
            logger.error("Failed to parse race calendar HTML")
            return races

        # Extract race links
        race_links = self._extract_race_links(soup)
        logger.debug(f"Found {len(race_links)} race links")

        # Process each race link to extract basic information
        for link_info in race_links:
            try:
                cname = link_info['cname']
                link_text = link_info['text']

                # Try to extract race information from CNAME and link text
                # CNAME format: pw01dde[競馬場コード][日付YYYYMMDD]/[レース番号]
                # Example: pw01dde0106202601021120260105/6C

                # Extract date from CNAME (this is a best-effort approach)
                date_match = re.search(r'(20\d{6})', cname)
                race_date = date
                if date_match:
                    try:
                        date_str = date_match.group(1)
                        race_date = datetime.strptime(date_str, '%Y%m%d')
                    except ValueError:
                        pass

                # Extract race number from CNAME (approximate)
                race_number_match = re.search(r'/(\d+)', cname)
                race_number = 1
                if race_number_match:
                    try:
                        race_number = int(race_number_match.group(1), 16)  # Hexadecimal
                    except ValueError:
                        race_number = 1

                # Extract basic info from link text
                race_name = clean_text(link_text) if link_text else "Unknown Race"

                # Extract distance from text (if present)
                distance = None
                distance_match = re.search(r'(\d{3,4})[mメートル]', link_text)
                if distance_match:
                    distance = int(distance_match.group(1))

                # Extract surface type (turf/dirt)
                surface = 'turf'  # Default
                if 'ダート' in link_text or 'ダ' in link_text:
                    surface = 'dirt'

                # Extract race class
                race_class = None
                if 'G1' in link_text:
                    race_class = 'G1'
                elif 'G2' in link_text:
                    race_class = 'G2'
                elif 'G3' in link_text:
                    race_class = 'G3'
                elif 'OP' in link_text or 'オープン' in link_text:
                    race_class = 'OP'

                # Generate jra_race_id (format: YYYYMMDD + track_code + race_number)
                # This is a simplified approach; actual ID may vary
                jra_race_id = f"{race_date.strftime('%Y%m%d')}01{race_number:02d}"

                # Extract track name (simplified - would need more sophisticated parsing)
                track = "Unknown"
                for track_name in ['東京', '中山', '京都', '阪神', '中京', '札幌', '函館', '福島', '新潟', '小倉']:
                    if track_name in link_text:
                        track = track_name
                        break

                race_info = {
                    'jra_race_id': jra_race_id,
                    'track': track,
                    'race_date': race_date,
                    'race_number': race_number,
                    'race_name': race_name,
                    'distance': distance,
                    'surface': surface,
                    'race_class': race_class,
                    'cname': cname  # Store CNAME for future use
                }

                # Filter by date if specified
                if race_date.date() == date.date():
                    races.append(race_info)

            except Exception as e:
                logger.warning(f"Failed to parse race info from link: {e}")
                continue

        logger.info(f"Scraped {len(races)} races for {date.strftime('%Y-%m-%d')}")
        return races

    def _extract_cname_from_onclick(self, onclick_str: str) -> Optional[str]:
        """
        Extract CNAME from onclick attribute.

        Args:
            onclick_str: onclick attribute string like "return doAction('/JRADB/accessK.html', 'pw04kmk005386/50');"

        Returns:
            CNAME string or None
        """
        if not onclick_str:
            return None

        import re
        # Extract CNAME from doAction second parameter
        match = re.search(r"doAction\([^,]+,\s*'([^']+)'\)", onclick_str)
        if match:
            return match.group(1)
        return None

    def scrape_race_card(self, race_id: str, cname: Optional[str] = None) -> Optional[Dict]:
        """
        Scrape race card (entries) for a specific race.

        Args:
            race_id: JRA race ID
            cname: Optional CNAME parameter for direct access

        Returns:
            Dictionary with race and entry information
        """
        from src.scrapers.utils import parse_japanese_number, clean_text
        import re

        logger.info(f"Scraping race card for race {race_id}")

        # Construct URL using CNAME if available
        if cname:
            race_url = f"{self.BASE_URL}/JRADB/accessD.html?CNAME={cname}"
        else:
            # Fallback: try to construct URL from race_id (may not work)
            race_url = f"{self.RACE_CARD_URL}?id={race_id}"
            logger.warning("CNAME not provided, using fallback URL construction")

        response = self._make_request(race_url)
        if not response:
            logger.error(f"Failed to fetch race card for {race_id}")
            return None

        soup = self._parse_html(response.text)
        if not soup:
            logger.error(f"Failed to parse race card HTML for {race_id}")
            return None

        # Initialize result structure with default values
        race_card = {
            'jra_race_id': race_id,
            'race_name': None,
            'track': None,
            'race_date': None,
            'distance': None,
            'surface': None,
            'race_class': None,
            'track_condition': None,
            'weather': None,
            'prize_money': None,
            'entries': []
        }

        # Extract race information from H1 tag
        # Format: "出馬表 2026年1月4日(日)1回中山1日 11レース"
        # Note: There are multiple H1 tags, need to find the one with race info
        h1_tags = soup.find_all('h1')
        h1_text = None
        for h1_tag in h1_tags:
            text = clean_text(h1_tag.get_text())
            # Look for H1 with race info (contains "レース" or "回")
            if 'レース' in text or ('回' in text and '日' in text):
                h1_text = text
                break

        if h1_text:
            logger.debug(f"H1 text: {h1_text}")

            # Extract date
            date_match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', h1_text)
            if date_match:
                try:
                    from datetime import datetime
                    year = int(date_match.group(1))
                    month = int(date_match.group(2))
                    day = int(date_match.group(3))
                    race_card['race_date'] = datetime(year, month, day)
                except ValueError:
                    pass

            # Extract track name (e.g., "中山", "東京", "京都", etc.)
            # Pattern: N回[競馬場]N日
            track_match = re.search(r'\d+回([^0-9]+?)\d+日', h1_text)
            if track_match:
                track_name = track_match.group(1)
                race_card['track'] = track_name
                logger.debug(f"Extracted track: {track_name}")

            # Extract race number (e.g., "11レース" -> 11)
            race_num_match = re.search(r'(\d+)レース', h1_text)
            if race_num_match:
                race_card['race_number'] = int(race_num_match.group(1))

        # Extract race details from race_title div
        # Format: "第75回日刊スポーツ賞中山金杯 4歳以上オープン(国際)(指定)芝・右コース:2,000メートル(外・外)"
        race_title_div = soup.find('div', class_='race_title')
        if race_title_div:
            title_text = clean_text(race_title_div.get_text())
            logger.debug(f"Race title text: {title_text}")

            # Extract race name (everything before "歳以上" or before course details)
            # Remove "第N回" prefix if present
            name_text = re.sub(r'^第\d+回', '', title_text)

            # Split at course details (e.g., "芝・", "ダート・", "障害")
            if 'コース' in name_text:
                parts = name_text.split('コース')
                if len(parts) > 0:
                    # Take everything before course info
                    before_course = parts[0]
                    # Remove trailing surface/direction info
                    name = re.sub(r'[芝ダート障害]・[左右内外]$', '', before_course).strip()
                    race_card['race_name'] = name
                    logger.debug(f"Extracted race name: {name}")

            # Extract distance (e.g., "2,000メートル" or "2000m")
            distance_match = re.search(r'[：:]\s*([0-9,]+)\s*[メートルm]', title_text)
            if distance_match:
                distance_str = distance_match.group(1).replace(',', '')
                try:
                    race_card['distance'] = int(distance_str)
                    logger.debug(f"Extracted distance: {race_card['distance']}m")
                except ValueError:
                    pass

            # Extract surface type (芝/ダート/障害)
            if '芝' in title_text:
                race_card['surface'] = 'turf'
            elif 'ダート' in title_text:
                race_card['surface'] = 'dirt'
            elif '障害' in title_text:
                race_card['surface'] = 'jump'

            # Extract race class (G1, G2, G3, etc.)
            if 'G1' in title_text or 'GⅠ' in title_text:
                race_card['race_class'] = 'G1'
            elif 'G2' in title_text or 'GⅡ' in title_text:
                race_card['race_class'] = 'G2'
            elif 'G3' in title_text or 'GⅢ' in title_text:
                race_card['race_class'] = 'G3'
            elif 'オープン' in title_text or 'OP' in title_text:
                race_card['race_class'] = 'Open'
            elif '未勝利' in title_text:
                race_card['race_class'] = 'Maiden'
            elif '新馬' in title_text:
                race_card['race_class'] = 'Newcomer'

        # Extract prize money from prize list
        prize_list = soup.find('ul', class_='prize')
        if prize_list:
            # Look for first place prize in the ordered list
            prize_ol = prize_list.find('ol')
            if prize_ol:
                first_place_li = prize_ol.find('li')
                if first_place_li:
                    # Extract number from span with class="num"
                    num_span = first_place_li.find('span', class_='num')
                    if num_span:
                        prize_text = clean_text(num_span.get_text())
                        prize_text = prize_text.replace(',', '')
                        try:
                            race_card['prize_money'] = int(prize_text) * 10000  # Convert 万円 to 円
                        except ValueError:
                            logger.warning(f"Failed to parse prize money: {prize_text}")

        # Track condition and weather are usually published closer to race time
        # For now, leave them as None

        # Extract entries from the main table (class="basic narrow-xy mt20")
        main_table = soup.find('table', class_='basic')

        if not main_table:
            logger.warning("Could not find main entry table")
            return race_card

        tbody = main_table.find('tbody')
        if not tbody:
            logger.warning("Could not find tbody in main table")
            return race_card

        rows = tbody.find_all('tr')

        for row in rows:
            try:
                entry_data = {}

                # Extract post position (枠) and horse number (馬番)
                waku_cell = row.find('td', class_='waku')
                num_cell = row.find('td', class_='num')

                if waku_cell:
                    waku_text = clean_text(waku_cell.get_text())
                    if waku_text and waku_text.isdigit():
                        entry_data['post_position'] = int(waku_text)

                if num_cell:
                    num_text = clean_text(num_cell.get_text())
                    if num_text and num_text.isdigit():
                        entry_data['horse_number'] = int(num_text)

                # Extract horse information
                horse_cell = row.find('td', class_='horse')
                if horse_cell:
                    # Horse name link
                    horse_link = horse_cell.find('div', class_='name_line')
                    if horse_link:
                        horse_a = horse_link.find('a')
                        if horse_a:
                            entry_data['horse_name'] = clean_text(horse_a.get_text())
                            # Extract horse ID from CNAME in href
                            href = horse_a.get('href', '')
                            cname = self._extract_cname_from_url(href)
                            if cname:
                                entry_data['jra_horse_id'] = cname
                            else:
                                entry_data['jra_horse_id'] = 'H_UNKNOWN'

                    # Trainer info
                    trainer_p = horse_cell.find('p', class_='trainer')
                    if trainer_p:
                        trainer_a = trainer_p.find('a')
                        if trainer_a:
                            entry_data['trainer_name'] = clean_text(trainer_a.get_text())
                            # Extract trainer ID from onclick
                            onclick = trainer_a.get('onclick', '')
                            cname = self._extract_cname_from_onclick(onclick)
                            if cname:
                                entry_data['jra_trainer_id'] = cname
                            else:
                                entry_data['jra_trainer_id'] = 'T_UNKNOWN'

                # Extract jockey information
                jockey_cell = row.find('td', class_='jockey')
                if jockey_cell:
                    # Weight
                    weight_p = jockey_cell.find('p', class_='weight')
                    if weight_p:
                        weight_text = clean_text(weight_p.get_text())
                        weight_match = re.search(r'(\d+\.?\d*)', weight_text)
                        if weight_match:
                            entry_data['weight'] = float(weight_match.group(1))

                    # Jockey name
                    jockey_p = jockey_cell.find('p', class_='jockey')
                    if jockey_p:
                        jockey_a = jockey_p.find('a')
                        if jockey_a:
                            entry_data['jockey_name'] = clean_text(jockey_a.get_text())
                            # Extract jockey ID from onclick
                            onclick = jockey_a.get('onclick', '')
                            cname = self._extract_cname_from_onclick(onclick)
                            if cname:
                                entry_data['jra_jockey_id'] = cname
                            else:
                                entry_data['jra_jockey_id'] = 'J_UNKNOWN'

                    # Age/Sex info (e.g., "牝5/鹿")
                    age_p = jockey_cell.find('p', class_='age')
                    if age_p:
                        age_text = clean_text(age_p.get_text())
                        # Parse sex (牡/牝/セン)
                        if '牡' in age_text:
                            entry_data['sex'] = '牡'
                        elif '牝' in age_text:
                            entry_data['sex'] = '牝'
                        elif 'セン' in age_text:
                            entry_data['sex'] = 'セン'

                # Only add entry if we have at least a horse name
                if entry_data.get('horse_name'):
                    # Set defaults for missing values
                    entry_data.setdefault('jra_horse_id', f"H{race_id}_{entry_data.get('horse_number', 0)}")
                    entry_data.setdefault('jra_jockey_id', 'J_UNKNOWN')
                    entry_data.setdefault('jra_trainer_id', 'T_UNKNOWN')
                    entry_data.setdefault('post_position', entry_data.get('horse_number', 1))
                    entry_data.setdefault('horse_number', 1)
                    entry_data.setdefault('weight', 57.0)
                    entry_data.setdefault('horse_weight', 500)
                    entry_data.setdefault('horse_weight_change', 0)
                    entry_data.setdefault('morning_odds', 10.0)

                    race_card['entries'].append(entry_data)

            except Exception as e:
                logger.debug(f"Failed to parse entry row: {e}")
                continue

        if not race_card['entries']:
            logger.warning(f"No entries found for race {race_id}")

        logger.info(f"Scraped {len(race_card['entries'])} entries for race {race_id}")
        return race_card

    def scrape_race_result(self, race_id: str, cname: Optional[str] = None) -> Optional[Dict]:
        """
        Scrape race result for a completed race.

        Args:
            race_id: JRA race ID
            cname: Optional CNAME parameter for direct access

        Returns:
            Dictionary with race results
        """
        from src.scrapers.utils import parse_japanese_number, parse_time, clean_text
        import re

        logger.info(f"Scraping race result for race {race_id}")

        # Construct URL using CNAME if available
        if cname:
            result_url = f"{self.BASE_URL}/JRADB/accessK.html?CNAME={cname}"
        else:
            # Fallback: try to access via standard result URL
            result_url = f"{self.RACE_RESULT_URL}?id={race_id}"
            logger.warning("CNAME not provided, using fallback URL construction")

        response = self._make_request(result_url)
        if not response:
            logger.error(f"Failed to fetch race result for {race_id}")
            return None

        soup = self._parse_html(response.text)
        if not soup:
            logger.error(f"Failed to parse race result HTML for {race_id}")
            return None

        # Initialize result structure
        result = {
            'jra_race_id': race_id,
            'results': [],
            'payouts': {}
        }

        # Extract race results from tables
        tables = soup.find_all('table')

        for table in tables:
            rows = table.find_all('tr')

            for row in rows:
                try:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) < 3:  # Skip if too few cells
                        continue

                    result_data = {}
                    cell_texts = [clean_text(cell.get_text()) for cell in cells]

                    for i, text in enumerate(cell_texts):
                        # Finish position (着順)
                        if re.match(r'^\d{1,2}$', text) and not result_data.get('finish_position'):
                            pos = parse_japanese_number(text)
                            if pos and pos <= 18:  # Max 18 horses in a race
                                result_data['finish_position'] = pos

                        # Horse number (馬番)
                        if re.match(r'^\d{1,2}$', text) and not result_data.get('horse_number'):
                            num = parse_japanese_number(text)
                            if num and num <= 18:
                                result_data['horse_number'] = num

                        # Finish time (タイム)
                        time_match = re.match(r'(\d+):(\d{2})\.(\d)', text)
                        if time_match:
                            finish_time = parse_time(text)
                            if finish_time:
                                result_data['finish_time'] = finish_time

                        # Margin (着差)
                        if any(keyword in text for keyword in ['クビ', 'アタマ', '馬身', 'ハナ', '大差']):
                            result_data['margin'] = text

                        # Final odds (確定オッズ)
                        odds_match = re.match(r'^(\d+\.?\d*)$', text)
                        if odds_match and not result_data.get('final_odds'):
                            odds = float(odds_match.group(1))
                            if 1.0 <= odds <= 999.9:
                                result_data['final_odds'] = odds

                        # Popularity (人気)
                        if re.match(r'^\d{1,2}$', text) and '人気' in str(cells[i]):
                            pop = parse_japanese_number(text)
                            if pop:
                                result_data['popularity'] = pop

                        # Running positions (コーナー通過順位)
                        # Pattern like "1-2-3-4" or "3-3-2-1"
                        positions_match = re.match(r'^(\d{1,2})-(\d{1,2})-(\d{1,2})-(\d{1,2})$', text)
                        if positions_match:
                            result_data['running_positions'] = [
                                int(positions_match.group(1)),
                                int(positions_match.group(2)),
                                int(positions_match.group(3)),
                                int(positions_match.group(4))
                            ]

                    # Look for race comment in the row
                    for cell in cells:
                        comment_text = clean_text(cell.get_text())
                        if len(comment_text) > 10 and not any(char.isdigit() for char in comment_text[:5]):
                            result_data['comment'] = comment_text

                    # Only add result if we have at least finish position and horse number
                    if result_data.get('finish_position') and result_data.get('horse_number'):
                        # Set defaults
                        result_data.setdefault('finish_time', 0.0)
                        result_data.setdefault('margin', '')
                        result_data.setdefault('final_odds', 0.0)
                        result_data.setdefault('popularity', 0)
                        result_data.setdefault('running_positions', [])
                        result_data.setdefault('comment', '')

                        result['results'].append(result_data)

                except Exception as e:
                    logger.debug(f"Failed to parse result row: {e}")
                    continue

        # Extract payouts (払戻金)
        # Look for tables or sections containing payout information
        info_text = soup.get_text()

        # Win (単勝)
        win_match = re.search(r'単勝.*?(\d+)\s+番.*?(\d{1,3}(?:,\d{3})*)\s*円', info_text)
        if win_match:
            horse_num = int(win_match.group(1))
            payout = int(win_match.group(2).replace(',', ''))
            result['payouts']['win'] = [(horse_num, payout)]

        # Place (複勝)
        place_matches = re.findall(r'複勝.*?(\d+)\s+番.*?(\d{1,3}(?:,\d{3})*)\s*円', info_text)
        if place_matches:
            result['payouts']['place'] = [(int(m[0]), int(m[1].replace(',', ''))) for m in place_matches]

        # Quinella (馬連)
        quinella_match = re.search(r'馬連.*?(\d+)\s*[-－]\s*(\d+).*?(\d{1,3}(?:,\d{3})*)\s*円', info_text)
        if quinella_match:
            horse1 = int(quinella_match.group(1))
            horse2 = int(quinella_match.group(2))
            payout = int(quinella_match.group(3).replace(',', ''))
            result['payouts']['quinella'] = [(horse1, horse2, payout)]

        if not result['results']:
            logger.warning(f"No results found for race {race_id}")

        logger.info(f"Scraped results for {len(result['results'])} horses in race {race_id}")
        return result

    def scrape_horse_profile(self, horse_id: str) -> Optional[Dict]:
        """
        Scrape detailed horse profile and history.

        Args:
            horse_id: JRA horse ID

        Returns:
            Dictionary with horse information
        """
        from src.scrapers.utils import clean_text
        import re

        logger.info(f"Scraping horse profile for {horse_id}")

        # Construct horse profile URL (this may need adjustment based on actual JRA URL structure)
        horse_url = f"{self.BASE_URL}/horse/{horse_id}/"

        response = self._make_request(horse_url)
        if not response:
            logger.error(f"Failed to fetch horse profile for {horse_id}")
            return None

        soup = self._parse_html(response.text)
        if not soup:
            logger.error(f"Failed to parse horse profile HTML for {horse_id}")
            return None

        # Initialize profile structure
        profile = {
            'jra_horse_id': horse_id,
            'name': None,
            'birth_date': None,
            'sex': None,
            'sire_name': None,
            'dam_name': None,
            'trainer_name': None,
            'past_performances': []
        }

        # Extract basic information
        info_text = soup.get_text()

        # Horse name (usually in title or h1)
        title_tag = soup.find('title')
        if title_tag:
            title_text = clean_text(title_tag.get_text())
            # Extract horse name from title (format may vary)
            name_match = re.search(r'([ぁ-んァ-ヶー一-龥a-zA-Z]+)', title_text)
            if name_match:
                profile['name'] = name_match.group(1)

        # Alternative: look for h1 tag
        if not profile['name']:
            h1_tag = soup.find('h1')
            if h1_tag:
                profile['name'] = clean_text(h1_tag.get_text())

        # Birth date (生年月日)
        birth_match = re.search(r'生年月日[：:]\s*(\d{4})年(\d{1,2})月(\d{1,2})日', info_text)
        if birth_match:
            try:
                year = int(birth_match.group(1))
                month = int(birth_match.group(2))
                day = int(birth_match.group(3))
                profile['birth_date'] = datetime(year, month, day)
            except ValueError:
                pass

        # Sex (性別)
        sex_match = re.search(r'性別[：:]\s*([牡牝セン])', info_text)
        if sex_match:
            profile['sex'] = sex_match.group(1)

        # Sire (父馬)
        sire_match = re.search(r'父[：:]\s*([ぁ-んァ-ヶー一-龥a-zA-Z]+)', info_text)
        if sire_match:
            profile['sire_name'] = sire_match.group(1)

        # Dam (母馬)
        dam_match = re.search(r'母[：:]\s*([ぁ-んァ-ヶー一-龥a-zA-Z]+)', info_text)
        if dam_match:
            profile['dam_name'] = dam_match.group(1)

        # Trainer (調教師)
        trainer_match = re.search(r'調教師[：:]\s*([ぁ-んァ-ヶー一-龥a-zA-Z\s]+)', info_text)
        if trainer_match:
            profile['trainer_name'] = clean_text(trainer_match.group(1))

        # Extract past performances from tables (simplified)
        tables = soup.find_all('table')

        for table in tables:
            rows = table.find_all('tr')

            for row in rows:
                try:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) < 4:  # Skip if too few cells
                        continue

                    # Try to identify performance rows (contains date, track, result, etc.)
                    cell_texts = [clean_text(cell.get_text()) for cell in cells]

                    # Look for date pattern
                    date_match = None
                    for text in cell_texts:
                        date_match = re.match(r'(\d{4})/(\d{1,2})/(\d{1,2})', text)
                        if date_match:
                            break

                    if date_match:
                        # This row likely contains a past performance
                        performance = {
                            'date': f"{date_match.group(1)}-{date_match.group(2):0>2}-{date_match.group(3):0>2}",
                            'track': None,
                            'finish_position': None,
                            'race_name': None
                        }

                        # Extract track name
                        for text in cell_texts:
                            for track_name in ['東京', '中山', '京都', '阪神', '中京', '札幌', '函館', '福島', '新潟', '小倉']:
                                if track_name in text:
                                    performance['track'] = track_name
                                    break

                        # Extract finish position (着順)
                        for text in cell_texts:
                            pos_match = re.match(r'^(\d{1,2})$', text)
                            if pos_match:
                                performance['finish_position'] = int(pos_match.group(1))
                                break

                        # Extract race name (usually longer text)
                        for text in cell_texts:
                            if len(text) > 5 and not re.match(r'^\d+$', text):
                                performance['race_name'] = text
                                break

                        profile['past_performances'].append(performance)

                except Exception as e:
                    logger.debug(f"Failed to parse performance row: {e}")
                    continue

        logger.info(f"Scraped profile for horse {horse_id} ({profile.get('name', 'Unknown')})")
        return profile

    def get_upcoming_races(self, days_ahead: int = 7) -> List[Dict]:
        """
        Get upcoming races for the next N days.

        Args:
            days_ahead: Number of days to look ahead

        Returns:
            List of upcoming race dictionaries
        """
        logger.info(f"Getting upcoming races for next {days_ahead} days")
        all_races = []

        today = datetime.now()
        for day_offset in range(days_ahead):
            target_date = today + timedelta(days=day_offset)
            races = self.scrape_race_calendar(target_date)
            all_races.extend(races)

        logger.info(f"Found {len(all_races)} upcoming races")
        return all_races

    def close(self):
        """Close the scraper session."""
        self.session.close()
        logger.info("JRA Scraper session closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Utility functions

def scrape_recent_results(days_back: int = 7) -> List[Dict]:
    """
    Scrape race results from the past N days.

    Args:
        days_back: Number of days to look back

    Returns:
        List of race result dictionaries
    """
    logger.info(f"Scraping results from past {days_back} days")
    results = []

    with JRAScraper() as scraper:
        today = datetime.now()
        for day_offset in range(days_back):
            target_date = today - timedelta(days=day_offset)
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
    return results


if __name__ == "__main__":
    # Test the scraper
    with JRAScraper() as scraper:
        # Test getting upcoming races
        upcoming = scraper.get_upcoming_races(days_ahead=1)
        print(f"Upcoming races: {len(upcoming)}")

        # Test scraping a specific race (placeholder)
        # result = scraper.scrape_race_result("202401010101")
        # print(f"Result: {result}")
