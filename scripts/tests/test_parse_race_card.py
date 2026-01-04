"""
Test script to verify race card HTML parsing logic.
"""
import sys
from pathlib import Path
import re
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bs4 import BeautifulSoup
from src.scrapers.utils import clean_text

# Read the saved HTML file
html_file = Path(__file__).parent.parent / 'data' / 'debug_race_card.html'

with open(html_file, 'r', encoding='shift_jis', errors='ignore') as f:
    html_content = f.read()

soup = BeautifulSoup(html_content, 'html.parser')

# Initialize result
race_card = {
    'jra_race_id': '202601010106',
    'race_name': None,
    'track': None,
    'race_date': None,
    'distance': None,
    'surface': None,
    'race_class': None,
    'entries': []
}

print("=" * 60)
print("EXTRACTING RACE INFORMATION")
print("=" * 60)

# Extract from H1
h1_tag = soup.find('h1')
if h1_tag:
    h1_text = clean_text(h1_tag.get_text())
    print(f"\nH1 text: {h1_text}")

    # Extract date
    date_match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', h1_text)
    if date_match:
        year = int(date_match.group(1))
        month = int(date_match.group(2))
        day = int(date_match.group(3))
        race_card['race_date'] = datetime(year, month, day)
        print(f"Extracted date: {race_card['race_date']}")
    else:
        print("Date match failed")

    # Extract track
    track_match = re.search(r'\d+回([^0-9]+?)\d+日', h1_text)
    if track_match:
        track_name = track_match.group(1)
        race_card['track'] = track_name
        print(f"Extracted track: {track_name}")
    else:
        print("Track match failed")

    # Extract race number
    race_num_match = re.search(r'(\d+)レース', h1_text)
    if race_num_match:
        race_card['race_number'] = int(race_num_match.group(1))
        print(f"Extracted race number: {race_card['race_number']}")
    else:
        print("Race number match failed")
else:
    print("H1 tag not found!")

# Extract from race_title
race_title_div = soup.find('div', class_='race_title')
if race_title_div:
    title_text = clean_text(race_title_div.get_text())
    print(f"\nRace title text: {title_text}")

    # Extract race name
    name_text = re.sub(r'^第\d+回', '', title_text)
    if 'コース' in name_text:
        parts = name_text.split('コース')
        if len(parts) > 0:
            before_course = parts[0]
            name = re.sub(r'[芝ダート障害]・[左右内外]$', '', before_course).strip()
            race_card['race_name'] = name
            print(f"Extracted race name: {name}")
    else:
        print("'コース' not found in title")

    # Extract distance
    distance_match = re.search(r'[：:]\s*([0-9,]+)\s*[メートルm]', title_text)
    if distance_match:
        distance_str = distance_match.group(1).replace(',', '')
        race_card['distance'] = int(distance_str)
        print(f"Extracted distance: {race_card['distance']}m")
    else:
        print("Distance match failed")

    # Extract surface
    if '芝' in title_text:
        race_card['surface'] = 'turf'
    elif 'ダート' in title_text:
        race_card['surface'] = 'dirt'
    elif '障害' in title_text:
        race_card['surface'] = 'jump'
    print(f"Extracted surface: {race_card.get('surface')}")

    # Extract race class
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
    print(f"Extracted race class: {race_card.get('race_class')}")
else:
    print("\nRace title div not found!")

print("\n" + "=" * 60)
print("FINAL RACE CARD:")
print("=" * 60)
for key, value in race_card.items():
    if key != 'entries':
        print(f"{key}: {value}")
