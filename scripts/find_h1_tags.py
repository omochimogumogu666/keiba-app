"""
Find H1 tags in debug HTML file.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bs4 import BeautifulSoup

html_file = Path(__file__).parent.parent / 'data' / 'debug_race_card.html'

with open(html_file, 'rb') as f:
    html_content = f.read()

# Try to decode as shift_jis
try:
    html_text = html_content.decode('shift_jis', errors='ignore')
except:
    html_text = html_content.decode('utf-8', errors='ignore')

soup = BeautifulSoup(html_text, 'lxml')

h1_tags = soup.find_all('h1')
print(f"Found {len(h1_tags)} H1 tags\n")

for i, h1 in enumerate(h1_tags):
    print(f"=== H1 Tag #{i} ===")
    print(f"Text: '{h1.get_text().strip()}'")
    print(f"HTML: {str(h1)[:300]}")
    print()

# Also check for elements that might contain race info
print("\n=== Looking for race info ===")
race_info_selectors = [
    ('#contents', 'Main content'),
    ('div.race_title', 'Race title'),
    ('div.data', 'Data sections'),
]

for selector, label in race_info_selectors:
    elements = soup.select(selector)
    print(f"\n{label} ({selector}): Found {len(elements)}")
    if elements and len(elements) > 0:
        text = elements[0].get_text().strip()[:200]
        print(f"  Text: {text}")
