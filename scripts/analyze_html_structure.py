"""
Analyze the structure of race card HTML to find race info fields.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bs4 import BeautifulSoup

html_file = Path(__file__).parent.parent / 'data' / 'debug_race_card.html'

with open(html_file, 'r', encoding='shift_jis', errors='ignore') as f:
    html_content = f.read()

soup = BeautifulSoup(html_content, 'html.parser')

# Print H1 tags
print("=" * 60)
print("H1 TAGS:")
print("=" * 60)
for h1 in soup.find_all('h1'):
    text = h1.get_text(strip=True)
    if text and len(text) > 3:
        print(f"{text[:200]}")
        print(f"Classes: {h1.get('class')}")
        print("-" * 40)

# Print H2 tags with race name potential
print("\n" + "=" * 60)
print("H2 TAGS:")
print("=" * 60)
for h2 in soup.find_all('h2'):
    text = h2.get_text(strip=True)
    if text and len(text) > 3:
        print(f"{text[:200]}")
        print(f"Classes: {h2.get('class')}")
        print("-" * 40)

# Look for div with race details
print("\n" + "=" * 60)
print("DIV.data TAGS:")
print("=" * 60)
for div in soup.find_all('div', class_='data'):
    text = div.get_text(strip=True)
    if text:
        print(f"{text[:300]}")
        print("-" * 40)

# Look for race title section
print("\n" + "=" * 60)
print("race_head / race_info sections:")
print("=" * 60)
for section in soup.find_all(['div', 'section'], class_=['race_head', 'race_info', 'race_title']):
    text = section.get_text(strip=True)
    if text:
        print(f"{text[:300]}")
        print(f"Classes: {section.get('class')}")
        print("-" * 40)

# Print tables with class='basic'
print("\n" + "=" * 60)
print("TABLE.basic:")
print("=" * 60)
for table in soup.find_all('table', class_='basic')[:1]:  # First one only
    # Get first few rows
    rows = table.find_all('tr')[:3]
    for i, row in enumerate(rows):
        cells = row.find_all(['th', 'td'])
        print(f"Row {i}: {[cell.get_text(strip=True)[:30] for cell in cells]}")
    print("-" * 40)

# Look for dd/dl elements (definition list - often used for race details)
print("\n" + "=" * 60)
print("DL/DT/DD elements:")
print("=" * 60)
for dl in soup.find_all('dl')[:3]:
    for dt in dl.find_all('dt'):
        dd = dt.find_next_sibling('dd')
        if dd:
            print(f"{dt.get_text(strip=True)}: {dd.get_text(strip=True)}")
    print("-" * 40)
