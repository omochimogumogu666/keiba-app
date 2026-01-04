"""Debug script to check netkeiba.com URL structure."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
import requests
from bs4 import BeautifulSoup

# Test URL structure
test_date = datetime(2025, 12, 28)
url = f"https://race.netkeiba.com/top/race_list.html?kaisai_date={test_date.strftime('%Y%m%d')}"

print(f"Testing URL: {url}")
print("="*80)

try:
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    response = requests.get(url, headers=headers, timeout=30)

    print(f"Status code: {response.status_code}")
    print(f"Response length: {len(response.content)} bytes")
    print(f"Encoding: {response.encoding}")

    # Decode with EUC-JP
    html_text = response.content.decode('euc-jp', errors='replace')
    soup = BeautifulSoup(html_text, 'lxml')

    # Try to find race list divs
    print("\nSearching for race list elements...")

    # Try different selectors
    selectors = [
        ('div.RaceList_DataItem', 'RaceList_DataItem divs'),
        ('div.RaceList', 'RaceList divs'),
        ('div[class*="Race"]', 'Divs with "Race" in class'),
        ('a[href*="/race/"]', 'Links to /race/'),
        ('a[href*="race_id="]', 'Links with race_id parameter'),
    ]

    for selector, description in selectors:
        elements = soup.select(selector)
        print(f"\n{description}: {len(elements)} found")
        if elements and len(elements) <= 5:
            for i, elem in enumerate(elements[:3], 1):
                print(f"  {i}. {elem.name} - {elem.get('class')} - {elem.get('href', 'N/A')[:80]}")

    # Save HTML for inspection
    output_file = 'data/debug_netkeiba_calendar.html'
    os.makedirs('data', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_text[:50000])  # First 50KB
    print(f"\nFirst 50KB of HTML saved to: {output_file}")
    print("Inspect this file to understand the HTML structure.")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
