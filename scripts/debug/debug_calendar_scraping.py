"""Debug script to check JRA calendar page structure."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from bs4 import BeautifulSoup
from datetime import datetime
from config.settings import Config

# Make request to JRA calendar page
url = "https://www.jra.go.jp/keiba/thisweek/"
print(f"Fetching: {url}")

response = requests.get(url, headers={'User-Agent': Config.USER_AGENT})
print(f"Status code: {response.status_code}")
print(f"Content-Type: {response.headers.get('Content-Type', 'Unknown')}")
print(f"Encoding detected: {response.encoding}")

# Decode as Shift_JIS
html_text = response.content.decode('shift_jis', errors='replace')
print(f"\nFirst 500 characters of HTML:")
print(html_text[:500])

# Save to file for inspection
output_file = "data/debug_calendar.html"
os.makedirs("data", exist_ok=True)
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(html_text)
print(f"\nFull HTML saved to: {output_file}")

# Parse with BeautifulSoup
soup = BeautifulSoup(html_text, 'lxml')

# Look for links
print(f"\nSearching for JRADB links...")
jradb_links = soup.find_all('a', href=lambda x: x and 'JRADB' in x)
print(f"Found {len(jradb_links)} JRADB links")

for i, link in enumerate(jradb_links[:10]):
    href = link.get('href', '')
    text = link.get_text(strip=True)
    print(f"{i+1}. {text[:50]} -> {href[:100]}")

# Look for CNAME parameters
print(f"\nSearching for CNAME parameters...")
cname_links = soup.find_all('a', href=lambda x: x and 'CNAME=' in x)
print(f"Found {len(cname_links)} links with CNAME")

for i, link in enumerate(cname_links[:10]):
    href = link.get('href', '')
    text = link.get_text(strip=True)
    print(f"{i+1}. {text[:50]} -> {href[:100]}")

# Check for race schedule structure
print(f"\nLooking for race schedule elements...")
print(f"Tables: {len(soup.find_all('table'))}")
print(f"Divs with class containing 'race': {len(soup.find_all('div', class_=lambda x: x and 'race' in str(x).lower()))}")
print(f"Divs with class containing 'schedule': {len(soup.find_all('div', class_=lambda x: x and 'schedule' in str(x).lower()))}")
