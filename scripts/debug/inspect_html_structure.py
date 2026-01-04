"""
Inspect HTML structure to find correct selectors.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bs4 import BeautifulSoup

# Output to file to avoid Windows console encoding issues
output_lines = []


def log(msg):
    """Log to both console and list."""
    output_lines.append(msg)
    try:
        print(msg)
    except UnicodeEncodeError:
        print("[Unicode characters]")


def inspect_race_card():
    """Inspect race card HTML."""
    log("=== RACE CARD STRUCTURE ===\n")

    html_file = Path(__file__).parent.parent / 'data' / 'debug_netkeiba_race_card.html'
    html = html_file.read_bytes().decode('euc-jp', errors='replace')
    soup = BeautifulSoup(html, 'lxml')

    # Find race name
    log("Looking for race name...")
    h1 = soup.find('h1')
    if h1:
        log(f"H1: {h1.get_text().strip()}")
        log(f"H1 classes: {h1.get('class')}")

    race_list = soup.find('div', class_='RaceList_Item02')
    if race_list:
        log(f"RaceList_Item02: {race_list.get_text().strip()[:100]}")

    # RaceData sections
    log("\n--- RaceData01 ---")
    race_data01 = soup.find('div', class_='RaceData01')
    if race_data01:
        log(f"Text: {race_data01.get_text().strip()}")
        # Check for spans
        spans = race_data01.find_all('span')
        for i, span in enumerate(spans):
            log(f"  Span {i}: class={span.get('class')}, text='{span.get_text().strip()}'")

    log("\n--- RaceData02 ---")
    race_data02 = soup.find('div', class_='RaceData02')
    if race_data02:
        log(f"Text: {race_data02.get_text().strip()}")
        # Check structure
        spans = race_data02.find_all('span')
        for i, span in enumerate(spans):
            log(f"  Span {i}: class={span.get('class')}, text='{span.get_text().strip()}'")


def inspect_race_result():
    """Inspect race result HTML."""
    log("\n\n=== RACE RESULT STRUCTURE ===\n")

    html_file = Path(__file__).parent.parent / 'data' / 'test_result.html'
    html = html_file.read_bytes().decode('euc-jp', errors='replace')
    soup = BeautifulSoup(html, 'lxml')

    # Find result table
    log("Looking for result table...")
    tables = soup.find_all('table')
    log(f"Found {len(tables)} tables")

    for i, table in enumerate(tables):
        classes = table.get('class', [])
        rows = table.find_all('tr', class_=lambda x: x and 'HorseList' in x)
        if rows:
            log(f"\nTable {i} (WITH HorseList rows):")
            log(f"  Classes: {classes}")
            log(f"  HorseList rows: {len(rows)}")

            # Check first row structure
            if rows:
                first_row = rows[0]
                tds = first_row.find_all('td')
                log(f"  First row has {len(tds)} columns")
                for j, td in enumerate(tds[:10]):
                    td_class = td.get('class', [])
                    text = td.get_text().strip()[:20]
                    log(f"    Col {j}: class={td_class}, text='{text}'")
                break


if __name__ == "__main__":
    inspect_race_card()
    inspect_race_result()

    # Save output to file
    output_file = Path(__file__).parent.parent / 'data' / 'html_structure_analysis.txt'
    output_file.write_text('\n'.join(output_lines), encoding='utf-8')
    log(f"\n=== Output saved to {output_file} ===")
