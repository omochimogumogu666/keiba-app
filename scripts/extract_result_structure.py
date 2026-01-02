"""
Extract and display race result structure with proper encoding.
"""
from bs4 import BeautifulSoup
import re

def extract_structure(filename):
    """Extract race result structure."""
    with open(filename, 'r', encoding='utf-8') as f:
        html = f.read()

    soup = BeautifulSoup(html, 'lxml')

    print("=" * 80)
    print("RACE RESULT PAGE STRUCTURE ANALYSIS")
    print("=" * 80)

    # Check if this is a result page or race card page
    page_text = soup.get_text()

    is_result = False
    result_keywords = ['着順', '払戻', '確定オッズ', '勝馬']
    for keyword in result_keywords:
        if keyword in page_text:
            is_result = True
            print(f"\nPage type: RESULT PAGE (found keyword: {keyword})")
            break

    if not is_result:
        print("\nPage type: RACE CARD / ENTRY PAGE (not a result page)")

    # Extract race title
    h1_tags = soup.find_all('h1')
    for h1 in h1_tags:
        text = h1.get_text(strip=True)
        if 'レース' in text or '回' in text:
            try:
                print(f"\nRace info: {text}")
            except UnicodeEncodeError:
                print("\nRace info: [Japanese text - cannot display in console]")
            break

    # Find main table with results
    main_table = soup.find('table', class_='basic')
    if main_table:
        print("\n" + "=" * 80)
        print("MAIN TABLE STRUCTURE")
        print("=" * 80)

        # Get header row
        thead = main_table.find('thead')
        if thead:
            headers = thead.find_all(['th', 'td'])
            print(f"\nHeaders ({len(headers)}):")
            for i, h in enumerate(headers):
                try:
                    print(f"  [{i}] {h.get_text(strip=True)}")
                except UnicodeEncodeError:
                    print(f"  [{i}] [Japanese text]")

        # Get first data row
        tbody = main_table.find('tbody')
        if tbody:
            rows = tbody.find_all('tr')
            print(f"\nTotal data rows: {len(rows)}")

            if rows:
                print("\nFirst row structure:")
                first_row = rows[0]
                cells = first_row.find_all(['td', 'th'])

                for i, cell in enumerate(cells):
                    cell_class = cell.get('class', [])
                    try:
                        cell_text = cell.get_text(strip=True)[:100]
                        print(f"  Cell[{i}] class={cell_class}")
                        print(f"    Text: {cell_text}")
                    except UnicodeEncodeError:
                        print(f"  Cell[{i}] class={cell_class}")
                        print(f"    Text: [Japanese text]")

                    # Look for links
                    links = cell.find_all('a')
                    if links:
                        for link in links:
                            href = link.get('href', '')
                            onclick = link.get('onclick', '')
                            if href:
                                print(f"      Link href: {href[:100]}")
                            if onclick:
                                print(f"      Link onclick: {onclick[:100]}")

    # Look for payout information
    print("\n" + "=" * 80)
    print("PAYOUT INFORMATION")
    print("=" * 80)

    payout_keywords = ['単勝', '複勝', '馬連', '馬単', 'ワイド', '三連複', '三連単']
    for keyword in payout_keywords:
        if keyword in page_text:
            try:
                print(f"\nFound: {keyword}")
            except UnicodeEncodeError:
                print(f"\nFound: [payout type]")

            # Find tables or divs containing this keyword
            for table in soup.find_all('table'):
                if keyword in table.get_text():
                    rows = table.find_all('tr')
                    print(f"  In table with {len(rows)} rows")
                    break

    # Save a clean sample to text file for inspection
    output_file = filename.replace('.html', '_structure.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("RACE RESULT PAGE ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        # Race title
        for h1 in h1_tags:
            text = h1.get_text(strip=True)
            if 'レース' in text or '回' in text:
                f.write(f"Race Title: {text}\n\n")
                break

        # Main table
        if main_table:
            f.write("MAIN TABLE\n")
            f.write("-" * 80 + "\n")

            # Headers
            if thead:
                headers = thead.find_all(['th', 'td'])
                f.write("Headers:\n")
                for i, h in enumerate(headers):
                    f.write(f"  [{i}] {h.get_text(strip=True)}\n")
                f.write("\n")

            # First 3 rows
            if tbody:
                rows = tbody.find_all('tr')[:3]
                for row_num, row in enumerate(rows):
                    f.write(f"\nRow {row_num + 1}:\n")
                    cells = row.find_all(['td', 'th'])
                    for i, cell in enumerate(cells):
                        cell_class = cell.get('class', [])
                        cell_text = cell.get_text(strip=True)
                        f.write(f"  Cell[{i}] class={cell_class}\n")
                        f.write(f"    {cell_text}\n")

    print(f"\n\nDetailed structure saved to: {output_file}")

if __name__ == "__main__":
    extract_structure("docs/debug_race_result_20260101.html")
