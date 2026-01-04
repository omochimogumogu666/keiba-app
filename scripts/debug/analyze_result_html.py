"""
Analyze race result HTML structure.
"""
from bs4 import BeautifulSoup

def analyze_html(filename):
    """Analyze race result HTML structure."""
    with open(filename, 'r', encoding='utf-8') as f:
        html = f.read()

    soup = BeautifulSoup(html, 'lxml')

    print("=" * 80)
    print(f"Analyzing: {filename}")
    print("=" * 80)

    # Check page title
    title = soup.find('title')
    if title:
        print(f"\nTitle: {title.get_text(strip=True)}")

    # Find all tables
    tables = soup.find_all('table')
    print(f"\nTotal tables found: {len(tables)}")

    # Analyze each table
    for i, table in enumerate(tables):
        print(f"\n--- Table {i+1} ---")

        # Get table classes
        table_class = table.get('class', [])
        print(f"Classes: {table_class}")

        # Count rows
        rows = table.find_all('tr')
        print(f"Rows: {len(rows)}")

        # Show first row headers
        if rows:
            first_row = rows[0]
            headers = first_row.find_all(['th', 'td'])
            if headers:
                print(f"First row cells ({len(headers)}): ", end="")
                for h in headers[:10]:  # First 10
                    text = h.get_text(strip=True)[:20]  # First 20 chars
                    print(f"[{text}]", end=" ")
                print()

        # Show second row if exists (data sample)
        if len(rows) > 1:
            second_row = rows[1]
            cells = second_row.find_all(['th', 'td'])
            if cells:
                print(f"Second row cells ({len(cells)}): ", end="")
                for c in cells[:10]:  # First 10
                    text = c.get_text(strip=True)[:20]  # First 20 chars
                    print(f"[{text}]", end=" ")
                print()

    # Look for specific keywords in divs and sections
    print("\n" + "=" * 80)
    print("Looking for result-related keywords...")
    print("=" * 80)

    keywords = ['着順', '結果', '払戻', 'オッズ', 'タイム', '人気']
    for keyword in keywords:
        elements = soup.find_all(string=lambda text: keyword in text if text else False)
        print(f"\n'{keyword}' found in {len(elements)} elements")
        if elements and len(elements) > 0:
            # Show first occurrence context
            elem = elements[0]
            parent = elem.parent
            if parent:
                print(f"  Parent tag: <{parent.name}> class={parent.get('class', [])}")
                print(f"  Text sample: {elem[:50]}")

    # Look for specific table classes that might contain results
    result_tables = soup.find_all('table', class_=lambda x: x and any(
        keyword in ' '.join(x).lower() for keyword in ['result', 'race', 'basic', 'pay']
    ))
    print(f"\n\nResult-related tables (by class): {len(result_tables)}")

    for i, table in enumerate(result_tables[:3]):  # Show first 3
        print(f"\n  Result Table {i+1}")
        print(f"  Classes: {table.get('class', [])}")
        rows = table.find_all('tr')
        print(f"  Rows: {len(rows)}")

if __name__ == "__main__":
    analyze_html("docs/debug_race_result_20260101.html")
