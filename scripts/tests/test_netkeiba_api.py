"""
Test script to explore netkeiba.com API endpoints.

このスクリプトはnetkeiba.comのURL構造とAPIエンドポイントを調査します。
- レース一覧ページへの直接アクセス
- カレンダーページの構造分析
- モバイルサイトの調査
- HTML内のrace_id抽出

The HTML shows that race data is loaded via JavaScript APIs.
"""
import requests
from datetime import datetime
import json
import re

BASE_URL = "https://race.netkeiba.com"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

def test_race_list_direct():
    """
    Try to get race list by accessing the race_list page directly
    for a specific date and track.
    """
    print("=" * 80)
    print("Testing Direct Race List Access")
    print("=" * 80)

    # Try different URL patterns
    test_urls = [
        # Calendar with date
        f"{BASE_URL}/top/race_list.html?kaisai_date=20260104",
        # Today's races
        f"{BASE_URL}/top/race_list.html",
        # Alternative pattern seen in netkeiba
        f"{BASE_URL}/race/race_list.html?kaisai_date=20260104",
    ]

    headers = {'User-Agent': USER_AGENT}

    for url in test_urls:
        print(f"\nTrying: {url}")
        try:
            response = requests.get(url, headers=headers, timeout=10)
            print(f"  Status: {response.status_code}")

            if response.status_code == 200:
                # Check if we got race data
                content = response.text
                if 'race_id=' in content:
                    print(f"  ✓ Found race_id in response!")
                    # Count occurrences
                    count = content.count('race_id=')
                    print(f"  Found {count} race_id references")

                    # Extract some race IDs
                    import re
                    race_ids = re.findall(r'race_id=(\d{12})', content)
                    if race_ids:
                        print(f"  Sample race IDs: {race_ids[:5]}")
                else:
                    print(f"  ✗ No race_id found - data likely loaded via JS")

        except requests.RequestException as e:
            print(f"  Network Error: {e}")
        except (ValueError, KeyError) as e:
            print(f"  Data Parsing Error: {e}")


def test_calendar_page_with_date_range():
    """
    Test accessing calendar page with date to see full month structure.
    """
    print("\n" + "=" * 80)
    print("Testing Calendar Page")
    print("=" * 80)

    url = f"{BASE_URL}/top/calendar.html"
    headers = {'User-Agent': USER_AGENT}

    print(f"\nFetching: {url}")

    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            content = response.text

            # Save for analysis
            with open('data/netkeiba_calendar_page.html', 'w', encoding='utf-8') as f:
                f.write(content)
            print("Saved to: data/netkeiba_calendar_page.html")

            # Check for race links
            import re
            race_links = re.findall(r'href="([^"]*race[^"]*)"', content)
            print(f"\nFound {len(race_links)} race-related links")

            # Look for date links
            date_links = re.findall(r'kaisai_date=(\d{8})', content)
            if date_links:
                print(f"\nFound kaisai_date parameters: {set(date_links)}")

    except Exception as e:
        print(f"Error: {e}")


def test_specific_track_date():
    """
    Test accessing race list for specific track and date.
    Based on netkeiba URL structure: /race/list/YYYYMMDD/
    """
    print("\n" + "=" * 80)
    print("Testing Track-Date Specific URLs")
    print("=" * 80)

    # Try URL pattern: /race/list/YYYYMMDD/
    test_dates = [
        "20260104",  # Today (might have races)
        "20260111",  # Next weekend
        "20251228",  # From debug file
    ]

    headers = {'User-Agent': USER_AGENT}

    for date in test_dates:
        url = f"{BASE_URL}/race/list/{date}/"
        print(f"\nTrying: {url}")

        try:
            response = requests.get(url, headers=headers, timeout=10)
            print(f"  Status: {response.status_code}")

            if response.status_code == 200:
                content = response.text

                # Look for race IDs
                import re
                race_ids = re.findall(r'race_id=(\d{12})', content)
                if race_ids:
                    print(f"  ✓ Found {len(race_ids)} race IDs!")
                    print(f"  Sample: {race_ids[:3]}")

                    # Save successful response
                    filename = f'data/netkeiba_list_{date}.html'
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"  Saved to: {filename}")
                else:
                    print(f"  ✗ No race IDs found")

        except requests.RequestException as e:
            print(f"  Network Error: {e}")
        except (ValueError, KeyError) as e:
            print(f"  Data Parsing Error: {e}")


def check_mobile_site():
    """
    Sometimes mobile sites have simpler structure without JavaScript.
    """
    print("\n" + "=" * 80)
    print("Testing Mobile Site")
    print("=" * 80)

    url = "https://race.sp.netkeiba.com/?pid=race_list&kaisai_date=20260104"

    headers = {
        'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15'
    }

    print(f"\nFetching: {url}")

    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            # Mobile site might use Shift-JIS or EUC-JP
            try:
                content = response.content.decode('euc-jp', errors='replace')
            except (UnicodeDecodeError, AttributeError):
                content = response.text

            # Save
            with open('data/netkeiba_mobile.html', 'w', encoding='utf-8') as f:
                f.write(content)
            print("Saved to: data/netkeiba_mobile.html")

            # Check for race IDs
            import re
            race_ids = re.findall(r'race_id=(\d{12})', content)
            if race_ids:
                print(f"✓ Found {len(race_ids)} race IDs in mobile site!")
                print(f"Sample: {race_ids[:5]}")
            else:
                print("✗ No race IDs in mobile site")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("NetKeiba API/URL Structure Test")
    print("=" * 80)
    print(f"Time: {datetime.now()}")
    print()

    # Test different approaches
    test_race_list_direct()

    import time
    time.sleep(2)

    test_calendar_page_with_date_range()

    time.sleep(2)

    test_specific_track_date()

    time.sleep(2)

    check_mobile_site()

    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)
