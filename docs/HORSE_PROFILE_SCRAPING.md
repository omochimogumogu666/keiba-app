# 馬プロフィールスクレイピング機能

## 概要

JRA競馬予想アプリケーションの馬プロフィールスクレイピング機能は、JRAの公式Webサイトから馬の詳細情報、血統情報、および過去の成績を取得します。

## 機能

### スクレイピング可能な情報

**基本情報:**
- 馬名
- 生年月日
- 性別 (牡/牝/セン)

**血統情報:**
- 父馬名・ID
- 母馬名・ID

**関係者情報:**
- 調教師名・ID
- 馬主名
- 生産者名

**通算成績:**
- 総レース数
- 勝利数
- 2着数
- 3着数
- 獲得賞金総額

**過去レース成績:**
- 日付
- 競馬場
- レース名
- 距離
- 馬場 (芝/ダート)
- 着順
- タイム
- 騎手名
- 斤量
- 馬体重

## 使い方

### 1. 基本的な使用方法

```python
from src.scrapers.jra_scraper import JRAScraper

# スクレイパーを初期化
with JRAScraper(delay=3) as scraper:
    # 馬のCNAMEパラメータを使用してプロフィールを取得
    horse_cname = "pw01sde3240230405/E3"  # レースカードから取得したCNAME
    profile = scraper.scrape_horse_profile(horse_cname)

    # プロフィール情報を表示
    print(f"Horse: {profile['name']}")
    print(f"Birth Date: {profile['birth_date']}")
    print(f"Sire: {profile['sire_name']}")
    print(f"Total Races: {profile['total_races']}")
```

### 2. レースカードから馬CNAMEを取得

馬のプロフィールをスクレイピングするには、まずレースカードから馬のCNAMEパラメータを取得する必要があります。

```python
from src.scrapers.jra_scraper import JRAScraper
from datetime import datetime

with JRAScraper(delay=3) as scraper:
    # 今日のレースカレンダーを取得
    today = datetime.now()
    races = scraper.scrape_race_calendar(today)

    # 最初のレースのレースカードを取得
    if races:
        race = races[0]
        race_card = scraper.scrape_race_card(
            race['jra_race_id'],
            cname=race['cname']
        )

        # 各馬のCNAMEを取得
        for entry in race_card['entries']:
            horse_cname = entry['jra_horse_id']
            horse_name = entry['horse_name']

            # 馬のプロフィールをスクレイピング
            profile = scraper.scrape_horse_profile(horse_cname)
            print(f"Scraped profile for {horse_name}")
```

### 3. データベースへの保存

```python
from src.scrapers.jra_scraper import JRAScraper
from src.data.database import save_horse_profile_to_db
from src.web.app import create_app

# Flaskアプリケーションコンテキストを作成
app = create_app('development')

with app.app_context():
    with JRAScraper(delay=3) as scraper:
        # 馬プロフィールをスクレイピング
        horse_cname = "pw01sde3240230405/E3"
        profile = scraper.scrape_horse_profile(horse_cname)

        # データベースに保存
        horse = save_horse_profile_to_db(profile)
        print(f"Saved horse: {horse.name} (ID: {horse.id})")
```

### 4. テストスクリプトの使用

馬プロフィールスクレイピングをテストするための専用スクリプトが用意されています。

```bash
# CNAMEなしで実行 (自動的にレースカードから取得)
python scripts/test_horse_profile_scraping.py

# 特定のCNAMEでテスト
python scripts/test_horse_profile_scraping.py pw01sde3240230405/E3

# データベースに保存
python scripts/test_horse_profile_scraping.py pw01sde3240230405/E3 --save
```

### 5. デバッグ用HTMLダンプ

JRAのHTML構造を確認する必要がある場合は、デバッグスクリプトを使用できます。

```bash
# 馬CNAMEを取得
python scripts/get_horse_cname.py

# HTMLをファイルに保存して分析
python scripts/debug_horse_profile.py <cname>
```

HTMLは `data/debug/horse_profile_<cname>.html` に保存されます。

## データ構造

### プロフィール辞書の構造

```python
{
    'jra_horse_id': 'pw01sde3240230405/E3',
    'name': 'イクイノックス',
    'birth_date': datetime(2019, 3, 23),
    'sex': '牡',
    'sire_name': 'キタサンブラック',
    'sire_id': 'pw01sde...',
    'dam_name': 'シャトーブランシュ',
    'dam_id': 'pw01sde...',
    'trainer_name': '木村哲也',
    'trainer_id': 'pw04kmk...',
    'owner': '...',
    'breeder': '...',
    'total_races': 15,
    'total_wins': 12,
    'total_places': 2,
    'total_shows': 1,
    'total_earnings': 2500000000,
    'past_performances': [
        {
            'date': '2023-11-26',
            'track': '東京',
            'race_name': 'ジャパンカップ',
            'distance': 2400,
            'surface': 'turf',
            'finish_position': 1,
            'finish_time': 142.5,
            'jockey': 'ルメール',
            'weight': 57.0,
            'horse_weight': 518
        },
        # ... more performances
    ]
}
```

### データベースモデルとの対応

`save_horse_profile_to_db()` 関数は、プロフィール辞書から以下のモデルにデータを保存します:

- **Horse**: 馬の基本情報、血統、調教師
- **Trainer**: 調教師情報 (存在しない場合は作成)
- **Horse (Sire)**: 父馬情報 (存在しない場合は作成)
- **Horse (Dam)**: 母馬情報 (存在しない場合は作成)

## 重要な注意事項

### 1. CNAMEパラメータの必要性

JRAの馬プロフィールページにアクセスするには、必ずCNAMEパラメータが必要です。CNAMEは:
- レースカードのスクレイピング時に取得できます
- プログラム的に生成することはできません
- 各馬、調教師、騎手ごとに一意です

### 2. レート制限の遵守

```python
# 必ず3秒以上の遅延を設定
scraper = JRAScraper(delay=3)  # 推奨: 3秒
```

- デフォルト: 3秒
- 最小推奨値: 2秒
- JRAサーバーへの負荷を最小限に抑えるため

### 3. エンコーディング

JRAのWebサイトは `Shift_JIS` エンコーディングを使用しています。スクレイパーは自動的に処理しますが、Windowsコンソールで日本語を表示する際はUnicodeエラーに注意してください。

```python
try:
    print(f"Horse: {profile['name']}")
except UnicodeEncodeError:
    print("Horse: [Unicode characters]")
```

### 4. HTML構造の変更

JRAは予告なくWebサイトのHTML構造を変更する可能性があります。スクレイピングが失敗した場合:

1. デバッグスクリプトでHTMLを保存
2. セレクターの更新が必要か確認
3. `scrape_horse_profile()` メソッドを更新

## テスト

### ユニットテスト

```bash
# 全テストを実行
pytest tests/test_scrapers/test_horse_profile.py -v

# 特定のテストクラスのみ
pytest tests/test_scrapers/test_horse_profile.py::TestHorseProfileParsing -v

# 統合テストをスキップ
pytest tests/test_scrapers/test_horse_profile.py -m "not integration" -v
```

### 統合テスト

```bash
# 実際のJRA Webサイトを使用した統合テスト
pytest tests/test_scrapers/test_horse_profile.py -m integration -v
```

**注意**: 統合テストは実際のJRA Webサイトにアクセスするため:
- 実行に時間がかかります
- レート制限を遵守します
- ネットワーク接続が必要です

## トラブルシューティング

### 問題: CNAMEが取得できない

**原因**: レースカレンダーにレースが見つからない

**解決策**:
```python
from datetime import timedelta

# 複数日先を検索
for days in range(7):
    target_date = datetime.now() + timedelta(days=days)
    races = scraper.scrape_race_calendar(target_date)
    if races:
        break
```

### 問題: プロフィールがNoneを返す

**原因**:
- 無効なCNAME
- JRAサーバーエラー
- HTML構造の変更

**解決策**:
1. CNAMEが正しいか確認
2. デバッグスクリプトでHTMLを確認
3. ログレベルをDEBUGに設定して詳細を確認

```python
from config.logging_config import setup_logging
setup_logging(log_level='DEBUG')
```

### 問題: 過去成績が抽出されない

**原因**:
- テーブルのクラス名が変更された
- HTMLの構造が変わった

**解決策**:
1. HTMLをファイルに保存
2. 実際のテーブル構造を確認
3. セレクターを更新

## API リファレンス

### `scrape_horse_profile(horse_id, cname=None)`

馬の詳細プロフィールをスクレイピングします。

**Parameters:**
- `horse_id` (str): JRA馬ID (CNAMEパラメータ)
- `cname` (str, optional): 明示的なCNAMEパラメータ (horse_idと同じ場合は省略可)

**Returns:**
- `dict` or `None`: 馬プロフィール情報の辞書、失敗時はNone

**Example:**
```python
profile = scraper.scrape_horse_profile('pw01sde3240230405/E3')
```

### `save_horse_profile_to_db(profile)`

馬プロフィールをデータベースに保存します。

**Parameters:**
- `profile` (dict): `scrape_horse_profile()` が返すプロフィール辞書

**Returns:**
- `Horse`: 保存されたHorseモデルインスタンス

**Example:**
```python
from src.data.database import save_horse_profile_to_db

horse = save_horse_profile_to_db(profile)
print(f"Saved: {horse.name} (ID: {horse.id})")
```

## 今後の改善予定

- [ ] 馬の過去成績を別テーブルに保存
- [ ] より詳細な血統情報 (母父、祖父母など)
- [ ] 馬の獲得賞金の年次推移
- [ ] レース毎の詳細コメント
- [ ] 並列スクレイピングによる高速化
- [ ] キャッシュ機能による再スクレイピングの削減

## 関連ファイル

- **スクレイパー**: [src/scrapers/jra_scraper.py](../src/scrapers/jra_scraper.py)
- **データベース**: [src/data/database.py](../src/data/database.py)
- **モデル**: [src/data/models.py](../src/data/models.py)
- **テスト**: [tests/test_scrapers/test_horse_profile.py](../tests/test_scrapers/test_horse_profile.py)
- **テストスクリプト**: [scripts/test_horse_profile_scraping.py](../scripts/test_horse_profile_scraping.py)
- **デバッグスクリプト**: [scripts/debug_horse_profile.py](../scripts/debug_horse_profile.py)
