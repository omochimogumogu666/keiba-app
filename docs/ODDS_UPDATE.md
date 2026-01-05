# オッズ自動更新機能

レース開始15分前に最新オッズを自動取得・更新する機能のドキュメントです。

## 概要

この機能は以下のことを実現します:

1. **レース開始時刻の記録**: 各レースの発走時刻をデータベースに保存
2. **最新オッズの取得**: Netkeibaから最新のオッズ情報をスクレイピング
3. **自動更新**: レース開始15分前になったら自動的にオッズを更新
4. **データベース保存**: 更新されたオッズと更新時刻を記録

## データベース変更点

### Raceモデル

新規フィールド:
- `post_time` (Time): レース発走時刻（例: 15:25）

### RaceEntryモデル

新規フィールド:
- `latest_odds` (Float): 最新オッズ（レース開始15分前に更新）
- `odds_updated_at` (DateTime): オッズ更新時刻

## 使用方法

### 1. 手動でオッズを更新

特定のレースのオッズを更新:

```bash
python scripts/update_odds.py --race-id 202601040511
```

今日の全レースのオッズを更新:

```bash
python scripts/update_odds.py --all
```

15分以内に開始するレースを自動検出して更新:

```bash
python scripts/update_odds.py
```

### 2. 自動スケジューラーを起動

レース開始15分前に自動的にオッズを更新するスケジューラーを起動:

```bash
python scripts/odds_update_scheduler.py
```

オプション設定:

```bash
# チェック間隔を5分に設定
python scripts/odds_update_scheduler.py --check-interval 5

# レース開始20分前から更新対象にする
python scripts/odds_update_scheduler.py --minutes-before 20

# 本番環境で実行
python scripts/odds_update_scheduler.py --env production
```

### 3. テスト実行

1回だけ実行してテスト:

```bash
python scripts/odds_update_scheduler.py --once
```

## コマンドラインオプション

### update_odds.py

| オプション | 説明 | デフォルト値 |
|-----------|------|-------------|
| `--race-id` | 更新対象のレースID | - |
| `--all` | 今日の全レースを更新 | False |
| `--minutes-before` | レース開始何分前まで対象にするか | 15 |
| `--scraping-delay` | スクレイピング時のリクエスト間隔(秒) | 3 |
| `--env` | 実行環境 (development/production/testing) | development |

### odds_update_scheduler.py

| オプション | 説明 | デフォルト値 |
|-----------|------|-------------|
| `--check-interval` | チェック間隔（分） | 3 |
| `--minutes-before` | レース開始何分前から更新対象にするか | 15 |
| `--scraping-delay` | スクレイピング時のリクエスト間隔(秒) | 3 |
| `--env` | 実行環境 (development/production/testing) | development |
| `--once` | 1回だけ実行して終了（テスト用） | False |

## 実装の詳細

### 1. オッズスクレイピング

`NetkeibaScraper.scrape_latest_odds(race_id)` メソッドが最新オッズを取得します。

- Netkeibaの出馬表ページから現在のオッズを取得
- 馬番とオッズのマッピングを返す
- レート制限: 3秒間隔（設定可能）

### 2. オッズ更新ロジック

`update_odds.py` スクリプトが以下の処理を実行:

1. データベースから該当レースを取得
2. Netkeibaから最新オッズをスクレイピング
3. 各馬のRaceEntryレコードを更新:
   - `latest_odds`: 最新オッズ値
   - `odds_updated_at`: 更新時刻（UTC）

### 3. 自動スケジューラー

`odds_update_scheduler.py` が定期的にチェック:

1. 今日のレースを取得（status='upcoming'）
2. レース開始時刻と現在時刻を比較
3. 15分以内に開始するレースを検出
4. 5分以内に更新済みのレースはスキップ（重複更新防止）
5. 該当レースのオッズを更新

### 4. 発走時刻の取得

`NetkeibaScraper._parse_post_time(text)` メソッドがHTMLから発走時刻を抽出:

- RaceData01要素から「発走 15:25」のような文字列を検索
- 正規表現で時刻部分を抽出: `HH:MM`
- `save_race_to_db()` 関数が文字列をtime型に変換して保存

## 運用例

### 開発環境での起動

```bash
# 3分ごとにチェック、15分前に更新
python scripts/odds_update_scheduler.py --check-interval 3
```

### 本番環境での起動

```bash
# systemdやsupervisorでバックグラウンド実行
python scripts/odds_update_scheduler.py --env production --check-interval 3
```

### デーモン化（Linux/Mac）

systemdサービスファイル例:

```ini
[Unit]
Description=Keiba Odds Update Scheduler
After=network.target

[Service]
Type=simple
User=keiba
WorkingDirectory=/path/to/keiba-app
ExecStart=/path/to/keiba-app/venv/bin/python scripts/odds_update_scheduler.py --env production
Restart=on-failure
RestartSec=60

[Install]
WantedBy=multi-user.target
```

## ログ出力

ログは標準的なロギングシステムを使用:

```
2026-01-04 14:45:00 - INFO - オッズ自動更新スケジューラー起動
2026-01-04 14:45:00 - INFO - Checking for races at 2026-01-04 14:45:00
2026-01-04 14:45:00 - INFO - Race starting in 12.5 minutes: 有馬記念
2026-01-04 14:45:01 - INFO - Updating odds for race: 有馬記念 (15:25)
2026-01-04 14:45:05 - INFO - Successfully updated odds for 18 horses in race 202601041011
2026-01-04 14:45:05 - INFO - Updated odds for 1 races
```

## 注意事項

1. **レート制限**: Netkeibaへのリクエスト間隔は最低3秒を推奨（デフォルト設定）
2. **発走時刻の精度**: Netkeibaから取得した発走時刻は変更される可能性があります
3. **重複更新の防止**: 5分以内に更新済みのレースは再更新されません
4. **エラーハンドリング**: スクレイピング失敗時はログに記録され、次回チェック時に再試行されます
5. **タイムゾーン**: データベースにはUTCで保存、表示時にJSTに変換してください

## トラブルシューティング

### オッズが更新されない

1. レースの`post_time`が設定されているか確認:
   ```python
   race = Race.query.filter_by(netkeiba_race_id='202601040511').first()
   print(race.post_time)
   ```

2. レースの`status`が'upcoming'になっているか確認:
   ```python
   print(race.status)
   ```

3. 手動で更新を試す:
   ```bash
   python scripts/update_odds.py --race-id 202601040511
   ```

### スケジューラーが起動しない

1. データベース接続を確認
2. 環境変数が正しく設定されているか確認
3. ログファイルを確認してエラーメッセージを確認

### オッズが取得できない

1. Netkeibaのサイト構造が変更された可能性があります
2. ネットワーク接続を確認
3. レースIDが正しいか確認

## 今後の拡張案

- [ ] オッズ変動の履歴を記録（時系列データ）
- [ ] オッズ変動のグラフ表示
- [ ] 急激なオッズ変動の通知機能
- [ ] 複数回のオッズ更新（10分前、5分前など）
- [ ] オッズデータのAPI提供

## 関連ファイル

- [src/data/models.py](../src/data/models.py:96) - Raceモデルの`post_time`フィールド
- [src/data/models.py](../src/data/models.py:136-137) - RaceEntryモデルの`latest_odds`, `odds_updated_at`フィールド
- [src/scrapers/netkeiba_scraper.py](../src/scrapers/netkeiba_scraper.py:346-365) - `_parse_post_time()`メソッド
- [src/scrapers/netkeiba_scraper.py](../src/scrapers/netkeiba_scraper.py:884-956) - `scrape_latest_odds()`メソッド
- [src/data/database.py](../src/data/database.py:230-297) - `save_race_to_db()`のpost_time処理
- [scripts/update_odds.py](../scripts/update_odds.py) - オッズ更新スクリプト
- [scripts/odds_update_scheduler.py](../scripts/odds_update_scheduler.py) - 自動スケジューラー
