# レース予想自動化ガイド

このドキュメントでは、今後のレースを自動的にスクレイピングして予想を生成する方法を説明します。

## 概要

予想自動化システムは以下の機能を提供します:

1. **今後のレーススクレイピング**: 指定日のレースカレンダーと出馬表を自動取得
2. **特徴量抽出**: 過去のデータから予測に必要な特徴量を自動生成
3. **予想生成**: 訓練済みMLモデルで各レースの予想を自動生成
4. **データベース保存**: 予想結果をデータベースに自動保存
5. **定期実行**: スケジューラーによる自動実行

## 必要な準備

### 1. 依存ライブラリのインストール

```bash
pip install -r requirements.txt
```

### 2. 訓練済みモデルの準備

予想を生成するには、事前に訓練されたモデルが必要です。

```bash
# 過去データをスクレイピング (初回のみ)
python scripts/scrape_historical_data.py --start-year 2023 --end-year 2024

# モデルを訓練
python scripts/train_model.py --model xgboost --task regression
```

訓練が完了すると、`data/models/` ディレクトリにモデルファイルが保存されます。

## 使い方

### 方法1: 今すぐ予想を生成 (手動実行)

#### 基本的な使い方

```bash
# 今日のレースを予想
python scripts/predict_upcoming_races.py
```

#### オプション付き実行例

```bash
# 明日のレースを予想
python scripts/predict_upcoming_races.py --date 2026-01-05

# 今日から3日後までのレースを予想
python scripts/predict_upcoming_races.py --days-ahead 3

# 特定のモデルを使用
python scripts/predict_upcoming_races.py --model-path data/models/xgboost_regression_20260104_120000.pkl

# CSVにも出力
python scripts/predict_upcoming_races.py --output-csv predictions.csv

# スクレイピングをスキップ (既にDBにデータがある場合)
python scripts/predict_upcoming_races.py --skip-scraping
```

#### コマンドラインオプション

| オプション | 説明 | デフォルト |
|----------|------|-----------|
| `--date` | レース日付 (YYYY-MM-DD) | 今日 |
| `--days-ahead` | 今日から何日後まで処理するか | 0 (今日のみ) |
| `--model-path` | 使用するモデルのパス | 最新のモデルを自動選択 |
| `--model-type` | モデルタイプ (xgboost/random_forest) | xgboost |
| `--output-csv` | CSV出力先 | なし (DBのみ) |
| `--skip-scraping` | スクレイピングをスキップ | False |
| `--scraping-delay` | リクエスト間隔(秒) | 3 |

### 方法2: 定期的に自動実行 (スケジューラー)

#### 即座に1回実行

```bash
python scripts/scheduler.py --schedule once
```

#### 毎日8時に自動実行

```bash
python scripts/scheduler.py --schedule daily --time 08:00
```

#### 毎週金曜日20時に自動実行

```bash
python scripts/scheduler.py --schedule weekly --day friday --time 20:00
```

#### スケジューラーのオプション

| オプション | 説明 | デフォルト |
|----------|------|-----------|
| `--schedule` | スケジュール頻度 (daily/weekly/once) | daily |
| `--time` | 実行時刻 (HH:MM) | 08:00 |
| `--day` | 週次実行の曜日 | - |
| `--days-ahead` | 何日後までのレースを処理するか | 1 |
| `--model-type` | モデルタイプ | xgboost |
| `--output-dir` | CSV出力先ディレクトリ | data/predictions |
| `--scraping-delay` | リクエスト間隔(秒) | 3 |

### 方法3: Windowsタスクスケジューラで自動実行

#### 1. バッチファイルを作成

`run_daily_predictions.bat` を作成:

```batch
@echo off
cd /d C:\Users\h01it\keiba-app
call venv\Scripts\activate
python scripts/predict_upcoming_races.py --days-ahead 1 --output-csv data/predictions/daily_predictions.csv
pause
```

#### 2. Windowsタスクスケジューラに登録

1. タスクスケジューラを起動 (`taskschd.msc`)
2. 「基本タスクの作成」をクリック
3. 名前: `競馬予想自動実行`
4. トリガー: `毎日` → 時刻を指定 (例: 08:00)
5. 操作: `プログラムの開始`
6. プログラム: 作成したバッチファイルのパス
7. 完了

### 方法4: Linux/Mac の cron で自動実行

#### crontabに追加

```bash
# crontabを編集
crontab -e

# 毎日8時に実行
0 8 * * * cd /path/to/keiba-app && /path/to/venv/bin/python scripts/predict_upcoming_races.py --days-ahead 1
```

## 出力結果

### データベース

予想結果は `predictions` テーブルに保存されます:

```sql
SELECT
    p.race_id,
    r.race_name,
    r.race_date,
    h.name as horse_name,
    p.predicted_position,
    p.win_probability,
    p.confidence_score
FROM predictions p
JOIN races r ON p.race_id = r.id
JOIN horses h ON p.horse_id = h.id
WHERE r.race_date >= CURRENT_DATE
ORDER BY r.race_date, p.race_id, p.predicted_position;
```

### CSV出力

`--output-csv` オプションを使用すると、予想結果がCSVファイルに保存されます:

```csv
race_id,horse_id,predicted_position,win_probability,confidence_score
123,456,1,0.35,0.35
123,789,2,0.28,0.28
...
```

## ワークフロー例

### 週末のレース予想を自動化

```bash
# 金曜日の夜に週末のレースをスクレイピング & 予想
python scripts/predict_upcoming_races.py --days-ahead 2 --output-csv weekend_predictions.csv
```

### 毎日の運用

1. **毎朝8時**: 今日と明日のレースを自動予想
   ```bash
   python scripts/scheduler.py --schedule daily --time 08:00 --days-ahead 1
   ```

2. **予想結果の確認**: Webインターフェースまたはデータベースで確認

3. **レース後**: 結果をスクレイピングしてモデル評価
   ```bash
   python scripts/scrape_and_save_results.py --date 2026-01-04
   ```

4. **定期的にモデル再訓練** (月1回など):
   ```bash
   python scripts/train_model.py --model xgboost --task regression
   ```

## トラブルシューティング

### モデルが見つからない

```
FileNotFoundError: xgboostモデルが見つかりません
```

**解決策**: モデルを訓練してください
```bash
python scripts/train_model.py --model xgboost
```

### レースが見つからない

```
WARNING: 2026-01-04のレースが見つかりませんでした
```

**原因**:
- 該当日にレースが開催されていない
- netkeibaでレース情報がまだ公開されていない

**解決策**: 日付を確認するか、レース情報が公開されるまで待つ

### スクレイピングエラー

```
ERROR: レース 202601010101 の処理中にエラー: HTTP 403
```

**原因**: アクセス制限、またはサイト構造の変更

**解決策**:
- `--scraping-delay` を増やす (例: `--scraping-delay 5`)
- しばらく時間を置いてから再実行

### データベース接続エラー

```
ERROR: (sqlite3.OperationalError) database is locked
```

**解決策**: 他のプロセスがデータベースを使用していないか確認

## ベストプラクティス

### 1. 適切なスクレイピング間隔

- デフォルトの3秒間隔を維持
- 負荷をかけすぎないよう注意

### 2. モデルの定期的な更新

- 月1回程度、最新データでモデルを再訓練
- 性能が低下していないか評価

### 3. ログの確認

- `logs/` ディレクトリのログファイルを定期的に確認
- エラーがないかチェック

### 4. バックアップ

- データベースを定期的にバックアップ
- 訓練済みモデルもバックアップ

## 次のステップ

1. **Webインターフェースで予想を表示**: `run.py` で起動
2. **予想精度の評価**: レース後に実際の結果と比較
3. **モデルの改善**: 新しい特徴量の追加、ハイパーパラメータ調整
4. **通知機能の追加**: 予想完了時にメールやSlackで通知

## 参考

- [train_model.py](../scripts/train_model.py) - モデル訓練スクリプト
- [generate_predictions.py](../scripts/generate_predictions.py) - 予想生成スクリプト
- [feature_engineering.py](../src/ml/feature_engineering.py) - 特徴量エンジニアリング
- [CLAUDE.md](../CLAUDE.md) - プロジェクト全体のドキュメント
