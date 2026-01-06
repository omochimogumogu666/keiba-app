# クイックスタートガイド

今後のレース予想を自動化するための最小限のステップを説明します。

## 前提条件

- Python 3.9以上がインストールされていること
- インターネット接続があること

## セットアップ（初回のみ）

### 1. プロジェクトのセットアップ

**クイックスタート（Docker、推奨）:**
```bash
# Mac/Linux
./quickstart.sh

# Windows
quickstart.bat
```

**手動セットアップ:**
```bash
# 仮想環境を作成
python -m venv venv

# 仮想環境をアクティベート
source venv/bin/activate    # Mac/Linux
# venv\Scripts\activate     # Windows

# 依存関係をインストール
pip install -r requirements.txt

# データベースを初期化
python scripts/init_db.py
```

### 2. 過去データの収集とモデル訓練

```bash
# 過去1年分のデータをスクレイピング（30分〜1時間程度）
python scripts/scrape_historical_data.py --years 1

# モデルを訓練（5〜10分程度）
python scripts/train_model.py --model xgboost --task regression
```

これで準備完了です！

## 日常的な使い方

### 今日のレースを予想する

```bash
# 今日のレースを自動スクレイピング & 予想
python scripts/predict_upcoming_races.py
```

実行すると:
1. 今日のレースカレンダーをスクレイピング
2. 各レースの出馬表をスクレイピング
3. データベースに保存
4. 最新の訓練済みモデルで予想を生成
5. 予想結果をデータベースに保存

### 週末のレースを予想する（金曜日の夜に実行）

```bash
# 今日から2日後（日曜日）までのレースを予想
python scripts/predict_upcoming_races.py --days-ahead 2 --output-csv weekend_predictions.csv
```

### 予想結果を確認する

```bash
# Webインターフェースを起動
python run.py
```

ブラウザで `http://localhost:5000` にアクセスして予想結果を確認できます。

## 自動実行の設定（オプション）

### 毎日自動的に予想を生成する

```bash
# 毎日朝8時に自動実行
python scripts/scheduler.py --schedule daily --time 08:00
```

このコマンドを実行すると、スケジューラーがバックグラウンドで待機し、毎日8時に自動的に:
1. 今後1日分のレースをスクレイピング
2. 予想を生成
3. 結果をデータベースとCSVに保存

### Mac/Linuxで常時起動させる（cron）

```bash
# crontab編集
crontab -e

# 毎日朝8時に実行（以下の行を追加）
0 8 * * * cd /path/to/keiba-app && source venv/bin/activate && python scripts/predict_upcoming_races.py
```

### Windowsで常時起動させる（タスクスケジューラ）

タスクスケジューラーに登録することで、Windowsの起動時に自動的にスケジューラーを開始できます。

1. `run_scheduler.bat` を作成:
```batch
@echo off
cd /d C:\path\to\keiba-app
call venv\Scripts\activate
python scripts/scheduler.py --schedule daily --time 08:00
```

2. Windowsタスクスケジューラーに登録:
   - タスクスケジューラを開く (`taskschd.msc`)
   - 「基本タスクの作成」
   - トリガー: 「ログオン時」
   - 操作: `run_scheduler.bat` のパスを指定

## トラブルシューティング

### Q: モデルが見つからないエラーが出る

```
FileNotFoundError: xgboostモデルが見つかりません
```

**A:** モデルを訓練してください:
```bash
python scripts/train_model.py --model xgboost --task regression
```

### Q: レースが見つからない

```
WARNING: 2026-01-04のレースが見つかりませんでした
```

**A:** 該当日にレースが開催されていない、またはまだ情報が公開されていません。開催日を確認してください。

### Q: スクレイピングが遅い

**A:** これは正常です。サーバーに負荷をかけないため、各リクエスト間に3秒の遅延を設けています。`--scraping-delay` で調整可能ですが、倫理的な理由から3秒以上を推奨します。

## 次のステップ

1. **定期的にモデルを更新**: 月1回程度、新しいデータでモデルを再訓練することをおすすめします
   ```bash
   python scripts/train_model.py --model xgboost --task regression
   ```

2. **予想精度を確認**: レース後に結果をスクレイピングして、予想の精度を評価できます
   ```bash
   python scripts/scrape_and_save_results.py --date 2026-01-04
   ```

3. **詳細な設定**: より詳しい使い方は [PREDICTION_AUTOMATION.md](docs/PREDICTION_AUTOMATION.md) を参照してください

## 免責事項

- このアプリケーションの予想は参考情報です
- 予想の正確性を保証するものではありません
- 実際の馬券購入は自己責任で行ってください
- ギャンブル依存症にご注意ください
