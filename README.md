# JRA競馬予想アプリケーション

JRA競馬データ（netkeiba.com経由）をスクレイピングし、機械学習モデルで予想を行うWebアプリケーション。

## 概要

このアプリケーションは以下の機能を提供します:
- ✅ netkeiba.comからのレースデータの自動収集（レースカレンダー、出馬表、レース結果）
- ✅ 機械学習モデルによる勝馬予想（RandomForest、XGBoost）
- ✅ Webインターフェースでの予想結果表示と統計情報
- ✅ RESTful APIによるデータアクセス
- ✅ パフォーマンス最適化とキャッシング機能
- ✅ UTF-8データベースによる完全な日本語サポート

## 技術スタック

- **バックエンド**: Python 3.9+, Flask, Flask-SQLAlchemy
- **機械学習**: scikit-learn, XGBoost, imbalanced-learn
- **データ処理**: pandas, numpy
- **スクレイピング**: BeautifulSoup4, Requests, cloudscraper
- **データベース**: PostgreSQL (本番), SQLite (開発)
- **フロントエンド**: Jinja2, Bootstrap 5
- **キャッシング**: Flask-Caching (SimpleCache/Redis)

## セットアップ

### オプション1: Docker環境（推奨）

Docker環境を使用すると、PostgreSQLデータベースを含む完全な環境を簡単にセットアップできます。

```bash
# Docker Composeでサービスを起動
docker compose up -d

# データベースマイグレーションを実行
docker compose exec web python scripts/init_docker_db.py

# ブラウザでアクセス
# http://localhost:5000
```

詳細は[Docker環境セットアップガイド](docs/DOCKER_SETUP.md)を参照してください。

### オプション2: ローカル環境

#### 1. 仮想環境の作成

```bash
python -m venv venv
```

#### 2. 仮想環境のアクティベート

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

#### 3. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 4. 環境変数の設定

`.env.example`を`.env`にコピーして、必要な環境変数を設定します。

```bash
cp .env.example .env
```

### 5. データベースの初期化

```bash
python scripts/init_db.py
```

**重要**: データベースはUTF-8エンコーディングで初期化されます。SQLiteの場合、デフォルトでUTF-8が使用されます。PostgreSQLを使用する場合は、データベース作成時に `ENCODING='UTF8'` を指定してください。

## 使用方法

### データのスクレイピング

**データソース**: このアプリケーションは[netkeiba.com](https://netkeiba.com)をデータソースとして使用します。

**今日のレースデータをスクレイピング:**
```bash
python scripts/scrape_data.py
```

**レース結果のスクレイピング（完全なワークフロー）:**
```bash
# 昨日のレース結果をスクレイピング（デフォルト）
python scripts/scrape_and_save_results.py

# 特定の日付のレース結果をスクレイピング
python scripts/scrape_and_save_results.py 2025-12-28
```

**過去の履歴データの一括スクレイピング（最適化版）:**

**⚡ スクレイピングの高速化設定**

デフォルトでは、以下の条件で絞り込み、スクレイピング時間を大幅に短縮します:
- **対象競馬場**: 東京、中山、阪神、京都、中京（主要5競馬場のみ）
- **対象レースクラス**: 2勝クラス以上（G1/G2/G3/OP/Listed/3勝/2勝）
  - 除外: 1勝クラス、未勝利、新馬

```bash
# 過去5年分のデータをスクレイピング（デフォルト、週末のみ、最適化版）
python scripts/scrape_historical_data.py

# 過去3年分のデータをスクレイピング
python scripts/scrape_historical_data.py --years 3

# 特定の期間のデータをスクレイピング
python scripts/scrape_historical_data.py --start-date 2021-01-01 --end-date 2023-12-31

# 週末以外も含めてすべての日をスクレイピング
python scripts/scrape_historical_data.py --years 5 --all-days
```

**スクレイピング効率の目安:**
- **従来版**: 全レース対象（週末約100レース/日、推定5分/日）→ 約5時間/年
- **最適化版**: 主要5場2勝以上（週末約30-40レース/日、推定2分/日）→ 約2時間/年
- **削減効果**: 約60%の時間短縮

**重要な注意事項:**
- netkeiba.comはEUC-JPエンコーディングを使用しますが、データベースにはUTF-8で保存されます
- リクエスト間に3秒の遅延を設けているため、大量データの取得には時間がかかります
- JRAレースは主に週末に開催されるため、デフォルトでは土日のみスクレイピングします
- 進捗は随時データベースに保存されるため、中断しても再開可能です
- スクレイピングはnetkeiba.comの利用規約を遵守してください
- 予想精度を重視する場合、主要競馬場・上級クラスのレースに絞ることが推奨されます

### 特徴量エンジニアリング

機械学習モデルの訓練前に、データベースから特徴量を抽出して前処理します。

**データベースから特徴量を抽出:**
```bash
python scripts/extract_features.py
```

**抽出される特徴量（40種類以上）:**
- 馬の過去成績統計（勝率、複勝率、平均着順）
- 距離別・馬場別・コース別の成績
- 騎手・調教師の統計情報
- レース固有の特徴（距離、馬場状態、天候、クラス）
- 最近の成績トレンド

**出力ファイル（`data/processed/`に保存）:**
- `features_raw.csv` - 生の特徴量データ
- `features_processed.csv` - スケーリング・正規化済みの特徴量
- `labels_regression.csv` - 回帰用ラベル（着順の予測）
- `labels_binary.csv` - 二値分類用ラベル（勝敗の予測）
- `labels_multiclass.csv` - 多クラス分類用ラベル（勝/複/他の予測）

すべてのCSVファイルはUTF-8エンコーディング（BOM付き）で保存され、Excelでも正しく開けます。

### モデルの訓練

機械学習モデルを訓練して、レース結果を予測します。

**RandomForestモデルの訓練（回帰タスク - 着順予測）:**
```bash
# データベースから直接訓練（推奨）
python scripts/train_model.py --model random_forest --task regression

# CSVファイルから訓練
python scripts/train_model.py --model random_forest --task regression --data-source csv
```

**XGBoostモデルの訓練（分類タスク - 勝敗予測）:**
```bash
python scripts/train_model.py --model xgboost --task classification
```

**両方のモデルを訓練:**
```bash
python scripts/train_model.py --model both --task regression
```

**訓練オプション:**
- `--model`: モデルタイプ（`random_forest`, `xgboost`, `both`）
- `--task`: タスクタイプ（`regression` - 着順予測、`classification` - 勝敗予測）
- `--data-source`: データソース（`database` - DB直接、`csv` - CSVファイル）
- `--output-dir`: モデル保存先ディレクトリ（デフォルト: `data/models/`）

訓練済みモデルは `data/models/` にタイムスタンプ付きで保存されます（例: `random_forest_regression_20260104_120000.pkl`）。

### モデルの定期再訓練

機械学習モデルを定期的に再訓練し、最新のデータで予想精度を維持・向上させます。

**今すぐ再訓練を実行:**
```bash
python scripts/model_retraining_scheduler.py --mode once
```

**定期再訓練のスケジュール設定:**
```bash
# 毎週月曜2時に自動再訓練（推奨）
python scripts/model_retraining_scheduler.py --mode schedule --interval weekly --time 02:00

# 毎日2時に自動再訓練
python scripts/model_retraining_scheduler.py --mode schedule --interval daily --time 02:00

# 毎月1日2時に自動再訓練
python scripts/model_retraining_scheduler.py --mode schedule --interval monthly --time 02:00
```

**再訓練の主な機能:**
- **自動データ取得**: データベースから最新のデータを自動抽出
- **モデルバージョニング**: 訓練済みモデルを自動的にバージョン管理
- **性能比較**: 新モデルと前回モデルのパフォーマンスを自動比較
- **自動デプロイ**: 性能が改善した場合のみ新モデルをアクティブ化（デフォルト: 5%以上改善）
- **モデル履歴管理**: 最新5バージョンを保持、古いモデルは自動削除
- **通知機能**: Email/Slack通知対応（オプション）

**設定ファイル（`config/retraining_config.json`）:**
```json
{
  "models_to_train": ["random_forest", "xgboost"],
  "task": "regression",
  "training_window_days": 365,
  "validation_split": 0.15,
  "test_split": 0.15,
  "min_samples": 1000,
  "performance_threshold": 0.05,
  "keep_last_n_models": 5,
  "models_dir": "data/models",
  "notification": {
    "enabled": false,
    "email_recipients": ["your@email.com"],
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "smtp_user": "your@email.com",
    "smtp_password": "your-password"
  }
}
```

**モデルレジストリ:**
再訓練されたすべてのモデルは `data/models/model_registry.json` に記録されます:
- モデルタイプと訓練日時
- パフォーマンス指標（RMSE, R2など）
- 前回モデルとの比較結果
- アクティブ/非アクティブステータス

### 予想の生成

#### 方法1: 自動スクレイピング + 予想生成（推奨）

今後のレースを自動的にスクレイピングして予想を生成します。

**今日のレースを予想:**
```bash
python scripts/predict_upcoming_races.py
```

**今日から3日後までのレースを予想:**
```bash
python scripts/predict_upcoming_races.py --days-ahead 3 --output-csv predictions.csv
```

**特定の日付のレースを予想:**
```bash
python scripts/predict_upcoming_races.py --date 2026-01-11
```

**定期的に自動実行（スケジューラー）:**
```bash
# 毎日8時に自動実行
python scripts/scheduler.py --schedule daily --time 08:00

# 毎週金曜日20時に自動実行
python scripts/scheduler.py --schedule weekly --day friday --time 20:00

# 即座に1回実行
python scripts/scheduler.py --schedule once
```

詳細な使い方は [PREDICTION_AUTOMATION.md](docs/PREDICTION_AUTOMATION.md) を参照してください。

#### 方法2: 手動で既存レースの予想を生成

**今日のレースを予想:**
```bash
python scripts/generate_predictions.py \
  --model-path data/models/random_forest_regression_20260104_120000.pkl \
  --save-to-db
```

**特定の日付のレースを予想:**
```bash
python scripts/generate_predictions.py \
  --model-path data/models/xgboost_regression_20260104_120000.pkl \
  --race-date 2026-01-11 \
  --save-to-db
```

**特定のレースを予想:**
```bash
python scripts/generate_predictions.py \
  --model-path data/models/random_forest_regression_20260104_120000.pkl \
  --race-id 123 \
  --save-to-db
```

**CSV形式で出力:**
```bash
python scripts/generate_predictions.py \
  --model-path data/models/random_forest_regression_20260104_120000.pkl \
  --output-csv predictions.csv
```

### 馬券シミュレーション

機械学習モデルの予測を使って、過去のレース結果で馬券購入をシミュレーションできます。

#### Webインターフェース

ブラウザで `http://localhost:5000/simulation/` にアクセスして、Webインターフェースでシミュレーションを実行できます。

**主な機能:**
- 期間指定によるシミュレーション（開始日〜終了日）
- 複数の馬券種に対応（単勝、複勝、馬連、馬単、ワイド、3連複、3連単）
- 購入戦略の設定（購入金額、予想確率上位N点、最小購入確率など）
- 結果表示
  - 回収率、的中率、総損益
  - 時系列グラフ（累積収支、回収率推移）
  - 馬券種別集計
  - 購入馬券一覧
- シミュレーション履歴の保存・閲覧

#### CLIスクリプト

コマンドラインからシミュレーションを実行:

**基本的な使い方:**
```bash
# 2024年1月1日〜12月31日のシミュレーション（単勝・複勝）
python scripts/run_betting_simulation.py \
  --start-date 2024-01-01 \
  --end-date 2024-12-31
```

**全馬券種でシミュレーション:**
```bash
python scripts/run_betting_simulation.py \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --bet-types win place quinella exacta wide trio trifecta \
  --bet-amount 100 \
  --top-n 5 \
  --name "2024年全馬券種シミュレーション"
```

**予測を使わないシミュレーション（均等確率）:**
```bash
python scripts/run_betting_simulation.py \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --no-predictions
```

**オプション:**
- `--start-date`: 開始日 (YYYY-MM-DD、必須)
- `--end-date`: 終了日 (YYYY-MM-DD、必須)
- `--name`: シミュレーション名（デフォルト: 自動生成）
- `--bet-types`: 馬券種のリスト（デフォルト: win place）
  - 選択肢: `win`（単勝）, `place`（複勝）, `quinella`（馬連）, `exacta`（馬単）, `wide`（ワイド）, `trio`（3連複）, `trifecta`（3連単）
- `--bet-amount`: 1点あたりの購入金額（デフォルト: 100円）
- `--top-n`: 予想確率上位N点を購入（デフォルト: 3）
- `--min-probability`: 最小購入確率閾値（デフォルト: 0.05）
- `--max-bets-per-race`: 1レースあたりの最大購入点数（デフォルト: 10）
- `--no-predictions`: 予測データを使わず均等確率でシミュレーション
- `--no-save`: 結果をデータベースに保存しない

**結果の確認:**

CLIで結果が表示され、データベースに保存された場合はWebインターフェース (`http://localhost:5000/simulation/history`) で詳細を確認できます。

### Webアプリケーションの起動

**開発サーバーの起動:**
```bash
python run.py
```

または

```bash
python -m src.web.app
```

ブラウザで `http://localhost:5000` にアクセス。

## プロジェクト構造

```
keiba-app/
├── config/                         # 設定ファイル
│   ├── settings.py                # 環境別設定（Development, Production, Testing）
│   ├── logging_config.py          # ロギング設定
│   └── retraining_config.json    # モデル再訓練設定
├── src/
│   ├── scrapers/                  # スクレイピングモジュール
│   │   ├── netkeiba_scraper.py   # Netkeibaスクレイパー（メイン）
│   │   └── utils.py              # スクレイピング共通ユーティリティ
│   ├── data/                      # データモデルと前処理
│   │   ├── models.py             # SQLAlchemyモデル（UTF-8対応）
│   │   └── database.py           # データベース操作（get-or-create）
│   ├── ml/                        # 機械学習モジュール
│   │   ├── feature_engineering.py # 特徴量エンジニアリング
│   │   ├── preprocessing.py       # データ前処理
│   │   ├── evaluation.py          # モデル評価
│   │   ├── betting_simulator.py   # 馬券シミュレーター
│   │   └── models/               # MLモデル実装
│   │       ├── base_model.py     # ベースモデルクラス
│   │       ├── random_forest.py  # RandomForestモデル
│   │       └── xgboost_model.py  # XGBoostモデル
│   ├── web/                       # Webアプリケーション
│   │   ├── app.py                # Flaskアプリケーションファクトリ
│   │   ├── cache.py              # キャッシング設定
│   │   ├── routes/               # ルーティング
│   │   │   ├── main.py          # メインページルート
│   │   │   ├── api.py           # REST APIエンドポイント
│   │   │   ├── predictions.py   # 予想表示ルート
│   │   │   ├── entities.py      # エンティティページルート
│   │   │   ├── search.py        # 検索機能ルート
│   │   │   └── simulation.py    # 馬券シミュレーションルート
│   │   └── templates/            # Jinja2テンプレート
│   │       └── simulation/       # シミュレーションテンプレート
│   └── utils/                     # ユーティリティ
│       ├── logger.py             # アプリケーションロガー
│       └── notification.py       # 通知機能（Email/Slack）
├── scripts/                       # 実行スクリプト
│   ├── init_db.py                # データベース初期化
│   ├── scrape_data.py            # レースデータスクレイピング
│   ├── scrape_and_save_results.py # レース結果スクレイピング
│   ├── scrape_historical_data.py  # 過去データ一括スクレイピング
│   ├── extract_features.py       # 特徴量抽出・前処理
│   ├── train_model.py            # モデル訓練
│   ├── model_retraining_scheduler.py # モデル定期再訓練スケジューラー
│   ├── generate_predictions.py   # 予想生成（既存レース用）
│   ├── predict_upcoming_races.py # 今後のレース自動予想（スクレイピング+予想）
│   ├── run_betting_simulation.py # 馬券シミュレーション実行
│   └── scheduler.py              # 定期実行スケジューラー
├── data/                          # データディレクトリ
│   ├── models/                   # 訓練済みモデル（.pkl）
│   ├── processed/                # 処理済み特徴量CSV
│   └── keiba.db                  # SQLiteデータベース（開発環境）
├── tests/                         # テストスイート
│   ├── test_scrapers/            # スクレイパーテスト
│   ├── test_models/              # データベースモデルテスト
│   └── test_ml/                  # 機械学習テスト
│       └── test_betting_simulator.py # 馬券シミュレータテスト
├── run.py                         # アプリケーションエントリーポイント
├── .env.example                   # 環境変数テンプレート
├── requirements.txt               # Pythonパッケージ依存関係
├── CLAUDE.md                      # Claude Code向け開発ガイド
└── README.md                      # このファイル
```

## 注意事項

### 法的配慮
- **必須**: netkeiba.comの利用規約を遵守してください
- **必須**: スクレイピングは適切な間隔で実行してください（デフォルト: 3秒の遅延）
- **必須**: 適切なUser-Agentヘッダーを設定してください（`.env`ファイルで設定）
- **禁止**: データの商用利用については事前にnetkeiba.comに確認してください
- **禁止**: 自動馬券購入機能の実装は行わないでください
- **推奨**: robots.txtを尊重し、サーバーに過度な負荷をかけないようにしてください

### 免責事項
- このアプリケーションの予想結果は参考情報です
- 予想の正確性を保証するものではありません
- 実際の馬券購入は自己責任で行ってください
- 本アプリケーションの使用による損失について、開発者は一切の責任を負いません
- ギャンブル依存症にはご注意ください

### データベースとエンコーディング
- データベースはUTF-8エンコーディングを使用します
- netkeiba.comのHTML（EUC-JP）は自動的にUTF-8に変換されます
- CSVファイルはUTF-8 BOM付きで保存され、Excelでも正しく開けます
- Windows環境でコンソール出力に文字化けが発生する場合がありますが、データベースには正しく保存されます

## ライセンス

MIT License

## 開発者向け

### テストの実行

**すべてのテストを実行:**
```bash
pytest
```

**カバレッジレポート付きでテスト:**
```bash
pytest --cov=src --cov-report=html
```

**特定のテストマークのみ実行:**
```bash
# ユニットテストのみ
pytest -m unit

# スクレイパーテストのみ
pytest -m scraper

# 遅いテストとインテグレーションテストを除外
pytest -m "not slow and not integration"
```

**特定のテストファイルを実行:**
```bash
pytest tests/test_scrapers/test_jra_scraper.py
```

### コードフォーマット

```bash
black .
```

### リント

```bash
flake8
```

## 開発ロードマップ

### 完了済み ✅
- [x] プロジェクト構造の作成
- [x] データベースモデルの実装（UTF-8対応）
- [x] netkeiba.comスクレイパーの実装（レースカレンダー、出馬表、レース結果）
- [x] データベース保存機能の実装（get-or-createパターン）
- [x] エンコーディング問題の解決（EUC-JP → UTF-8変換）
- [x] テストスイートの実装（スクレイピング、データベース操作、ML）
- [x] 特徴量エンジニアリング（40+特徴量）
- [x] データ前処理パイプライン
- [x] 機械学習モデルの実装（RandomForest、XGBoost）
- [x] モデル評価システム（RMSE, MAE, ROI, Hit Rate）
- [x] 予想生成機能
- [x] Webインターフェースの完成（レスポンシブデザイン）
- [x] REST APIエンドポイントの実装
- [x] パフォーマンス最適化とキャッシング
- [x] 検索機能（馬名、騎手名、調教師名）
- [x] 統計・分析ページ（予想精度、モデル別統計）
- [x] 馬券シミュレーション機能（全7馬券種対応、Web/CLI両対応）

### 今後の予定 📋
- [ ] デプロイメント（Heroku/AWS/Railway）
- [x] バックグラウンドタスク自動実行（schedule）
  - [x] 定期的なスクレイピング
  - [x] 自動予想生成
  - [x] モデルの定期再訓練
- [ ] ユーザー認証システム
- [ ] お気に入り機能
- [x] 馬券シミュレーション機能
- [ ] モバイルアプリ化（React Native/Flutter）

### 実装済み機能の詳細

#### 1. スクレイピングシステム
- ✅ **データソース**: netkeiba.com（モバイル版とPC版を併用）
- ✅ **エンコーディング処理**: EUC-JP → UTF-8自動変換
- ✅ **取得可能データ**:
  - レースカレンダー（開催日、開催地、レースID）
  - 出馬表（馬情報、騎手、調教師、斤量、馬体重、オッズ）
  - レース結果（着順、タイム、着差、確定オッズ、人気順）
- ✅ **レート制限**: 3秒遅延（サーバー負荷軽減）
- ✅ **エラーハンドリング**: リトライ機能、タイムアウト処理

#### 2. データベース設計
- ✅ **モデル**: Track, Horse, Jockey, Trainer, Race, RaceEntry, RaceResult, Prediction
- ✅ **エンコーディング**: 完全UTF-8対応
- ✅ **データ整合性**: get-or-createパターンによる重複防止
- ✅ **リレーションシップ**: 適切な外部キー設定
- ✅ **インデックス**: クエリパフォーマンス最適化

#### 3. 機械学習パイプライン
**特徴量エンジニアリング（40+特徴量）:**
- 馬の成績統計（全体、距離別、馬場別、コース別）
- 騎手・調教師の成績統計
- レース条件（距離、馬場状態、天候、クラス、頭数）
- 最近のフォーム（直近3走、5走の平均成績）
- 馬体重変動、年齢、性別

**モデル:**
- RandomForest（回帰・分類）
- XGBoost（回帰・分類）
- ハイパーパラメータチューニング対応

**評価指標:**
- 回帰: RMSE, MAE, R², ランク相関（Spearman, Kendall）
- 分類: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- 実用指標: 的中率、回収率（ROI）

#### 4. Webインターフェース
**ページ構成:**
- 📊 ダッシュボード（今週のレース、最近の結果、統計サマリー）
- 🏇 レース一覧・詳細（フィルター、ソート機能）
- 🔮 予想一覧・詳細（勝率順、信頼度スコア表示）
- 📈 予想精度分析（的中率、回収率、モデル別比較）
- 🐴 エンティティページ（馬、騎手、調教師の詳細統計）
- 🔍 検索機能（複数エンティティ対応）

**デザイン:**
- Bootstrap 5ベース
- レスポンシブ対応（モバイル・タブレット・デスクトップ）
- 直感的なUI/UX

#### 5. REST API
**エンドポイント:**
```
GET /api/races              # レース一覧
GET /api/races/<id>         # レース詳細
GET /api/predictions        # 予想一覧
GET /api/predictions/<id>   # 予想詳細
GET /api/horses/<id>        # 馬プロフィール
GET /api/jockeys/<id>       # 騎手プロフィール
GET /api/trainers/<id>      # 調教師プロフィール
```

**特徴:**
- JSON形式のレスポンス
- クエリパラメータによるフィルタリング
- ページネーション対応
- エラーハンドリング

#### 6. パフォーマンス最適化
- ✅ Flask-Cachingによるクエリキャッシング（TTL: 5分）
- ✅ SQLAlchemy eager loading（N+1問題解決）
- ✅ ページネーション（デフォルト: 20件/ページ）
- ✅ データベースインデックス最適化

#### 7. テスト
- ✅ **ユニットテスト**: 個別機能のテスト
- ✅ **インテグレーションテスト**: エンドツーエンドテスト
- ✅ **カバレッジ**: 主要機能をカバー
- ✅ **CI対応**: pytest実行可能

---

詳細なドキュメントとアーキテクチャ説明は [CLAUDE.md](CLAUDE.md) を参照してください。
