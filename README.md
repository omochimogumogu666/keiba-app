# JRA競馬予想アプリケーション

JRA競馬データ（netkeiba.com経由）をスクレイピングし、機械学習モデルで予想を行うWebアプリケーション。

## 概要

このアプリケーションは以下の機能を提供します:
- ✅ netkeiba.comからのレースデータの自動収集（レースカレンダー、出馬表、レース結果、馬プロフィール）
- ✅ 機械学習モデルによる勝馬予想（RandomForest、XGBoost）
- ✅ Webインターフェースでの予想結果表示
- ✅ RESTful APIによるデータアクセス
- ✅ パフォーマンス最適化とキャッシング機能

## 技術スタック

- **バックエンド**: Python 3.9+, Flask, Flask-SQLAlchemy
- **機械学習**: scikit-learn, XGBoost, imbalanced-learn
- **データ処理**: pandas, numpy
- **スクレイピング**: BeautifulSoup4, Requests, cloudscraper
- **データベース**: PostgreSQL (本番), SQLite (開発)
- **フロントエンド**: Jinja2, Bootstrap 5
- **キャッシング**: Flask-Caching (SimpleCache/Redis)

## セットアップ

### 1. 仮想環境の作成

```bash
python -m venv venv
```

### 2. 仮想環境のアクティベート

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### 3. 依存関係のインストール

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

## 使用方法

### データのスクレイピング

**レースカレンダーと出馬表のスクレイピング:**
```bash
python scripts/scrape_data.py
```

**レース結果のスクレイピングと保存（完全なワークフロー）:**
```bash
# 昨日のレース結果をスクレイピング（デフォルト）
python scripts/scrape_and_save_results.py

# 特定の日付のレース結果をスクレイピング
python scripts/scrape_and_save_results.py 2026-01-01
```

**過去の履歴データの一括スクレイピング:**
```bash
# 過去5年分のデータをスクレイピング（デフォルト、週末のみ）
python scripts/scrape_historical_data.py

# 過去3年分のデータをスクレイピング
python scripts/scrape_historical_data.py --years 3

# 特定の期間のデータをスクレイピング
python scripts/scrape_historical_data.py --start-date 2021-01-01 --end-date 2023-12-31

# 週末以外も含めてすべての日をスクレイピング
python scripts/scrape_historical_data.py --years 5 --all-days
```

注意事項:
- JRAレースは主に週末に開催されるため、デフォルトでは土日のみスクレイピングします
- リクエスト間に3秒の遅延を設けているため、大量データの取得には時間がかかります
- 進捗は随時保存されるため、中断しても再開可能です

### 特徴量エンジニアリング

**データベースから特徴量を抽出:**
```bash
python scripts/extract_features.py
```

出力ファイル（`data/processed/`に保存）:
- `features_raw.csv` - 生の特徴量
- `features_processed.csv` - 正規化された特徴量
- `labels_regression.csv` - 回帰用ラベル（着順）
- `labels_binary.csv` - 二値分類用ラベル（勝敗）
- `labels_multiclass.csv` - 多クラス分類用ラベル（勝/複/他）

### モデルの訓練

**RandomForestモデルの訓練（回帰）:**
```bash
# データベースから直接訓練
python scripts/train_model.py --model random_forest --task regression

# CSVファイルから訓練
python scripts/train_model.py --model random_forest --task regression --data-source csv
```

**XGBoostモデルの訓練（分類）:**
```bash
python scripts/train_model.py --model xgboost --task classification
```

**両方のモデルを訓練:**
```bash
python scripts/train_model.py --model both --task regression
```

訓練済みモデルは `data/models/` に保存されます。

### 予想の生成

**訓練済みモデルで予想を生成:**
```bash
# 今日のレースを予想
python scripts/generate_predictions.py --model-path data/models/random_forest_regression_20260103_120000.pkl --save-to-db

# 特定のレースを予想
python scripts/generate_predictions.py --model-path data/models/xgboost_regression_20260103_120000.pkl --race-id 123 --save-to-db

# CSV出力
python scripts/generate_predictions.py --model-path data/models/random_forest_regression_20260103_120000.pkl --output-csv predictions.csv
```

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
│   └── logging_config.py          # ロギング設定
├── src/
│   ├── scrapers/                  # スクレイピングモジュール
│   │   ├── netkeiba_scraper.py   # Netkeibaスクレイパー（メイン）
│   │   └── jra_scraper.py        # JRAスクレイパー（非推奨、CNAME対応）
│   ├── data/                      # データモデルと前処理
│   │   ├── models.py             # SQLAlchemyモデル
│   │   └── database.py           # データベース操作（get-or-create）
│   ├── ml/                        # 機械学習モデル
│   │   ├── feature_engineering.py # 特徴量エンジニアリング
│   │   ├── models/               # MLモデル（RandomForest, XGBoost）
│   │   └── prediction.py         # 予想生成
│   ├── web/                       # Webアプリケーション
│   │   ├── app.py                # Flaskアプリケーションファクトリ
│   │   ├── routes/               # ルート（メイン、API、エンティティ）
│   │   └── templates/            # Jinja2テンプレート
│   └── utils/                     # ユーティリティ
│       └── logger.py             # ロガー
├── scripts/                       # 実行スクリプト
│   ├── init_db.py                # データベース初期化
│   ├── scrape_data.py            # レースデータスクレイピング
│   ├── scrape_and_save_results.py # レース結果スクレイピング（完全ワークフロー）
│   ├── extract_features.py       # 特徴量抽出
│   ├── train_model.py            # モデル訓練
│   └── generate_predictions.py   # 予想生成
├── data/                          # データファイル
│   ├── models/                   # 訓練済みモデル
│   └── processed/                # 処理済み特徴量CSV
├── tests/                         # テスト
│   ├── test_scrapers/            # スクレイパーテスト
│   ├── test_models/              # データベースモデルテスト
│   └── test_ml/                  # 機械学習テスト
├── notebooks/                     # Jupyter notebooks
├── run.py                         # アプリケーションエントリーポイント
├── .env.example                   # 環境変数テンプレート
└── requirements.txt               # Pythonパッケージ依存関係
```

## 注意事項

### 法的配慮
- **必須**: netkeiba.comの利用規約を遵守してください
- **必須**: スクレイピングは適切な間隔で実行してください（デフォルト: 3秒の遅延）
- **必須**: 適切なUser-Agentヘッダーを設定してください
- **禁止**: データの商用利用については事前にnetkeiba.comに確認してください
- **禁止**: 自動馬券購入機能の実装

### 免責事項
- このアプリケーションの予想結果は参考情報です
- 予想の正確性を保証するものではありません
- 実際の馬券購入は自己責任で行ってください
- 本アプリケーションの使用による損失について、開発者は一切の責任を負いません

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

## ロードマップ

- [x] プロジェクト構造の作成
- [x] データベースモデルの実装
- [x] データスクレイパーの実装（レースカレンダー、出馬表、レース結果、馬プロフィール）
- [x] データベース保存機能の実装（get-or-createパターン）
- [x] テストの実装（スクレイピング、データベース操作）
- [x] 特徴量エンジニアリング
- [x] 機械学習モデルの訓練
- [x] 予想機能の実装
- [x] Webインターフェースの完成
- [x] REST APIエンドポイントの実装
- [x] パフォーマンス最適化とキャッシング
- [ ] デプロイ（Heroku/AWS）
- [ ] バックグラウンドタスク自動実行（スクレイピング、予想生成）

### 実装済み機能

#### スクレイピング
- ✅ レースカレンダーの取得（netkeiba.com mobile経由）
- ✅ 出馬表（レースカード）の取得
- ✅ レース結果の取得
- ✅ 馬プロフィールの取得
- ✅ シンプルなURL構造（CNAMEパラメータ不要）

#### データベース
- ✅ Track, Horse, Jockey, Trainer, Race, RaceEntry, RaceResult, Prediction モデル
- ✅ get-or-create パターンによる重複防止
- ✅ 完全な保存ワークフロー

#### 機械学習
- ✅ 特徴量エンジニアリング（40+特徴量）
  - 馬の過去成績統計（勝率、複勝率、距離別・馬場別・コース別）
  - 騎手・調教師の統計
  - レース固有の特徴（距離、馬場状態、天候、クラス）
  - 最近の成績トレンド
- ✅ データ前処理パイプライン（欠損値処理、正規化、特徴選択）
- ✅ RandomForestモデル（回帰・分類）
- ✅ XGBoostモデル（回帰・分類）
- ✅ モデル評価指標（RMSE, MAE, ROI, Hit Rate, ランク相関）
- ✅ モデル永続化とバージョニング

#### テスト
- ✅ ユニットテスト（パース機能、特徴量抽出、MLモデル）
- ✅ インテグレーションテスト（データベース保存）
- ✅ 包括的なテストスイート（43テスト）
- ✅ pytest による自動テスト

#### Webインターフェース
- ✅ レスポンシブデザイン（Bootstrap 5）
- ✅ ホームページ（今週のレース、最近の結果）
- ✅ レース一覧（フィルター機能付き）
- ✅ レース詳細ページ（出走馬情報、レース結果表示）
- ✅ 予想一覧・詳細ページ（勝率、信頼度スコア表示）
- ✅ 予想精度ページ（単勝的中率、複勝的中率、モデル別統計）
- ✅ 馬・騎手・調教師の一覧・詳細ページ（成績統計付き）
- ✅ 検索機能（馬名、騎手名、調教師名、レース名）

#### REST API
- ✅ `/api/predictions` - 予想データ取得（日付、レースIDフィルター対応）
- ✅ `/api/predictions/<id>` - 個別予想データ取得
- ✅ `/api/races` - レース一覧取得（日付、開催地フィルター対応）
- ✅ `/api/races/<id>` - レース詳細取得（出走馬情報含む）
- ✅ `/api/horses/<id>` - 馬プロフィール取得
- ✅ `/api/jockeys/<id>` - 騎手プロフィール取得
- ✅ `/api/trainers/<id>` - 調教師プロフィール取得

#### パフォーマンス最適化
- ✅ Flask-Cachingによるクエリ結果キャッシング
- ✅ データベースクエリの最適化（eager loading）
- ✅ ページネーション実装
- ✅ 静的ファイルの効率的な配信

詳細なドキュメントは [CLAUDE.md](CLAUDE.md) を参照してください。
