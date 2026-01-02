# JRA競馬予想アプリケーション

JRAの競馬データをスクレイピングし、機械学習モデルで予想を行うWebアプリケーション。

## 概要

このアプリケーションは以下の機能を提供します:
- JRAウェブサイトからのレースデータの自動収集
- 機械学習モデルによる勝馬予想
- Webインターフェースでの予想結果表示
- REST APIによるデータアクセス

## 技術スタック

- **バックエンド**: Python 3.9+, Flask
- **機械学習**: scikit-learn, XGBoost
- **データ処理**: pandas, numpy
- **スクレイピング**: BeautifulSoup4, Requests
- **データベース**: PostgreSQL (本番), SQLite (開発)
- **フロントエンド**: Jinja2, Bootstrap 5

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

```bash
python -m src.web.app
```

ブラウザで `http://localhost:5000` にアクセス。

## プロジェクト構造

```
keiba-app/
├── config/              # 設定ファイル
├── src/
│   ├── scrapers/       # スクレイピングモジュール
│   ├── data/           # データモデルと前処理
│   ├── ml/             # 機械学習モデル
│   ├── web/            # Webアプリケーション
│   └── utils/          # ユーティリティ
├── scripts/            # 実行スクリプト
├── data/               # データファイル
├── tests/              # テスト
└── notebooks/          # Jupyter notebooks
```

## 注意事項

### 法的配慮
- JRAウェブサイトの利用規約を遵守してください
- スクレイピングは適切な間隔で実行してください（2-5秒の遅延）
- データの商用利用については事前にJRAに確認してください

### 免責事項
- このアプリケーションの予想結果は参考情報です
- 予想の正確性を保証するものではありません
- 実際の馬券購入は自己責任で行ってください

## ライセンス

MIT License

## 開発者向け

### テストの実行

```bash
pytest
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
- [ ] APIエンドポイントの拡張
- [ ] デプロイ

### 実装済み機能

#### スクレイピング
- ✅ レースカレンダーの取得（CNAME付き）
- ✅ 出馬表（レースカード）の取得
- ✅ レース結果の取得
- ✅ 馬プロフィールの取得
- ✅ CNAMEパラメータの自動抽出

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
- ✅ REST API エンドポイント（予想データ取得）

詳細なドキュメントは [CLAUDE.md](CLAUDE.md) を参照してください。
