# プロジェクト構造

このドキュメントは、JRA競馬予想アプリケーションのディレクトリ構造とファイルの役割を説明します。

## ディレクトリ構造

```
keiba-app/
├── .claude/                    # Claude Code設定ファイル
│   └── settings.local.json     # ローカル設定
│
├── config/                     # 設定ファイル
│   ├── examples/               # 設定ファイルの例
│   │   ├── .env.example        # 環境変数の例
│   │   ├── .env.personal.example
│   │   ├── personal_settings.json
│   │   └── retraining_config.json
│   ├── __init__.py
│   ├── logging_config.py       # ログ設定
│   └── settings.py             # アプリケーション設定
│
├── data/                       # データディレクトリ
│   ├── backups/                # データベースバックアップ
│   ├── debug/                  # デバッグ用HTMLファイル
│   ├── models/                 # 訓練済み機械学習モデル
│   ├── processed/              # 処理済みデータ
│   ├── raw/                    # 生データ
│   └── temp/                   # 一時ファイル
│
├── docs/                       # ドキュメント
│   ├── API.md                  # API仕様
│   ├── DOCKER_SETUP.md         # Docker環境構築
│   ├── HORSE_PROFILE_SCRAPING.md
│   ├── IMPLEMENTATION_STATUS.md
│   ├── IMPROVEMENT_PLAN.md     # 改善計画
│   ├── MODEL_RETRAINING.md     # モデル再訓練
│   ├── ODDS_UPDATE.md          # オッズ更新
│   ├── PREDICTION_AUTOMATION.md
│   ├── PROJECT_STRUCTURE.md    # このファイル
│   └── RequirementsDefinition.md # 要件定義
│
├── logs/                       # ログファイル
│
├── migrations/                 # データベースマイグレーション
│   └── env.py
│
├── notebooks/                  # Jupyter Notebook
│
├── scripts/                    # ユーティリティスクリプト
│   ├── README.md               # スクリプト説明
│   ├── docker_migrate.sh       # Dockerマイグレーション
│   ├── extract_features.py     # 特徴量抽出
│   ├── generate_predictions.py # 予測生成
│   ├── init_db.py              # データベース初期化
│   ├── init_docker_db.py       # Docker用DB初期化
│   ├── model_retraining_scheduler.py
│   ├── odds_update_scheduler.py
│   ├── predict_upcoming_races.py
│   ├── run_betting_simulation.py
│   ├── scheduler.py            # スケジューラー
│   ├── scrape_and_save_results.py
│   ├── scrape_data.py          # データスクレイピング
│   ├── scrape_historical_data.py
│   ├── train_model.py          # モデル訓練
│   └── update_odds.py          # オッズ更新
│
├── src/                        # ソースコード
│   ├── data/                   # データ関連
│   │   ├── database.py         # データベース操作
│   │   └── models.py           # データモデル
│   ├── ml/                     # 機械学習
│   │   ├── feature_engineering.py
│   │   ├── models/             # MLモデル
│   │   └── preprocessing.py    # 前処理
│   ├── scrapers/               # スクレイパー
│   │   └── netkeiba_scraper.py # Netkeibaスクレイパー
│   ├── utils/                  # ユーティリティ
│   │   └── notification.py     # 通知機能
│   └── web/                    # Webアプリケーション
│       └── app.py              # Flaskアプリ
│
├── tests/                      # テストコード
│   ├── test_config.py
│   ├── test_models/
│   ├── test_scrapers/
│   └── conftest.py
│
├── .dockerignore               # Docker除外ファイル
├── .env                        # 環境変数（Git管理外）
├── .gitignore                  # Git除外ファイル
├── CLAUDE.md                   # Claude Code向けガイド
├── docker-compose.yml          # Docker Compose設定
├── Dockerfile                  # Dockerイメージ定義
├── manage.py                   # 管理コマンド
├── pytest.ini                  # pytest設定
├── quickstart.bat              # クイックスタート（Windows）
├── QUICKSTART.md               # クイックスタートガイド
├── README.md                   # プロジェクト概要
├── requirements.txt            # Python依存パッケージ
└── run.py                      # アプリケーション起動
```

## 主要ファイルの役割

### ルートディレクトリ

| ファイル | 説明 |
|---------|------|
| `CLAUDE.md` | Claude Codeを使用する際の開発ガイドライン |
| `README.md` | プロジェクト全体の概要説明 |
| `QUICKSTART.md` | 初回セットアップの手順書 |
| `requirements.txt` | Python依存パッケージリスト |
| `run.py` | Flaskアプリケーション起動スクリプト |
| `manage.py` | データベース管理コマンド |
| `pytest.ini` | テスト実行設定 |
| `docker-compose.yml` | Docker環境定義 |
| `Dockerfile` | Dockerイメージビルド定義 |

### config/

アプリケーションの設定ファイルを格納します。

- `settings.py`: 環境別設定クラス（Development, Production, Testing）
- `logging_config.py`: ロギング設定
- `examples/`: 設定ファイルのテンプレート（Git管理）

### data/

データファイルを格納します。

- `raw/`: スクレイピングした生データ
- `processed/`: 特徴量エンジニアリング後のデータ
- `models/`: 訓練済みMLモデル（`.pkl`, `.joblib`など）
- `debug/`: デバッグ用HTML・テキストファイル
- `temp/`: 一時ファイル
- `backups/`: データベースバックアップ

**注意**: `data/`配下のファイルは基本的にGit管理外です。

### docs/

プロジェクトドキュメントを格納します。

- `PROJECT_STRUCTURE.md`: このファイル - ディレクトリ構造の説明
- `RequirementsDefinition.md`: 要件定義書
- `IMPROVEMENT_PLAN.md`: 機能改善計画
- `IMPLEMENTATION_STATUS.md`: 実装状況
- その他、機能別ドキュメント

### scripts/

運用・保守用のスクリプトを格納します。

| スクリプト | 説明 |
|-----------|------|
| `init_db.py` | データベース初期化 |
| `scrape_data.py` | データスクレイピング |
| `scrape_historical_data.py` | 過去データの一括取得 |
| `train_model.py` | 機械学習モデル訓練 |
| `predict_upcoming_races.py` | 今後のレース予測 |
| `scheduler.py` | 定期実行スケジューラー |
| `update_odds.py` | オッズ情報更新 |

### src/

アプリケーションのソースコードを格納します。

```
src/
├── data/           # データ層
│   ├── models.py   # SQLAlchemyモデル定義
│   └── database.py # DB操作関数
├── scrapers/       # スクレイパー
│   └── netkeiba_scraper.py
├── ml/             # 機械学習
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   └── models/
├── web/            # Webアプリケーション
│   └── app.py
└── utils/          # ユーティリティ
    └── notification.py
```

### tests/

テストコードを格納します。

- `test_models/`: データモデルのテスト
- `test_scrapers/`: スクレイパーのテスト
- `test_web/`: Webアプリのテスト
- `conftest.py`: pytest共通設定

## ベストプラクティス

### ファイルの配置

1. **設定ファイル**: `config/`または`config/examples/`に配置
2. **ドキュメント**: `docs/`に配置
3. **デバッグファイル**: `data/debug/`に配置
4. **一時ファイル**: `data/temp/`に配置
5. **ログファイル**: `logs/`に配置

### Git管理

- `.env`ファイルは**絶対にコミットしない**
- 大きなデータファイル（`.db`, `.pkl`など）はコミットしない
- デバッグ用HTMLファイルはコミットしない
- 個人用設定ファイル（`README_PERSONAL.md`など）はコミットしない

### 命名規則

- **スクリプト**: 動詞で始める（例: `scrape_data.py`, `train_model.py`）
- **モジュール**: 小文字＋アンダースコア（例: `feature_engineering.py`）
- **クラス**: パスカルケース（例: `NetkeibaScraper`）
- **関数**: 小文字＋アンダースコア（例: `get_or_create_horse()`）

## データフロー

```
1. スクレイピング
   netkeiba.com → NetkeibaScraper → data/raw/

2. データベース保存
   data/raw/ → database.py → keiba.db

3. 特徴量エンジニアリング
   keiba.db → feature_engineering.py → data/processed/

4. モデル訓練
   data/processed/ → train_model.py → data/models/

5. 予測
   keiba.db + data/models/ → predict.py → predictions
```

## 開発ワークフロー

1. **新機能開発**: `src/`配下にコードを追加
2. **テスト作成**: `tests/`配下にテストを追加
3. **ドキュメント更新**: `docs/`のドキュメントを更新
4. **スクリプト作成**: 必要に応じて`scripts/`にスクリプト追加

## 参考資料

- プロジェクト概要: [README.md](../README.md)
- 開発ガイド: [CLAUDE.md](../CLAUDE.md)
- クイックスタート: [QUICKSTART.md](../QUICKSTART.md)
- 要件定義: [RequirementsDefinition.md](RequirementsDefinition.md)
