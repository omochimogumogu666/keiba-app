# CLAUDE.md

JRA競馬予想アプリケーション - Horse racing prediction app with ML models.
**Stack**: Python 3.9+, Flask, SQLAlchemy, BeautifulSoup4, scikit-learn, XGBoost

##　Rules to Always Follow

-　Always perform environment setup on Docker; do not install locally.

## Quick Start

```bash
./quickstart.sh  # Docker (recommended)
python run.py    # Local server: http://localhost:5001
```

## Core Commands

```bash
# Scraping (Optimized: 主要5場・2勝クラス以上に限定)
python scripts/scrape_and_save_results.py 2025-12-28
python scripts/scrape_historical_data.py --years 5  # 東京/中山/阪神/京都/中京, 2勝クラス以上

# ML Pipeline
python scripts/extract_features.py
python scripts/train_model.py --model both --task regression
python scripts/predict_upcoming_races.py --days-ahead 3

# Testing
pytest -m "not slow and not integration"
```

## Critical Patterns

### 1. Netkeiba Scraper
- **Source**: netkeiba.com (EUC-JP encoding)
- **Race ID**: `YYYYKKRRDDNN` (12桁)
- **URLs**:
  - Calendar: `race.sp.netkeiba.com/?pid=race_list&kaisai_date=YYYYMMDD`
  - Card: `race.netkeiba.com/race/shutuba.html?race_id=YYYYKKRRDDNN`
  - Result: `race.netkeiba.com/race/result.html?race_id=YYYYKKRRDDNN`
- **Rate Limit**: 3秒 delay必須
- **Class**: `NetkeibaScraper` in `src/scrapers/netkeiba_scraper.py`
- **Optimization**:
  - **Tracks**: 東京、中山、阪神、京都、中京 (主要5場)
  - **Race Class**: 2勝クラス以上 (G1/G2/G3/OP/Listed/3勝/2勝) - 除外: 1勝/未勝利/新馬
  - **Speed**: ~60% time reduction vs full scraping

### 2. Database
- **Schema**: `tracks → races → race_entries → race_results`
- **Get-or-Create**: 必ず使用 (`get_or_create_track/horse/jockey/trainer`)
- **Save Functions**: `save_race_to_db()`, `save_race_entries_to_db()`, `save_race_results_to_db()`
- **Config**: `DevelopmentConfig` (SQLite), `ProductionConfig` (PostgreSQL), `TestingConfig`

### 3. Flask App
```python
from src.web.app import create_app
app = create_app('development')
with app.app_context():
    db.create_all()
```

### 4. ML Pipeline
- **Features**: 40+ features (`src/ml/feature_engineering.py`)
- **Models**: RandomForest, XGBoost (`src/ml/models/`)
- **Betting**: 7券種対応 (`src/ml/betting_simulator.py`)
- **Retraining**: 自動バージョン管理 (`scripts/model_retraining_scheduler.py`)

## Data Flow

### Scraping → DB
```python
with NetkeibaScraper(delay=3) as scraper:
    races = scraper.scrape_race_calendar(date)
    for race_info in races:
        race_card = scraper.scrape_race_card(race_info['race_id'])
        race = save_race_to_db(race_data)
        entries = save_race_entries_to_db(race.id, race_card['entries'])
```

**注意**:
- `save_race_to_db()`: track名(string)を渡す
- `save_race_entries_to_db()`: DB race.idを渡す(jra_race_idではない)
- 全save関数は自動でcreate/update判定

## Testing

```python
pytest -m "not slow and not integration"  # 高速テスト
pytest -m integration  # 実際のサイトにアクセス

# DB tests: TestingConfigを使用
app = create_app('testing')
with app.app_context():
    # test operations
```

## Gotchas

1. **Unicode**: Windows `cp932`でエラー時は `try-except` でラップ
2. **Foreign Key順**: Track → Horse/Jockey/Trainer → Race → RaceEntry → RaceResult
3. **Encoding**: Netkeiba = EUC-JP, JRA(deprecated) = Shift_JIS
4. **HTML変更**: スクレイピング失敗時は `debug_html.py` でHTML保存して調査

## Legal & Ethics

- **必須**: `SCRAPING_DELAY >= 2`秒 (default: 3s)
- **必須**: `USER_AGENT` header設定
- **禁止**: 自動投票機能、JRAデータの商用利用
- **必須**: 予想は参考情報のみである旨を明記

## Implementation Status

**Complete**: Netkeiba scraper, DB models, ML pipeline (40+ features, RF/XGBoost), Betting simulation (7券種), Web UI, REST API, Caching, Auto-retraining, Odds update, Tests

**Pending**: User auth, Mobile app, Advanced viz, Real-time notifications, Ensemble models

## Key Files

**Core**:
- `src/scrapers/netkeiba_scraper.py` - Netkeibaスクレイパー
- `src/data/{database,models}.py` - DB操作とORM
- `config/settings.py` - 環境設定

**ML**:
- `src/ml/feature_engineering.py` - 特徴量抽出(40+)
- `src/ml/models/{random_forest,xgboost_model}.py` - モデル実装
- `src/ml/betting_simulator.py` - 馬券シミュレーション

**Web**:
- `src/web/app.py` - Flask factory
- `src/web/routes/{main,api,simulation}.py` - ルート

**Scripts**:
- `scripts/scrape_and_save_results.py` - 完全スクレイピング
- `scripts/predict_upcoming_races.py` - 自動予想(推奨)
- `scripts/model_retraining_scheduler.py` - 自動再学習

**Tests**: `tests/test_{scrapers,models,ml,api}/`
