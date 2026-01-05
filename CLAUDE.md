# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JRA競馬予想アプリケーション (JRA Horse Racing Prediction Application) - A web application that scrapes JRA (Japan Racing Association) horse racing data and uses machine learning models to predict race outcomes.

**Tech Stack**: Python 3.9+, Flask, SQLAlchemy, BeautifulSoup4, scikit-learn, XGBoost

## Essential Commands

### Setup & Development
```bash
# Initial setup
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
cp .env.example .env

# Database initialization
python scripts/init_db.py

# Run web application
python run.py
# or
python -m src.web.app

# Data scraping
python scripts/scrape_data.py

# Historical data scraping (bulk collection)
python scripts/scrape_historical_data.py --years 5

# Complete workflow (scrape races + results for a specific date)
python scripts/scrape_and_save_results.py 2025-12-28
```

### Testing
```bash
# Run all tests with coverage
pytest

# Run specific test types
pytest -m unit
pytest -m integration
pytest -m scraper

# Run tests without slow/integration tests
pytest -m "not slow and not integration"

# Run single test file
pytest tests/test_scrapers/test_netkeiba_scraper.py

# Generate coverage report
pytest --cov=src --cov-report=html
```

### Code Quality
```bash
# Format code
black .

# Lint code
flake8
```

### Machine Learning & Predictions
```bash
# Feature extraction
python scripts/extract_features.py

# Train models
python scripts/train_model.py --model random_forest --task regression
python scripts/train_model.py --model xgboost --task classification
python scripts/train_model.py --model both --task regression

# Generate predictions for upcoming races (auto-scrape + predict)
python scripts/predict_upcoming_races.py
python scripts/predict_upcoming_races.py --days-ahead 3 --date 2026-01-11

# Generate predictions for existing races in DB
python scripts/generate_predictions.py --model-path data/models/model.pkl --save-to-db

# Automated retraining (run once or schedule periodic retraining)
python scripts/model_retraining_scheduler.py --mode once
python scripts/model_retraining_scheduler.py --mode schedule --interval weekly --time 02:00
```

### Betting Simulation
```bash
# Run betting simulation via CLI
python scripts/run_betting_simulation.py \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --bet-types win place quinella \
  --bet-amount 100 \
  --top-n 3

# Without predictions (uniform probability baseline)
python scripts/run_betting_simulation.py \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --no-predictions

# View simulation results: http://localhost:5000/simulation/
```

### Task Scheduling
```bash
# Schedule daily prediction generation
python scripts/scheduler.py --schedule daily --time 08:00

# Schedule weekly predictions (Friday 20:00)
python scripts/scheduler.py --schedule weekly --day friday --time 20:00

# Run once immediately
python scripts/scheduler.py --schedule once

# Schedule periodic odds updates
python scripts/odds_update_scheduler.py --schedule daily --time 09:00
```

## Critical Architecture Patterns

### 1. Netkeiba.com Scraping Architecture (CURRENT)

**Primary Data Source**: The application uses [netkeiba.com](https://netkeiba.com) as the primary data source. This is more stable and reliable than scraping JRA's official site directly.

**URL Structure**:
- **Race Calendar**: Uses mobile site (`https://race.sp.netkeiba.com/?pid=race_list&kaisai_date=YYYYMMDD`)
  - PC version loads data via JavaScript, mobile version has static HTML
  - Race IDs are embedded directly in the HTML
- **Race Card (Shutuba)**: `https://race.netkeiba.com/race/shutuba.html?race_id=YYYYKKRRDDNN`
- **Race Result**: `https://race.netkeiba.com/race/result.html?race_id=YYYYKKRRDDNN`
- **Horse Profile**: `https://db.netkeiba.com/horse/HORSE_ID/`

**Race ID Format**: 12-digit ID format: `YYYYKKRRDDNN`
- YYYY = Year (4 digits)
- KK = Kaisai code / Track code (01=札幌, 02=函館, ..., 06=中山, ...)
- RR = Meeting number (2 digits)
- DD = Day number (2 digits)
- NN = Race number (2 digits)

**HTML Structure Specifics**:
```python
# Mobile calendar: RaceList with links containing race_id parameter
race_links = soup.find_all('a', href=re.compile(r'race_id=\d{12}'))

# Race card table: Shutuba_Table RaceTable01 ShutubaTable
# Entry rows: tr.HorseList
# - Waku (post position): td.Waku1, td.Waku2, ...
# - Horse number: td.Umaban1, td.Umaban2, ...
# - Horse info: td.HorseInfo > a (href=/horse/HORSE_ID)
# - Jockey: td.Jockey > a (href=/jockey/result/recent/JOCKEY_ID/)
# - Trainer: td.Trainer > a (href=/trainer/result/recent/TRAINER_ID/)
# - Weight (斤量): td after td.Barei
# - Horse weight: td.Weight (format: "482 (+2)" or "500(0)")
# - Morning odds: td.Popular > span#odds-X_YY

# Result table: Result_Table or ResultRaceShutuba
# Result rows: tr.HorseList (same structure as shutuba)
# - Finish position: td.Chakujun or td.Result
# - Time: td.Time
# - Margin: td.Sa (着差)
# - Final odds: td.Odds
# - Popularity: td.Ninki
```

**Key Implementation Details**:
- `NetkeibaScraper`: Main scraper class in `src/scrapers/netkeiba_scraper.py`
- Encoding: **EUC-JP** (not Shift_JIS like JRA)
- No CNAME parameters needed - much simpler URL structure
- Horse weight parsing handles both `"482(+2)"` and `"482 (+2)"` formats (with/without spaces)
- ID extraction regex patterns:
  - Horse: `/horse/(\d+)` → extracts 10-digit horse ID
  - Jockey: `/jockey/[^/]+/[^/]+/(\d+)` → extracts 5-digit jockey ID
  - Trainer: `/trainer/[^/]+/[^/]+/(\d+)` → extracts 5-digit trainer ID

**Rate Limiting**: Always use 3-second delay between requests (configured in `SCRAPING_DELAY`). This is both ethical and respects netkeiba's terms of service.

### 1b. JRA Scraping Architecture (DEPRECATED)

**NOTE**: Direct JRA scraping has been replaced by Netkeiba scraper. JRA's CNAME system is complex and fragile.

<details>
<summary>Historical JRA Scraper Documentation (for reference only)</summary>

**CNAME Parameter System**: JRA uses a complex CNAME parameter system for URLs that cannot be programmatically generated. The scraper must:
- Extract CNAMEs dynamically from JRA's race calendar links
- Handle two extraction methods:
  - URL parameters: `?CNAME=pw01dde0106202601011120260104/6C`
  - JavaScript onclick: `doAction('/JRADB/accessK.html', 'pw04kmk005386/50')`

**HTML Structure Specifics**:
```python
# Race card table: class="basic"
# Horse info: td.horse > div.name_line > a (horse name + CNAME in href)
# Trainer info: p.trainer > a (onclick contains CNAME)
# Jockey info: p.jockey > a (onclick contains CNAME)
```
</details>

### 2. Database Model Relationships

The application uses SQLAlchemy with a normalized relational schema:

**Core Entity Pattern**:
```
tracks (1) → (*) races (1) → (*) race_entries
                                      ↓
horses (*) ← (1) trainers         (1) race_results
  ↓
(*) race_entries (*) ← jockeys
  ↓
(*) predictions
```

**Get-or-Create Pattern**: Always use get-or-create functions to prevent duplicates:
- `get_or_create_track(name, **kwargs)`
- `get_or_create_horse(jra_horse_id, name, **kwargs)`
- `get_or_create_jockey(jra_jockey_id, name)`
- `get_or_create_trainer(jra_trainer_id, name, **kwargs)`

These are defined in `src/data/database.py` and critical for maintaining data integrity.

### 3. Configuration Management

Uses environment-aware configuration classes in `config/settings.py`:
- `DevelopmentConfig`: SQLite, debug mode, SQL echo enabled
- `ProductionConfig`: PostgreSQL (via DATABASE_URL), debug off
- `TestingConfig`: In-memory SQLite for isolated testing

**Access config**:
```python
from config.settings import get_config
config = get_config()  # Auto-detects from FLASK_ENV
```

**Critical settings**:
- `SCRAPING_DELAY`: Minimum delay between requests (default: 3s)
- `USER_AGENT`: Must be set to avoid blocking
- `MAX_RETRIES`: Retry count for failed requests (default: 3)

### 4. Flask Application Factory

Uses the application factory pattern in `src/web/app.py`:
```python
from src.web.app import create_app

app = create_app('development')  # or 'production', 'testing'
```

Always create app within Flask app context for database operations:
```python
with app.app_context():
    db.create_all()
    # database operations here
```

### 5. Logging System

Centralized logging configuration in `config/logging_config.py`:
```python
from config.logging_config import setup_logging
from src.utils.logger import get_app_logger

setup_logging(log_level='INFO')
logger = get_app_logger(__name__)
```

Use structured logging levels:
- `DEBUG`: Detailed scraping/parsing steps
- `INFO`: High-level operations (race scraped, entries saved)
- `WARNING`: Recoverable issues (missing data, using defaults)
- `ERROR`: Failures requiring attention

### 6. Betting Simulation Architecture

The application includes a comprehensive betting simulation engine for strategy backtesting.

**Core Components**:
- `BettingSimulator` class in `src/ml/betting_simulator.py`
- Supports 7 bet types: win (単勝), place (複勝), quinella (馬連), exacta (馬単), wide (ワイド), trio (3連複), trifecta (3連単)
- Database models: `SimulationRun`, `SimulationBet`

**Workflow**:
```python
from src.ml.betting_simulator import BettingSimulator, BettingStrategy, BetType

# Define strategy
strategy = BettingStrategy(
    bet_types=[BetType.WIN, BetType.PLACE],
    bet_amount=100,
    top_n=3,  # Bet on top 3 predicted horses
    min_probability=0.1,
    max_bets_per_race=10
)

# Run simulation
simulator = BettingSimulator(strategy, session)
result = simulator.run_simulation(
    start_date='2024-01-01',
    end_date='2024-12-31',
    use_predictions=True  # Use ML predictions or uniform probability
)

# Access results
print(f"Recovery rate: {result.recovery_rate:.2%}")
print(f"Hit rate: {result.hit_rate:.2%}")
```

**Important**:
- Simulations require race results and payout data to be in the database
- Use `use_predictions=False` to establish a baseline with uniform probabilities
- Results include per-bet-type statistics and time-series data
- Web interface available at `/simulation/` for interactive simulations

### 7. Prediction Automation

**Automatic Race Scraping + Prediction**:
The `predict_upcoming_races.py` script combines scraping and prediction in one workflow:

```python
# Scrapes upcoming races and generates predictions automatically
python scripts/predict_upcoming_races.py --days-ahead 3
```

This is the **recommended** approach for production use because it:
- Automatically fetches race data for future dates
- Extracts features and generates predictions
- Saves everything to database
- No manual data preparation needed

**Manual Prediction** (for existing DB races):
```python
python scripts/generate_predictions.py \
  --model-path data/models/model.pkl \
  --race-id 123 \
  --save-to-db
```

### 8. Model Retraining System

**Automated Model Retraining** with version control and performance comparison:

```bash
# Run retraining once
python scripts/model_retraining_scheduler.py --mode once

# Schedule weekly retraining (recommended for production)
python scripts/model_retraining_scheduler.py --mode schedule --interval weekly --time 02:00
```

**Key Features**:
- Automatic data extraction from database (configurable time window)
- Model versioning with timestamps
- Performance comparison with previous models
- Auto-deployment only if new model improves by threshold (default: 5%)
- Model registry tracking in `data/models/model_registry.json`
- Keeps last N models (default: 5), auto-cleanup of old models
- Optional email/Slack notifications

**Configuration** (`config/retraining_config.json`):
- `models_to_train`: Which models to retrain (random_forest, xgboost)
- `training_window_days`: How much historical data to use
- `performance_threshold`: Minimum improvement to deploy new model
- `keep_last_n_models`: Model history retention

### 9. Odds Update Mechanism

**Real-time Odds Updates** for races:

```bash
# Update odds for today's races
python scripts/update_odds.py

# Update odds for specific date
python scripts/update_odds.py --date 2026-01-11

# Schedule periodic updates (every 30 minutes on race days)
python scripts/odds_update_scheduler.py --schedule daily --time 09:00
```

**How it works**:
- Scrapes latest odds from netkeiba.com
- Updates `morning_odds` field in `RaceEntry` table
- Only updates races that haven't started yet
- Used by prediction models for more accurate probabilities

## Data Flow Patterns

### Scraping → Database Pipeline
```python
# 1. Scrape with JRAScraper context manager
with JRAScraper(delay=3) as scraper:
    # Get race calendar (includes CNAME)
    races = scraper.scrape_race_calendar(date)

    for race_info in races:
        # Scrape race card
        race_card = scraper.scrape_race_card(race_info['jra_race_id'], cname=race_info['cname'])

        # Scrape race result (only for completed races)
        race_result = scraper.scrape_race_result(race_info['jra_race_id'], cname=race_info['cname'])

# 2. Save to database using dedicated functions
race = save_race_to_db(race_data)
entries = save_race_entries_to_db(race.id, race_card['entries'])
results = save_race_results_to_db(race.id, race_result['results'])  # After race completion
```

**Important**:
- `save_race_to_db()` expects track name (string), not track_id
- `save_race_entries_to_db()` expects database race.id, NOT jra_race_id
- `save_race_results_to_db()` expects database race.id and matches by horse_number
- All save functions handle create/update logic automatically
- Race results only available after race completion

### Feature Engineering Pattern (ML Pipeline)
When implementing feature engineering (`src/ml/feature_engineering.py`):
- Extract features from `race_entries` JOIN `horses` JOIN `jockeys` JOIN `trainers`
- Calculate historical stats: win rate, place rate by distance/surface/track
- Use pandas for aggregations before converting to ML features
- Store processed features in `data/processed/` directory

## Testing Approach

### Test Structure
- `tests/test_scrapers/`: Scraper unit and integration tests
- `tests/test_models/`: Database model tests
- `tests/test_web/`: Flask route tests

### Markers Usage
```python
@pytest.mark.unit
def test_parse_distance():
    # Fast, no I/O

@pytest.mark.integration
def test_scrape_real_race():
    # Hits actual JRA website

@pytest.mark.slow
def test_train_model():
    # Long-running tests
```

### Testing Scrapers
- **Unit tests**: Mock HTML responses, test parsing logic
- **Integration tests**: Use actual JRA pages (mark with `@pytest.mark.integration`)
- Keep saved HTML fixtures in `tests/fixtures/` for repeatable parsing tests
- Handle Windows `cp932` codec issues in console output with try-except blocks

### Testing Database
Always use `TestingConfig` for isolated in-memory database:
```python
from src.web.app import create_app

def test_save_race():
    app = create_app('testing')
    with app.app_context():
        # test database operations
```

## Common Gotchas

### 1. Unicode Handling on Windows
Windows console uses `cp932` codec which cannot display all Japanese characters. Wrap print statements:
```python
try:
    print(f"馬名: {horse_name}")
except UnicodeEncodeError:
    print("馬名: [Unicode characters]")
```

### 2. JRA HTML Structure Changes
JRA may change HTML structure without notice. When scraping fails:
- Use `debug_html.py` pattern to save actual HTML to file
- Inspect saved HTML to identify new selectors
- Update selectors in `scrape_race_card()` accordingly

### 3. CNAME Dependency
Race card and result scraping **requires** CNAME parameter:
```python
# ✓ Correct - scrape calendar first to get CNAME
races = scraper.scrape_race_calendar(date)
for race in races:
    race_card = scraper.scrape_race_card(race['jra_race_id'], cname=race['cname'])
    race_result = scraper.scrape_race_result(race['jra_race_id'], cname=race['cname'])

# ✗ Wrong - missing CNAME will fail
race_card = scraper.scrape_race_card(race_id)  # Will use fallback URL that doesn't work
```

**CNAME URL Patterns**:
- Race card (出馬表): `/JRADB/accessD.html?CNAME={cname}` - Pattern: `pw01dde*`
- Race result (結果): `/JRADB/accessS.html?CNAME={cname}` - Pattern: `pw01sde*` or `pw01sli*`
- Horse profile: `/JRADB/accessS.html?CNAME={cname}` - Pattern varies by entity type

### 4. Foreign Key Order
When saving data, respect foreign key dependencies:
1. Track (no dependencies)
2. Horse, Jockey, Trainer (no dependencies)
3. Race (depends on Track)
4. RaceEntry (depends on Race, Horse, Jockey)
5. RaceResult (depends on RaceEntry)
6. Prediction (depends on Race, Horse)

Use the provided `save_race_to_db()` and `save_race_entries_to_db()` functions - they handle this automatically.

## Legal & Ethical Requirements

- **MANDATORY**: Set `SCRAPING_DELAY >= 2` seconds (default: 3s)
- **MANDATORY**: Set proper `USER_AGENT` header
- **MANDATORY**: Respect JRA's `robots.txt`
- **NEVER**: Implement automatic betting/wagering features
- **NEVER**: Commercialize JRA data without permission
- **ALWAYS**: Include disclaimer that predictions are for reference only

## Current Implementation Status

**Completed**:
- ✅ Database models and migrations (UTF-8 support)
- ✅ Netkeiba.com scraper (race calendar, cards, results, horse profiles)
- ✅ Database save pipeline with get-or-create pattern
- ✅ Race result and payout scraping (`scrape_race_result()`, `save_payouts_to_db()`)
- ✅ Horse profile scraping with pedigree data (`scrape_horse_profile()`)
- ✅ Feature engineering (40+ features) (`src/ml/feature_engineering.py`)
- ✅ ML model training (RandomForest, XGBoost) (`src/ml/models/`)
- ✅ Model evaluation and metrics (`src/ml/evaluation.py`)
- ✅ Prediction generation (`scripts/generate_predictions.py`)
- ✅ Automated prediction for upcoming races (`scripts/predict_upcoming_races.py`)
- ✅ **Betting simulation engine** with 7 bet types (`src/ml/betting_simulator.py`)
- ✅ **Web interface with simulation UI** (`/simulation/`, `/simulation/history`)
- ✅ Complete REST API endpoints (`/api/races`, `/api/predictions`, `/api/simulation/run`)
- ✅ Performance optimization and caching (Flask-Caching)
- ✅ **Model retraining scheduler** with auto-deployment (`scripts/model_retraining_scheduler.py`)
- ✅ **Odds update automation** (`scripts/odds_update_scheduler.py`)
- ✅ Task scheduling system (`scripts/scheduler.py`)
- ✅ Configuration and logging system
- ✅ Comprehensive test suite (scrapers, models, ML, API)

**Pending/Future Enhancements**:
- ⏳ User authentication and authorization
- ⏳ Favorites/bookmarking feature
- ⏳ Mobile app (React Native/Flutter)
- ⏳ Advanced visualization (charts, graphs)
- ⏳ Deployment automation (Docker/Kubernetes)
- ⏳ Real-time notifications (push, email)
- ⏳ Multi-model ensemble predictions

## Key Files Reference

### Core Implementation
- `src/scrapers/netkeiba_scraper.py` - Main scraper (netkeiba.com, EUC-JP)
- `src/scrapers/jra_scraper.py` - Legacy scraper (DEPRECATED - CNAME issues)
- `src/data/database.py` - Get-or-create functions and save logic
- `src/data/models.py` - SQLAlchemy ORM models (14 tables including SimulationRun/SimulationBet)
- `config/settings.py` - Environment configurations
- `config/retraining_config.json` - Model retraining configuration

### Machine Learning
- `src/ml/feature_engineering.py` - Feature extraction (40+ features)
- `src/ml/preprocessing.py` - Data preprocessing and scaling
- `src/ml/evaluation.py` - Model evaluation metrics
- `src/ml/betting_simulator.py` - Betting simulation engine (7 bet types)
- `src/ml/models/base_model.py` - Abstract base model class
- `src/ml/models/random_forest.py` - RandomForest implementation
- `src/ml/models/xgboost_model.py` - XGBoost implementation

### Web Application
- `src/web/app.py` - Flask application factory
- `src/web/cache.py` - Caching configuration
- `src/web/routes/main.py` - Main pages (home, races, race detail)
- `src/web/routes/predictions.py` - Prediction display pages
- `src/web/routes/entities.py` - Horse/Jockey/Trainer pages
- `src/web/routes/search.py` - Search functionality
- `src/web/routes/api.py` - REST API endpoints
- `src/web/routes/simulation.py` - Betting simulation UI and API

### Scripts
- `scripts/scrape_data.py` - Basic race data scraping
- `scripts/scrape_and_save_results.py` - Complete workflow: scrape races, cards, and results for a single date
- `scripts/scrape_historical_data.py` - Bulk scraping for multiple years (weekends only by default)
- `scripts/extract_features.py` - Feature extraction for ML training
- `scripts/train_model.py` - Model training workflow
- `scripts/generate_predictions.py` - Prediction generation for existing races
- `scripts/predict_upcoming_races.py` - Auto-scrape + predict upcoming races (recommended)
- `scripts/run_betting_simulation.py` - CLI betting simulation
- `scripts/model_retraining_scheduler.py` - Automated model retraining with versioning
- `scripts/update_odds.py` - Odds update utility
- `scripts/odds_update_scheduler.py` - Scheduled odds updates
- `scripts/scheduler.py` - General task scheduler
- `scripts/init_db.py` - Database initialization

### Tests
- `tests/test_scrapers/test_netkeiba_scraper.py` - Tests for netkeiba scraper
- `tests/test_scrapers/test_jra_scraper.py` - Tests for legacy JRA scraper
- `tests/test_models/` - Database model tests
- `tests/test_ml/test_betting_simulator.py` - Betting simulation tests
- `tests/test_api/` - API endpoint tests

### Documentation
- `RequirementsDefinition.md` - Detailed requirements and data model specifications
- `CLAUDE.md` - This file - development guidelines and patterns
