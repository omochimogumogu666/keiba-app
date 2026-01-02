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
pytest tests/test_scrapers/test_jra_scraper.py

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

## Critical Architecture Patterns

### 1. JRA Scraping Architecture

**CNAME Parameter System**: JRA uses a complex CNAME parameter system for URLs that cannot be programmatically generated. The scraper must:
- Extract CNAMEs dynamically from JRA's race calendar links
- Handle two extraction methods:
  - URL parameters: `?CNAME=pw01dde0106202601011120260104/6C`
  - JavaScript onclick: `doAction('/JRADB/accessK.html', 'pw04kmk005386/50')`

**Key Implementation Details**:
- `_extract_cname_from_url()`: Extracts CNAME from URL query strings
- `_extract_cname_from_onclick()`: Parses CNAME from onclick attributes using regex
- `_extract_race_links()`: Finds all race links with CNAMEs from calendar pages

**HTML Structure Specifics**:
```python
# Race card table: class="basic"
# Horse info: td.horse > div.name_line > a (horse name + CNAME in href)
# Trainer info: p.trainer > a (onclick contains CNAME)
# Jockey info: p.jockey > a (onclick contains CNAME)
# Prize money: ul.prize > ol > li > span.num
```

**Rate Limiting**: Always use 3-second delay between requests (configured in `SCRAPING_DELAY`). This is both ethical and required by JRA's terms of service.

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
- ✅ Database models and migrations
- ✅ JRA scraper foundation (race calendar, race cards)
- ✅ Database save pipeline with get-or-create pattern
- ✅ Race result scraping and saving (`scrape_race_result()`, `save_race_results_to_db()`)
- ✅ Horse profile scraping (`scrape_horse_profile()`)
- ✅ Basic web interface structure
- ✅ Configuration and logging system
- ✅ Comprehensive test suite for scraping and database operations

**Pending**:
- ⏳ Feature engineering for ML (`src/ml/feature_engineering.py`)
- ⏳ ML model training and prediction (`src/ml/models/`)
- ⏳ Complete web interface with predictions display
- ⏳ REST API endpoints for external access
- ⏳ Performance optimization and caching

## Key Files Reference

### Core Implementation
- `src/scrapers/jra_scraper.py` - Main scraper implementation with CNAME handling
- `src/data/database.py` - Get-or-create functions and save logic
- `src/data/models.py` - SQLAlchemy ORM models
- `config/settings.py` - Environment configurations

### Scripts
- `scripts/scrape_and_save_results.py` - Complete workflow: scrape races, cards, and results, then save to DB
- `scripts/scrape_data.py` - General data scraping script
- `scripts/debug_race_result_html.py` - Debug script to fetch and analyze race result HTML
- `scripts/get_horse_cname.py` - Utility to extract horse CNAMEs

### Tests
- `tests/test_scrapers/test_jra_scraper.py` - Tests for race calendar and race card scraping
- `tests/test_scrapers/test_race_results.py` - Tests for race result scraping and database saving
- `tests/test_models/` - Database model tests

### Documentation
- `RequirementsDefinition.md` - Detailed requirements and data model specifications
- `CLAUDE.md` - This file - development guidelines and patterns
