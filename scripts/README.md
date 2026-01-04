# Scripts Directory

This directory contains various scripts for the JRA Horse Racing Prediction Application.

## Directory Structure

```
scripts/
├── README.md                      # This file
├── init_db.py                     # Database initialization
├── scrape_data.py                 # General data scraping script
├── scrape_and_save_results.py     # Scrape races, cards, and results for a single date
├── scrape_historical_data.py      # Bulk scraping for historical data (multiple years)
├── extract_features.py            # Feature extraction for ML models
├── train_model.py                 # ML model training
├── generate_predictions.py        # Generate race predictions
├── debug/                         # Debug and analysis scripts
├── tests/                         # Ad-hoc test scripts
└── utils/                         # Database and data utilities
```

## Main Scripts (Production)

### Database Setup
- **`init_db.py`**: Initialize the database schema
  ```bash
  python scripts/init_db.py
  ```

### Data Collection
- **`scrape_data.py`**: General-purpose data scraping script
  ```bash
  python scripts/scrape_data.py
  ```

- **`scrape_and_save_results.py`**: Complete workflow for a single date
  - Scrapes race calendar, race cards, and race results
  - Saves all data to database
  ```bash
  python scripts/scrape_and_save_results.py --date 20260104
  ```

- **`scrape_historical_data.py`**: Bulk historical data scraping
  - Scrapes multiple years of data (weekends only by default)
  - Useful for building training dataset
  ```bash
  python scripts/scrape_historical_data.py --start-year 2020 --end-year 2024
  ```

### Machine Learning
- **`extract_features.py`**: Extract features from database for ML training
  ```bash
  python scripts/extract_features.py
  ```

- **`train_model.py`**: Train machine learning models
  ```bash
  python scripts/train_model.py
  ```

- **`generate_predictions.py`**: Generate predictions for upcoming races
  ```bash
  python scripts/generate_predictions.py --date 20260105
  ```

## Debug Scripts (`debug/`)

Debugging and HTML analysis scripts used during development:

- `debug_*.py`: Various debugging scripts for scraper development
- `analyze_*.py`: HTML structure analysis tools
- `inspect_*.py`: HTML inspection utilities
- `extract_result_structure.py`: Race result HTML structure analyzer

**Note**: These scripts are primarily for development and troubleshooting. Most are not needed for regular operation.

## Test Scripts (`tests/`)

Ad-hoc test scripts created during development:

- `test_*.py`: Various feature-specific test scripts
- These complement the formal test suite in `tests/` (project root)

**Note**: For comprehensive testing, use pytest with the main test suite:
```bash
pytest tests/
```

## Utility Scripts (`utils/`)

Database and data management utilities:

- **`reset_database.py`**: Reset/recreate database (⚠️ destructive!)
- **`check_db_data.py`**: Verify database data integrity
- **`check_db_encoding.py`**: Check database encoding issues
- **`export_db_utf8.py`**: Export database with UTF-8 encoding
- **`analyze_db_issues.py`**: Analyze database issues
- **`find_recent_race.py`**: Find recent races in database
- **`fetch_recent_race_result.py`**: Fetch and display recent race results
- **`get_horse_cname.py`**: Extract horse CNAME from JRA (legacy)

## Usage Guidelines

### For Normal Operation
Use only the main scripts in the root `scripts/` directory:
1. Initialize database: `init_db.py`
2. Scrape data: `scrape_historical_data.py` or `scrape_and_save_results.py`
3. Train model: `extract_features.py` → `train_model.py`
4. Generate predictions: `generate_predictions.py`

### For Development/Debugging
- Use scripts in `debug/` for troubleshooting scraper issues
- Use scripts in `utils/` for database management
- Use scripts in `tests/` for quick feature testing

### For Testing
Use pytest with the main test suite:
```bash
pytest                           # Run all tests
pytest -m unit                   # Unit tests only
pytest -m integration            # Integration tests only
pytest tests/test_scrapers/      # Scraper tests only
```

## Important Notes

- **Rate Limiting**: All scraper scripts respect `SCRAPING_DELAY` (default: 3 seconds)
- **Environment**: Ensure `.env` file is configured before running scripts
- **Database**: Most scripts require database to be initialized (`init_db.py`)
- **Dependencies**: Run `pip install -r requirements.txt` before using any script

## See Also

- [CLAUDE.md](../CLAUDE.md) - Development guidelines and architecture patterns
- [README.md](../README.md) - Project overview and setup instructions
- [RequirementsDefinition.md](../RequirementsDefinition.md) - Detailed requirements
