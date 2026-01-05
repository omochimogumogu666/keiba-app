#!/usr/bin/env python
"""
Initialize database in Docker environment.
This script should be run inside the Docker container.
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask
from flask_migrate import upgrade, init as migrate_init, migrate as create_migration
from config.settings import get_config
from config.logging_config import setup_logging
from src.data.models import db, Track
from src.utils.logger import get_app_logger

# Setup logging
setup_logging(log_level='INFO')
logger = get_app_logger(__name__)


def create_app():
    """Create Flask application for database initialization."""
    from src.web.app import create_app as app_factory
    return app_factory(os.getenv('FLASK_ENV', 'production'))


def wait_for_db():
    """Wait for database to be ready."""
    import time
    import psycopg2

    db_url = os.getenv('DATABASE_URL')
    if not db_url or not db_url.startswith('postgresql'):
        logger.info("Not using PostgreSQL, skipping wait")
        return

    logger.info("Waiting for PostgreSQL to be ready...")
    max_retries = 30
    retry_count = 0

    while retry_count < max_retries:
        try:
            # Try to connect to the database
            app = create_app()
            with app.app_context():
                db.engine.connect()
            logger.info("PostgreSQL is ready!")
            return
        except Exception as e:
            retry_count += 1
            logger.info(f"Database not ready (attempt {retry_count}/{max_retries}): {e}")
            time.sleep(1)

    raise Exception("Database did not become ready in time")


def run_migrations():
    """Run database migrations."""
    app = create_app()

    with app.app_context():
        logger.info("Running database migrations...")

        # Check if migrations directory exists
        migrations_dir = Path(__file__).parent.parent / 'migrations'

        if not migrations_dir.exists():
            logger.info("Migrations directory not found. Creating initial migration...")
            migrate_init()
            create_migration(message="Initial database schema")

        # Run migrations
        upgrade()
        logger.info("Database migrations completed successfully")


def add_sample_tracks():
    """Add sample track data to database."""
    logger.info("Adding sample track data...")

    app = create_app()
    with app.app_context():
        # JRA major racetracks
        tracks = [
            {'name': '東京', 'location': '東京都府中市', 'surface_types': ['turf', 'dirt']},
            {'name': '中山', 'location': '千葉県船橋市', 'surface_types': ['turf', 'dirt']},
            {'name': '京都', 'location': '京都府京都市', 'surface_types': ['turf', 'dirt']},
            {'name': '阪神', 'location': '兵庫県宝塚市', 'surface_types': ['turf', 'dirt']},
            {'name': '中京', 'location': '愛知県豊明市', 'surface_types': ['turf', 'dirt']},
            {'name': '新潟', 'location': '新潟県新潟市', 'surface_types': ['turf', 'dirt']},
            {'name': '福島', 'location': '福島県福島市', 'surface_types': ['turf', 'dirt']},
            {'name': '札幌', 'location': '北海道札幌市', 'surface_types': ['turf', 'dirt']},
            {'name': '函館', 'location': '北海道函館市', 'surface_types': ['turf']},
            {'name': '小倉', 'location': '福岡県北九州市', 'surface_types': ['turf', 'dirt']},
        ]

        for track_data in tracks:
            # Check if track already exists
            existing_track = Track.query.filter_by(name=track_data['name']).first()
            if not existing_track:
                track = Track(**track_data)
                db.session.add(track)
                logger.info(f"Added track: {track_data['name']}")
            else:
                logger.info(f"Track already exists: {track_data['name']}")

        db.session.commit()
        logger.info(f"Sample track data added: {len(tracks)} tracks")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Initialize database in Docker')
    parser.add_argument(
        '--skip-sample-data',
        action='store_true',
        help='Skip adding sample track data'
    )

    args = parser.parse_args()

    try:
        # Wait for database to be ready
        wait_for_db()

        # Run migrations
        run_migrations()

        # Add sample data
        if not args.skip_sample_data:
            add_sample_tracks()

        logger.info("Database initialization complete")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        sys.exit(1)
