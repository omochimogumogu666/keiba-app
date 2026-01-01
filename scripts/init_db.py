"""
Initialize database with tables and sample data.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask
from config.settings import get_config
from config.logging_config import setup_logging
from src.data.models import db, Track
from src.utils.logger import get_app_logger

# Setup logging
setup_logging(log_level='INFO')
logger = get_app_logger(__name__)


def create_app():
    """Create Flask application for database initialization."""
    app = Flask(__name__)
    config = get_config()
    app.config.from_object(config)
    config.init_app(app)

    # Initialize database
    db.init_app(app)

    return app


def init_database():
    """Initialize database tables."""
    app = create_app()

    with app.app_context():
        logger.info("Creating database tables...")
        db.create_all()
        logger.info("Database tables created successfully")

        # Add sample track data
        add_sample_tracks()


def add_sample_tracks():
    """Add sample track data to database."""
    logger.info("Adding sample track data...")

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


def reset_database():
    """Drop all tables and recreate them."""
    app = create_app()

    with app.app_context():
        logger.warning("Dropping all database tables...")
        db.drop_all()
        logger.info("All tables dropped")

        logger.info("Recreating database tables...")
        db.create_all()
        logger.info("Database tables recreated successfully")

        # Add sample data
        add_sample_tracks()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Initialize database')
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Drop all tables and recreate (WARNING: destroys all data)'
    )

    args = parser.parse_args()

    if args.reset:
        response = input("WARNING: This will delete all data. Continue? (yes/no): ")
        if response.lower() == 'yes':
            reset_database()
        else:
            logger.info("Reset cancelled")
    else:
        init_database()

    logger.info("Database initialization complete")
