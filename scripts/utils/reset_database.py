"""Reset database to clean state."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.web.app import create_app
from src.data.models import db

def reset_database():
    """Drop all tables and recreate them."""
    app = create_app('development')

    with app.app_context():
        print("Resetting database...")
        print("WARNING: This will delete ALL data!")

        # Drop all tables
        print("Dropping all tables...")
        db.drop_all()
        print("[OK] All tables dropped")

        # Recreate all tables
        print("Creating tables...")
        db.create_all()
        print("[OK] All tables created")

        print("\nDatabase has been reset successfully!")

if __name__ == "__main__":
    reset_database()
