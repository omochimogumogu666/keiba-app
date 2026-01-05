#!/usr/bin/env python
"""
Database migration management script.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flask_migrate import Migrate, init, migrate, upgrade, downgrade
from src.web.app import create_app
from src.data.models import db

# Create Flask app
app = create_app(os.getenv('FLASK_ENV', 'development'))

if __name__ == '__main__':
    with app.app_context():
        print("Flask-Migrate management script")
        print("Use: python manage.py [command]")
        print("Available commands: init, migrate, upgrade, downgrade")
