"""Check database for encoding issues."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.web.app import create_app
from src.data.models import db, Track, Race, Horse, Jockey, Trainer, RaceEntry

app = create_app('development')

with app.app_context():
    print("Database Statistics:")
    print(f"Tracks: {db.session.query(Track).count()}")
    print(f"Races: {db.session.query(Race).count()}")
    print(f"Horses: {db.session.query(Horse).count()}")
    print(f"Jockeys: {db.session.query(Jockey).count()}")
    print(f"Trainers: {db.session.query(Trainer).count()}")
    print(f"Race Entries: {db.session.query(RaceEntry).count()}")

    print("\nSample Track Names:")
    for track in db.session.query(Track).limit(5).all():
        try:
            print(f"  - {track.name}")
        except UnicodeEncodeError:
            print(f"  - [Encoding issue] ID: {track.id}")

    print("\nSample Horse Names:")
    for horse in db.session.query(Horse).limit(5).all():
        try:
            print(f"  - {horse.name}")
        except UnicodeEncodeError:
            print(f"  - [Encoding issue] ID: {horse.id}, JRA ID: {horse.jra_horse_id}")

    print("\nSample Jockey Names:")
    for jockey in db.session.query(Jockey).limit(5).all():
        try:
            print(f"  - {jockey.name}")
        except UnicodeEncodeError:
            print(f"  - [Encoding issue] ID: {jockey.id}, JRA ID: {jockey.jra_jockey_id}")

    print("\nRaw encoding check (first horse):")
    first_horse = db.session.query(Horse).first()
    if first_horse:
        print(f"  Name type: {type(first_horse.name)}")
        print(f"  Name repr: {repr(first_horse.name)}")
        print(f"  Name bytes (UTF-8): {first_horse.name.encode('utf-8') if first_horse.name else 'None'}")
