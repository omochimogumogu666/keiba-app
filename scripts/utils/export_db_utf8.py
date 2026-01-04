"""Export database contents to UTF-8 file to verify encoding."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.web.app import create_app
from src.data.models import db, Track, Race, Horse, Jockey, Trainer

app = create_app('development')

with app.app_context():
    output_file = "data/database_check.txt"
    os.makedirs("data", exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("DATABASE CONTENT VERIFICATION (UTF-8)\n")
        f.write("="*80 + "\n\n")

        f.write(f"Track Names ({db.session.query(Track).count()} total):\n")
        for track in db.session.query(Track).all():
            f.write(f"  - {track.name}\n")

        f.write(f"\nRace Names ({db.session.query(Race).count()} total):\n")
        for race in db.session.query(Race).all():
            f.write(f"  - {race.race_name} ({race.race_date})\n")

        f.write(f"\nHorse Names ({db.session.query(Horse).count()} total):\n")
        for horse in db.session.query(Horse).limit(10).all():
            f.write(f"  - {horse.name} (ID: {horse.jra_horse_id})\n")

        f.write(f"\nJockey Names ({db.session.query(Jockey).count()} total):\n")
        for jockey in db.session.query(Jockey).limit(10).all():
            f.write(f"  - {jockey.name} (ID: {jockey.jra_jockey_id})\n")

        f.write(f"\nTrainer Names ({db.session.query(Trainer).count()} total):\n")
        for trainer in db.session.query(Trainer).limit(10).all():
            f.write(f"  - {trainer.name} (ID: {trainer.jra_trainer_id})\n")

    print(f"Database contents exported to: {output_file}")
    print("Please open this file in a UTF-8 capable text editor to view correct Japanese characters.")
