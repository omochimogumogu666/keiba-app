"""Check what data is in the database."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.web.app import create_app
from src.data.models import db, Race, RaceEntry, RaceResult, Horse, Jockey, Trainer, Track

def main():
    """Check database data."""
    app = create_app('development')

    with app.app_context():
        print("=" * 80)
        print("DATABASE CONTENTS")
        print("=" * 80)

        # Count all records
        track_count = Track.query.count()
        horse_count = Horse.query.count()
        jockey_count = Jockey.query.count()
        trainer_count = Trainer.query.count()
        race_count = Race.query.count()
        entry_count = RaceEntry.query.count()
        result_count = RaceResult.query.count()

        print(f"\nRecord counts:")
        print(f"  - Tracks: {track_count}")
        print(f"  - Horses: {horse_count}")
        print(f"  - Jockeys: {jockey_count}")
        print(f"  - Trainers: {trainer_count}")
        print(f"  - Races: {race_count}")
        print(f"  - Race Entries: {entry_count}")
        print(f"  - Race Results: {result_count}")

        # Show sample data
        if race_count > 0:
            print("\nSample race data:")
            races = Race.query.limit(3).all()
            for race in races:
                print(f"\n  Race ID: {race.id}")
                print(f"    Netkeiba ID: {race.netkeiba_race_id}")
                print(f"    Track: {race.track.name}")
                print(f"    Date: {race.race_date}")
                print(f"    Race Number: {race.race_number}")
                print(f"    Distance: {race.distance}m" if race.distance else "    Distance: Unknown")
                print(f"    Surface: {race.surface}")
                print(f"    Status: {race.status}")
                print(f"    Entries: {len(race.race_entries)}")

        if entry_count > 0:
            print("\nSample entry data (first 3):")
            entries = RaceEntry.query.limit(3).all()
            for entry in entries:
                print(f"\n  Entry ID: {entry.id}")
                print(f"    Horse: {entry.horse.name}")
                print(f"    Jockey: {entry.jockey.name}")
                print(f"    Horse Number: {entry.horse_number}")
                print(f"    Post Position: {entry.post_position}")
                print(f"    Weight: {entry.weight}kg")
                print(f"    Horse Weight: {entry.horse_weight}kg")

        print("\n" + "=" * 80)
        print("CHECK COMPLETE")
        print("=" * 80)

if __name__ == "__main__":
    main()
