"""Analyze database issues in detail."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.web.app import create_app
from src.data.models import db, Race, RaceEntry, RaceResult, Horse, Jockey, Trainer, Track
from sqlalchemy import func, and_

def main():
    """Analyze database for issues."""
    app = create_app('development')

    with app.app_context():
        print("=" * 80)
        print("DATABASE ISSUE ANALYSIS")
        print("=" * 80)

        # Issue 1: NULL fields analysis
        print("\n1. NULL FIELD ANALYSIS")
        print("-" * 80)

        races = Race.query.all()
        if races:
            null_counts = {
                'race_name': 0,
                'distance': 0,
                'surface': 0,
                'track_condition': 0,
                'weather': 0,
                'race_class': 0,
                'course_type': 0,
                'track_variant': 0
            }

            for race in races:
                if not race.race_name: null_counts['race_name'] += 1
                if not race.distance: null_counts['distance'] += 1
                if not race.surface: null_counts['surface'] += 1
                if not race.track_condition: null_counts['track_condition'] += 1
                if not race.weather: null_counts['weather'] += 1
                if not race.race_class: null_counts['race_class'] += 1
                if not race.course_type: null_counts['course_type'] += 1
                if not race.track_variant: null_counts['track_variant'] += 1

            print(f"\nRaces table (total: {len(races)}):")
            for field, count in null_counts.items():
                percentage = (count / len(races)) * 100
                print(f"  {field}: {count} NULL ({percentage:.1f}%)")

        # Horse NULL analysis
        horses = Horse.query.all()
        if horses:
            horse_null_counts = {
                'netkeiba_horse_id': 0,
                'birth_date': 0,
                'sex': 0,
                'sire_id': 0,
                'dam_id': 0,
                'trainer_id': 0
            }

            for horse in horses:
                if not horse.netkeiba_horse_id: horse_null_counts['netkeiba_horse_id'] += 1
                if not horse.birth_date: horse_null_counts['birth_date'] += 1
                if not horse.sex: horse_null_counts['sex'] += 1
                if not horse.sire_id: horse_null_counts['sire_id'] += 1
                if not horse.dam_id: horse_null_counts['dam_id'] += 1
                if not horse.trainer_id: horse_null_counts['trainer_id'] += 1

            print(f"\nHorses table (total: {len(horses)}):")
            for field, count in horse_null_counts.items():
                percentage = (count / len(horses)) * 100
                print(f"  {field}: {count} NULL ({percentage:.1f}%)")

        # Issue 2: Duplicate analysis
        print("\n\n2. DUPLICATE DATA ANALYSIS")
        print("-" * 80)

        # Check for duplicate horses by name
        duplicate_horses = db.session.query(
            Horse.name, func.count(Horse.id)
        ).group_by(Horse.name).having(func.count(Horse.id) > 1).all()

        if duplicate_horses:
            print(f"\nDuplicate horses by name: {len(duplicate_horses)}")
            for name, count in duplicate_horses[:5]:
                print(f"  '{name}': {count} entries")
                horses_with_name = Horse.query.filter_by(name=name).all()
                for h in horses_with_name:
                    print(f"    - ID: {h.id}, Netkeiba ID: {h.netkeiba_horse_id}")
        else:
            print("\nNo duplicate horses found")

        # Check for duplicate tracks
        duplicate_tracks = db.session.query(
            Track.name, func.count(Track.id)
        ).group_by(Track.name).having(func.count(Track.id) > 1).all()

        if duplicate_tracks:
            print(f"\nDuplicate tracks: {len(duplicate_tracks)}")
            for name, count in duplicate_tracks:
                print(f"  '{name}': {count} entries")
        else:
            print("\nNo duplicate tracks found (UNIQUE constraint working)")

        # Issue 3: Race results analysis
        print("\n\n3. RACE RESULTS ANALYSIS")
        print("-" * 80)

        total_entries = RaceEntry.query.count()
        entries_with_results = db.session.query(RaceEntry).join(
            RaceResult, RaceEntry.id == RaceResult.race_entry_id
        ).count()

        print(f"\nTotal race entries: {total_entries}")
        print(f"Entries with results: {entries_with_results}")
        print(f"Entries WITHOUT results: {total_entries - entries_with_results}")

        if total_entries > 0:
            percentage = (entries_with_results / total_entries) * 100
            print(f"Completion rate: {percentage:.1f}%")

        # Sample a race entry to see what data we have
        sample_entry = RaceEntry.query.first()
        if sample_entry:
            print(f"\nSample entry (ID: {sample_entry.id}):")
            print(f"  Horse: {sample_entry.horse.name}")
            print(f"  Jockey: {sample_entry.jockey.name}")
            print(f"  Post position: {sample_entry.post_position}")
            print(f"  Horse number: {sample_entry.horse_number}")
            print(f"  Weight: {sample_entry.weight}kg")
            print(f"  Horse weight: {sample_entry.horse_weight}kg ({sample_entry.horse_weight_change:+d}kg)" if sample_entry.horse_weight_change else f"  Horse weight: {sample_entry.horse_weight}kg")
            print(f"  Morning odds: {sample_entry.morning_odds}")
            print(f"  Has result: {sample_entry.result is not None}")

        # Issue 4: ID meaningfulness
        print("\n\n4. ID USAGE ANALYSIS")
        print("-" * 80)

        print("\nCurrent ID strategy:")
        print("  - Auto-increment integer IDs (database-generated)")
        print("  - Separate netkeiba_race_id for external reference")
        print("  - Separate netkeiba_horse_id, netkeiba_jockey_id for external reference")

        sample_race = Race.query.first()
        if sample_race:
            print(f"\nSample Race:")
            print(f"  Internal ID: {sample_race.id} (database auto-increment)")
            print(f"  Netkeiba ID: {sample_race.netkeiba_race_id} (external reference)")
            print(f"  Usage: Internal ID for foreign keys, Netkeiba ID for deduplication")

        # Check ID usage in relationships
        print(f"\nRelationship usage:")
        if sample_race:
            print(f"  Race {sample_race.id} has {len(sample_race.race_entries)} entries")
            print(f"  - Entries reference race via race_id={sample_race.id}")

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)

if __name__ == "__main__":
    main()
