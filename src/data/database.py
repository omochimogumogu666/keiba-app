"""
Database connection and session management.

Provides get-or-create functions and save operations for scraped data.
"""
from contextlib import contextmanager
from datetime import datetime, time
from typing import Any, Dict, List, Optional, TypeVar

from sqlalchemy.exc import SQLAlchemyError

from src.data.models import db
from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)

# Type variable for generic model types
T = TypeVar('T')


# Constants for race status
class RaceStatus:
    """Race status constants."""
    UPCOMING = 'upcoming'
    IN_PROGRESS = 'in_progress'
    COMPLETED = 'completed'
    CANCELLED = 'cancelled'


def init_db(app) -> None:
    """
    Initialize database with Flask app.

    Args:
        app: Flask application instance
    """
    db.init_app(app)
    with app.app_context():
        db.create_all()
        logger.info("Database initialized successfully")


def _parse_post_time(post_time_value: Any) -> Optional[time]:
    """
    Parse post time from various formats.

    Args:
        post_time_value: Post time as string ('HH:MM') or time object

    Returns:
        time object or None if parsing fails
    """
    if not post_time_value:
        return None

    if isinstance(post_time_value, time):
        return post_time_value

    if isinstance(post_time_value, str):
        try:
            return datetime.strptime(post_time_value, '%H:%M').time()
        except ValueError:
            logger.warning(f"Invalid post_time format: {post_time_value}")
            return None

    return None


@contextmanager
def get_session():
    """
    Get database session context manager.

    Usage:
        with get_session() as session:
            session.query(Horse).all()

    Note:
        Flask-SQLAlchemy manages session lifecycle, so we don't close
        the session explicitly. The session is scoped to the request.
    """
    try:
        yield db.session
        db.session.commit()
    except SQLAlchemyError as e:
        db.session.rollback()
        logger.error(f"Database session error: {e}")
        raise


def save_to_db(obj: T) -> T:
    """
    Save object to database.

    Args:
        obj: SQLAlchemy model instance

    Returns:
        Saved object

    Raises:
        SQLAlchemyError: If database operation fails
    """
    try:
        db.session.add(obj)
        db.session.commit()
        logger.debug(f"Saved to database: {obj}")
        return obj
    except SQLAlchemyError as e:
        db.session.rollback()
        logger.error(f"Error saving to database: {e}")
        raise


def bulk_save_to_db(objects: List[T]) -> int:
    """
    Bulk save objects to database.

    Args:
        objects: List of SQLAlchemy model instances

    Returns:
        Number of objects saved

    Raises:
        SQLAlchemyError: If database operation fails
    """
    try:
        db.session.bulk_save_objects(objects)
        db.session.commit()
        count = len(objects)
        logger.info(f"Bulk saved {count} objects to database")
        return count
    except SQLAlchemyError as e:
        db.session.rollback()
        logger.error(f"Error bulk saving to database: {e}")
        raise


def delete_from_db(obj: T) -> None:
    """
    Delete object from database.

    Args:
        obj: SQLAlchemy model instance

    Raises:
        SQLAlchemyError: If database operation fails
    """
    try:
        db.session.delete(obj)
        db.session.commit()
        logger.debug(f"Deleted from database: {obj}")
    except SQLAlchemyError as e:
        db.session.rollback()
        logger.error(f"Error deleting from database: {e}")
        raise


# Get or create functions

def get_or_create_track(name, **kwargs):
    """
    Get or create a track by name.

    Args:
        name: Track name
        **kwargs: Additional track attributes

    Returns:
        Track instance
    """
    from src.data.models import Track

    track = Track.query.filter_by(name=name).first()
    if not track:
        track = Track(name=name, **kwargs)
        db.session.add(track)
        db.session.commit()
        logger.info(f"Created new track: {name}")
    return track


def get_or_create_horse(netkeiba_horse_id, name=None, **kwargs):
    """
    Get or create a horse by netkeiba ID.

    Args:
        netkeiba_horse_id: Netkeiba horse ID (required)
        name: Horse name
        **kwargs: Additional horse attributes

    Returns:
        Horse instance
    """
    from src.data.models import Horse

    horse = Horse.query.filter_by(netkeiba_horse_id=netkeiba_horse_id).first()

    if not horse:
        horse = Horse(
            netkeiba_horse_id=netkeiba_horse_id,
            name=name,
            **kwargs
        )
        db.session.add(horse)
        db.session.commit()
        logger.info(f"Created new horse: {name} (netkeiba:{netkeiba_horse_id})")
    return horse


def get_or_create_jockey(netkeiba_jockey_id, name=None):
    """
    Get or create a jockey by netkeiba ID.

    Args:
        netkeiba_jockey_id: Netkeiba jockey ID (required)
        name: Jockey name

    Returns:
        Jockey instance
    """
    from src.data.models import Jockey

    jockey = Jockey.query.filter_by(netkeiba_jockey_id=netkeiba_jockey_id).first()

    if not jockey:
        jockey = Jockey(
            netkeiba_jockey_id=netkeiba_jockey_id,
            name=name
        )
        db.session.add(jockey)
        db.session.commit()
        logger.info(f"Created new jockey: {name} (netkeiba:{netkeiba_jockey_id})")
    return jockey


def get_or_create_trainer(netkeiba_trainer_id, name=None, **kwargs):
    """
    Get or create a trainer by netkeiba ID.

    Args:
        netkeiba_trainer_id: Netkeiba trainer ID (required)
        name: Trainer name
        **kwargs: Additional trainer attributes

    Returns:
        Trainer instance
    """
    from src.data.models import Trainer

    trainer = Trainer.query.filter_by(netkeiba_trainer_id=netkeiba_trainer_id).first()

    if not trainer:
        trainer = Trainer(
            netkeiba_trainer_id=netkeiba_trainer_id,
            name=name,
            **kwargs
        )
        db.session.add(trainer)
        db.session.commit()
        logger.info(f"Created new trainer: {name} (netkeiba:{netkeiba_trainer_id})")
    return trainer


# Scraped data saving functions

def save_race_to_db(race_data: Dict[str, Any]) -> 'Race':
    """
    Save race data to database.

    Args:
        race_data: Dictionary with race information. Required keys:
            - 'netkeiba_race_id': Unique race identifier
            - 'track': Track name (string)
            - 'race_date': Race date

    Returns:
        Race instance

    Raises:
        KeyError: If required keys are missing
        SQLAlchemyError: If database operation fails
    """
    from src.data.models import Race

    # Validate required fields
    required_keys = ['netkeiba_race_id', 'track', 'race_date']
    for key in required_keys:
        if key not in race_data:
            raise KeyError(f"Missing required key in race_data: {key}")

    # Get or create track
    track = get_or_create_track(race_data['track'])

    # Parse post_time using helper function
    post_time = _parse_post_time(race_data.get('post_time'))

    # Check if race already exists by netkeiba ID
    race = Race.query.filter_by(netkeiba_race_id=race_data['netkeiba_race_id']).first()

    if not race:
        race = Race(
            netkeiba_race_id=race_data['netkeiba_race_id'],
            track_id=track.id,
            race_date=race_data['race_date'],
            post_time=post_time,
            race_number=race_data.get('race_number', 1),
            race_name=race_data.get('race_name'),
            distance=race_data.get('distance'),
            surface=race_data.get('surface', 'turf'),
            race_class=race_data.get('race_class'),
            status=race_data.get('status', RaceStatus.UPCOMING),
            kaisai_code=race_data.get('kaisai_code'),
            meeting_number=race_data.get('meeting_number'),
            day_number=race_data.get('day_number'),
            course_type=race_data.get('course_type'),
            track_variant=race_data.get('track_variant'),
            weather=race_data.get('weather'),
            track_condition=race_data.get('track_condition'),
            prize_money=race_data.get('prize_money')
        )
        db.session.add(race)
        db.session.commit()
        logger.info(f"Created new race: {race_data['netkeiba_race_id']}")
    else:
        # Update existing race
        race.track_id = track.id
        race.race_date = race_data['race_date']
        if post_time:
            race.post_time = post_time
        race.race_number = race_data.get('race_number', race.race_number)
        race.race_name = race_data.get('race_name', race.race_name)
        race.distance = race_data.get('distance', race.distance)
        race.surface = race_data.get('surface', race.surface)
        race.race_class = race_data.get('race_class', race.race_class)
        race.status = race_data.get('status', race.status)
        race.kaisai_code = race_data.get('kaisai_code', race.kaisai_code)
        race.meeting_number = race_data.get('meeting_number', race.meeting_number)
        race.day_number = race_data.get('day_number', race.day_number)
        race.course_type = race_data.get('course_type', race.course_type)
        race.track_variant = race_data.get('track_variant', race.track_variant)
        race.weather = race_data.get('weather', race.weather)
        race.track_condition = race_data.get('track_condition', race.track_condition)
        race.prize_money = race_data.get('prize_money', race.prize_money)
        db.session.commit()
        logger.info(f"Updated race: {race_data['netkeiba_race_id']}")

    return race


def save_race_entries_to_db(race_id, entries):
    """
    Save race entries (horses) to database.

    Args:
        race_id: Database race ID (not netkeiba ID)
        entries: List of entry dictionaries

    Returns:
        List of RaceEntry instances
    """
    from src.data.models import RaceEntry

    race_entries = []

    for entry_data in entries:
        # Validate required fields
        if not entry_data.get('netkeiba_horse_id'):
            logger.warning(f"Entry missing horse ID, skipping: {entry_data}")
            continue

        if not entry_data.get('netkeiba_jockey_id'):
            logger.warning(f"Entry missing jockey ID for horse {entry_data.get('horse_name')}, skipping")
            continue

        if not entry_data.get('netkeiba_trainer_id'):
            logger.warning(f"Entry missing trainer ID for horse {entry_data.get('horse_name')}, skipping")
            continue

        # Get or create horse, jockey, trainer using netkeiba IDs
        horse = get_or_create_horse(
            netkeiba_horse_id=entry_data['netkeiba_horse_id'],
            name=entry_data.get('horse_name')
        )

        jockey = get_or_create_jockey(
            netkeiba_jockey_id=entry_data['netkeiba_jockey_id'],
            name=entry_data.get('jockey_name', 'Unknown')
        )

        trainer = get_or_create_trainer(
            netkeiba_trainer_id=entry_data['netkeiba_trainer_id'],
            name=entry_data.get('trainer_name', 'Unknown')
        )

        # Check if entry already exists
        entry = RaceEntry.query.filter_by(
            race_id=race_id,
            horse_id=horse.id
        ).first()

        if not entry:
            entry = RaceEntry(
                race_id=race_id,
                horse_id=horse.id,
                jockey_id=jockey.id,
                post_position=entry_data.get('post_position'),
                horse_number=entry_data.get('horse_number'),
                weight=entry_data.get('weight'),
                horse_weight=entry_data.get('horse_weight'),
                horse_weight_change=entry_data.get('horse_weight_change'),
                morning_odds=entry_data.get('morning_odds')
            )
            db.session.add(entry)
            logger.debug(f"Created entry for horse {entry_data['horse_name']} in race {race_id}")
        else:
            # Update existing entry
            entry.jockey_id = jockey.id
            entry.post_position = entry_data.get('post_position', entry.post_position)
            entry.horse_number = entry_data.get('horse_number', entry.horse_number)
            entry.weight = entry_data.get('weight', entry.weight)
            entry.horse_weight = entry_data.get('horse_weight', entry.horse_weight)
            entry.horse_weight_change = entry_data.get('horse_weight_change', entry.horse_weight_change)
            entry.morning_odds = entry_data.get('morning_odds', entry.morning_odds)
            logger.debug(f"Updated entry for horse {entry_data['horse_name']} in race {race_id}")

        race_entries.append(entry)

    db.session.commit()
    logger.info(f"Saved {len(race_entries)} entries for race {race_id}")

    return race_entries


def save_race_results_to_db(race_id, results):
    """
    Save race results to database.

    Args:
        race_id: Database race ID (not netkeiba ID)
        results: List of result dictionaries

    Returns:
        List of RaceResult instances
    """
    from src.data.models import RaceEntry, RaceResult

    race_results = []

    for result_data in results:
        # Find the race entry by horse number
        entry = RaceEntry.query.filter_by(
            race_id=race_id,
            horse_number=result_data['horse_number']
        ).first()

        if not entry:
            logger.warning(f"Race entry not found for horse number {result_data['horse_number']} in race {race_id}")
            continue

        # Check if result already exists
        result = RaceResult.query.filter_by(race_entry_id=entry.id).first()

        if not result:
            result = RaceResult(
                race_entry_id=entry.id,
                finish_position=result_data.get('finish_position'),
                finish_time=result_data.get('finish_time'),
                margin=result_data.get('margin'),
                final_odds=result_data.get('final_odds'),
                popularity=result_data.get('popularity'),
                running_positions=result_data.get('running_positions'),
                comment=result_data.get('comment')
            )
            db.session.add(result)
            logger.debug(f"Created result for entry {entry.id}")
        else:
            # Update existing result
            result.finish_position = result_data.get('finish_position', result.finish_position)
            result.finish_time = result_data.get('finish_time', result.finish_time)
            result.margin = result_data.get('margin', result.margin)
            result.final_odds = result_data.get('final_odds', result.final_odds)
            result.popularity = result_data.get('popularity', result.popularity)
            result.running_positions = result_data.get('running_positions', result.running_positions)
            result.comment = result_data.get('comment', result.comment)
            logger.debug(f"Updated result for entry {entry.id}")

        race_results.append(result)

    db.session.commit()
    logger.info(f"Saved {len(race_results)} results for race {race_id}")

    return race_results


def save_horse_profile_to_db(profile):
    """
    Save horse profile to database, including pedigree and trainer information.

    Args:
        profile: Dictionary with horse profile information from scraper

    Returns:
        Horse instance
    """
    from src.data.models import Horse

    # First, handle sire if present
    sire = None
    if profile.get('sire_id') and profile.get('sire_name'):
        sire = get_or_create_horse(
            netkeiba_horse_id=profile['sire_id'],
            name=profile['sire_name']
        )

    # Handle dam if present
    dam = None
    if profile.get('dam_id') and profile.get('dam_name'):
        dam = get_or_create_horse(
            netkeiba_horse_id=profile['dam_id'],
            name=profile['dam_name']
        )

    # Handle trainer if present
    trainer = None
    if profile.get('trainer_id') and profile.get('trainer_name'):
        trainer = get_or_create_trainer(
            netkeiba_trainer_id=profile['trainer_id'],
            name=profile['trainer_name']
        )

    # Get or create the main horse
    horse = Horse.query.filter_by(netkeiba_horse_id=profile['netkeiba_horse_id']).first()

    if not horse:
        # Create new horse with all available information
        horse = Horse(
            netkeiba_horse_id=profile['netkeiba_horse_id'],
            name=profile.get('name'),
            birth_date=profile.get('birth_date'),
            sex=profile.get('sex'),
            sire_id=sire.id if sire else None,
            dam_id=dam.id if dam else None,
            trainer_id=trainer.id if trainer else None
        )
        db.session.add(horse)
        logger.info(f"Created new horse profile: {profile.get('name')} ({profile['netkeiba_horse_id']})")
    else:
        # Update existing horse with new information
        updated = False

        if profile.get('name') and horse.name != profile['name']:
            horse.name = profile['name']
            updated = True

        if profile.get('birth_date') and horse.birth_date != profile['birth_date']:
            horse.birth_date = profile['birth_date']
            updated = True

        if profile.get('sex') and horse.sex != profile['sex']:
            horse.sex = profile['sex']
            updated = True

        if sire and horse.sire_id != sire.id:
            horse.sire_id = sire.id
            updated = True

        if dam and horse.dam_id != dam.id:
            horse.dam_id = dam.id
            updated = True

        if trainer and horse.trainer_id != trainer.id:
            horse.trainer_id = trainer.id
            updated = True

        if updated:
            logger.info(f"Updated horse profile: {profile.get('name')} ({profile['netkeiba_horse_id']})")
        else:
            logger.debug(f"Horse profile unchanged: {profile.get('name')} ({profile['netkeiba_horse_id']})")

    db.session.commit()

    return horse


def save_payouts_to_db(race_id, payouts_data):
    """
    Save payout data for a race.

    Args:
        race_id: Database race ID (not netkeiba ID)
        payouts_data: Dictionary with bet types and payout lists
                     Format: {'win': [{'combination': '1', 'payout': 600}], ...}

    Returns:
        List of Payout instances
    """
    from src.data.models import Payout

    payout_objects = []

    for bet_type, payout_list in payouts_data.items():
        for payout_info in payout_list:
            # Check if payout already exists
            existing = Payout.query.filter_by(
                race_id=race_id,
                bet_type=bet_type,
                combination=payout_info['combination']
            ).first()

            if not existing:
                payout = Payout(
                    race_id=race_id,
                    bet_type=bet_type,
                    combination=payout_info['combination'],
                    payout=payout_info['payout'],
                    popularity=payout_info.get('popularity')
                )
                db.session.add(payout)
                logger.debug(f"Created payout: {bet_type} {payout_info['combination']} = {payout_info['payout']}円")
            else:
                # Update existing payout
                existing.payout = payout_info['payout']
                existing.popularity = payout_info.get('popularity', existing.popularity)
                logger.debug(f"Updated payout: {bet_type} {payout_info['combination']} = {payout_info['payout']}円")

            payout_objects.append(existing if existing else payout)

    db.session.commit()
    logger.info(f"Saved {len(payout_objects)} payouts for race {race_id}")

    return payout_objects


def update_race_status_from_results():
    """
    レース結果が存在するレースのステータスを自動的に'completed'に更新する。

    Returns:
        更新されたレース数
    """
    from src.data.models import Race, RaceEntry, RaceResult

    # レース結果があるのにステータスがupcomingのレースを検索
    races_to_update = db.session.query(Race).join(
        RaceEntry, Race.id == RaceEntry.race_id
    ).join(
        RaceResult, RaceEntry.id == RaceResult.race_entry_id
    ).filter(
        Race.status == RaceStatus.UPCOMING
    ).distinct().all()

    updated_count = 0
    for race in races_to_update:
        race.status = RaceStatus.COMPLETED
        logger.info(f"レース {race.netkeiba_race_id} ({race.race_name}) のステータスを 'completed' に更新")
        updated_count += 1

    db.session.commit()

    if updated_count > 0:
        logger.info(f"{updated_count}件のレースステータスを更新しました")
    else:
        logger.debug("更新が必要なレースはありませんでした")

    return updated_count
