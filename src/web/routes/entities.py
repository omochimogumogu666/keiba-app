"""
Entity routes for horses, jockeys, and trainers.
"""
from flask import Blueprint, render_template, request
from sqlalchemy import desc, func
from src.data.models import (
    db, Horse, Jockey, Trainer, RaceEntry, RaceResult, Race
)
from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)

entities_bp = Blueprint('entities', __name__, url_prefix='/entities')


@entities_bp.route('/horses')
def horse_list():
    """List all horses."""
    page = request.args.get('page', 1, type=int)
    per_page = 50

    # Get horses with race count
    horses = Horse.query.outerjoin(RaceEntry).group_by(Horse.id).order_by(
        desc(func.count(RaceEntry.id))
    ).paginate(page=page, per_page=per_page, error_out=False)

    return render_template(
        'entities/horse_list.html',
        horses=horses
    )


@entities_bp.route('/horses/<int:horse_id>')
def horse_detail(horse_id):
    """Horse detail page."""
    horse = Horse.query.get_or_404(horse_id)

    # Get race entries with race and result information
    entries = RaceEntry.query.filter_by(horse_id=horse_id).join(
        Race
    ).order_by(Race.race_date.desc()).limit(20).all()

    # Get race results
    results = db.session.query(RaceResult, RaceEntry, Race).join(
        RaceEntry, RaceResult.race_entry_id == RaceEntry.id
    ).join(
        Race, RaceResult.race_id == Race.id
    ).filter(
        RaceEntry.horse_id == horse_id
    ).order_by(Race.race_date.desc()).limit(20).all()

    # Calculate statistics
    total_races = RaceEntry.query.filter_by(horse_id=horse_id).count()
    total_results = db.session.query(RaceResult).join(
        RaceEntry
    ).filter(RaceEntry.horse_id == horse_id).count()

    wins = db.session.query(RaceResult).join(
        RaceEntry
    ).filter(
        RaceEntry.horse_id == horse_id,
        RaceResult.finish_position == 1
    ).count()

    places = db.session.query(RaceResult).join(
        RaceEntry
    ).filter(
        RaceEntry.horse_id == horse_id,
        RaceResult.finish_position <= 3
    ).count()

    win_rate = wins / total_results if total_results > 0 else 0
    place_rate = places / total_results if total_results > 0 else 0

    stats = {
        'total_races': total_races,
        'total_results': total_results,
        'wins': wins,
        'places': places,
        'win_rate': win_rate,
        'place_rate': place_rate
    }

    return render_template(
        'entities/horse_detail.html',
        horse=horse,
        entries=entries,
        results=results,
        stats=stats
    )


@entities_bp.route('/jockeys')
def jockey_list():
    """List all jockeys."""
    page = request.args.get('page', 1, type=int)
    per_page = 50

    # Get jockeys with race count
    jockeys = Jockey.query.outerjoin(RaceEntry).group_by(Jockey.id).order_by(
        desc(func.count(RaceEntry.id))
    ).paginate(page=page, per_page=per_page, error_out=False)

    return render_template(
        'entities/jockey_list.html',
        jockeys=jockeys
    )


@entities_bp.route('/jockeys/<int:jockey_id>')
def jockey_detail(jockey_id):
    """Jockey detail page."""
    jockey = Jockey.query.get_or_404(jockey_id)

    # Get recent race entries
    entries = RaceEntry.query.filter_by(jockey_id=jockey_id).join(
        Race
    ).order_by(Race.race_date.desc()).limit(20).all()

    # Calculate statistics
    total_races = RaceEntry.query.filter_by(jockey_id=jockey_id).count()
    total_results = db.session.query(RaceResult).join(
        RaceEntry
    ).filter(RaceEntry.jockey_id == jockey_id).count()

    wins = db.session.query(RaceResult).join(
        RaceEntry
    ).filter(
        RaceEntry.jockey_id == jockey_id,
        RaceResult.finish_position == 1
    ).count()

    places = db.session.query(RaceResult).join(
        RaceEntry
    ).filter(
        RaceEntry.jockey_id == jockey_id,
        RaceResult.finish_position <= 3
    ).count()

    win_rate = wins / total_results if total_results > 0 else 0
    place_rate = places / total_results if total_results > 0 else 0

    stats = {
        'total_races': total_races,
        'total_results': total_results,
        'wins': wins,
        'places': places,
        'win_rate': win_rate,
        'place_rate': place_rate
    }

    return render_template(
        'entities/jockey_detail.html',
        jockey=jockey,
        entries=entries,
        stats=stats
    )


@entities_bp.route('/trainers')
def trainer_list():
    """List all trainers."""
    page = request.args.get('page', 1, type=int)
    per_page = 50

    # Get trainers with horse count
    trainers = Trainer.query.outerjoin(RaceEntry).group_by(Trainer.id).order_by(
        desc(func.count(RaceEntry.id))
    ).paginate(page=page, per_page=per_page, error_out=False)

    return render_template(
        'entities/trainer_list.html',
        trainers=trainers
    )


@entities_bp.route('/trainers/<int:trainer_id>')
def trainer_detail(trainer_id):
    """Trainer detail page."""
    trainer = Trainer.query.get_or_404(trainer_id)

    # Get recent race entries
    entries = RaceEntry.query.filter_by(trainer_id=trainer_id).join(
        Race
    ).order_by(Race.race_date.desc()).limit(20).all()

    # Calculate statistics
    total_races = RaceEntry.query.filter_by(trainer_id=trainer_id).count()
    total_results = db.session.query(RaceResult).join(
        RaceEntry
    ).filter(RaceEntry.trainer_id == trainer_id).count()

    wins = db.session.query(RaceResult).join(
        RaceEntry
    ).filter(
        RaceEntry.trainer_id == trainer_id,
        RaceResult.finish_position == 1
    ).count()

    places = db.session.query(RaceResult).join(
        RaceEntry
    ).filter(
        RaceEntry.trainer_id == trainer_id,
        RaceResult.finish_position <= 3
    ).count()

    win_rate = wins / total_results if total_results > 0 else 0
    place_rate = places / total_results if total_results > 0 else 0

    stats = {
        'total_races': total_races,
        'total_results': total_results,
        'wins': wins,
        'places': places,
        'win_rate': win_rate,
        'place_rate': place_rate
    }

    return render_template(
        'entities/trainer_detail.html',
        trainer=trainer,
        entries=entries,
        stats=stats
    )
