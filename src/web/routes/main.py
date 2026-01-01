"""
Main routes for the web application.
"""
from flask import Blueprint, render_template, request
from datetime import datetime, timedelta
from src.data.models import db, Race, Track
from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    """Home page."""
    # Get upcoming races (next 7 days)
    today = datetime.now().date()
    week_later = today + timedelta(days=7)

    upcoming_races = Race.query.filter(
        Race.race_date >= today,
        Race.race_date <= week_later,
        Race.status == 'upcoming'
    ).order_by(Race.race_date, Race.race_number).limit(10).all()

    # Get recent completed races
    recent_races = Race.query.filter(
        Race.status == 'completed'
    ).order_by(Race.race_date.desc()).limit(5).all()

    return render_template(
        'index.html',
        upcoming_races=upcoming_races,
        recent_races=recent_races
    )


@main_bp.route('/races')
def race_list():
    """List all races."""
    # Get filter parameters
    date_str = request.args.get('date')
    track_id = request.args.get('track')
    status = request.args.get('status', 'upcoming')

    # Build query
    query = Race.query

    if date_str:
        try:
            target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            query = query.filter(Race.race_date == target_date)
        except ValueError:
            logger.warning(f"Invalid date format: {date_str}")

    if track_id:
        query = query.filter(Race.track_id == track_id)

    if status:
        query = query.filter(Race.status == status)

    races = query.order_by(Race.race_date.desc(), Race.race_number).all()

    # Get all tracks for filter dropdown
    tracks = Track.query.all()

    return render_template(
        'race_list.html',
        races=races,
        tracks=tracks,
        selected_date=date_str,
        selected_track=track_id,
        selected_status=status
    )


@main_bp.route('/races/<int:race_id>')
def race_detail(race_id):
    """Race detail page."""
    race = Race.query.get_or_404(race_id)

    # Get race entries with horse and jockey information
    entries = race.race_entries

    return render_template(
        'race_detail.html',
        race=race,
        entries=entries
    )


@main_bp.route('/about')
def about():
    """About page."""
    return render_template('about.html')
