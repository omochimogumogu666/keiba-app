"""
Main routes for the web application.
"""
from flask import Blueprint, render_template, request
from datetime import datetime, timedelta
from sqlalchemy import func
from src.data.models import db, Race, Track, RaceResult, RaceEntry, Payout
from src.data.statistics import calculate_prediction_accuracy
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

    # Get statistics for charts
    # 週別レース数統計
    weeks_ago = today - timedelta(days=56)  # 8週間分
    try:
        # PostgreSQL用のクエリ (to_char関数を使用)
        race_stats = db.session.query(
            func.to_char(Race.race_date, 'IYYY-IW').label('week'),
            func.count(Race.id).label('count')
        ).filter(
            Race.race_date >= weeks_ago,
            Race.status == 'completed'
        ).group_by('week').order_by('week').all()
    except Exception:
        # SQLite用のクエリ (strftime関数を使用)
        race_stats = db.session.query(
            func.strftime('%Y-%W', Race.race_date).label('week'),
            func.count(Race.id).label('count')
        ).filter(
            Race.race_date >= weeks_ago,
            Race.status == 'completed'
        ).group_by('week').order_by('week').all()

    chart_labels = [f'第{i+1}週' for i in range(len(race_stats))] if race_stats else []
    chart_data = [stat.count for stat in race_stats] if race_stats else []

    # 競馬場別レース数
    track_stats = db.session.query(
        Track.name,
        func.count(Race.id).label('count')
    ).join(Race).filter(
        Race.race_date >= weeks_ago,
        Race.status == 'completed'
    ).group_by(Track.name).order_by(func.count(Race.id).desc()).limit(5).all()

    track_labels = [stat.name for stat in track_stats] if track_stats else []
    track_data = [stat.count for stat in track_stats] if track_stats else []

    # 予測精度・回収率の計算
    try:
        prediction_stats = calculate_prediction_accuracy(days=30)
        prediction_accuracy = prediction_stats.get('win_accuracy', 0)
        recovery_rate = prediction_stats.get('roi', 0)
    except Exception as e:
        logger.warning(f"Could not calculate prediction accuracy: {e}")
        prediction_accuracy = 0
        recovery_rate = 0

    return render_template(
        'index.html',
        upcoming_races=upcoming_races,
        recent_races=recent_races,
        chart_labels=chart_labels,
        chart_data=chart_data,
        track_labels=track_labels,
        track_data=track_data,
        prediction_accuracy=prediction_accuracy,
        recovery_rate=recovery_rate
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

    # Get payouts for completed races
    payouts = None
    if race.status == 'completed':
        payouts = Payout.query.filter_by(race_id=race.id).all()
        # Group payouts by bet type
        payout_groups = {
            'win': [],
            'place': [],
            'bracket_quinella': [],
            'quinella': [],
            'exacta': [],
            'wide': [],
            'trio': [],
            'trifecta': []
        }
        for payout in payouts:
            if payout.bet_type in payout_groups:
                payout_groups[payout.bet_type].append(payout)
        payouts = payout_groups

    # 馬・騎手の追加統計情報を取得
    horse_stats = {}
    jockey_stats = {}
    horse_recent_races = {}

    for entry in entries:
        # 馬の直近5レース成績
        if entry.horse:
            recent_results = RaceResult.query.join(RaceEntry).join(Race).filter(
                RaceEntry.horse_id == entry.horse.id,
                Race.status == 'completed',
                Race.id != race.id
            ).order_by(Race.race_date.desc()).limit(5).all()

            horse_recent_races[entry.horse.id] = [
                {
                    'position': r.finish_position,
                    'race_name': r.race_entry.race.race_name[:10] if r.race_entry.race.race_name else '',
                    'date': r.race_entry.race.race_date.strftime('%m/%d') if r.race_entry.race.race_date else ''
                }
                for r in recent_results
            ]

            # 馬の通算成績
            all_results = RaceResult.query.join(RaceEntry).join(Race).filter(
                RaceEntry.horse_id == entry.horse.id,
                Race.status == 'completed'
            ).all()

            total = len(all_results)
            wins = sum(1 for r in all_results if r.finish_position == 1)
            places = sum(1 for r in all_results if r.finish_position and r.finish_position <= 3)
            horse_stats[entry.horse.id] = {
                'total': total,
                'wins': wins,
                'places': places,
                'win_rate': round(wins / total * 100, 1) if total > 0 else 0,
                'place_rate': round(places / total * 100, 1) if total > 0 else 0
            }

        # 騎手の成績
        if entry.jockey:
            jockey_results = RaceResult.query.join(RaceEntry).join(Race).filter(
                RaceEntry.jockey_id == entry.jockey.id,
                Race.status == 'completed',
                Race.race_date >= datetime.now().date() - timedelta(days=365)
            ).all()

            total = len(jockey_results)
            wins = sum(1 for r in jockey_results if r.finish_position == 1)
            places = sum(1 for r in jockey_results if r.finish_position and r.finish_position <= 3)
            jockey_stats[entry.jockey.id] = {
                'total': total,
                'wins': wins,
                'places': places,
                'win_rate': round(wins / total * 100, 1) if total > 0 else 0,
                'place_rate': round(places / total * 100, 1) if total > 0 else 0
            }

    return render_template(
        'race_detail.html',
        race=race,
        entries=entries,
        payouts=payouts,
        horse_stats=horse_stats,
        jockey_stats=jockey_stats,
        horse_recent_races=horse_recent_races
    )


@main_bp.route('/results')
def results_list():
    """List all race results."""
    # Get filter parameters
    date_str = request.args.get('date')
    track_id = request.args.get('track')
    page = request.args.get('page', 1, type=int)
    per_page = 20

    # Build query for completed races
    query = Race.query.filter(Race.status == 'completed')

    if date_str:
        try:
            target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            query = query.filter(Race.race_date == target_date)
        except ValueError:
            logger.warning(f"Invalid date format: {date_str}")

    if track_id:
        query = query.filter(Race.track_id == track_id)

    # Get paginated results
    pagination = query.order_by(Race.race_date.desc(), Race.race_number).paginate(
        page=page, per_page=per_page, error_out=False
    )
    races = pagination.items

    # Get all tracks for filter dropdown
    tracks = Track.query.all()

    # Get statistics
    stats = {
        'total_races': Race.query.filter(Race.status == 'completed').count(),
        'date_range': db.session.query(
            func.min(Race.race_date),
            func.max(Race.race_date)
        ).filter(Race.status == 'completed').first()
    }

    return render_template(
        'results_list.html',
        races=races,
        pagination=pagination,
        tracks=tracks,
        selected_date=date_str,
        selected_track=track_id,
        stats=stats
    )


@main_bp.route('/about')
def about():
    """About page."""
    return render_template('about.html')
