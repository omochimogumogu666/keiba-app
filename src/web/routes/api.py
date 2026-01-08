"""
API routes for external access.
Provides RESTful API endpoints for horses, jockeys, trainers, races, and predictions.
"""
from flask import Blueprint, jsonify, request
from sqlalchemy import desc, func, or_
from datetime import datetime
from src.data.models import (
    db, Horse, Jockey, Trainer, Race, Track, RaceEntry, RaceResult, Prediction
)
from src.web.cache import cache
from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)

api_bp = Blueprint('api', __name__, url_prefix='/api')


# Helper functions
def get_pagination_params():
    """Get pagination parameters from request."""
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 20, type=int), 100)  # Max 100 items per page
    return page, per_page


def paginate_response(query, page, per_page):
    """Paginate query and return response with metadata."""
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)

    return {
        'data': pagination.items,
        'meta': {
            'page': page,
            'per_page': per_page,
            'total_items': pagination.total,
            'total_pages': pagination.pages,
            'has_next': pagination.has_next,
            'has_prev': pagination.has_prev
        }
    }


def error_response(message, status_code=400):
    """Return error response."""
    return jsonify({'error': message}), status_code


# Race endpoints
@api_bp.route('/races', methods=['GET'])
@cache.cached(timeout=300, query_string=True)  # Cache for 5 minutes
def get_races():
    """
    Get list of races with optional filtering.

    Query parameters:
    - date: Filter by race date (YYYY-MM-DD)
    - track_id: Filter by track ID
    - status: Filter by race status (upcoming, in_progress, completed, cancelled)
    - page: Page number (default: 1)
    - per_page: Items per page (default: 20, max: 100)
    - sort: Sort field (date, race_number)
    - order: Sort order (asc, desc)
    """
    query = Race.query

    # Apply filters
    date_str = request.args.get('date')
    if date_str:
        try:
            target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            query = query.filter(Race.race_date == target_date)
        except ValueError:
            return error_response(f"Invalid date format: {date_str}. Use YYYY-MM-DD")

    track_id = request.args.get('track_id', type=int)
    if track_id:
        query = query.filter(Race.track_id == track_id)

    status = request.args.get('status')
    if status:
        valid_statuses = ['upcoming', 'in_progress', 'completed', 'cancelled']
        if status not in valid_statuses:
            return error_response(f"Invalid status. Must be one of: {', '.join(valid_statuses)}")
        query = query.filter(Race.status == status)

    # Apply sorting
    sort_field = request.args.get('sort', 'date')
    sort_order = request.args.get('order', 'desc')

    if sort_field == 'date':
        order_by = Race.race_date.desc() if sort_order == 'desc' else Race.race_date.asc()
    elif sort_field == 'race_number':
        order_by = Race.race_number.desc() if sort_order == 'desc' else Race.race_number.asc()
    else:
        order_by = Race.race_date.desc()

    query = query.order_by(order_by)

    # Paginate
    page, per_page = get_pagination_params()
    result = paginate_response(query, page, per_page)

    # Format response
    races = [{
        'id': race.id,
        'netkeiba_race_id': race.netkeiba_race_id,
        'race_name': race.race_name,
        'race_number': race.race_number,
        'race_date': race.race_date.isoformat() if race.race_date else None,
        'track_id': race.track_id,
        'track_name': race.track.name if race.track else None,
        'distance': race.distance,
        'surface': race.surface,
        'weather': race.weather,
        'track_condition': race.track_condition,
        'status': race.status,
        'prize_money': race.prize_money,
        'race_class': race.race_class
    } for race in result['data']]

    return jsonify({
        'races': races,
        'meta': result['meta']
    })


@api_bp.route('/races/<int:race_id>', methods=['GET'])
def get_race(race_id):
    """
    Get detailed information about a specific race.

    Query parameters:
    - include_entries: Include race entries (true/false, default: false)
    - include_results: Include race results (true/false, default: false)
    - include_predictions: Include predictions (true/false, default: false)
    """
    from sqlalchemy.orm import joinedload
    race = Race.query.options(joinedload(Race.track)).get_or_404(race_id)

    race_data = {
        'id': race.id,
        'netkeiba_race_id': race.netkeiba_race_id,
        'race_name': race.race_name,
        'race_number': race.race_number,
        'race_date': race.race_date.isoformat() if race.race_date else None,
        'track': {
            'id': race.track.id,
            'name': race.track.name,
            'location': race.track.location
        } if race.track else None,
        'distance': race.distance,
        'surface': race.surface,
        'weather': race.weather,
        'track_condition': race.track_condition,
        'status': race.status,
        'prize_money': race.prize_money,
        'race_class': race.race_class
    }

    # Include entries if requested
    if request.args.get('include_entries', 'false').lower() == 'true':
        entries = RaceEntry.query.filter_by(race_id=race_id).all()
        race_data['entries'] = [{
            'id': entry.id,
            'horse_number': entry.horse_number,
            'post_position': entry.post_position,
            'horse': {
                'id': entry.horse.id,
                'name': entry.horse.name,
                'netkeiba_horse_id': entry.horse.netkeiba_horse_id
            } if entry.horse else None,
            'jockey': {
                'id': entry.jockey.id,
                'name': entry.jockey.name,
                'netkeiba_jockey_id': entry.jockey.netkeiba_jockey_id
            } if entry.jockey else None,
            'trainer': {
                'id': entry.horse.trainer.id,
                'name': entry.horse.trainer.name,
                'netkeiba_trainer_id': entry.horse.trainer.netkeiba_trainer_id
            } if entry.horse and entry.horse.trainer else None,
            'weight': entry.weight,
            'horse_weight': entry.horse_weight,
            'morning_odds': entry.morning_odds
        } for entry in entries]

    # Include results if requested
    if request.args.get('include_results', 'false').lower() == 'true':
        results = db.session.query(RaceResult).join(RaceEntry).filter(
            RaceEntry.race_id == race_id
        ).all()
        race_data['results'] = [{
            'horse_number': result.race_entry.horse_number if result.race_entry else None,
            'finish_position': result.finish_position,
            'final_odds': result.final_odds,
            'popularity': result.popularity,
            'finish_time': result.finish_time,
            'margin': result.margin
        } for result in results]

    # Include predictions if requested
    if request.args.get('include_predictions', 'false').lower() == 'true':
        predictions = Prediction.query.filter_by(race_id=race_id).order_by(
            Prediction.predicted_position
        ).all()
        race_data['predictions'] = [{
            'horse_id': pred.horse_id,
            'horse_name': pred.horse.name if pred.horse else None,
            'predicted_position': pred.predicted_position,
            'win_probability': pred.win_probability,
            'confidence_score': pred.confidence_score,
            'model_name': pred.model_name
        } for pred in predictions]

    return jsonify(race_data)


# Horse endpoints
@api_bp.route('/horses', methods=['GET'])
def get_horses():
    """
    Get list of horses with optional filtering.

    Query parameters:
    - search: Search by horse name
    - page: Page number (default: 1)
    - per_page: Items per page (default: 20, max: 100)
    - sort: Sort field (name, race_count)
    - order: Sort order (asc, desc)
    """
    query = Horse.query

    # Search filter
    search = request.args.get('search')
    if search:
        query = query.filter(Horse.name.contains(search))

    # Apply sorting
    sort_field = request.args.get('sort', 'name')
    sort_order = request.args.get('order', 'asc')

    if sort_field == 'name':
        order_by = Horse.name.desc() if sort_order == 'desc' else Horse.name.asc()
    elif sort_field == 'race_count':
        query = query.outerjoin(RaceEntry).group_by(Horse.id)
        order_by = func.count(RaceEntry.id).desc() if sort_order == 'desc' else func.count(RaceEntry.id).asc()
    else:
        order_by = Horse.name.asc()

    query = query.order_by(order_by)

    # Paginate
    page, per_page = get_pagination_params()
    result = paginate_response(query, page, per_page)

    # Format response
    horses = [{
        'id': horse.id,
        'netkeiba_horse_id': horse.netkeiba_horse_id,
        'name': horse.name,
        'birth_date': horse.birth_date.isoformat() if horse.birth_date else None,
        'sex': horse.sex
    } for horse in result['data']]

    return jsonify({
        'horses': horses,
        'meta': result['meta']
    })


@api_bp.route('/horses/<int:horse_id>', methods=['GET'])
def get_horse(horse_id):
    """
    Get detailed information about a specific horse.

    Query parameters:
    - include_stats: Include statistics (true/false, default: false)
    - include_races: Include recent races (true/false, default: false)
    - race_limit: Number of recent races to include (default: 10)
    """
    horse = Horse.query.get_or_404(horse_id)

    horse_data = {
        'id': horse.id,
        'netkeiba_horse_id': horse.netkeiba_horse_id,
        'name': horse.name,
        'birth_date': horse.birth_date.isoformat() if horse.birth_date else None,
        'sex': horse.sex,
        'trainer_id': horse.trainer_id
    }

    # Include statistics if requested
    if request.args.get('include_stats', 'false').lower() == 'true':
        total_races = RaceEntry.query.filter_by(horse_id=horse_id).count()
        total_results = db.session.query(RaceResult).join(RaceEntry).filter(
            RaceEntry.horse_id == horse_id
        ).count()

        wins = db.session.query(RaceResult).join(RaceEntry).filter(
            RaceEntry.horse_id == horse_id,
            RaceResult.finish_position == 1
        ).count()

        places = db.session.query(RaceResult).join(RaceEntry).filter(
            RaceEntry.horse_id == horse_id,
            RaceResult.finish_position <= 3
        ).count()

        win_rate = wins / total_results if total_results > 0 else 0
        place_rate = places / total_results if total_results > 0 else 0

        horse_data['statistics'] = {
            'total_races': total_races,
            'total_results': total_results,
            'wins': wins,
            'places': places,
            'win_rate': round(win_rate, 3),
            'place_rate': round(place_rate, 3)
        }

    # Include recent races if requested
    if request.args.get('include_races', 'false').lower() == 'true':
        race_limit = request.args.get('race_limit', 10, type=int)
        entries = RaceEntry.query.filter_by(horse_id=horse_id).join(
            Race
        ).order_by(Race.race_date.desc()).limit(race_limit).all()

        horse_data['recent_races'] = [{
            'race_id': entry.race.id,
            'race_name': entry.race.race_name,
            'race_date': entry.race.race_date.isoformat() if entry.race.race_date else None,
            'track_name': entry.race.track.name if entry.race.track else None,
            'horse_number': entry.horse_number,
            'jockey_name': entry.jockey.name if entry.jockey else None
        } for entry in entries]

    return jsonify(horse_data)


# Jockey endpoints
@api_bp.route('/jockeys', methods=['GET'])
def get_jockeys():
    """
    Get list of jockeys with optional filtering.

    Query parameters:
    - search: Search by jockey name
    - page: Page number (default: 1)
    - per_page: Items per page (default: 20, max: 100)
    - sort: Sort field (name, race_count)
    - order: Sort order (asc, desc)
    """
    query = Jockey.query

    # Search filter
    search = request.args.get('search')
    if search:
        query = query.filter(Jockey.name.contains(search))

    # Apply sorting
    sort_field = request.args.get('sort', 'name')
    sort_order = request.args.get('order', 'asc')

    if sort_field == 'name':
        order_by = Jockey.name.desc() if sort_order == 'desc' else Jockey.name.asc()
    elif sort_field == 'race_count':
        query = query.outerjoin(RaceEntry).group_by(Jockey.id)
        order_by = func.count(RaceEntry.id).desc() if sort_order == 'desc' else func.count(RaceEntry.id).asc()
    else:
        order_by = Jockey.name.asc()

    query = query.order_by(order_by)

    # Paginate
    page, per_page = get_pagination_params()
    result = paginate_response(query, page, per_page)

    # Format response
    jockeys = [{
        'id': jockey.id,
        'netkeiba_jockey_id': jockey.netkeiba_jockey_id,
        'name': jockey.name
    } for jockey in result['data']]

    return jsonify({
        'jockeys': jockeys,
        'meta': result['meta']
    })


@api_bp.route('/jockeys/<int:jockey_id>', methods=['GET'])
def get_jockey(jockey_id):
    """
    Get detailed information about a specific jockey.

    Query parameters:
    - include_stats: Include statistics (true/false, default: false)
    - include_races: Include recent races (true/false, default: false)
    - race_limit: Number of recent races to include (default: 10)
    """
    jockey = Jockey.query.get_or_404(jockey_id)

    jockey_data = {
        'id': jockey.id,
        'netkeiba_jockey_id': jockey.netkeiba_jockey_id,
        'name': jockey.name
    }

    # Include statistics if requested
    if request.args.get('include_stats', 'false').lower() == 'true':
        total_races = RaceEntry.query.filter_by(jockey_id=jockey_id).count()
        total_results = db.session.query(RaceResult).join(RaceEntry).filter(
            RaceEntry.jockey_id == jockey_id
        ).count()

        wins = db.session.query(RaceResult).join(RaceEntry).filter(
            RaceEntry.jockey_id == jockey_id,
            RaceResult.finish_position == 1
        ).count()

        places = db.session.query(RaceResult).join(RaceEntry).filter(
            RaceEntry.jockey_id == jockey_id,
            RaceResult.finish_position <= 3
        ).count()

        win_rate = wins / total_results if total_results > 0 else 0
        place_rate = places / total_results if total_results > 0 else 0

        jockey_data['statistics'] = {
            'total_races': total_races,
            'total_results': total_results,
            'wins': wins,
            'places': places,
            'win_rate': round(win_rate, 3),
            'place_rate': round(place_rate, 3)
        }

    # Include recent races if requested
    if request.args.get('include_races', 'false').lower() == 'true':
        race_limit = request.args.get('race_limit', 10, type=int)
        entries = RaceEntry.query.filter_by(jockey_id=jockey_id).join(
            Race
        ).order_by(Race.race_date.desc()).limit(race_limit).all()

        jockey_data['recent_races'] = [{
            'race_id': entry.race.id,
            'race_name': entry.race.race_name,
            'race_date': entry.race.race_date.isoformat() if entry.race.race_date else None,
            'track_name': entry.race.track.name if entry.race.track else None,
            'horse_name': entry.horse.name if entry.horse else None,
            'horse_number': entry.horse_number
        } for entry in entries]

    return jsonify(jockey_data)


# Trainer endpoints
@api_bp.route('/trainers', methods=['GET'])
def get_trainers():
    """
    Get list of trainers with optional filtering.

    Query parameters:
    - search: Search by trainer name
    - page: Page number (default: 1)
    - per_page: Items per page (default: 20, max: 100)
    - sort: Sort field (name, race_count)
    - order: Sort order (asc, desc)
    """
    query = Trainer.query

    # Search filter
    search = request.args.get('search')
    if search:
        query = query.filter(Trainer.name.contains(search))

    # Apply sorting
    sort_field = request.args.get('sort', 'name')
    sort_order = request.args.get('order', 'asc')

    if sort_field == 'name':
        order_by = Trainer.name.desc() if sort_order == 'desc' else Trainer.name.asc()
    elif sort_field == 'race_count':
        query = query.outerjoin(Horse).outerjoin(RaceEntry, Horse.id == RaceEntry.horse_id).group_by(Trainer.id)
        order_by = func.count(RaceEntry.id).desc() if sort_order == 'desc' else func.count(RaceEntry.id).asc()
    else:
        order_by = Trainer.name.asc()

    query = query.order_by(order_by)

    # Paginate
    page, per_page = get_pagination_params()
    result = paginate_response(query, page, per_page)

    # Format response
    trainers = [{
        'id': trainer.id,
        'netkeiba_trainer_id': trainer.netkeiba_trainer_id,
        'name': trainer.name,
        'stable': trainer.stable
    } for trainer in result['data']]

    return jsonify({
        'trainers': trainers,
        'meta': result['meta']
    })


@api_bp.route('/trainers/<int:trainer_id>', methods=['GET'])
def get_trainer(trainer_id):
    """
    Get detailed information about a specific trainer.

    Query parameters:
    - include_stats: Include statistics (true/false, default: false)
    - include_races: Include recent races (true/false, default: false)
    - race_limit: Number of recent races to include (default: 10)
    """
    trainer = Trainer.query.get_or_404(trainer_id)

    trainer_data = {
        'id': trainer.id,
        'netkeiba_trainer_id': trainer.netkeiba_trainer_id,
        'name': trainer.name,
        'stable': trainer.stable
    }

    # Include statistics if requested
    if request.args.get('include_stats', 'false').lower() == 'true':
        total_races = db.session.query(RaceEntry).join(Horse).filter(
            Horse.trainer_id == trainer_id
        ).count()
        total_results = db.session.query(RaceResult).join(RaceEntry).join(Horse).filter(
            Horse.trainer_id == trainer_id
        ).count()

        wins = db.session.query(RaceResult).join(RaceEntry).join(Horse).filter(
            Horse.trainer_id == trainer_id,
            RaceResult.finish_position == 1
        ).count()

        places = db.session.query(RaceResult).join(RaceEntry).join(Horse).filter(
            Horse.trainer_id == trainer_id,
            RaceResult.finish_position <= 3
        ).count()

        win_rate = wins / total_results if total_results > 0 else 0
        place_rate = places / total_results if total_results > 0 else 0

        trainer_data['statistics'] = {
            'total_races': total_races,
            'total_results': total_results,
            'wins': wins,
            'places': places,
            'win_rate': round(win_rate, 3),
            'place_rate': round(place_rate, 3)
        }

    # Include recent races if requested
    if request.args.get('include_races', 'false').lower() == 'true':
        race_limit = request.args.get('race_limit', 10, type=int)
        entries = db.session.query(RaceEntry).join(Horse).join(Race).filter(
            Horse.trainer_id == trainer_id
        ).order_by(Race.race_date.desc()).limit(race_limit).all()

        trainer_data['recent_races'] = [{
            'race_id': entry.race.id,
            'race_name': entry.race.race_name,
            'race_date': entry.race.race_date.isoformat() if entry.race.race_date else None,
            'track_name': entry.race.track.name if entry.race.track else None,
            'horse_name': entry.horse.name if entry.horse else None,
            'jockey_name': entry.jockey.name if entry.jockey else None
        } for entry in entries]

    return jsonify(trainer_data)


# Prediction endpoints
@api_bp.route('/predictions', methods=['GET'])
def get_predictions():
    """
    Get list of predictions with optional filtering.

    Query parameters:
    - race_id: Filter by race ID
    - model_name: Filter by model name
    - page: Page number (default: 1)
    - per_page: Items per page (default: 20, max: 100)
    """
    query = Prediction.query

    # Apply filters
    race_id = request.args.get('race_id', type=int)
    if race_id:
        query = query.filter(Prediction.race_id == race_id)

    model_name = request.args.get('model_name')
    if model_name:
        query = query.filter(Prediction.model_name == model_name)

    # Order by race date and predicted position
    query = query.join(Race).order_by(
        Race.race_date.desc(),
        Prediction.predicted_position.asc()
    )

    # Paginate
    page, per_page = get_pagination_params()
    result = paginate_response(query, page, per_page)

    # Format response
    predictions = [{
        'id': pred.id,
        'race_id': pred.race_id,
        'race_name': pred.race.race_name if pred.race else None,
        'horse_id': pred.horse_id,
        'horse_name': pred.horse.name if pred.horse else None,
        'predicted_position': pred.predicted_position,
        'win_probability': pred.win_probability,
        'confidence_score': pred.confidence_score,
        'model_name': pred.model_name,
        'created_at': pred.created_at.isoformat() if pred.created_at else None
    } for pred in result['data']]

    return jsonify({
        'predictions': predictions,
        'meta': result['meta']
    })


@api_bp.route('/predictions/race/<int:race_id>', methods=['GET'])
def get_race_predictions(race_id):
    """
    Get predictions for a specific race.
    """
    race = Race.query.get_or_404(race_id)

    predictions = Prediction.query.filter_by(race_id=race_id).order_by(
        Prediction.predicted_position
    ).all()

    # Get race entries for additional information
    entries = {entry.horse_id: entry for entry in race.race_entries}

    prediction_data = []
    for pred in predictions:
        entry = entries.get(pred.horse_id)
        prediction_data.append({
            'horse_id': pred.horse_id,
            'horse_name': pred.horse.name if pred.horse else None,
            'horse_number': entry.horse_number if entry else None,
            'jockey_name': entry.jockey.name if entry and entry.jockey else None,
            'predicted_position': pred.predicted_position,
            'win_probability': pred.win_probability,
            'confidence_score': pred.confidence_score,
            'model_name': pred.model_name,
            'morning_odds': entry.morning_odds if entry else None
        })

    return jsonify({
        'race': {
            'id': race.id,
            'race_name': race.race_name,
            'race_date': race.race_date.isoformat() if race.race_date else None,
            'track_name': race.track.name if race.track else None
        },
        'predictions': prediction_data
    })


# Track endpoints
@api_bp.route('/tracks', methods=['GET'])
def get_tracks():
    """
    Get list of all tracks.
    """
    tracks = Track.query.order_by(Track.name).all()

    track_data = [{
        'id': track.id,
        'name': track.name,
        'location': track.location,
        'surface_types': track.surface_types
    } for track in tracks]

    return jsonify({
        'tracks': track_data,
        'meta': {
            'total_items': len(track_data)
        }
    })


@api_bp.route('/tracks/<int:track_id>', methods=['GET'])
def get_track(track_id):
    """
    Get information about a specific track.

    Query parameters:
    - include_races: Include recent races (true/false, default: false)
    - race_limit: Number of recent races to include (default: 10)
    """
    track = Track.query.get_or_404(track_id)

    track_data = {
        'id': track.id,
        'name': track.name,
        'location': track.location,
        'surface_types': track.surface_types
    }

    # Include recent races if requested
    if request.args.get('include_races', 'false').lower() == 'true':
        race_limit = request.args.get('race_limit', 10, type=int)
        races = Race.query.filter_by(track_id=track_id).order_by(
            Race.race_date.desc()
        ).limit(race_limit).all()

        track_data['recent_races'] = [{
            'id': race.id,
            'race_name': race.race_name,
            'race_date': race.race_date.isoformat() if race.race_date else None,
            'race_number': race.race_number,
            'distance': race.distance,
            'surface': race.surface
        } for race in races]

    return jsonify(track_data)


# Error handlers
@api_bp.errorhandler(404)
def api_not_found(error):
    """Handle 404 errors for API."""
    return jsonify({'error': 'Resource not found'}), 404


@api_bp.errorhandler(500)
def api_internal_error(error):
    """Handle 500 errors for API."""
    db.session.rollback()
    return jsonify({'error': 'Internal server error'}), 500
