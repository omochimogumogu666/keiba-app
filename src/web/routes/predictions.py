"""
Prediction routes for the web application.
"""
from flask import Blueprint, render_template, jsonify
from sqlalchemy.orm import joinedload
from src.data.models import db, Race, Prediction, RaceEntry, RaceResult
from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)

predictions_bp = Blueprint('predictions', __name__, url_prefix='/predictions')


@predictions_bp.route('/')
def predictions_index():
    """Predictions index page."""
    # Get races with predictions
    races_with_predictions = db.session.query(Race).join(Prediction).distinct().all()

    return render_template(
        'predictions/index.html',
        races=races_with_predictions
    )


@predictions_bp.route('/race/<int:race_id>')
def race_predictions(race_id):
    """Show predictions for a specific race."""
    race = Race.query.get_or_404(race_id)

    # Get predictions for this race, ordered by predicted position
    # Eagerly load horse relationship to avoid lazy loading issues
    predictions = Prediction.query.options(
        joinedload(Prediction.horse)
    ).filter_by(race_id=race_id).order_by(
        Prediction.predicted_position
    ).all()

    # Get race entries for additional information
    entries = {entry.horse_id: entry for entry in race.race_entries}

    # Combine predictions with entry information
    prediction_data = []
    for pred in predictions:
        entry = entries.get(pred.horse_id)
        if entry:
            prediction_data.append({
                'prediction': pred,
                'entry': entry,
                'horse': pred.horse,
                'jockey': entry.jockey if entry else None
            })

    return render_template(
        'predictions/race.html',
        race=race,
        predictions=prediction_data
    )


@predictions_bp.route('/api/race/<int:race_id>')
def api_race_predictions(race_id):
    """API endpoint for race predictions."""
    race = Race.query.get_or_404(race_id)

    # Eagerly load horse relationship
    predictions = Prediction.query.options(
        joinedload(Prediction.horse)
    ).filter_by(race_id=race_id).order_by(
        Prediction.predicted_position
    ).all()

    result = {
        'race_id': race_id,
        'race_name': race.race_name,
        'race_date': race.race_date.isoformat() if race.race_date else None,
        'predictions': [
            {
                'horse_id': p.horse_id,
                'horse_name': p.horse.name if p.horse else None,
                'predicted_position': p.predicted_position,
                'win_probability': p.win_probability,
                'confidence_score': p.confidence_score,
                'model_name': p.model_name
            }
            for p in predictions
        ]
    }

    return jsonify(result)


@predictions_bp.route('/accuracy')
def prediction_accuracy():
    """Show prediction accuracy statistics."""
    from sqlalchemy import func
    from src.data.models import RaceResult

    # Basic statistics
    total_predictions = Prediction.query.count()
    total_races = db.session.query(Race).join(Prediction).distinct().count()

    # Calculate accuracy - compare predictions with actual results
    # Find predictions where we have actual results
    # Join through RaceEntry to connect Prediction and RaceResult
    predictions_with_results = db.session.query(
        Prediction, RaceResult
    ).join(
        RaceEntry,
        (Prediction.race_id == RaceEntry.race_id) &
        (Prediction.horse_id == RaceEntry.horse_id)
    ).join(
        RaceResult,
        RaceResult.race_entry_id == RaceEntry.id
    ).all()

    # Win accuracy (predicted position 1 == actual position 1)
    win_predictions = [p for p, r in predictions_with_results if p.predicted_position == 1]
    win_correct = [p for p, r in predictions_with_results
                   if p.predicted_position == 1 and r.finish_position == 1]
    win_accuracy = len(win_correct) / len(win_predictions) if win_predictions else 0

    # Place accuracy (predicted top 3 == actual top 3)
    place_predictions = [p for p, r in predictions_with_results if p.predicted_position <= 3]
    place_correct = [p for p, r in predictions_with_results
                     if p.predicted_position <= 3 and r.finish_position <= 3]
    place_accuracy = len(place_correct) / len(place_predictions) if place_predictions else 0

    # Top pick win rate (how often our #1 pick wins)
    top_pick_wins = len(win_correct)
    top_pick_total = len(win_predictions)

    # Model performance breakdown
    model_stats = db.session.query(
        Prediction.model_name,
        func.count(Prediction.id).label('count')
    ).group_by(Prediction.model_name).all()

    stats = {
        'total_predictions': total_predictions,
        'total_races': total_races,
        'predictions_with_results': len(predictions_with_results),
        'win_accuracy': win_accuracy,
        'place_accuracy': place_accuracy,
        'top_pick_wins': top_pick_wins,
        'top_pick_total': top_pick_total,
        'model_stats': model_stats
    }

    return render_template(
        'predictions/accuracy.html',
        stats=stats
    )
