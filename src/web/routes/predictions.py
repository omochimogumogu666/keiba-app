"""
Prediction routes for the web application.
"""
from flask import Blueprint, render_template, jsonify
from src.data.models import db, Race, Prediction, RaceEntry
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
    predictions = Prediction.query.filter_by(race_id=race_id).order_by(
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

    predictions = Prediction.query.filter_by(race_id=race_id).order_by(
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
    # TODO: Implement accuracy calculation
    # Compare predictions with actual results

    stats = {
        'total_predictions': Prediction.query.count(),
        'total_races': db.session.query(Race).join(Prediction).distinct().count(),
        # Add more statistics
    }

    return render_template(
        'predictions/accuracy.html',
        stats=stats
    )
