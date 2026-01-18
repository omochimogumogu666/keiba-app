"""
Prediction routes for the web application.

予測画面のルートと、予測タスク実行APIを提供します。
"""
import json
import time
from datetime import datetime, date
from flask import Blueprint, render_template, jsonify, request, Response, current_app
from sqlalchemy.orm import joinedload
from src.data.models import db, Race, Prediction, RaceEntry, RaceResult
from src.web.prediction_manager import prediction_task_manager
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


# ============================================================================
# 予測タスク実行API
# ============================================================================

@predictions_bp.route('/generate')
def generate_predictions_page():
    """予測生成画面を表示"""
    running_task = prediction_task_manager.get_running_task()
    recent_tasks = prediction_task_manager.get_recent_tasks(limit=5)

    # 今日のレース（upcoming）を取得
    today = date.today()
    today_races = db.session.query(Race).filter(
        Race.race_date == today,
        Race.status == 'upcoming'
    ).order_by(Race.race_number).all()

    # 今日の予測済みレースを取得
    today_predicted_races = db.session.query(Race).join(Prediction).filter(
        Race.race_date == today
    ).distinct().order_by(Race.race_number).all()

    return render_template(
        'predictions/generate.html',
        running_task=running_task,
        recent_tasks=recent_tasks,
        today_races=today_races,
        today_predicted_races=today_predicted_races,
        today=today
    )


@predictions_bp.route('/api/generate/start', methods=['POST'])
def api_start_prediction():
    """
    予測タスクを開始

    Request JSON:
    {
        "date": "2025-01-17",  // optional, default: today
        "model_type": "xgboost",  // optional, default: xgboost
        "skip_scraping": false  // optional, default: false
    }

    Response JSON:
    {
        "success": true,
        "task_id": "uuid-string"
    }
    """
    try:
        data = request.get_json(silent=True) or {}

        # 日付をパース
        date_str = data.get('date')
        if date_str:
            try:
                target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': '日付形式が不正です (YYYY-MM-DD)'
                }), 400
        else:
            target_date = date.today()

        model_type = data.get('model_type', 'xgboost')
        if model_type not in ('xgboost', 'random_forest'):
            return jsonify({
                'success': False,
                'error': 'モデルタイプは xgboost または random_forest を指定してください'
            }), 400

        skip_scraping = data.get('skip_scraping', False)
        scraper_delay = max(2, data.get('scraper_delay', 3))

        # タスクを開始
        success, result, error_type = prediction_task_manager.start_task(
            app=current_app._get_current_object(),
            target_date=target_date,
            model_type=model_type,
            skip_scraping=skip_scraping,
            scraper_delay=scraper_delay
        )

        if success:
            logger.info(f"Started prediction task: {result}")
            return jsonify({
                'success': True,
                'task_id': result
            })
        else:
            status_code = 409 if error_type == 'concurrent_limit' else 400
            return jsonify({
                'success': False,
                'error': result,
                'error_type': error_type
            }), status_code

    except Exception as e:
        logger.exception("Error starting prediction task")
        return jsonify({
            'success': False,
            'error': f'内部エラー: {str(e)}'
        }), 500


@predictions_bp.route('/api/generate/progress/<task_id>')
def api_prediction_progress_stream(task_id):
    """
    SSEエンドポイント - リアルタイム予測進捗配信

    Returns Server-Sent Events stream with progress updates.
    """
    def generate():
        last_update = None

        while True:
            progress = prediction_task_manager.get_progress(task_id)

            if not progress:
                yield f"event: error\ndata: {json.dumps({'error': 'タスクが見つかりません'})}\n\n"
                break

            current_update = progress.get('updated_at')
            if current_update != last_update:
                yield f"data: {json.dumps(progress)}\n\n"
                last_update = current_update

            if progress['status'] in ('completed', 'failed', 'cancelled'):
                yield f"event: complete\ndata: {json.dumps(progress)}\n\n"
                break

            time.sleep(0.5)

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )


@predictions_bp.route('/api/generate/status/<task_id>')
def api_prediction_status(task_id):
    """タスク状態を取得（非ストリーミング）"""
    progress = prediction_task_manager.get_progress(task_id)

    if not progress:
        return jsonify({
            'success': False,
            'error': 'タスクが見つかりません'
        }), 404

    return jsonify({
        'success': True,
        'task': progress
    })


@predictions_bp.route('/api/generate/cancel/<task_id>', methods=['POST'])
def api_cancel_prediction(task_id):
    """タスクをキャンセル"""
    success = prediction_task_manager.cancel_task(task_id)

    if success:
        logger.info(f"Cancelled prediction task: {task_id}")
        return jsonify({
            'success': True,
            'message': 'キャンセルをリクエストしました'
        })
    else:
        return jsonify({
            'success': False,
            'error': 'タスクが見つからないか、既に完了しています'
        }), 404


@predictions_bp.route('/api/generate/running')
def api_get_running_prediction():
    """実行中のタスクを取得"""
    running_task = prediction_task_manager.get_running_task()

    return jsonify({
        'success': True,
        'task': running_task
    })
