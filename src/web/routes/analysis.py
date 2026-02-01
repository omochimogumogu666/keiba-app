"""
Analysis routes for prediction accuracy analysis.

予測精度分析、モデル比較、時系列パフォーマンス追跡のルート。
"""
from flask import Blueprint, render_template, request, jsonify
from datetime import datetime, date, timedelta
from sqlalchemy import func
from src.data.models import db, PredictionAccuracy
from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)

analysis_bp = Blueprint('analysis', __name__, url_prefix='/analysis')


@analysis_bp.route('/accuracy')
def accuracy_dashboard():
    """精度分析ダッシュボード画面"""
    return render_template('analysis/accuracy.html')


@analysis_bp.route('/api/accuracy/summary')
def get_accuracy_summary():
    """
    集計サマリーAPIエンドポイント

    Query Parameters:
        - start_date (required): YYYY-MM-DD
        - end_date (required): YYYY-MM-DD
        - model_name (optional): モデル名フィルター
        - aggregation (optional): daily/weekly/monthly (default: daily)

    Returns:
        {
            "period": {"start": "2026-01-01", "end": "2026-01-31"},
            "overall": {
                "total_races": 100,
                "total_predictions": 1600,
                "win_accuracy": 25.5,
                "top3_accuracy": 62.3,
                "roi": -12.8
            }
        }
    """
    try:
        # パラメータ取得
        start_str = request.args.get('start_date')
        end_str = request.args.get('end_date')
        model_name = request.args.get('model_name')
        aggregation = request.args.get('aggregation', 'daily')

        if not start_str or not end_str:
            return jsonify({'error': 'start_date and end_date are required'}), 400

        start_date = datetime.strptime(start_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_str, '%Y-%m-%d').date()

        # クエリ構築
        query = PredictionAccuracy.query.filter(
            PredictionAccuracy.aggregation_type == aggregation,
            PredictionAccuracy.aggregation_date.between(start_date, end_date),
            PredictionAccuracy.track_id.is_(None),
            PredictionAccuracy.race_class.is_(None)
        )

        if model_name:
            query = query.filter(PredictionAccuracy.model_name == model_name)

        records = query.all()

        if not records:
            return jsonify({
                'period': {'start': start_str, 'end': end_str},
                'overall': {
                    'total_races': 0,
                    'total_predictions': 0,
                    'win_accuracy': 0.0,
                    'top3_accuracy': 0.0,
                    'roi': 0.0
                }
            })

        # 集計
        total_races = sum(r.total_races for r in records)
        total_predictions = sum(r.total_predictions for r in records)
        win_predictions = sum(r.win_predictions for r in records)
        win_hits = sum(r.win_hits for r in records)
        top3_predictions = sum(r.top3_predictions for r in records)
        top3_hits = sum(r.top3_hits for r in records)
        total_bet = sum(r.total_bet_amount for r in records)
        total_return = sum(r.total_return_amount for r in records)

        # 平均計算
        win_accuracy = (win_hits / win_predictions * 100) if win_predictions > 0 else 0.0
        top3_accuracy = (top3_hits / top3_predictions * 100) if top3_predictions > 0 else 0.0
        roi = ((total_return - total_bet) / total_bet * 100) if total_bet > 0 else 0.0

        return jsonify({
            'period': {'start': start_str, 'end': end_str},
            'overall': {
                'total_races': total_races,
                'total_predictions': total_predictions,
                'win_accuracy': round(win_accuracy, 2),
                'top3_accuracy': round(top3_accuracy, 2),
                'roi': round(roi, 2)
            }
        })

    except ValueError as e:
        return jsonify({'error': f'Invalid date format: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Error in accuracy summary API: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


@analysis_bp.route('/api/accuracy/timeseries')
def get_accuracy_timeseries():
    """
    時系列データAPIエンドポイント（Chart.js用）

    Query Parameters:
        - start_date, end_date (required)
        - model_name (optional)
        - aggregation (optional): daily/weekly/monthly (default: daily)

    Returns:
        {
            "labels": ["2026-01-01", "2026-01-02", ...],
            "datasets": [
                {"label": "的中率", "data": [25.0, 30.5, ...]},
                {"label": "複勝圏内率", "data": [60.0, 65.2, ...]},
                {"label": "ROI", "data": [-15.2, -8.3, ...]}
            ]
        }
    """
    try:
        start_str = request.args.get('start_date')
        end_str = request.args.get('end_date')
        model_name = request.args.get('model_name')
        aggregation = request.args.get('aggregation', 'daily')

        if not start_str or not end_str:
            return jsonify({'error': 'start_date and end_date are required'}), 400

        start_date = datetime.strptime(start_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_str, '%Y-%m-%d').date()

        # クエリ
        query = PredictionAccuracy.query.filter(
            PredictionAccuracy.aggregation_type == aggregation,
            PredictionAccuracy.aggregation_date.between(start_date, end_date),
            PredictionAccuracy.track_id.is_(None),
            PredictionAccuracy.race_class.is_(None)
        )

        if model_name:
            query = query.filter(PredictionAccuracy.model_name == model_name)

        records = query.order_by(PredictionAccuracy.aggregation_date).all()

        # Chart.js形式に変換
        labels = [r.aggregation_date.strftime('%Y-%m-%d') for r in records]
        win_accuracy_data = [r.win_accuracy or 0.0 for r in records]
        top3_accuracy_data = [r.top3_accuracy or 0.0 for r in records]
        roi_data = [r.roi or 0.0 for r in records]

        return jsonify({
            'labels': labels,
            'datasets': [
                {
                    'label': '的中率 (%)',
                    'data': win_accuracy_data,
                    'borderColor': 'rgb(75, 192, 192)',
                    'backgroundColor': 'rgba(75, 192, 192, 0.2)'
                },
                {
                    'label': '複勝圏内率 (%)',
                    'data': top3_accuracy_data,
                    'borderColor': 'rgb(54, 162, 235)',
                    'backgroundColor': 'rgba(54, 162, 235, 0.2)'
                },
                {
                    'label': 'ROI (%)',
                    'data': roi_data,
                    'borderColor': 'rgb(255, 99, 132)',
                    'backgroundColor': 'rgba(255, 99, 132, 0.2)'
                }
            ]
        })

    except ValueError as e:
        return jsonify({'error': f'Invalid date format: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Error in timeseries API: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


@analysis_bp.route('/api/accuracy/models')
def get_model_comparison():
    """
    モデル比較APIエンドポイント

    Query Parameters:
        - start_date, end_date (required)
        - aggregation (optional): daily/weekly/monthly (default: daily)

    Returns:
        {
            "models": [
                {
                    "model_name": "xgboost",
                    "win_accuracy": 26.2,
                    "top3_accuracy": 64.1,
                    "roi": -10.5
                },
                ...
            ]
        }
    """
    try:
        start_str = request.args.get('start_date')
        end_str = request.args.get('end_date')
        aggregation = request.args.get('aggregation', 'daily')

        if not start_str or not end_str:
            return jsonify({'error': 'start_date and end_date are required'}), 400

        start_date = datetime.strptime(start_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_str, '%Y-%m-%d').date()

        # モデル別に集計
        query = db.session.query(
            PredictionAccuracy.model_name,
            func.sum(PredictionAccuracy.win_predictions).label('total_win_pred'),
            func.sum(PredictionAccuracy.win_hits).label('total_win_hits'),
            func.sum(PredictionAccuracy.top3_predictions).label('total_top3_pred'),
            func.sum(PredictionAccuracy.top3_hits).label('total_top3_hits'),
            func.sum(PredictionAccuracy.total_bet_amount).label('total_bet'),
            func.sum(PredictionAccuracy.total_return_amount).label('total_return')
        ).filter(
            PredictionAccuracy.aggregation_type == aggregation,
            PredictionAccuracy.aggregation_date.between(start_date, end_date),
            PredictionAccuracy.track_id.is_(None),
            PredictionAccuracy.race_class.is_(None)
        ).group_by(PredictionAccuracy.model_name).all()

        models = []
        for row in query:
            win_acc = (row.total_win_hits / row.total_win_pred * 100) if row.total_win_pred > 0 else 0.0
            top3_acc = (row.total_top3_hits / row.total_top3_pred * 100) if row.total_top3_pred > 0 else 0.0
            roi = ((row.total_return - row.total_bet) / row.total_bet * 100) if row.total_bet > 0 else 0.0

            models.append({
                'model_name': row.model_name,
                'win_accuracy': round(win_acc, 2),
                'top3_accuracy': round(top3_acc, 2),
                'roi': round(roi, 2)
            })

        return jsonify({'models': models})

    except ValueError as e:
        return jsonify({'error': f'Invalid date format: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Error in models API: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500
