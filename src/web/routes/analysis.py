"""
Analysis routes for prediction performance analysis.

予測精度分析、モデル比較、パフォーマンス追跡のルート。
"""
from flask import Blueprint, render_template, request, jsonify
from src.data.statistics import (
    calculate_prediction_accuracy,
    calculate_roi_by_model,
    calculate_daily_performance,
    calculate_track_accuracy,
    get_model_comparison_summary
)
from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)

analysis_bp = Blueprint('analysis', __name__, url_prefix='/analysis')


@analysis_bp.route('/')
def index():
    """分析ダッシュボード"""
    try:
        # 全体の精度統計（過去30日）
        overall_stats = calculate_prediction_accuracy(days=30)

        # モデル別比較
        model_comparison = get_model_comparison_summary(days=30)

        # 日別パフォーマンス
        daily_performance = calculate_daily_performance(days=30)

        # 競馬場別精度
        track_accuracy = calculate_track_accuracy(days=90)

        return render_template(
            'predictions/analysis.html',
            overall_stats=overall_stats,
            model_comparison=model_comparison,
            daily_performance=daily_performance,
            track_accuracy=track_accuracy
        )

    except Exception as e:
        logger.error(f"Error loading analysis dashboard: {e}")
        return render_template(
            'predictions/analysis.html',
            overall_stats={},
            model_comparison=[],
            daily_performance=[],
            track_accuracy={},
            error=str(e)
        )


@analysis_bp.route('/api/accuracy')
def api_accuracy():
    """精度データAPI"""
    try:
        model_name = request.args.get('model')
        days = request.args.get('days', 30, type=int)

        stats = calculate_prediction_accuracy(model_name=model_name, days=days)

        return jsonify({
            'success': True,
            'data': stats
        })

    except Exception as e:
        logger.error(f"Error in accuracy API: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@analysis_bp.route('/api/daily')
def api_daily():
    """日別パフォーマンスAPI"""
    try:
        model_name = request.args.get('model')
        days = request.args.get('days', 30, type=int)

        data = calculate_daily_performance(model_name=model_name, days=days)

        return jsonify({
            'success': True,
            'data': data
        })

    except Exception as e:
        logger.error(f"Error in daily API: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@analysis_bp.route('/api/models')
def api_models():
    """モデル比較API"""
    try:
        days = request.args.get('days', 30, type=int)

        data = get_model_comparison_summary(days=days)

        return jsonify({
            'success': True,
            'data': data
        })

    except Exception as e:
        logger.error(f"Error in models API: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@analysis_bp.route('/api/tracks')
def api_tracks():
    """競馬場別精度API"""
    try:
        model_name = request.args.get('model')
        days = request.args.get('days', 90, type=int)

        data = calculate_track_accuracy(model_name=model_name, days=days)

        return jsonify({
            'success': True,
            'data': data
        })

    except Exception as e:
        logger.error(f"Error in tracks API: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
