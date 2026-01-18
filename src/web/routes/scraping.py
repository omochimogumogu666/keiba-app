"""
Scraping routes for web-based scraping execution with real-time progress.

Provides endpoints for starting, monitoring, and cancelling scraping tasks
using Server-Sent Events (SSE) for real-time progress updates.
"""
import json
import time
from datetime import datetime
from flask import Blueprint, request, jsonify, render_template, Response, current_app
from src.web.scraping_manager import task_manager
from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)

scraping_bp = Blueprint('scraping', __name__, url_prefix='/scraping')


@scraping_bp.route('/')
def index():
    """スクレイピング管理画面"""
    running_task = task_manager.get_running_task()
    recent_tasks = task_manager.get_recent_tasks(limit=5)
    return render_template('scraping/index.html',
                          running_task=running_task,
                          recent_tasks=recent_tasks)


@scraping_bp.route('/api/start', methods=['POST'])
def api_start_scraping():
    """
    スクレイピングタスクを開始

    Request JSON:
    {
        "start_date": "2025-01-01",
        "end_date": "2025-01-31",
        "weekends_only": true
    }

    Response JSON:
    {
        "success": true,
        "task_id": "uuid-string"
    }
    """
    try:
        data = request.get_json(silent=True)

        if not data:
            return jsonify({'success': False, 'error': 'リクエストボディが必要です'}), 400

        # Parse dates
        try:
            start_date_str = data.get('start_date')
            end_date_str = data.get('end_date')

            if not start_date_str or not end_date_str:
                return jsonify({
                    'success': False,
                    'error': '開始日と終了日は必須です'
                }), 400

            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

        except ValueError as e:
            return jsonify({
                'success': False,
                'error': f'日付形式が不正です (YYYY-MM-DD): {e}'
            }), 400

        # Validate date range
        if start_date > end_date:
            return jsonify({
                'success': False,
                'error': '開始日は終了日より前である必要があります'
            }), 400

        # Prevent scraping future dates
        if end_date > datetime.now():
            return jsonify({
                'success': False,
                'error': '未来の日付はスクレイピングできません'
            }), 400

        weekends_only = data.get('weekends_only', True)
        scraper_delay = data.get('scraper_delay', 3)

        # Ensure minimum delay for rate limiting
        if scraper_delay < 2:
            scraper_delay = 2

        # Start task
        success, result, error_type = task_manager.start_task(
            app=current_app._get_current_object(),
            start_date=start_date,
            end_date=end_date,
            weekends_only=weekends_only,
            scraper_delay=scraper_delay
        )

        if success:
            logger.info(f"Started scraping task: {result}")
            return jsonify({
                'success': True,
                'task_id': result
            })
        else:
            return jsonify({
                'success': False,
                'error': result,
                'error_type': error_type
            }), 409 if error_type == 'concurrent_limit' else 400

    except Exception as e:
        logger.exception("Error starting scraping task")
        return jsonify({
            'success': False,
            'error': f'内部エラー: {str(e)}'
        }), 500


@scraping_bp.route('/api/progress/<task_id>')
def api_progress_stream(task_id):
    """
    SSEエンドポイント - リアルタイムプログレス配信

    Returns Server-Sent Events stream with progress updates.
    """
    def generate():
        last_update = None

        while True:
            progress = task_manager.get_progress(task_id)

            if not progress:
                yield f"event: error\ndata: {json.dumps({'error': 'タスクが見つかりません'})}\n\n"
                break

            # Only send if there's an update
            current_update = progress.get('updated_at')
            if current_update != last_update:
                yield f"data: {json.dumps(progress)}\n\n"
                last_update = current_update

            # Stop if task is finished
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


@scraping_bp.route('/api/status/<task_id>')
def api_get_status(task_id):
    """
    タスク状態を取得（非ストリーミング）

    Response JSON:
    {
        "task_id": "uuid",
        "status": "running|completed|failed|cancelled",
        "progress": {...},
        "error": null,
        "stats": {...}
    }
    """
    progress = task_manager.get_progress(task_id)

    if not progress:
        return jsonify({
            'success': False,
            'error': 'タスクが見つかりません'
        }), 404

    return jsonify({
        'success': True,
        'task': progress
    })


@scraping_bp.route('/api/cancel/<task_id>', methods=['POST'])
def api_cancel_task(task_id):
    """
    タスクをキャンセル

    Response JSON:
    {
        "success": true,
        "message": "キャンセルをリクエストしました"
    }
    """
    success = task_manager.cancel_task(task_id)

    if success:
        logger.info(f"Cancelled scraping task: {task_id}")
        return jsonify({
            'success': True,
            'message': 'キャンセルをリクエストしました'
        })
    else:
        return jsonify({
            'success': False,
            'error': 'タスクが見つからないか、既に完了しています'
        }), 404


@scraping_bp.route('/api/running')
def api_get_running():
    """
    実行中のタスクを取得

    Response JSON:
    {
        "success": true,
        "task": {...} or null
    }
    """
    running_task = task_manager.get_running_task()

    return jsonify({
        'success': True,
        'task': running_task
    })


@scraping_bp.route('/api/history')
def api_get_history():
    """
    最近のタスク履歴を取得

    Response JSON:
    {
        "success": true,
        "tasks": [...]
    }
    """
    limit = request.args.get('limit', 10, type=int)
    limit = min(limit, 50)

    tasks = task_manager.get_recent_tasks(limit=limit)

    return jsonify({
        'success': True,
        'tasks': tasks
    })
