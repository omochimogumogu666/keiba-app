"""
Training routes for web-based model training with real-time progress.

Provides endpoints for starting, monitoring, and cancelling training tasks
using Server-Sent Events (SSE) for real-time progress updates.
"""
import json
import time
from flask import Blueprint, request, jsonify, render_template, Response, current_app
from src.web.training_manager import training_manager
from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)

training_bp = Blueprint('training', __name__, url_prefix='/training')


@training_bp.route('/')
def index():
    """モデル学習管理画面"""
    running_task = training_manager.get_running_task()
    recent_tasks = training_manager.get_recent_tasks(limit=5)
    saved_models = training_manager.get_saved_models()
    return render_template('training/index.html',
                          running_task=running_task,
                          recent_tasks=recent_tasks,
                          saved_models=saved_models)


@training_bp.route('/api/start', methods=['POST'])
def api_start_training():
    """
    学習タスクを開始

    Request JSON:
    {
        "model_type": "random_forest",
        "task_type": "regression",
        "data_source": "database",
        "batch_size": 64,
        "n_epochs": 100,
        "learning_rate": 0.001,
        "save_model": true
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

        # Validate model type
        model_type = data.get('model_type', 'random_forest')
        if model_type not in ('random_forest', 'xgboost', 'neural_network'):
            return jsonify({
                'success': False,
                'error': f'無効なモデルタイプ: {model_type}'
            }), 400

        # Validate task type
        task_type = data.get('task_type', 'regression')
        if task_type not in ('regression', 'classification'):
            return jsonify({
                'success': False,
                'error': f'無効なタスクタイプ: {task_type}'
            }), 400

        # Extract parameters
        data_source = data.get('data_source', 'database')
        batch_size = data.get('batch_size', 64)
        n_epochs = data.get('n_epochs', 100)
        learning_rate = data.get('learning_rate', 0.001)
        save_model = data.get('save_model', True)

        # Validate numeric parameters
        try:
            batch_size = int(batch_size)
            n_epochs = int(n_epochs)
            learning_rate = float(learning_rate)

            if batch_size < 1 or batch_size > 1024:
                raise ValueError("バッチサイズは1-1024の範囲で指定してください")
            if n_epochs < 1 or n_epochs > 1000:
                raise ValueError("エポック数は1-1000の範囲で指定してください")
            if learning_rate <= 0 or learning_rate > 1:
                raise ValueError("学習率は0より大きく1以下で指定してください")

        except (ValueError, TypeError) as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 400

        # Start task
        success, result, error_type = training_manager.start_task(
            app=current_app._get_current_object(),
            model_type=model_type,
            task_type=task_type,
            data_source=data_source,
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            save_model=save_model
        )

        if success:
            logger.info(f"Started training task: {result}")
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
        logger.exception("Error starting training task")
        return jsonify({
            'success': False,
            'error': f'内部エラー: {str(e)}'
        }), 500


@training_bp.route('/api/progress/<task_id>')
def api_progress_stream(task_id):
    """
    SSEエンドポイント - リアルタイムプログレス配信

    Returns Server-Sent Events stream with progress updates.
    """
    def generate():
        last_update = None

        while True:
            progress = training_manager.get_progress(task_id)

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


@training_bp.route('/api/status/<task_id>')
def api_get_status(task_id):
    """
    タスク状態を取得（非ストリーミング）

    Response JSON:
    {
        "task_id": "uuid",
        "status": "running|completed|failed|cancelled",
        "progress": {...},
        "result": {...},
        "error": null
    }
    """
    progress = training_manager.get_progress(task_id)

    if not progress:
        return jsonify({
            'success': False,
            'error': 'タスクが見つかりません'
        }), 404

    return jsonify({
        'success': True,
        'task': progress
    })


@training_bp.route('/api/cancel/<task_id>', methods=['POST'])
def api_cancel_task(task_id):
    """
    タスクをキャンセル

    Response JSON:
    {
        "success": true,
        "message": "キャンセルをリクエストしました"
    }
    """
    success = training_manager.cancel_task(task_id)

    if success:
        logger.info(f"Cancelled training task: {task_id}")
        return jsonify({
            'success': True,
            'message': 'キャンセルをリクエストしました'
        })
    else:
        return jsonify({
            'success': False,
            'error': 'タスクが見つからないか、既に完了しています'
        }), 404


@training_bp.route('/api/running')
def api_get_running():
    """
    実行中のタスクを取得

    Response JSON:
    {
        "success": true,
        "task": {...} or null
    }
    """
    running_task = training_manager.get_running_task()

    return jsonify({
        'success': True,
        'task': running_task
    })


@training_bp.route('/api/history')
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

    tasks = training_manager.get_recent_tasks(limit=limit)

    return jsonify({
        'success': True,
        'tasks': tasks
    })


@training_bp.route('/api/models')
def api_get_models():
    """
    保存済みモデル一覧を取得

    Response JSON:
    {
        "success": true,
        "models": [...]
    }
    """
    models = training_manager.get_saved_models()

    return jsonify({
        'success': True,
        'models': models
    })
