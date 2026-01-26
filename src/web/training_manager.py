"""
Training task manager with thread-safe progress tracking.

This module manages ML model training tasks executed in background threads,
providing real-time progress updates via SSE.
"""
import os
import threading
import uuid
from datetime import datetime, timedelta
from typing import Callable, Dict, Optional, Any
from flask import Flask

from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)


class TrainingTaskManager:
    """
    Thread-safe manager for model training tasks.

    Handles task lifecycle including start, progress tracking, and cancellation.
    """

    # Maximum number of concurrent tasks
    MAX_CONCURRENT_TASKS = 1

    # Task TTL for cleanup (hours)
    TASK_TTL_HOURS = 24

    def __init__(self):
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._threads: Dict[str, threading.Thread] = {}

    def start_task(
        self,
        app: Flask,
        model_type: str,
        task_type: str,
        data_source: str = 'database',
        batch_size: int = 64,
        n_epochs: int = 100,
        learning_rate: float = 0.001,
        save_model: bool = True
    ) -> tuple[bool, str, Optional[str]]:
        """
        Start a new training task.

        Args:
            app: Flask application instance
            model_type: Model type (random_forest, xgboost, neural_network)
            task_type: Task type (regression, classification)
            data_source: Data source (database, csv)
            batch_size: Batch size for neural network
            n_epochs: Number of epochs for neural network
            learning_rate: Learning rate for neural network
            save_model: Whether to save the trained model

        Returns:
            Tuple of (success, task_id or error_message, error_type)
        """
        with self._lock:
            # Check for running tasks
            running_tasks = [
                tid for tid, task in self._tasks.items()
                if task['status'] == 'running'
            ]

            if len(running_tasks) >= self.MAX_CONCURRENT_TASKS:
                return False, "既に学習タスクが実行中です", "concurrent_limit"

            # Generate task ID
            task_id = str(uuid.uuid4())

            # Initialize task state
            self._tasks[task_id] = {
                'task_id': task_id,
                'status': 'running',
                'started_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'params': {
                    'model_type': model_type,
                    'task_type': task_type,
                    'data_source': data_source,
                    'batch_size': batch_size,
                    'n_epochs': n_epochs,
                    'learning_rate': learning_rate,
                    'save_model': save_model,
                },
                'progress': {
                    'phase': 'initializing',
                    'phase_text': '初期化中...',
                    'percent_complete': 0,
                    'current_epoch': 0,
                    'total_epochs': n_epochs if model_type == 'neural_network' else 0,
                    'train_loss': None,
                    'val_loss': None,
                    'samples_loaded': 0,
                    'features_count': 0,
                },
                'result': None,
                'error': None,
                'cancelled': False,
            }

        # Start worker thread
        thread = threading.Thread(
            target=self._training_worker,
            args=(app, task_id, model_type, task_type, data_source,
                  batch_size, n_epochs, learning_rate, save_model),
            daemon=True
        )
        self._threads[task_id] = thread
        thread.start()

        logger.info(f"Started training task {task_id} for {model_type}")
        return True, task_id, None

    def _training_worker(
        self,
        app: Flask,
        task_id: str,
        model_type: str,
        task_type: str,
        data_source: str,
        batch_size: int,
        n_epochs: int,
        learning_rate: float,
        save_model: bool
    ):
        """Worker function for training thread."""
        def progress_callback(data: Dict[str, Any]):
            self._update_progress(task_id, data)

        def cancel_check() -> bool:
            with self._lock:
                task = self._tasks.get(task_id)
                return task.get('cancelled', False) if task else True

        try:
            with app.app_context():
                result = self._run_training(
                    task_id=task_id,
                    model_type=model_type,
                    task_type=task_type,
                    data_source=data_source,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    learning_rate=learning_rate,
                    save_model=save_model,
                    progress_callback=progress_callback,
                    cancel_check=cancel_check
                )

                with self._lock:
                    if task_id in self._tasks:
                        if self._tasks[task_id].get('cancelled'):
                            self._tasks[task_id]['status'] = 'cancelled'
                        else:
                            self._tasks[task_id]['status'] = 'completed'
                            self._tasks[task_id]['result'] = result
                        self._tasks[task_id]['updated_at'] = datetime.now().isoformat()

                logger.info(f"Training task {task_id} completed")

        except Exception as e:
            logger.exception(f"Training task {task_id} failed")
            with self._lock:
                if task_id in self._tasks:
                    self._tasks[task_id]['status'] = 'failed'
                    self._tasks[task_id]['error'] = str(e)
                    self._tasks[task_id]['updated_at'] = datetime.now().isoformat()

    def _run_training(
        self,
        task_id: str,
        model_type: str,
        task_type: str,
        data_source: str,
        batch_size: int,
        n_epochs: int,
        learning_rate: float,
        save_model: bool,
        progress_callback: Callable,
        cancel_check: Callable
    ) -> Dict[str, Any]:
        """
        Run the actual training process.

        Returns result dictionary with metrics and model path.
        """
        import time
        start_time = time.time()

        from src.data.models import db
        from src.ml.feature_engineering import FeatureExtractor
        from src.ml.preprocessing import (
            FeaturePreprocessor,
            handle_missing_values,
            create_target_variable,
            split_by_date
        )
        from src.ml.models.random_forest import RandomForestRaceModel
        from src.ml.models.xgboost_model import XGBoostRaceModel
        from src.ml.models import TORCH_AVAILABLE
        from src.ml.evaluation import (
            evaluate_regression_model,
            evaluate_classification_model,
            evaluate_ranking_performance
        )

        # Phase 1: Loading data
        progress_callback({
            'event': 'phase_change',
            'phase': 'loading',
            'phase_text': 'データを読み込み中...',
            'percent_complete': 5
        })

        if cancel_check():
            return None

        # Load data from database
        extractor = FeatureExtractor(db.session, lookback_days=730)
        X, y = extractor.extract_features_for_training()

        if X.empty:
            raise ValueError("学習データが見つかりません")

        progress_callback({
            'event': 'data_loaded',
            'phase': 'loading',
            'phase_text': f'{len(X)}件のデータを読み込みました',
            'percent_complete': 15,
            'samples_loaded': len(X),
            'features_count': len(X.columns)
        })

        if cancel_check():
            return None

        # Phase 2: Preprocessing
        progress_callback({
            'event': 'phase_change',
            'phase': 'preprocessing',
            'phase_text': '前処理中...',
            'percent_complete': 20
        })

        # Handle missing values
        X = handle_missing_values(X, strategy='median')

        # Split data by date
        if 'race_date' in X.columns:
            all_dates = sorted(X['race_date'].unique())
            train_end_idx = int(len(all_dates) * 0.7)
            val_end_idx = int(len(all_dates) * 0.85)
            train_end_date = all_dates[train_end_idx]
            val_end_date = all_dates[val_end_idx]

            train_df, test_df, val_df = split_by_date(
                X.assign(target=y),
                date_column='race_date',
                train_end_date=train_end_date,
                val_end_date=val_end_date
            )

            X_train = train_df.drop('target', axis=1)
            y_train = train_df['target']
            X_val = val_df.drop('target', axis=1) if val_df is not None else None
            y_val = val_df['target'] if val_df is not None else None
            X_test = test_df.drop('target', axis=1)
            y_test = test_df['target']

            # race_dateは分割にのみ使用するため、学習前に削除
            # (datetime64型のままだとモデルがエラーを出す)
            for df in [X_train, X_val, X_test]:
                if df is not None and 'race_date' in df.columns:
                    df.drop('race_date', axis=1, inplace=True)
        else:
            from sklearn.model_selection import train_test_split
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42
            )

        if cancel_check():
            return None

        # Preprocess features
        preprocessor = FeaturePreprocessor()
        X_train_scaled = preprocessor.fit_transform(X_train)
        X_val_scaled = preprocessor.transform(X_val) if X_val is not None else None
        X_test_scaled = preprocessor.transform(X_test, keep_identifiers=True)

        # Create target variable for classification
        if task_type == 'classification':
            y_train = create_target_variable(y_train, target_type='binary_win')
            y_val = create_target_variable(y_val, target_type='binary_win') if y_val is not None else None
            y_test = create_target_variable(y_test, target_type='binary_win')

        progress_callback({
            'event': 'preprocessing_complete',
            'phase': 'preprocessing',
            'phase_text': f'前処理完了 (Train: {len(X_train)}, Val: {len(X_val) if X_val is not None else 0}, Test: {len(X_test)})',
            'percent_complete': 30
        })

        if cancel_check():
            return None

        # Phase 3: Training
        progress_callback({
            'event': 'phase_change',
            'phase': 'training',
            'phase_text': f'{model_type}モデルを学習中...',
            'percent_complete': 35
        })

        # Initialize model
        if model_type == 'random_forest':
            model = RandomForestRaceModel(task=task_type)
        elif model_type == 'xgboost':
            model = XGBoostRaceModel(task=task_type)
        elif model_type == 'neural_network':
            if not TORCH_AVAILABLE:
                raise ValueError("PyTorchがインストールされていません")
            from src.ml.models.neural_network import NeuralNetworkRaceModel
            model = NeuralNetworkRaceModel(
                task=task_type,
                batch_size=batch_size,
                n_epochs=n_epochs,
                learning_rate=learning_rate
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Train model with progress callback
        train_metrics = model.train(
            X_train_scaled,
            y_train,
            X_val_scaled,
            y_val,
            progress_callback=progress_callback,
            cancel_check=cancel_check
        )

        if cancel_check():
            return None

        # Phase 4: Evaluation
        progress_callback({
            'event': 'phase_change',
            'phase': 'evaluating',
            'phase_text': 'モデルを評価中...',
            'percent_complete': 85
        })

        y_test_pred = model.predict(X_test_scaled)

        if task_type == 'regression':
            test_metrics = evaluate_regression_model(y_test, y_test_pred, model_type)
            if 'race_id' in X_test.columns:
                ranking_metrics = evaluate_ranking_performance(
                    y_test, y_test_pred, X_test['race_id']
                )
                test_metrics.update(ranking_metrics)
        else:
            test_metrics = evaluate_classification_model(y_test, y_test_pred, model_type)

        # Get feature importance
        feature_importance = None
        if hasattr(model, 'get_feature_importance'):
            importance_series = model.get_feature_importance()
            if importance_series is not None and not importance_series.empty:
                # Series を DataFrame に変換してから to_dict('records') を使用
                importance_df = importance_series.head(15).reset_index()
                importance_df.columns = ['feature', 'importance']
                feature_importance = importance_df.to_dict('records')

        if cancel_check():
            return None

        # Phase 5: Saving
        model_path = None
        if save_model:
            progress_callback({
                'event': 'phase_change',
                'phase': 'saving',
                'phase_text': 'モデルを保存中...',
                'percent_complete': 95
            })

            output_dir = 'data/models'
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            ext = '.pt' if model_type == 'neural_network' else '.pkl'
            model_path = os.path.join(
                output_dir,
                f"{model_type}_{task_type}_{timestamp}{ext}"
            )
            model.save(model_path)

        # Complete
        training_time = time.time() - start_time
        progress_callback({
            'event': 'complete',
            'phase': 'complete',
            'phase_text': '学習完了',
            'percent_complete': 100
        })

        return {
            'model_path': model_path,
            'model_type': model_type,
            'task_type': task_type,
            'train_samples': len(X_train),
            'val_samples': len(X_val) if X_val is not None else 0,
            'test_samples': len(X_test),
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_importance': feature_importance,
            'training_time_seconds': round(training_time, 2)
        }

    def _update_progress(self, task_id: str, data: Dict[str, Any]):
        """Update task progress from callback data."""
        with self._lock:
            if task_id not in self._tasks:
                return

            task = self._tasks[task_id]
            task['updated_at'] = datetime.now().isoformat()

            # Update progress based on event type
            event = data.get('event', '')

            if 'phase' in data:
                task['progress']['phase'] = data['phase']

            if 'phase_text' in data:
                task['progress']['phase_text'] = data['phase_text']

            if 'percent_complete' in data:
                task['progress']['percent_complete'] = data['percent_complete']

            if 'samples_loaded' in data:
                task['progress']['samples_loaded'] = data['samples_loaded']

            if 'features_count' in data:
                task['progress']['features_count'] = data['features_count']

            # Neural network epoch progress
            if event == 'epoch_complete':
                task['progress']['current_epoch'] = data.get('epoch', 0)
                task['progress']['total_epochs'] = data.get('total_epochs', 0)
                task['progress']['train_loss'] = data.get('train_loss')
                task['progress']['val_loss'] = data.get('val_loss')

                # Calculate percent based on epoch for NN
                if task['progress']['total_epochs'] > 0:
                    epoch_progress = data.get('epoch', 0) / task['progress']['total_epochs']
                    # Training phase is 35-85%
                    task['progress']['percent_complete'] = 35 + (epoch_progress * 50)

    def get_progress(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current progress for a task.

        Args:
            task_id: Task identifier

        Returns:
            Task state dictionary or None if not found
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                return task.copy()
            return None

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task.

        Args:
            task_id: Task identifier

        Returns:
            True if cancellation was requested, False if task not found
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task['status'] == 'running':
                task['cancelled'] = True
                logger.info(f"Cancellation requested for task {task_id}")
                return True
            return False

    def get_running_task(self) -> Optional[Dict[str, Any]]:
        """Get currently running task if any."""
        with self._lock:
            for task in self._tasks.values():
                if task['status'] == 'running':
                    return task.copy()
            return None

    def get_recent_tasks(self, limit: int = 10) -> list[Dict[str, Any]]:
        """
        Get recent tasks sorted by start time.

        Args:
            limit: Maximum number of tasks to return

        Returns:
            List of task dictionaries
        """
        with self._lock:
            tasks = list(self._tasks.values())
            tasks.sort(key=lambda x: x['started_at'], reverse=True)
            return [t.copy() for t in tasks[:limit]]

    def cleanup_old_tasks(self):
        """Remove tasks older than TTL."""
        cutoff = datetime.now() - timedelta(hours=self.TASK_TTL_HOURS)
        cutoff_str = cutoff.isoformat()

        with self._lock:
            to_remove = [
                tid for tid, task in self._tasks.items()
                if task['status'] != 'running' and task['started_at'] < cutoff_str
            ]

            for tid in to_remove:
                del self._tasks[tid]
                if tid in self._threads:
                    del self._threads[tid]

            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} old training tasks")

    def get_saved_models(self) -> list[Dict[str, Any]]:
        """
        Get list of saved models in the models directory.

        Returns:
            List of model info dictionaries
        """
        models_dir = 'data/models'
        models = []

        if not os.path.exists(models_dir):
            return models

        for filename in os.listdir(models_dir):
            if filename.endswith(('.pkl', '.pt')):
                filepath = os.path.join(models_dir, filename)
                stat = os.stat(filepath)

                # Parse filename: modeltype_tasktype_timestamp.ext
                parts = filename.rsplit('.', 1)[0].split('_')
                if len(parts) >= 3:
                    model_type = parts[0]
                    task_type = parts[1]
                    timestamp = '_'.join(parts[2:])
                else:
                    model_type = 'unknown'
                    task_type = 'unknown'
                    timestamp = ''

                models.append({
                    'filename': filename,
                    'model_type': model_type,
                    'task_type': task_type,
                    'timestamp': timestamp,
                    'size_bytes': stat.st_size,
                    'created_at': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })

        # Sort by creation time (newest first)
        models.sort(key=lambda x: x['created_at'], reverse=True)
        return models


# Global singleton instance
training_manager = TrainingTaskManager()
