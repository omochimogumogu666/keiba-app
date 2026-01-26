"""
XGBoost model for horse racing prediction.
"""
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

from src.ml.models.base_model import BaseRaceModel
from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)


def _load_optimized_params(model_type: str, task: str, params_dir: str = 'data/models'):
    """最適化済みパラメータを読み込むヘルパー関数"""
    from src.ml.hyperparameter_tuning import find_latest_params, load_best_params
    params_path = find_latest_params(model_type, task, params_dir)
    if params_path is None:
        return None
    return load_best_params(params_path)


class XGBoostRaceModel(BaseRaceModel):
    """
    XGBoost model for race prediction.

    Supports both classification (win/place prediction) and
    regression (finish position prediction).
    """

    def __init__(
        self,
        task: str = 'regression',
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        version: str = "1.0",
        **kwargs
    ):
        """
        Initialize XGBoost model.

        Args:
            task: 'regression' or 'classification'
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns for each tree
            random_state: Random seed for reproducibility
            version: Model version
            **kwargs: Additional XGBoost parameters
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost is not installed. Install it with: pip install xgboost"
            )

        super().__init__(model_name=f'XGBoost_{task}', version=version)

        self.task = task
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'random_state': random_state,
            **kwargs
        }

        # Initialize appropriate model based on task
        if task == 'regression':
            self.model = xgb.XGBRegressor(**self.params)
        elif task == 'classification':
            self.model = xgb.XGBClassifier(**self.params)
        else:
            raise ValueError(f"Unknown task: {task}. Must be 'regression' or 'classification'")

    @classmethod
    def from_optimized_params(
        cls,
        task: str = 'regression',
        params_dir: str = 'data/models',
        version: str = "optimized"
    ) -> 'XGBoostRaceModel':
        """
        最適化済みパラメータからモデルを作成する。

        Args:
            task: 'regression' or 'classification'
            params_dir: パラメータ保存ディレクトリ
            version: モデルバージョン

        Returns:
            最適化パラメータで初期化されたXGBoostRaceModel

        Raises:
            FileNotFoundError: 最適化パラメータが見つからない場合
        """
        params = _load_optimized_params('xgboost', task, params_dir)
        if params is None:
            raise FileNotFoundError(
                f"Optimized parameters not found for xgboost_{task} in {params_dir}. "
                f"Run 'python scripts/optimize_hyperparameters.py --model xgboost --task {task}' first."
            )
        logger.info(f"Creating XGBoostRaceModel with optimized params: {params}")
        return cls(task=task, version=version, **params)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        early_stopping_rounds: Optional[int] = 10,
        verbose: bool = False,
        progress_callback: Optional[Any] = None,
        cancel_check: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the XGBoost model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            early_stopping_rounds: Rounds for early stopping (None = disabled)
            verbose: Whether to print training progress
            progress_callback: Progress callback function (optional)
            cancel_check: Cancel check function (optional)
            **kwargs: Additional training parameters

        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training {self.model_name} on {len(X_train)} samples")

        # キャンセルチェック
        if cancel_check and cancel_check():
            return {}

        # Store feature columns
        self.feature_columns = X_train.columns.tolist()

        # 進捗コールバック（学習開始）
        if progress_callback:
            progress_callback({
                'event': 'training_start',
                'model_type': 'xgboost'
            })

        # Prepare evaluation set for early stopping
        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
        else:
            eval_set = [(X_train, y_train)]

        # Train model
        train_start = datetime.utcnow()

        fit_params = {
            'eval_set': eval_set,
            'verbose': verbose
        }

        if early_stopping_rounds is not None and X_val is not None:
            fit_params['early_stopping_rounds'] = early_stopping_rounds

        self.model.fit(X_train, y_train, **fit_params)

        train_time = (datetime.utcnow() - train_start).total_seconds()

        # 進捗コールバック（学習完了）
        if progress_callback:
            progress_callback({
                'event': 'training_complete',
                'model_type': 'xgboost',
                'training_time_seconds': train_time,
                'best_iteration': getattr(self.model, 'best_iteration', None)
            })
        self.is_trained = True

        # Calculate training metrics
        y_train_pred = self.model.predict(X_train)

        metrics = {
            'training_time_seconds': train_time,
            'n_samples_train': len(X_train),
            'n_features': len(self.feature_columns),
            'task': self.task,
            'best_iteration': getattr(self.model, 'best_iteration', None)
        }

        if self.task == 'regression':
            train_mse = mean_squared_error(y_train, y_train_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            train_rmse = np.sqrt(train_mse)

            metrics.update({
                'train_mse': train_mse,
                'train_mae': train_mae,
                'train_rmse': train_rmse
            })

            logger.info(f"Training RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")

        elif self.task == 'classification':
            train_acc = accuracy_score(y_train, y_train_pred)
            metrics['train_accuracy'] = train_acc

            logger.info(f"Training accuracy: {train_acc:.4f}")

        # Validation metrics if provided
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            metrics['n_samples_val'] = len(X_val)

            if self.task == 'regression':
                val_mse = mean_squared_error(y_val, y_val_pred)
                val_mae = mean_absolute_error(y_val, y_val_pred)
                val_rmse = np.sqrt(val_mse)

                metrics.update({
                    'val_mse': val_mse,
                    'val_mae': val_mae,
                    'val_rmse': val_rmse
                })

                logger.info(f"Validation RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}")

            elif self.task == 'classification':
                val_acc = accuracy_score(y_val, y_val_pred)
                metrics['val_accuracy'] = val_acc

                logger.info(f"Validation accuracy: {val_acc:.4f}")

        # Store metadata
        self.training_metadata = {
            'trained_at': datetime.utcnow().isoformat(),
            'metrics': metrics,
            'params': self.params
        }

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Features DataFrame

        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        X_prepared = self._prepare_features(X)
        predictions = self.model.predict(X_prepared)

        return predictions

    def predict_top_n(
        self,
        X: pd.DataFrame,
        n: int = 3,
        return_probabilities: bool = False
    ) -> pd.DataFrame:
        """
        Predict top N horses for each race.

        For regression: Ranks horses by predicted finish position
        For classification: Ranks horses by win probability

        Args:
            X: Features DataFrame (must include 'race_id' column)
            n: Number of top horses to return per race
            return_probabilities: Whether to include probabilities (classification only)

        Returns:
            DataFrame with race_id, horse_id, rank, and optionally probability
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        if 'race_id' not in X.columns or 'horse_id' not in X.columns:
            raise ValueError("X must include 'race_id' and 'horse_id' columns")

        # Get predictions
        if self.task == 'regression':
            predictions = self.predict(X)
            X_with_pred = X[['race_id', 'horse_id']].copy()
            X_with_pred['predicted_position'] = predictions

            # Rank by predicted position (lower is better)
            X_with_pred['rank'] = X_with_pred.groupby('race_id')['predicted_position'].rank(method='first')

        elif self.task == 'classification':
            # Get win probabilities
            probas = self.predict_proba(X)
            # Assuming class 1 is "win" for binary, or highest class for multiclass
            if probas.shape[1] == 2:
                win_proba = probas[:, 1]
            else:
                # For multiclass, use highest class probability
                win_proba = probas[:, -1]

            X_with_pred = X[['race_id', 'horse_id']].copy()
            X_with_pred['win_probability'] = win_proba

            # Rank by win probability (higher is better)
            X_with_pred['rank'] = X_with_pred.groupby('race_id')['win_probability'].rank(
                method='first', ascending=False
            )

        # Filter top N per race
        top_n = X_with_pred[X_with_pred['rank'] <= n].copy()
        top_n = top_n.sort_values(['race_id', 'rank'])

        return top_n

    def save(self, filepath: str) -> None:
        """
        Save model to file.

        Extends base class to save task attribute.

        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'version': self.version,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained,
            'training_metadata': self.training_metadata,
            'task': self.task,  # Save task attribute
            'params': self.params  # Save model parameters
        }

        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """
        Load model from file.

        Extends base class to load task attribute.

        Args:
            filepath: Path to load the model from
        """
        import os
        import pickle

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.version = model_data['version']
        self.feature_columns = model_data['feature_columns']
        self.is_trained = model_data['is_trained']
        self.training_metadata = model_data.get('training_metadata', {})
        self.task = model_data.get('task', 'regression')  # Load task attribute with default
        self.params = model_data.get('params', {})  # Load model parameters

        logger.info(f"Model loaded from {filepath} (task: {self.task})")
