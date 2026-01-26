"""
Random Forest model for horse racing prediction.
"""
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from datetime import datetime

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


class RandomForestRaceModel(BaseRaceModel):
    """
    Random Forest model for race prediction.

    Supports both classification (win/place prediction) and
    regression (finish position prediction).
    """

    def __init__(
        self,
        task: str = 'regression',
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
        version: str = "1.0",
        **kwargs
    ):
        """
        Initialize Random Forest model.

        Args:
            task: 'regression' or 'classification'
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (None = unlimited)
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at leaf node
            random_state: Random seed for reproducibility
            version: Model version
            **kwargs: Additional sklearn RandomForest parameters
        """
        super().__init__(model_name=f'RandomForest_{task}', version=version)

        self.task = task
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'random_state': random_state,
            **kwargs
        }

        # Initialize appropriate model based on task
        if task == 'regression':
            self.model = RandomForestRegressor(**self.params)
        elif task == 'classification':
            self.model = RandomForestClassifier(**self.params)
        else:
            raise ValueError(f"Unknown task: {task}. Must be 'regression' or 'classification'")

    @classmethod
    def from_optimized_params(
        cls,
        task: str = 'regression',
        params_dir: str = 'data/models',
        version: str = "optimized"
    ) -> 'RandomForestRaceModel':
        """
        最適化済みパラメータからモデルを作成する。

        Args:
            task: 'regression' or 'classification'
            params_dir: パラメータ保存ディレクトリ
            version: モデルバージョン

        Returns:
            最適化パラメータで初期化されたRandomForestRaceModel

        Raises:
            FileNotFoundError: 最適化パラメータが見つからない場合
        """
        params = _load_optimized_params('random_forest', task, params_dir)
        if params is None:
            raise FileNotFoundError(
                f"Optimized parameters not found for random_forest_{task} in {params_dir}. "
                f"Run 'python scripts/optimize_hyperparameters.py --model random_forest --task {task}' first."
            )
        logger.info(f"Creating RandomForestRaceModel with optimized params: {params}")
        return cls(task=task, version=version, **params)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        progress_callback: Optional[Any] = None,
        cancel_check: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the Random Forest model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
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
                'model_type': 'random_forest'
            })

        # Train model
        train_start = datetime.utcnow()
        self.model.fit(X_train, y_train)
        train_time = (datetime.utcnow() - train_start).total_seconds()

        # 進捗コールバック（学習完了）
        if progress_callback:
            progress_callback({
                'event': 'training_complete',
                'model_type': 'random_forest',
                'training_time_seconds': train_time
            })

        self.is_trained = True

        # Calculate training metrics
        y_train_pred = self.model.predict(X_train)

        metrics = {
            'training_time_seconds': train_time,
            'n_samples_train': len(X_train),
            'n_features': len(self.feature_columns),
            'task': self.task
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
