"""
Base model class for horse racing prediction.

This module provides an abstract base class that all prediction models
must inherit from, ensuring a consistent interface.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import os

from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)


class BaseRaceModel(ABC):
    """
    Abstract base class for race prediction models.

    All prediction models (RandomForest, XGBoost, etc.) should inherit
    from this class and implement the required methods.
    """

    def __init__(self, model_name: str, version: str = "1.0"):
        """
        Initialize base model.

        Args:
            model_name: Name of the model (e.g., 'RandomForest', 'XGBoost')
            version: Model version string
        """
        self.model_name = model_name
        self.version = version
        self.model = None
        self.is_trained = False
        self.feature_columns = None
        self.training_metadata = {}

    @abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model on training data.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            **kwargs: Additional model-specific parameters

        Returns:
            Dictionary with training metrics
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Features DataFrame

        Returns:
            Array of predictions
        """
        pass

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities (for classification models).

        Args:
            X: Features DataFrame

        Returns:
            Array of class probabilities
        """
        if not hasattr(self.model, 'predict_proba'):
            raise NotImplementedError(
                f"{self.model_name} does not support probability predictions"
            )

        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        X_prepared = self._prepare_features(X)
        return self.model.predict_proba(X_prepared)

    def get_feature_importance(self) -> pd.Series:
        """
        Get feature importance scores.

        Returns:
            Series with feature names as index and importance scores as values
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")

        if not hasattr(self.model, 'feature_importances_'):
            raise NotImplementedError(
                f"{self.model_name} does not support feature importance"
            )

        importances = self.model.feature_importances_
        return pd.Series(
            importances,
            index=self.feature_columns,
            name='importance'
        ).sort_values(ascending=False)

    def save(self, filepath: str) -> None:
        """
        Save model to file.

        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'version': self.version,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained,
            'training_metadata': self.training_metadata
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """
        Load model from file.

        Args:
            filepath: Path to load the model from
        """
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

        logger.info(f"Model loaded from {filepath}")

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for prediction.

        Ensures feature columns match training data.

        Args:
            X: Input features

        Returns:
            Prepared features DataFrame
        """
        if self.feature_columns is None:
            raise ValueError("Model has not been trained yet")

        # Check for missing columns
        missing_cols = set(self.feature_columns) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")

        # Select and order columns to match training data
        return X[self.feature_columns]

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.

        Returns:
            Dictionary with model metadata
        """
        return {
            'model_name': self.model_name,
            'version': self.version,
            'is_trained': self.is_trained,
            'n_features': len(self.feature_columns) if self.feature_columns else 0,
            'feature_columns': self.feature_columns,
            'training_metadata': self.training_metadata
        }

    def __repr__(self):
        status = "trained" if self.is_trained else "untrained"
        return f"<{self.model_name} v{self.version} ({status})>"
