"""
Machine Learning module for horse racing prediction.

This module provides:
- Feature engineering and extraction
- Data preprocessing pipelines
- ML models (RandomForest, XGBoost)
- Model evaluation metrics
"""
from src.ml.feature_engineering import FeatureExtractor
from src.ml.preprocessing import FeaturePreprocessor
from src.ml.models import (
    BaseRaceModel,
    RandomForestRaceModel,
    XGBoostRaceModel,
    XGBOOST_AVAILABLE
)
from src.ml.evaluation import (
    evaluate_regression_model,
    evaluate_classification_model,
    calculate_roi
)

__all__ = [
    'FeatureExtractor',
    'FeaturePreprocessor',
    'BaseRaceModel',
    'RandomForestRaceModel',
    'XGBoostRaceModel',
    'XGBOOST_AVAILABLE',
    'evaluate_regression_model',
    'evaluate_classification_model',
    'calculate_roi'
]
