"""
ML models module for race prediction.
"""
from src.ml.models.base_model import BaseRaceModel
from src.ml.models.random_forest import RandomForestRaceModel

# XGBoost is optional
try:
    from src.ml.models.xgboost_model import XGBoostRaceModel
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBoostRaceModel = None
    XGBOOST_AVAILABLE = False

# PyTorch Neural Network is optional
try:
    from src.ml.models.neural_network import NeuralNetworkRaceModel
    TORCH_AVAILABLE = True
except ImportError:
    NeuralNetworkRaceModel = None
    TORCH_AVAILABLE = False

__all__ = [
    'BaseRaceModel',
    'RandomForestRaceModel',
    'XGBoostRaceModel',
    'XGBOOST_AVAILABLE',
    'NeuralNetworkRaceModel',
    'TORCH_AVAILABLE'
]
