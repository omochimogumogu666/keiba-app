"""
Tests for ML models.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import os
import tempfile

from src.ml.models.random_forest import RandomForestRaceModel
from src.ml.models.xgboost_model import XGBoostRaceModel, XGBOOST_AVAILABLE


@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    n_samples = 100

    X = pd.DataFrame({
        'distance': np.random.randint(1200, 2400, n_samples),
        'horse_weight': np.random.randint(450, 520, n_samples),
        'weight': np.random.uniform(52, 58, n_samples),
        'horse_win_rate': np.random.uniform(0, 0.5, n_samples),
        'jockey_win_rate': np.random.uniform(0, 0.4, n_samples),
        'morning_odds': np.random.uniform(2, 20, n_samples)
    })

    # Regression target: finish positions (1-10)
    y_regression = np.random.randint(1, 11, n_samples)

    # Classification target: win (1) or not (0)
    y_classification = (y_regression == 1).astype(int)

    return X, y_regression, y_classification


@pytest.mark.unit
class TestRandomForestRaceModel:
    """Test RandomForestRaceModel class."""

    def test_initialization_regression(self):
        """Test model initialization for regression."""
        model = RandomForestRaceModel(task='regression', n_estimators=10)

        assert model.model_name == 'RandomForest_regression'
        assert model.task == 'regression'
        assert not model.is_trained
        assert model.params['n_estimators'] == 10

    def test_initialization_classification(self):
        """Test model initialization for classification."""
        model = RandomForestRaceModel(task='classification', n_estimators=10)

        assert model.model_name == 'RandomForest_classification'
        assert model.task == 'classification'
        assert not model.is_trained

    def test_initialization_invalid_task(self):
        """Test model initialization with invalid task."""
        with pytest.raises(ValueError, match="Unknown task"):
            RandomForestRaceModel(task='invalid')

    def test_train_regression(self, sample_data):
        """Test training regression model."""
        X, y_regression, _ = sample_data

        model = RandomForestRaceModel(task='regression', n_estimators=10, random_state=42)
        metrics = model.train(X, y_regression)

        assert model.is_trained
        assert 'train_mse' in metrics
        assert 'train_mae' in metrics
        assert 'train_rmse' in metrics
        assert metrics['n_samples_train'] == len(X)
        assert metrics['n_features'] == len(X.columns)

    def test_train_classification(self, sample_data):
        """Test training classification model."""
        X, _, y_classification = sample_data

        model = RandomForestRaceModel(task='classification', n_estimators=10, random_state=42)
        metrics = model.train(X, y_classification)

        assert model.is_trained
        assert 'train_accuracy' in metrics
        assert metrics['n_samples_train'] == len(X)

    def test_train_with_validation(self, sample_data):
        """Test training with validation set."""
        X, y_regression, _ = sample_data

        # Split data
        split_idx = 80
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y_regression[:split_idx], y_regression[split_idx:]

        model = RandomForestRaceModel(task='regression', n_estimators=10, random_state=42)
        metrics = model.train(X_train, y_train, X_val, y_val)

        assert 'val_mse' in metrics
        assert 'val_mae' in metrics
        assert metrics['n_samples_val'] == len(X_val)

    def test_predict_before_training(self, sample_data):
        """Test prediction before training raises error."""
        X, _, _ = sample_data

        model = RandomForestRaceModel(task='regression')

        with pytest.raises(ValueError, match="must be trained"):
            model.predict(X)

    def test_predict_regression(self, sample_data):
        """Test regression prediction."""
        X, y_regression, _ = sample_data

        model = RandomForestRaceModel(task='regression', n_estimators=10, random_state=42)
        model.train(X, y_regression)

        predictions = model.predict(X)

        assert len(predictions) == len(X)
        assert isinstance(predictions, np.ndarray)

    def test_predict_classification(self, sample_data):
        """Test classification prediction."""
        X, _, y_classification = sample_data

        model = RandomForestRaceModel(task='classification', n_estimators=10, random_state=42)
        model.train(X, y_classification)

        predictions = model.predict(X)

        assert len(predictions) == len(X)
        assert all(p in [0, 1] for p in predictions)

    def test_predict_proba(self, sample_data):
        """Test probability prediction."""
        X, _, y_classification = sample_data

        model = RandomForestRaceModel(task='classification', n_estimators=10, random_state=42)
        model.train(X, y_classification)

        probas = model.predict_proba(X)

        assert probas.shape == (len(X), 2)
        assert np.allclose(probas.sum(axis=1), 1.0)

    def test_get_feature_importance(self, sample_data):
        """Test feature importance extraction."""
        X, y_regression, _ = sample_data

        model = RandomForestRaceModel(task='regression', n_estimators=10, random_state=42)
        model.train(X, y_regression)

        importance = model.get_feature_importance()

        assert len(importance) == len(X.columns)
        assert all(imp >= 0 for imp in importance.values)
        assert importance.index.tolist() == sorted(X.columns, key=lambda c: importance[c], reverse=True)

    def test_save_and_load(self, sample_data):
        """Test model saving and loading."""
        X, y_regression, _ = sample_data

        model = RandomForestRaceModel(task='regression', n_estimators=10, random_state=42)
        model.train(X, y_regression)

        predictions_before = model.predict(X)

        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.pkl')
            model.save(model_path)

            assert os.path.exists(model_path)

            # Load model
            loaded_model = RandomForestRaceModel()
            loaded_model.load(model_path)

            assert loaded_model.is_trained
            assert loaded_model.model_name == model.model_name
            assert loaded_model.feature_columns == model.feature_columns

            # Predictions should be identical
            predictions_after = loaded_model.predict(X)
            np.testing.assert_array_equal(predictions_before, predictions_after)

    def test_save_untrained_model_raises_error(self):
        """Test saving untrained model raises error."""
        model = RandomForestRaceModel(task='regression')

        with pytest.raises(ValueError, match="Cannot save untrained"):
            model.save('/tmp/test_model.pkl')

    def test_get_model_info(self, sample_data):
        """Test getting model information."""
        X, y_regression, _ = sample_data

        model = RandomForestRaceModel(task='regression', n_estimators=10, version='2.0')
        model.train(X, y_regression)

        info = model.get_model_info()

        assert info['model_name'] == 'RandomForest_regression'
        assert info['version'] == '2.0'
        assert info['is_trained'] is True
        assert info['n_features'] == len(X.columns)


@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not installed")
@pytest.mark.unit
class TestXGBoostRaceModel:
    """Test XGBoostRaceModel class."""

    def test_initialization(self):
        """Test XGBoost model initialization."""
        model = XGBoostRaceModel(task='regression', n_estimators=10)

        assert model.model_name == 'XGBoost_regression'
        assert model.task == 'regression'
        assert not model.is_trained

    def test_train_regression(self, sample_data):
        """Test training XGBoost regression model."""
        X, y_regression, _ = sample_data

        model = XGBoostRaceModel(task='regression', n_estimators=10, random_state=42)
        metrics = model.train(X, y_regression, verbose=False)

        assert model.is_trained
        assert 'train_mse' in metrics
        assert 'train_rmse' in metrics

    def test_train_classification(self, sample_data):
        """Test training XGBoost classification model."""
        X, _, y_classification = sample_data

        model = XGBoostRaceModel(task='classification', n_estimators=10, random_state=42)
        metrics = model.train(X, y_classification, verbose=False)

        assert model.is_trained
        assert 'train_accuracy' in metrics

    def test_predict(self, sample_data):
        """Test XGBoost prediction."""
        X, y_regression, _ = sample_data

        model = XGBoostRaceModel(task='regression', n_estimators=10, random_state=42)
        model.train(X, y_regression, verbose=False)

        predictions = model.predict(X)

        assert len(predictions) == len(X)
        assert isinstance(predictions, np.ndarray)


@pytest.mark.integration
def test_model_not_installed():
    """Test that proper error is raised when XGBoost not installed."""
    if XGBOOST_AVAILABLE:
        pytest.skip("XGBoost is installed")

    with pytest.raises(ImportError, match="XGBoost is not installed"):
        XGBoostRaceModel(task='regression')
