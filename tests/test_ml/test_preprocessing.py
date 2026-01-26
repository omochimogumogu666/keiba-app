"""
Tests for preprocessing module.
"""
import pytest
import pandas as pd
import numpy as np

from src.ml.preprocessing import (
    FeaturePreprocessor,
    handle_missing_values,
    remove_outliers,
    create_target_variable,
    select_features,
    split_by_date
)


@pytest.fixture
def sample_features():
    """Create sample feature DataFrame."""
    np.random.seed(42)
    data = {
        'race_id': [1, 2, 3, 4, 5],
        'horse_id': [10, 20, 30, 40, 50],
        'distance': [1600, 2000, 1800, 1600, 2000],
        'horse_weight': [480, 460, 470, 485, 475],
        'weight': [55.0, 54.0, 56.0, 55.0, 54.5],
        'horse_win_rate': [0.2, 0.3, np.nan, 0.15, 0.25],
        'jockey_win_rate': [0.25, np.nan, 0.22, 0.28, 0.24],
        'morning_odds': [3.5, 5.2, 4.1, 3.8, 6.0]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_labels():
    """Create sample label Series."""
    return pd.Series([1, 3, 2, 5, 4], name='finish_position')


@pytest.mark.unit
class TestFeaturePreprocessor:
    """Test FeaturePreprocessor class."""

    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = FeaturePreprocessor()
        assert preprocessor.scaler is not None
        assert preprocessor.imputer is not None
        assert not preprocessor.is_fitted

    def test_fit_transform(self, sample_features):
        """Test fit and transform."""
        preprocessor = FeaturePreprocessor()
        X_transformed = preprocessor.fit_transform(sample_features)

        assert isinstance(X_transformed, pd.DataFrame)
        assert len(X_transformed) == len(sample_features)
        assert preprocessor.is_fitted

        # ID columns should be excluded in fit_transform (for training)
        assert 'race_id' not in X_transformed.columns
        assert 'horse_id' not in X_transformed.columns

        # Feature columns should be present
        assert 'distance' in X_transformed.columns
        assert 'horse_weight' in X_transformed.columns

    def test_transform_with_keep_identifiers(self, sample_features):
        """Test that transform with keep_identifiers preserves ID columns."""
        preprocessor = FeaturePreprocessor()
        preprocessor.fit(sample_features)
        X_transformed = preprocessor.transform(sample_features, keep_identifiers=True)

        # ID columns should be preserved when keep_identifiers=True
        assert 'race_id' in X_transformed.columns
        assert 'horse_id' in X_transformed.columns

        # Feature columns should also be present
        assert 'distance' in X_transformed.columns
        assert 'horse_weight' in X_transformed.columns

    def test_transform_before_fit_raises_error(self, sample_features):
        """Test that transform raises error before fit."""
        preprocessor = FeaturePreprocessor()

        with pytest.raises(ValueError, match="must be fitted"):
            preprocessor.transform(sample_features)

    def test_get_feature_columns(self, sample_features):
        """Test feature column selection."""
        preprocessor = FeaturePreprocessor()
        feature_cols = preprocessor._get_feature_columns(sample_features)

        # Should exclude ID columns
        assert 'race_id' not in feature_cols
        assert 'horse_id' not in feature_cols

        # Should include numerical features
        assert 'distance' in feature_cols
        assert 'horse_weight' in feature_cols


@pytest.mark.unit
def test_handle_missing_values(sample_features):
    """Test missing value imputation."""
    df_filled = handle_missing_values(sample_features, strategy='median')

    # Check no missing values remain
    assert df_filled.isnull().sum().sum() == 0

    # Check original data still has missing values
    assert sample_features.isnull().sum().sum() > 0


@pytest.mark.unit
def test_remove_outliers():
    """Test outlier removal."""
    # Create data with outliers
    data = {
        'value1': [1, 2, 3, 4, 5, 100],  # 100 is outlier
        'value2': [10, 12, 11, 13, 12, 14]
    }
    df = pd.DataFrame(data)

    # Remove outliers using IQR method
    df_clean = remove_outliers(df, method='iqr', threshold=1.5)

    # Outlier should be removed
    assert len(df_clean) < len(df)
    assert 100 not in df_clean['value1'].values


@pytest.mark.unit
def test_create_target_variable_regression(sample_labels):
    """Test regression target creation."""
    target = create_target_variable(sample_labels, target_type='regression')

    # Should be same as input
    assert all(target == sample_labels)


@pytest.mark.unit
def test_create_target_variable_binary():
    """Test binary classification target."""
    finish_positions = pd.Series([1, 2, 3, 4, 5])
    target = create_target_variable(finish_positions, target_type='binary_win')

    # Only position 1 should be 1, rest 0
    assert target[0] == 1
    assert all(target[1:] == 0)


@pytest.mark.unit
def test_create_target_variable_multiclass():
    """Test multi-class target creation."""
    finish_positions = pd.Series([1, 2, 3, 4, 5])
    target = create_target_variable(finish_positions, target_type='multiclass_top3')

    # Position 1 -> 2 (win)
    assert target[0] == 2

    # Positions 2-3 -> 1 (place)
    assert target[1] == 1
    assert target[2] == 1

    # Positions 4+ -> 0 (other)
    assert target[3] == 0
    assert target[4] == 0


@pytest.mark.unit
def test_select_features_variance(sample_features, sample_labels):
    """Test feature selection by variance."""
    # Remove ID columns for this test
    X = sample_features.drop(['race_id', 'horse_id'], axis=1)

    selected = select_features(X, sample_labels, method='variance', threshold=0.0)

    assert isinstance(selected, list)
    assert len(selected) > 0


@pytest.mark.unit
def test_select_features_correlation(sample_features, sample_labels):
    """Test feature selection by correlation."""
    # Remove ID columns
    X = sample_features.drop(['race_id', 'horse_id'], axis=1)

    selected = select_features(X, sample_labels, method='correlation', top_k=3)

    assert isinstance(selected, list)
    assert len(selected) == 3


@pytest.mark.unit
def test_split_by_date():
    """Test date-based data splitting."""
    data = {
        'race_date': pd.date_range('2024-01-01', periods=10, freq='D'),
        'feature1': range(10),
        'feature2': range(10, 20)
    }
    df = pd.DataFrame(data)

    # Split with validation set
    train, test, val = split_by_date(
        df,
        date_column='race_date',
        train_end_date='2024-01-05',
        val_end_date='2024-01-08'
    )

    assert len(train) == 5  # Jan 1-5
    assert len(val) == 3    # Jan 6-8
    assert len(test) == 2   # Jan 9-10

    # Split without validation set
    train, test, val = split_by_date(
        df,
        date_column='race_date',
        train_end_date='2024-01-07'
    )

    assert len(train) == 7  # Jan 1-7
    assert len(test) == 3   # Jan 8-10
    assert val is None
