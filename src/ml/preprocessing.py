"""
Data preprocessing utilities for ML pipeline.

This module provides functions to clean, normalize, and prepare
feature data for model training and prediction.
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)


class FeaturePreprocessor:
    """
    Preprocess features for machine learning models.

    Handles:
    - Missing value imputation
    - Feature scaling/normalization
    - Outlier detection and handling
    - Feature selection
    """

    def __init__(self):
        """Initialize preprocessor with scalers and imputers."""
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame) -> 'FeaturePreprocessor':
        """
        Fit preprocessor on training data.

        Args:
            X: Training features DataFrame

        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting preprocessor on {len(X)} samples with {len(X.columns)} features")

        # Store feature columns (excluding identifiers)
        self.feature_columns = self._get_feature_columns(X)

        # Fit imputer and scaler on numerical features only
        numerical_features = X[self.feature_columns]

        self.imputer.fit(numerical_features)
        imputed_data = self.imputer.transform(numerical_features)

        self.scaler.fit(imputed_data)

        self.is_fitted = True
        logger.info("Preprocessor fitted successfully")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted preprocessor.

        Args:
            X: Features DataFrame to transform

        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        logger.debug(f"Transforming {len(X)} samples")

        # Keep identifier columns
        id_columns = [col for col in X.columns if col.endswith('_id')]
        identifiers = X[id_columns].copy() if id_columns else pd.DataFrame()

        # Transform numerical features
        numerical_features = X[self.feature_columns]

        # Impute missing values
        imputed_data = self.imputer.transform(numerical_features)

        # Scale features
        scaled_data = self.scaler.transform(imputed_data)

        # Create transformed DataFrame
        X_transformed = pd.DataFrame(
            scaled_data,
            columns=self.feature_columns,
            index=X.index
        )

        # Add back identifiers
        if not identifiers.empty:
            X_transformed = pd.concat([identifiers.reset_index(drop=True), X_transformed], axis=1)

        return X_transformed

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit preprocessor and transform data in one step.

        Args:
            X: Features DataFrame

        Returns:
            Transformed DataFrame
        """
        return self.fit(X).transform(X)

    def _get_feature_columns(self, X: pd.DataFrame) -> List[str]:
        """
        Get list of feature columns (excluding identifiers).

        Args:
            X: Features DataFrame

        Returns:
            List of feature column names
        """
        # Exclude ID columns from scaling
        exclude_patterns = ['_id', 'race_entry_id']
        feature_cols = [
            col for col in X.columns
            if not any(pattern in col for pattern in exclude_patterns)
        ]

        return feature_cols


def handle_missing_values(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """
    Handle missing values in DataFrame.

    Args:
        df: Input DataFrame
        strategy: Imputation strategy ('median', 'mean', 'most_frequent', or 'constant')

    Returns:
        DataFrame with missing values filled
    """
    logger.info(f"Handling missing values with strategy: {strategy}")

    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        logger.warning(f"Found {missing_counts.sum()} missing values across {(missing_counts > 0).sum()} columns")

    # Separate numerical and categorical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    df_filled = df.copy()

    # Impute numerical columns
    if len(numerical_cols) > 0:
        num_imputer = SimpleImputer(strategy=strategy)
        df_filled[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

    # Impute categorical columns with most frequent
    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_filled[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    logger.info("Missing values handled successfully")
    return df_filled


def remove_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'iqr',
    threshold: float = 1.5
) -> pd.DataFrame:
    """
    Remove outliers from DataFrame.

    Args:
        df: Input DataFrame
        columns: Columns to check for outliers (None = all numerical columns)
        method: Outlier detection method ('iqr' or 'zscore')
        threshold: Threshold for outlier detection (1.5 for IQR, 3.0 for z-score)

    Returns:
        DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    logger.info(f"Removing outliers from {len(columns)} columns using {method} method")

    df_clean = df.copy()
    initial_rows = len(df_clean)

    for col in columns:
        if method == 'iqr':
            # IQR method
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
            df_clean = df_clean[mask]

        elif method == 'zscore':
            # Z-score method
            z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
            mask = z_scores <= threshold
            df_clean = df_clean[mask]

    removed_rows = initial_rows - len(df_clean)
    logger.info(f"Removed {removed_rows} outlier rows ({removed_rows/initial_rows*100:.2f}%)")

    return df_clean


def create_target_variable(
    finish_positions: pd.Series,
    target_type: str = 'regression'
) -> pd.Series:
    """
    Create target variable from finish positions.

    Args:
        finish_positions: Series of finish positions
        target_type: Type of target ('regression', 'binary_win', 'multiclass_top3')

    Returns:
        Target variable Series
    """
    if target_type == 'regression':
        # Predict exact finish position
        return finish_positions

    elif target_type == 'binary_win':
        # Binary classification: win (1) or not (0)
        return (finish_positions == 1).astype(int)

    elif target_type == 'multiclass_top3':
        # Multi-class: win (2), place 2-3 (1), or other (0)
        target = pd.Series(0, index=finish_positions.index)
        target[finish_positions == 1] = 2  # Win
        target[finish_positions.isin([2, 3])] = 1  # Place
        return target

    else:
        raise ValueError(f"Unknown target_type: {target_type}")


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = 'variance',
    threshold: float = 0.01,
    top_k: Optional[int] = None
) -> List[str]:
    """
    Select most important features.

    Args:
        X: Features DataFrame
        y: Target variable
        method: Feature selection method ('variance', 'correlation')
        threshold: Variance threshold or correlation threshold
        top_k: Select top K features (None = use threshold)

    Returns:
        List of selected feature names
    """
    logger.info(f"Selecting features using {method} method")

    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()

    if method == 'variance':
        # Remove low variance features
        variances = X[numerical_features].var()

        if top_k:
            selected = variances.nlargest(top_k).index.tolist()
        else:
            selected = variances[variances > threshold].index.tolist()

    elif method == 'correlation':
        # Select features with high correlation to target
        correlations = X[numerical_features].corrwith(y).abs()

        if top_k:
            selected = correlations.nlargest(top_k).index.tolist()
        else:
            selected = correlations[correlations > threshold].index.tolist()

    else:
        raise ValueError(f"Unknown method: {method}")

    logger.info(f"Selected {len(selected)} features")
    return selected


def split_by_date(
    df: pd.DataFrame,
    date_column: str,
    train_end_date: str,
    val_end_date: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Split data by date for time-series cross-validation.

    Args:
        df: Input DataFrame with date column
        date_column: Name of date column
        train_end_date: End date for training set (format: 'YYYY-MM-DD')
        val_end_date: End date for validation set (None = no validation set)

    Returns:
        Tuple of (train_df, test_df, val_df or None)
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    train_end = pd.to_datetime(train_end_date)

    train_df = df[df[date_column] <= train_end]

    if val_end_date:
        val_end = pd.to_datetime(val_end_date)
        val_df = df[(df[date_column] > train_end) & (df[date_column] <= val_end)]
        test_df = df[df[date_column] > val_end]
        return train_df, test_df, val_df
    else:
        test_df = df[df[date_column] > train_end]
        return train_df, test_df, None
