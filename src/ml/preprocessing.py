"""
Data preprocessing utilities for ML pipeline.

このモジュールはMLパイプライン用のデータ前処理ユーティリティを提供します。
- 欠損値処理（ドメイン対応）
- 特徴量スケーリング/正規化
- 外れ値検出・処理
- 特徴量選択

This module provides functions to clean, normalize, and prepare
feature data for model training and prediction.
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)


# ============================================================================
# ドメイン対応欠損値処理戦略
# ============================================================================
RACING_IMPUTATION_STRATEGIES: Dict[str, Any] = {
    # 統計系特徴量 - 未知=保守的に0.0
    'horse_win_rate': 0.0,
    'horse_place_rate': 0.0,
    'horse_distance_win_rate': 0.0,
    'horse_surface_win_rate': 0.0,
    'horse_track_win_rate': 0.0,
    'jockey_win_rate': 0.0,
    'jockey_place_rate': 0.0,
    'trainer_win_rate': 0.0,
    'trainer_place_rate': 0.0,

    # カウント系特徴量 - 未知=0
    'horse_total_races': 0,
    'horse_distance_races': 0,
    'horse_surface_races': 0,
    'horse_track_races': 0,
    'jockey_total_races': 0,
    'trainer_total_races': 0,

    # 体重系特徴量 - 典型的な値
    'horse_weight': 480.0,  # JRA平均馬体重
    'horse_weight_change': 0.0,  # 変化なし
    'weight': 55.0,  # 標準斤量

    # 最近のパフォーマンス - 中立的な値
    'recent_avg_position': 8.0,  # 中団
    'recent_best_position': 5,  # 中程度
    'last_race_position': 8,  # 中団
    'days_since_last_race': 60,  # 中程度の休養
    'recent_avg_odds': 20.0,  # 中程度のオッズ
    'horse_avg_finish_position': 8.0,  # 中団

    # オッズ系 - 中程度のオッズ
    'morning_odds': 20.0,
    'final_odds': 20.0,
    'odds_rank': 8.0,  # 中程度の人気

    # 枠順・馬番 - 中央
    'post_position': 4.0,
    'horse_number': 8.0,

    # デフォルト（未定義カラムはmedian）
    '__default__': 'median'
}
"""競馬固有の欠損値補完戦略

理由:
- 統計系: 未知の馬/騎手は保守的に0.0（過大評価を防ぐ）
- 体重系: 典型的なJRA馬の値（480kg、斤量55kg）
- パフォーマンス系: 中立的な値（中団想定）
- これにより新馬や海外馬の過大評価を防ぎ、予測精度を向上
"""


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

        # Use add_indicator=False and keep_empty_features=True to maintain column count
        self.imputer = SimpleImputer(strategy='median', keep_empty_features=True)
        self.imputer.fit(numerical_features)
        imputed_data = self.imputer.transform(numerical_features)

        self.scaler.fit(imputed_data)

        self.is_fitted = True
        logger.info("Preprocessor fitted successfully")

        return self

    def transform(self, X: pd.DataFrame, keep_identifiers: bool = False) -> pd.DataFrame:
        """
        Transform features using fitted preprocessor.

        Args:
            X: Features DataFrame to transform
            keep_identifiers: If True, keep ID columns in output (for prediction/evaluation).
                             If False, exclude ID columns (for training).

        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        logger.debug(f"Transforming {len(X)} samples")

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

        # Optionally add back identifiers (for evaluation purposes only)
        if keep_identifiers:
            id_columns = [col for col in X.columns if col.endswith('_id') or col == 'race_date']
            if id_columns:
                identifiers = X[id_columns].copy()
                X_transformed = pd.concat([identifiers.reset_index(drop=True), X_transformed.reset_index(drop=True)], axis=1)

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
        Get list of feature columns (excluding identifiers and non-numeric columns).

        Args:
            X: Features DataFrame

        Returns:
            List of feature column names
        """
        # Exclude ID columns and date columns from scaling
        exclude_patterns = ['_id', 'race_entry_id', 'race_date']

        # Get numeric columns only (excludes string, datetime, etc.)
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        feature_cols = [
            col for col in numeric_cols
            if not any(pattern in col for pattern in exclude_patterns)
        ]

        # Log warning if non-numeric columns were found and excluded
        non_numeric_cols = set(X.columns) - set(numeric_cols)
        if non_numeric_cols:
            logger.warning(f"Excluded non-numeric columns from features: {non_numeric_cols}")

        return feature_cols


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'median',
    use_domain_aware: bool = True
) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.

    ドメイン対応モード(use_domain_aware=True)では、競馬固有のデフォルト値を使用:
    - 統計系（勝率等）: 0.0（未知の馬/騎手は保守的に評価）
    - 体重系: 典型的なJRA馬の値（480kg、斤量55kg）
    - パフォーマンス系: 中立的な値（中団想定）
    - 未定義カラム: medianでフォールバック

    Args:
        df: Input DataFrame
        strategy: Imputation strategy ('median', 'mean', 'most_frequent', or 'constant')
        use_domain_aware: If True, use RACING_IMPUTATION_STRATEGIES for domain-specific defaults

    Returns:
        DataFrame with missing values filled
    """
    logger.info(f"Handling missing values with strategy: {strategy}, domain_aware: {use_domain_aware}")

    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        logger.warning(f"Found {missing_counts.sum()} missing values across {(missing_counts > 0).sum()} columns")

    # Separate numerical and categorical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    df_filled = df.copy()

    # ドメイン対応欠損値処理
    if use_domain_aware:
        domain_filled_cols = []
        remaining_numerical_cols = []

        for col in numerical_cols:
            if col in RACING_IMPUTATION_STRATEGIES:
                # ドメイン固有のデフォルト値を使用
                default_value = RACING_IMPUTATION_STRATEGIES[col]
                df_filled[col] = df_filled[col].fillna(default_value)
                domain_filled_cols.append(col)
            else:
                remaining_numerical_cols.append(col)

        if domain_filled_cols:
            logger.info(f"Applied domain-aware imputation to {len(domain_filled_cols)} columns")

        # 残りのカラムはmedianで処理
        if remaining_numerical_cols:
            default_strategy = RACING_IMPUTATION_STRATEGIES.get('__default__', 'median')
            num_imputer = SimpleImputer(strategy=default_strategy)
            df_filled[remaining_numerical_cols] = num_imputer.fit_transform(df[remaining_numerical_cols])
    else:
        # 従来の一括処理
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
