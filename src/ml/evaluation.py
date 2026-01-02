"""
Model evaluation metrics and utilities for horse racing prediction.
"""
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)


def evaluate_regression_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Evaluate regression model performance.

    Args:
        y_true: True finish positions
        y_pred: Predicted finish positions
        model_name: Name of the model being evaluated

    Returns:
        Dictionary with evaluation metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    # R² score (coefficient of determination)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2_score': r2_score,
        'mape': mape
    }

    logger.info(f"{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2_score:.4f}")

    return metrics


def evaluate_classification_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    average: str = 'binary'
) -> Dict[str, float]:
    """
    Evaluate classification model performance.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model being evaluated
        average: Averaging method for multiclass ('binary', 'macro', 'micro', 'weighted')

    Returns:
        Dictionary with evaluation metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    logger.info(
        f"{model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
        f"Recall: {recall:.4f}, F1: {f1:.4f}"
    )

    return metrics


def evaluate_top_n_accuracy(
    y_true: np.ndarray,
    predictions_df: pd.DataFrame,
    n: int = 3
) -> Dict[str, float]:
    """
    Evaluate top-N prediction accuracy.

    Calculates how often the actual winner is in the top N predictions.

    Args:
        y_true: True finish positions (1-indexed)
        predictions_df: DataFrame with 'race_id', 'horse_id', 'rank' columns
        n: Number of top predictions to consider

    Returns:
        Dictionary with top-N metrics
    """
    # This is a placeholder - actual implementation would need race_id mapping
    # For now, return basic structure
    metrics = {
        f'top_{n}_accuracy': 0.0,
        f'top_{n}_hit_rate': 0.0
    }

    return metrics


def calculate_hit_rate(
    y_true_positions: pd.Series,
    predicted_ranks: pd.Series,
    threshold: int = 3
) -> float:
    """
    Calculate hit rate for top-N predictions.

    Args:
        y_true_positions: True finish positions
        predicted_ranks: Predicted ranks (1 = predicted winner)
        threshold: Top N threshold (default: 3 for top-3)

    Returns:
        Hit rate (proportion of actual winners in top N predictions)
    """
    # Winner is position 1
    actual_winners = (y_true_positions == 1)

    # Predicted as top N
    predicted_top_n = (predicted_ranks <= threshold)

    # Hit rate: proportion of winners that were predicted in top N
    hits = (actual_winners & predicted_top_n).sum()
    total_winners = actual_winners.sum()

    hit_rate = hits / total_winners if total_winners > 0 else 0.0

    return hit_rate


def calculate_roi(
    predictions_df: pd.DataFrame,
    results_df: pd.DataFrame,
    stake: float = 100.0
) -> Dict[str, float]:
    """
    Calculate Return on Investment for betting strategy.

    Args:
        predictions_df: DataFrame with 'race_id', 'horse_id', 'rank' columns
        results_df: DataFrame with 'race_id', 'horse_id', 'finish_position', 'final_odds'
        stake: Betting stake per race

    Returns:
        Dictionary with ROI metrics
    """
    # Merge predictions with results
    merged = predictions_df.merge(
        results_df,
        on=['race_id', 'horse_id'],
        how='inner'
    )

    # Calculate wins (predicted rank 1 and actual position 1)
    wins = merged[(merged['rank'] == 1) & (merged['finish_position'] == 1)]

    total_races = predictions_df['race_id'].nunique()
    total_bet = total_races * stake
    total_return = (wins['final_odds'] * stake).sum()

    roi = ((total_return - total_bet) / total_bet * 100) if total_bet > 0 else 0.0
    profit = total_return - total_bet

    metrics = {
        'total_races': total_races,
        'total_bet': total_bet,
        'total_return': total_return,
        'profit': profit,
        'roi_percentage': roi,
        'win_rate': len(wins) / total_races if total_races > 0 else 0.0
    }

    logger.info(
        f"ROI Analysis - Races: {total_races}, ROI: {roi:.2f}%, "
        f"Profit: {profit:.2f}, Win Rate: {metrics['win_rate']:.2%}"
    )

    return metrics


def evaluate_ranking_performance(
    y_true: pd.Series,
    y_pred: pd.Series,
    race_ids: pd.Series
) -> Dict[str, float]:
    """
    Evaluate ranking performance using rank correlation metrics.

    Args:
        y_true: True finish positions
        y_pred: Predicted finish positions
        race_ids: Race IDs for grouping

    Returns:
        Dictionary with ranking metrics
    """
    from scipy.stats import spearmanr, kendalltau

    # Create DataFrame for easier manipulation
    df = pd.DataFrame({
        'race_id': race_ids,
        'true_position': y_true,
        'pred_position': y_pred
    })

    # Calculate correlation per race
    race_correlations = []

    for race_id in df['race_id'].unique():
        race_df = df[df['race_id'] == race_id]

        if len(race_df) > 1:  # Need at least 2 horses
            spearman_corr, _ = spearmanr(
                race_df['true_position'],
                race_df['pred_position']
            )
            kendall_corr, _ = kendalltau(
                race_df['true_position'],
                race_df['pred_position']
            )

            race_correlations.append({
                'spearman': spearman_corr,
                'kendall': kendall_corr
            })

    # Average correlations across races
    avg_spearman = np.mean([r['spearman'] for r in race_correlations]) if race_correlations else 0.0
    avg_kendall = np.mean([r['kendall'] for r in race_correlations]) if race_correlations else 0.0

    metrics = {
        'avg_spearman_correlation': avg_spearman,
        'avg_kendall_tau': avg_kendall,
        'n_races_evaluated': len(race_correlations)
    }

    logger.info(
        f"Ranking Performance - Spearman: {avg_spearman:.4f}, "
        f"Kendall Tau: {avg_kendall:.4f}"
    )

    return metrics


def generate_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List] = None
) -> pd.DataFrame:
    """
    Generate confusion matrix as DataFrame.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of label names (optional)

    Returns:
        Confusion matrix as DataFrame
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    cm_df = pd.DataFrame(
        cm,
        index=[f'True_{label}' for label in labels],
        columns=[f'Pred_{label}' for label in labels]
    )

    return cm_df


def cross_validate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    metric: str = 'rmse'
) -> Dict[str, float]:
    """
    Perform time-series cross-validation.

    Args:
        model: Model instance with train() and predict() methods
        X: Features DataFrame (must include 'race_date' column)
        y: Target Series
        n_splits: Number of CV splits
        metric: Metric to evaluate ('rmse', 'mae', 'accuracy')

    Returns:
        Dictionary with CV metrics
    """
    if 'race_date' not in X.columns:
        raise ValueError("X must include 'race_date' column for time-series CV")

    # Sort by date
    df = X.copy()
    df['target'] = y
    df = df.sort_values('race_date')

    # Split into n_splits folds
    fold_size = len(df) // n_splits
    scores = []

    for i in range(n_splits - 1):
        # Train on all data up to fold i+1
        train_end_idx = (i + 1) * fold_size
        test_end_idx = (i + 2) * fold_size

        train_df = df.iloc[:train_end_idx]
        test_df = df.iloc[train_end_idx:test_end_idx]

        X_train = train_df.drop('target', axis=1)
        y_train = train_df['target']
        X_test = test_df.drop('target', axis=1)
        y_test = test_df['target']

        # Train and predict
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metric
        if metric == 'rmse':
            score = np.sqrt(mean_squared_error(y_test, y_pred))
        elif metric == 'mae':
            score = mean_absolute_error(y_test, y_pred)
        elif metric == 'accuracy':
            score = accuracy_score(y_test, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        scores.append(score)

    cv_metrics = {
        f'cv_{metric}_mean': np.mean(scores),
        f'cv_{metric}_std': np.std(scores),
        f'cv_{metric}_scores': scores,
        'n_splits': n_splits
    }

    logger.info(
        f"Cross-Validation {metric.upper()}: "
        f"{cv_metrics[f'cv_{metric}_mean']:.4f} ± {cv_metrics[f'cv_{metric}_std']:.4f}"
    )

    return cv_metrics
