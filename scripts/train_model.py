"""
Script to train machine learning models for race prediction.

This script demonstrates the complete ML pipeline:
1. Load training data from database or CSV
2. Split data into train/validation/test sets
3. Train models (RandomForest and XGBoost)
4. Evaluate model performance
5. Save trained models
"""
import os
import sys
from datetime import datetime, timedelta
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.web.app import create_app
from src.ml.feature_engineering import FeatureExtractor, load_features_from_csv
from src.ml.preprocessing import (
    FeaturePreprocessor,
    handle_missing_values,
    create_target_variable,
    split_by_date
)
from src.ml.models.random_forest import RandomForestRaceModel
from src.ml.models.xgboost_model import XGBoostRaceModel
from src.ml.models import TORCH_AVAILABLE
if TORCH_AVAILABLE:
    from src.ml.models.neural_network import NeuralNetworkRaceModel
from src.ml.evaluation import (
    evaluate_regression_model,
    evaluate_classification_model,
    evaluate_ranking_performance
)
from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train race prediction models')

    parser.add_argument(
        '--model',
        type=str,
        default='random_forest',
        choices=['random_forest', 'xgboost', 'neural_network', 'both', 'all'],
        help='Model type to train (both=RF+XGB, all=RF+XGB+NN)'
    )

    parser.add_argument(
        '--task',
        type=str,
        default='regression',
        choices=['regression', 'classification'],
        help='Task type (regression for finish position, classification for win/loss)'
    )

    parser.add_argument(
        '--data-source',
        type=str,
        default='database',
        choices=['database', 'csv'],
        help='Data source (database or CSV files)'
    )

    parser.add_argument(
        '--csv-path',
        type=str,
        default='data/processed/features_processed.csv',
        help='Path to CSV file with features (if using CSV source)'
    )

    parser.add_argument(
        '--labels-path',
        type=str,
        default='data/processed/labels_regression.csv',
        help='Path to CSV file with labels (if using CSV source)'
    )

    parser.add_argument(
        '--train-end-date',
        type=str,
        help='End date for training set (YYYY-MM-DD format)'
    )

    parser.add_argument(
        '--val-end-date',
        type=str,
        help='End date for validation set (YYYY-MM-DD format)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/models',
        help='Directory to save trained models'
    )

    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save trained models (evaluation only)'
    )

    # Neural Network specific parameters
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for neural network training'
    )

    parser.add_argument(
        '--n-epochs',
        type=int,
        default=100,
        help='Number of epochs for neural network training'
    )

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate for neural network training'
    )

    return parser.parse_args()


def load_data_from_database(app, min_date=None, max_date=None):
    """Load training data from database."""
    logger.info("Loading data from database")

    with app.app_context():
        from src.data.models import db

        extractor = FeatureExtractor(db.session, lookback_days=730)

        # Extract features and labels
        X, y = extractor.extract_features_for_training(
            min_date=min_date,
            max_date=max_date
        )

        if X.empty:
            raise ValueError("No training data found in database")

        logger.info(f"Loaded {len(X)} samples from database")

        return X, y


def load_data_from_csv(features_path, labels_path):
    """Load training data from CSV files."""
    logger.info(f"Loading data from CSV: {features_path}, {labels_path}")

    import pandas as pd

    X = pd.read_csv(features_path, encoding='utf-8-sig')
    y = pd.read_csv(labels_path, encoding='utf-8-sig').squeeze()

    logger.info(f"Loaded {len(X)} samples from CSV")

    return X, y


def main():
    """Main training pipeline."""
    args = parse_args()

    print("="*80)
    print("RACE PREDICTION MODEL TRAINING")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Task: {args.task}")
    print(f"Data Source: {args.data_source}")
    print("="*80)

    # Load data
    if args.data_source == 'database':
        app = create_app()
        X, y = load_data_from_database(app)
    else:
        X, y = load_data_from_csv(args.csv_path, args.labels_path)

    # Handle missing values
    logger.info("Handling missing values")
    X = handle_missing_values(X, strategy='median')

    # Split data by date if race_date column exists
    if 'race_date' in X.columns:
        logger.info("Splitting data by date")

        # Default dates if not provided
        if args.train_end_date is None:
            # Use 70% of data for training
            all_dates = sorted(X['race_date'].unique())
            train_end_idx = int(len(all_dates) * 0.7)
            args.train_end_date = all_dates[train_end_idx]

        if args.val_end_date is None:
            # Use 15% for validation
            all_dates = sorted(X['race_date'].unique())
            val_end_idx = int(len(all_dates) * 0.85)
            args.val_end_date = all_dates[val_end_idx]

        train_df, test_df, val_df = split_by_date(
            X.assign(target=y),
            date_column='race_date',
            train_end_date=args.train_end_date,
            val_end_date=args.val_end_date
        )

        X_train = train_df.drop('target', axis=1)
        y_train = train_df['target']
        X_val = val_df.drop('target', axis=1) if val_df is not None else None
        y_val = val_df['target'] if val_df is not None else None
        X_test = test_df.drop('target', axis=1)
        y_test = test_df['target']

    else:
        # Simple split if no date column
        from sklearn.model_selection import train_test_split

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

    logger.info(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

    # Preprocess features (exclude ID columns from training data)
    logger.info("Preprocessing features")
    preprocessor = FeaturePreprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_val_scaled = preprocessor.transform(X_val) if X_val is not None else None
    # Keep identifiers in test set for ranking evaluation
    X_test_scaled = preprocessor.transform(X_test, keep_identifiers=True)

    # Create target variable based on task
    if args.task == 'classification':
        y_train = create_target_variable(y_train, target_type='binary_win')
        y_val = create_target_variable(y_val, target_type='binary_win') if y_val is not None else None
        y_test = create_target_variable(y_test, target_type='binary_win')

    # Train models
    models_to_train = []

    if args.model in ['random_forest', 'both', 'all']:
        models_to_train.append(('RandomForest', RandomForestRaceModel, {}))

    if args.model in ['xgboost', 'both', 'all']:
        try:
            models_to_train.append(('XGBoost', XGBoostRaceModel, {}))
        except ImportError:
            logger.warning("XGBoost not available, skipping")

    if args.model in ['neural_network', 'all']:
        if TORCH_AVAILABLE:
            nn_params = {
                'batch_size': args.batch_size,
                'n_epochs': args.n_epochs,
                'learning_rate': args.learning_rate
            }
            models_to_train.append(('NeuralNetwork', NeuralNetworkRaceModel, nn_params))
        else:
            logger.warning("PyTorch not available, skipping neural network")

    results = {}

    for model_name, ModelClass, extra_params in models_to_train:
        print(f"\n{'='*80}")
        print(f"Training {model_name} model")
        print(f"{'='*80}")

        # Initialize model
        model = ModelClass(task=args.task, **extra_params)

        # Train
        train_metrics = model.train(
            X_train_scaled,
            y_train,
            X_val_scaled,
            y_val
        )

        # Evaluate on test set
        y_test_pred = model.predict(X_test_scaled)

        if args.task == 'regression':
            test_metrics = evaluate_regression_model(y_test, y_test_pred, model_name)

            # Ranking performance
            if 'race_id' in X_test.columns:
                ranking_metrics = evaluate_ranking_performance(
                    y_test,
                    y_test_pred,
                    X_test['race_id']
                )
                test_metrics.update(ranking_metrics)

        else:
            test_metrics = evaluate_classification_model(y_test, y_test_pred, model_name)

        # Feature importance
        if hasattr(model, 'get_feature_importance'):
            importance = model.get_feature_importance()
            print(f"\nTop 10 Important Features:")
            print(importance.head(10))

        # Save model
        if not args.no_save:
            os.makedirs(args.output_dir, exist_ok=True)
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            # Use .pt extension for neural networks, .pkl for others
            ext = '.pt' if model_name == 'NeuralNetwork' else '.pkl'
            model_path = os.path.join(
                args.output_dir,
                f"{model_name.lower()}_{args.task}_{timestamp}{ext}"
            )
            model.save(model_path)
            print(f"\nModel saved to: {model_path}")

        results[model_name] = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        }

    # Summary
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")

    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        if args.task == 'regression':
            print(f"  Test RMSE: {metrics['test_metrics']['rmse']:.4f}")
            print(f"  Test MAE: {metrics['test_metrics']['mae']:.4f}")
            print(f"  Test R2: {metrics['test_metrics']['r2_score']:.4f}")
        else:
            print(f"  Test Accuracy: {metrics['test_metrics']['accuracy']:.4f}")
            print(f"  Test F1 Score: {metrics['test_metrics']['f1_score']:.4f}")

    print(f"\n{'='*80}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
