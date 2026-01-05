"""
Model Retraining Scheduler

Automatically retrain machine learning models on a scheduled basis.
Supports:
- Configurable retraining intervals (daily, weekly, monthly)
- Model versioning and performance tracking
- Automatic model comparison and deployment
- Email notifications on completion (optional)
"""
import os
import sys
import time
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import schedule
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.web.app import create_app
from src.ml.feature_engineering import FeatureExtractor
from src.ml.preprocessing import FeaturePreprocessor, handle_missing_values
from src.ml.models.random_forest import RandomForestRaceModel
from src.ml.models.xgboost_model import XGBoostRaceModel
from src.ml.evaluation import evaluate_regression_model
from src.utils.logger import get_app_logger
from src.utils.notification import send_notification

logger = get_app_logger(__name__)


class ModelRetrainingManager:
    """Manages model retraining workflow."""

    def __init__(self, config_path='config/retraining_config.json'):
        """
        Initialize retraining manager.

        Args:
            config_path: Path to retraining configuration file
        """
        self.config = self._load_config(config_path)
        self.app = create_app()
        self.models_dir = Path(self.config.get('models_dir', 'data/models'))
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Model registry to track versions
        self.registry_path = self.models_dir / 'model_registry.json'
        self.registry = self._load_registry()

    def _load_config(self, config_path):
        """Load retraining configuration."""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {
                'models_to_train': ['random_forest', 'xgboost'],
                'task': 'regression',
                'training_window_days': 365,
                'validation_split': 0.15,
                'test_split': 0.15,
                'min_samples': 1000,
                'performance_threshold': 0.05,  # 5% improvement required
                'keep_last_n_models': 5,
                'models_dir': 'data/models'
            }

    def _load_registry(self):
        """Load model registry from file."""
        if self.registry_path.exists():
            with open(self.registry_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'models': []}

    def _save_registry(self):
        """Save model registry to file."""
        with open(self.registry_path, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, indent=2, ensure_ascii=False)

    def _get_training_data(self):
        """
        Extract training data from database.

        Returns:
            Tuple of (X, y, dates)
        """
        logger.info("Extracting training data from database")

        with self.app.app_context():
            from src.data.models import db

            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.config['training_window_days'])

            extractor = FeatureExtractor(db.session, lookback_days=730)

            X, y = extractor.extract_features_for_training(
                min_date=start_date.strftime('%Y-%m-%d'),
                max_date=end_date.strftime('%Y-%m-%d')
            )

            if X.empty or len(X) < self.config['min_samples']:
                raise ValueError(
                    f"Insufficient training data: {len(X)} samples "
                    f"(minimum: {self.config['min_samples']})"
                )

            logger.info(f"Extracted {len(X)} samples from {start_date.date()} to {end_date.date()}")

            return X, y

    def _split_data(self, X, y):
        """
        Split data into train/validation/test sets.

        Args:
            X: Features DataFrame
            y: Target Series

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        from src.ml.preprocessing import split_by_date

        if 'race_date' not in X.columns:
            raise ValueError("race_date column required for time-based split")

        # Calculate split dates
        all_dates = sorted(X['race_date'].unique())
        n_dates = len(all_dates)

        train_end_idx = int(n_dates * (1 - self.config['validation_split'] - self.config['test_split']))
        val_end_idx = int(n_dates * (1 - self.config['test_split']))

        train_end_date = all_dates[train_end_idx]
        val_end_date = all_dates[val_end_idx]

        logger.info(f"Splitting data: train until {train_end_date}, val until {val_end_date}")

        train_df, test_df, val_df = split_by_date(
            X.assign(target=y),
            date_column='race_date',
            train_end_date=train_end_date,
            val_end_date=val_end_date
        )

        X_train = train_df.drop('target', axis=1)
        y_train = train_df['target']
        X_val = val_df.drop('target', axis=1) if val_df is not None else None
        y_val = val_df['target'] if val_df is not None else None
        X_test = test_df.drop('target', axis=1)
        y_test = test_df['target']

        logger.info(
            f"Split sizes - Train: {len(X_train)}, "
            f"Val: {len(X_val) if X_val is not None else 0}, "
            f"Test: {len(X_test)}"
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _train_model(self, model_type, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        Train a single model and evaluate performance.

        Args:
            model_type: Type of model ('random_forest' or 'xgboost')
            X_train, X_val, X_test: Feature DataFrames
            y_train, y_val, y_test: Target Series

        Returns:
            Dict with model info and metrics
        """
        logger.info(f"Training {model_type} model")

        # Select model class
        if model_type == 'random_forest':
            ModelClass = RandomForestRaceModel
        elif model_type == 'xgboost':
            ModelClass = XGBoostRaceModel
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Initialize model
        model = ModelClass(task=self.config['task'])

        # Train
        start_time = time.time()
        train_metrics = model.train(X_train, y_train, X_val, y_val)
        training_time = time.time() - start_time

        # Evaluate on test set
        y_test_pred = model.predict(X_test)
        test_metrics = evaluate_regression_model(y_test, y_test_pred, model_type)

        # Generate timestamp and model path
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        model_filename = f"{model_type}_{self.config['task']}_{timestamp}.pkl"
        model_path = self.models_dir / model_filename

        # Save model
        model.save(str(model_path))

        logger.info(
            f"Model saved: {model_filename} "
            f"(Test RMSE: {test_metrics['rmse']:.4f}, "
            f"R2: {test_metrics['r2_score']:.4f})"
        )

        # Return model info
        return {
            'model_type': model_type,
            'task': self.config['task'],
            'timestamp': timestamp,
            'filename': model_filename,
            'path': str(model_path),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'training_time_seconds': round(training_time, 2),
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        }

    def _compare_with_previous(self, new_model_info):
        """
        Compare new model with previous best model.

        Args:
            new_model_info: Dict with new model information

        Returns:
            Dict with comparison results
        """
        model_type = new_model_info['model_type']

        # Find previous models of same type
        previous_models = [
            m for m in self.registry['models']
            if m['model_type'] == model_type and m.get('is_active', False)
        ]

        if not previous_models:
            logger.info(f"No previous {model_type} model found, deploying new model")
            return {
                'has_previous': False,
                'should_deploy': True,
                'improvement': None
            }

        # Get most recent active model
        previous_model = sorted(
            previous_models,
            key=lambda m: m['timestamp'],
            reverse=True
        )[0]

        # Compare performance (using RMSE for regression)
        new_rmse = new_model_info['test_metrics']['rmse']
        prev_rmse = previous_model['test_metrics']['rmse']

        improvement = (prev_rmse - new_rmse) / prev_rmse

        logger.info(
            f"Performance comparison for {model_type}:\n"
            f"  Previous RMSE: {prev_rmse:.4f}\n"
            f"  New RMSE: {new_rmse:.4f}\n"
            f"  Improvement: {improvement*100:.2f}%"
        )

        # Decide whether to deploy
        threshold = self.config['performance_threshold']
        should_deploy = improvement >= threshold

        if should_deploy:
            logger.info(f"New model shows improvement >= {threshold*100}%, will deploy")
        else:
            logger.info(f"New model improvement < {threshold*100}%, keeping previous model")

        return {
            'has_previous': True,
            'previous_model': previous_model['filename'],
            'previous_rmse': prev_rmse,
            'new_rmse': new_rmse,
            'improvement': improvement,
            'should_deploy': should_deploy
        }

    def _update_registry(self, model_info, is_active):
        """
        Update model registry with new model.

        Args:
            model_info: Dict with model information
            is_active: Whether this model should be marked as active
        """
        # Mark previous models as inactive if deploying new one
        if is_active:
            for model in self.registry['models']:
                if model['model_type'] == model_info['model_type']:
                    model['is_active'] = False

        # Add new model to registry
        model_info['is_active'] = is_active
        model_info['created_at'] = datetime.utcnow().isoformat()
        self.registry['models'].append(model_info)

        # Clean up old models (keep only last N)
        self._cleanup_old_models(model_info['model_type'])

        self._save_registry()

    def _cleanup_old_models(self, model_type):
        """
        Remove old model files, keeping only the last N versions.

        Args:
            model_type: Type of model to clean up
        """
        keep_n = self.config['keep_last_n_models']

        # Get all models of this type, sorted by timestamp
        type_models = sorted(
            [m for m in self.registry['models'] if m['model_type'] == model_type],
            key=lambda m: m['timestamp'],
            reverse=True
        )

        # Models to remove
        to_remove = type_models[keep_n:]

        for model in to_remove:
            # Delete file
            model_path = Path(model['path'])
            if model_path.exists():
                model_path.unlink()
                logger.info(f"Deleted old model: {model['filename']}")

            # Remove from registry
            self.registry['models'].remove(model)

    def retrain_models(self):
        """Execute full retraining workflow."""
        logger.info("="*80)
        logger.info("STARTING MODEL RETRAINING")
        logger.info("="*80)

        try:
            # 1. Get training data
            X, y = self._get_training_data()

            # 2. Handle missing values
            logger.info("Handling missing values")
            X = handle_missing_values(X, strategy='median')

            # 3. Split data
            X_train, X_val, X_test, y_train, y_val, y_test = self._split_data(X, y)

            # 4. Preprocess features
            logger.info("Preprocessing features")
            preprocessor = FeaturePreprocessor()
            X_train_scaled = preprocessor.fit_transform(X_train)
            X_val_scaled = preprocessor.transform(X_val) if X_val is not None else None
            X_test_scaled = preprocessor.transform(X_test)

            # 5. Train each model
            results = []

            for model_type in self.config['models_to_train']:
                try:
                    logger.info(f"\n{'='*80}")
                    logger.info(f"Training {model_type.upper()}")
                    logger.info(f"{'='*80}")

                    # Train model
                    model_info = self._train_model(
                        model_type,
                        X_train_scaled, X_val_scaled, X_test_scaled,
                        y_train, y_val, y_test
                    )

                    # Compare with previous model
                    comparison = self._compare_with_previous(model_info)
                    model_info['comparison'] = comparison

                    # Update registry
                    self._update_registry(model_info, comparison['should_deploy'])

                    results.append(model_info)

                except Exception as e:
                    logger.error(f"Failed to train {model_type}: {e}", exc_info=True)

            # 6. Summary
            self._print_summary(results)

            logger.info("="*80)
            logger.info("RETRAINING COMPLETED SUCCESSFULLY")
            logger.info("="*80)

            # 7. Send notification
            if self.config.get('notification', {}).get('enabled', False):
                send_notification(
                    self.config['notification'],
                    results,
                    status='success'
                )

            return results

        except Exception as e:
            logger.error(f"Retraining failed: {e}", exc_info=True)

            # Send failure notification
            if self.config.get('notification', {}).get('enabled', False):
                error_info = {
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }
                send_notification(
                    self.config['notification'],
                    error_info,
                    status='failure'
                )

            raise

    def _print_summary(self, results):
        """Print summary of retraining results."""
        print(f"\n{'='*80}")
        print("RETRAINING SUMMARY")
        print(f"{'='*80}")

        for result in results:
            print(f"\n{result['model_type'].upper()}:")
            print(f"  Filename: {result['filename']}")
            print(f"  Training samples: {result['train_samples']}")
            print(f"  Test samples: {result['test_samples']}")
            print(f"  Training time: {result['training_time_seconds']}s")
            print(f"  Test RMSE: {result['test_metrics']['rmse']:.4f}")
            print(f"  Test R2: {result['test_metrics']['r2_score']:.4f}")

            comparison = result['comparison']
            if comparison['has_previous']:
                print(f"  Improvement: {comparison['improvement']*100:.2f}%")
                print(f"  Deployed: {'Yes' if comparison['should_deploy'] else 'No'}")
            else:
                print(f"  Deployed: Yes (first model)")

        print(f"\n{'='*80}\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Model Retraining Scheduler')

    parser.add_argument(
        '--mode',
        type=str,
        default='once',
        choices=['once', 'schedule'],
        help='Run mode: once (immediate) or schedule (daemon)'
    )

    parser.add_argument(
        '--interval',
        type=str,
        default='weekly',
        choices=['daily', 'weekly', 'monthly'],
        help='Retraining interval for schedule mode'
    )

    parser.add_argument(
        '--time',
        type=str,
        default='02:00',
        help='Time to run retraining (HH:MM format, for schedule mode)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/retraining_config.json',
        help='Path to retraining configuration file'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    manager = ModelRetrainingManager(config_path=args.config)

    if args.mode == 'once':
        # Run immediately
        logger.info("Running retraining once")
        manager.retrain_models()

    else:
        # Schedule mode
        logger.info(f"Scheduling retraining: {args.interval} at {args.time}")

        if args.interval == 'daily':
            schedule.every().day.at(args.time).do(manager.retrain_models)
        elif args.interval == 'weekly':
            schedule.every().monday.at(args.time).do(manager.retrain_models)
        elif args.interval == 'monthly':
            # Run on first day of each month
            def monthly_job():
                if datetime.now().day == 1:
                    manager.retrain_models()
            schedule.every().day.at(args.time).do(monthly_job)

        logger.info("Scheduler started, waiting for scheduled time...")

        # Run scheduler loop
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
    except Exception as e:
        logger.error(f"Scheduler failed: {e}", exc_info=True)
        sys.exit(1)
