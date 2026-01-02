"""
Script to extract features from race data for ML training.

This script demonstrates how to use the FeatureExtractor to prepare
training data from completed races in the database.
"""
import os
import sys
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.web.app import create_app
from src.ml.feature_engineering import FeatureExtractor, save_features_to_csv
from src.ml.preprocessing import (
    FeaturePreprocessor,
    handle_missing_values,
    create_target_variable
)
from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)


def main():
    """Extract features and prepare training data."""
    # Create Flask app and get database session
    app = create_app()

    with app.app_context():
        from src.data.models import db

        # Initialize feature extractor
        extractor = FeatureExtractor(db.session, lookback_days=730)

        # Define date range for training data
        # Example: Extract features for races in the last year
        max_date = datetime.utcnow()
        min_date = max_date - timedelta(days=365)

        logger.info(f"Extracting features for races from {min_date.date()} to {max_date.date()}")

        # Extract features and labels
        X, y = extractor.extract_features_for_training(
            min_date=min_date,
            max_date=max_date
        )

        if X.empty:
            logger.warning("No features extracted. Make sure you have completed races with results in the database.")
            return

        logger.info(f"Extracted {len(X)} samples with {len(X.columns)} features")
        logger.info(f"Feature columns: {X.columns.tolist()}")

        # Handle missing values
        X_filled = handle_missing_values(X, strategy='median')

        # Save raw features to CSV
        output_dir = 'data/processed'
        os.makedirs(output_dir, exist_ok=True)

        features_path = os.path.join(output_dir, 'features_raw.csv')
        save_features_to_csv(X_filled, features_path)

        # Save labels
        labels_path = os.path.join(output_dir, 'labels.csv')
        y.to_csv(labels_path, index=False, encoding='utf-8-sig')
        logger.info(f"Saved labels to {labels_path}")

        # Preprocess features (scaling and normalization)
        preprocessor = FeaturePreprocessor()
        X_processed = preprocessor.fit_transform(X_filled)

        # Save processed features
        processed_path = os.path.join(output_dir, 'features_processed.csv')
        save_features_to_csv(X_processed, processed_path)

        # Create different target variables for different model types
        # 1. Regression target (exact finish position)
        y_regression = create_target_variable(y, target_type='regression')
        y_regression.to_csv(
            os.path.join(output_dir, 'labels_regression.csv'),
            index=False,
            encoding='utf-8-sig'
        )

        # 2. Binary classification target (win or not)
        y_binary = create_target_variable(y, target_type='binary_win')
        y_binary.to_csv(
            os.path.join(output_dir, 'labels_binary.csv'),
            index=False,
            encoding='utf-8-sig'
        )

        # 3. Multi-class target (win/place/other)
        y_multiclass = create_target_variable(y, target_type='multiclass_top3')
        y_multiclass.to_csv(
            os.path.join(output_dir, 'labels_multiclass.csv'),
            index=False,
            encoding='utf-8-sig'
        )

        logger.info("Feature extraction completed successfully!")
        logger.info(f"Training data saved to {output_dir}/")

        # Print summary statistics
        print("\n" + "="*60)
        print("FEATURE EXTRACTION SUMMARY")
        print("="*60)
        print(f"Total samples: {len(X)}")
        print(f"Total features: {len(X.columns)}")
        print(f"Date range: {min_date.date()} to {max_date.date()}")
        print(f"\nLabel distribution:")
        print(y.value_counts().sort_index().head(10))
        print(f"\nWin rate: {(y == 1).sum() / len(y) * 100:.2f}%")
        print(f"Top-3 rate: {(y <= 3).sum() / len(y) * 100:.2f}%")
        print("\nOutput files:")
        print(f"  - {features_path}")
        print(f"  - {processed_path}")
        print(f"  - {labels_path}")
        print("="*60)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}", exc_info=True)
        sys.exit(1)
