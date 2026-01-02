"""
Script to generate predictions for upcoming races and save to database.

This script:
1. Loads a trained model from file
2. Extracts features for upcoming races
3. Generates predictions
4. Saves predictions to the database
"""
import os
import sys
from datetime import datetime
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.web.app import create_app
from src.ml.feature_engineering import FeatureExtractor
from src.ml.preprocessing import FeaturePreprocessor, handle_missing_values
from src.ml.models.random_forest import RandomForestRaceModel
from src.ml.models.xgboost_model import XGBoostRaceModel
from src.data.models import db, Prediction
from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate race predictions')

    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained model file'
    )

    parser.add_argument(
        '--race-id',
        type=int,
        help='Specific race ID to predict (optional, default: all upcoming races)'
    )

    parser.add_argument(
        '--race-date',
        type=str,
        help='Date to predict races for (YYYY-MM-DD format, default: today)'
    )

    parser.add_argument(
        '--save-to-db',
        action='store_true',
        help='Save predictions to database'
    )

    parser.add_argument(
        '--output-csv',
        type=str,
        help='Path to save predictions as CSV'
    )

    return parser.parse_args()


def load_model(model_path):
    """
    Load trained model from file.

    Args:
        model_path: Path to model file

    Returns:
        Loaded model instance
    """
    logger.info(f"Loading model from {model_path}")

    # Determine model type from filename
    if 'xgboost' in model_path.lower():
        model = XGBoostRaceModel()
    else:
        model = RandomForestRaceModel()

    model.load(model_path)

    logger.info(f"Loaded {model.model_name} v{model.version}")

    return model


def get_races_to_predict(app, race_id=None, race_date=None):
    """
    Get list of races to predict.

    Args:
        app: Flask app instance
        race_id: Specific race ID (optional)
        race_date: Race date (optional)

    Returns:
        List of race IDs
    """
    with app.app_context():
        from src.data.models import Race

        query = db.session.query(Race).filter(Race.status == 'upcoming')

        if race_id:
            query = query.filter(Race.id == race_id)
        elif race_date:
            from datetime import datetime
            date_obj = datetime.strptime(race_date, '%Y-%m-%d').date()
            query = query.filter(Race.race_date == date_obj)
        else:
            # Default: today's races
            today = datetime.utcnow().date()
            query = query.filter(Race.race_date == today)

        races = query.all()

        logger.info(f"Found {len(races)} races to predict")

        return [race.id for race in races]


def generate_predictions_for_race(extractor, model, race_id):
    """
    Generate predictions for a single race.

    Args:
        extractor: FeatureExtractor instance
        model: Trained model instance
        race_id: Race ID

    Returns:
        DataFrame with predictions
    """
    # Extract features
    X = extractor.extract_features_for_race(race_id)

    if X.empty:
        logger.warning(f"No entries found for race {race_id}")
        return None

    # Handle missing values
    X_filled = handle_missing_values(X, strategy='median')

    # Make predictions
    if model.task == 'regression':
        # Predict finish positions
        predictions = model.predict(X_filled)

        X_filled['predicted_position'] = predictions

        # Calculate win probability based on predicted position
        # Lower position = higher probability
        max_position = X_filled['predicted_position'].max()
        X_filled['win_probability'] = 1 - (X_filled['predicted_position'] - 1) / max_position
        X_filled['confidence_score'] = X_filled['win_probability']

    else:
        # Classification: predict win probability
        probas = model.predict_proba(X_filled)

        if probas.shape[1] == 2:
            win_proba = probas[:, 1]
        else:
            win_proba = probas[:, -1]

        X_filled['win_probability'] = win_proba
        X_filled['confidence_score'] = win_proba

        # Assign predicted positions based on probability ranking
        X_filled['predicted_position'] = X_filled['win_probability'].rank(
            ascending=False, method='first'
        ).astype(int)

    return X_filled[['race_id', 'horse_id', 'predicted_position', 'win_probability', 'confidence_score']]


def save_predictions_to_db(predictions_df, model_name, model_version):
    """
    Save predictions to database.

    Args:
        predictions_df: DataFrame with predictions
        model_name: Name of the model
        model_version: Version of the model
    """
    logger.info(f"Saving {len(predictions_df)} predictions to database")

    from src.data.models import db

    for _, row in predictions_df.iterrows():
        # Check if prediction already exists
        existing = db.session.query(Prediction).filter_by(
            race_id=int(row['race_id']),
            horse_id=int(row['horse_id']),
            model_name=model_name,
            model_version=model_version
        ).first()

        if existing:
            # Update existing prediction
            existing.predicted_position = int(row['predicted_position'])
            existing.win_probability = float(row['win_probability'])
            existing.confidence_score = float(row['confidence_score'])
        else:
            # Create new prediction
            prediction = Prediction(
                race_id=int(row['race_id']),
                horse_id=int(row['horse_id']),
                predicted_position=int(row['predicted_position']),
                win_probability=float(row['win_probability']),
                confidence_score=float(row['confidence_score']),
                model_name=model_name,
                model_version=model_version
            )
            db.session.add(prediction)

    db.session.commit()

    logger.info("Predictions saved to database")


def main():
    """Main prediction generation pipeline."""
    args = parse_args()

    print("="*80)
    print("RACE PREDICTION GENERATION")
    print("="*80)

    # Load model
    model = load_model(args.model_path)

    print(f"Model: {model.model_name} v{model.version}")
    print(f"Task: {model.task}")
    print(f"Features: {len(model.feature_columns)}")

    # Create Flask app
    app = create_app()

    # Get races to predict
    race_ids = get_races_to_predict(app, args.race_id, args.race_date)

    if not race_ids:
        print("No upcoming races found to predict")
        return

    print(f"Races to predict: {len(race_ids)}")
    print("="*80)

    # Generate predictions
    all_predictions = []

    with app.app_context():
        extractor = FeatureExtractor(db.session)

        for i, race_id in enumerate(race_ids, 1):
            logger.info(f"Generating predictions for race {race_id} ({i}/{len(race_ids)})")

            predictions = generate_predictions_for_race(extractor, model, race_id)

            if predictions is not None:
                all_predictions.append(predictions)

                # Display top 3 predictions for this race
                top_3 = predictions.nsmallest(3, 'predicted_position')
                print(f"\nRace {race_id} - Top 3 Predictions:")
                for _, row in top_3.iterrows():
                    print(f"  {int(row['predicted_position'])}. Horse {int(row['horse_id'])} "
                          f"(Win Prob: {row['win_probability']:.2%}, "
                          f"Confidence: {row['confidence_score']:.2%})")

    if not all_predictions:
        print("\nNo predictions generated")
        return

    # Combine all predictions
    import pandas as pd
    combined_predictions = pd.concat(all_predictions, ignore_index=True)

    print(f"\n{'='*80}")
    print(f"Generated {len(combined_predictions)} predictions for {len(race_ids)} races")
    print(f"{'='*80}")

    # Save to database
    if args.save_to_db:
        with app.app_context():
            save_predictions_to_db(
                combined_predictions,
                model.model_name,
                model.version
            )
            print(f"\nPredictions saved to database")

    # Save to CSV
    if args.output_csv:
        combined_predictions.to_csv(args.output_csv, index=False, encoding='utf-8-sig')
        print(f"Predictions saved to {args.output_csv}")

    print(f"\n{'='*80}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Prediction generation failed: {e}", exc_info=True)
        sys.exit(1)
