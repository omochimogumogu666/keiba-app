"""
ハイパーパラメータ最適化スクリプト

Optunaを使用してXGBoostとRandomForestのハイパーパラメータを最適化し、
最適なモデルを保存する。

使用例:
    python scripts/optimize_hyperparameters.py --model xgboost --task regression --n-trials 100
    python scripts/optimize_hyperparameters.py --model random_forest --task classification --n-trials 50
    python scripts/optimize_hyperparameters.py --model both --task regression --n-trials 100
"""
import os
import sys
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.web.app import create_app
from src.ml.feature_engineering import FeatureExtractor
from src.ml.preprocessing import (
    FeaturePreprocessor,
    handle_missing_values,
    create_target_variable
)
from src.ml.hyperparameter_tuning import (
    run_hyperparameter_search,
    save_optimization_results,
    get_default_params
)
from src.ml.models.random_forest import RandomForestRaceModel
from src.ml.models.xgboost_model import XGBoostRaceModel
from src.ml.evaluation import evaluate_regression_model, evaluate_classification_model
from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description='ハイパーパラメータ最適化スクリプト',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用例:
  python scripts/optimize_hyperparameters.py --model xgboost --n-trials 100
  python scripts/optimize_hyperparameters.py --model both --task classification
        '''
    )

    parser.add_argument(
        '--model',
        type=str,
        default='xgboost',
        choices=['xgboost', 'random_forest', 'both'],
        help='最適化するモデル（default: xgboost）'
    )

    parser.add_argument(
        '--task',
        type=str,
        default='regression',
        choices=['regression', 'classification'],
        help='タスク（default: regression）'
    )

    parser.add_argument(
        '--n-trials',
        type=int,
        default=100,
        help='Optuna試行回数（default: 100）'
    )

    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='クロスバリデーション分割数（default: 5）'
    )

    parser.add_argument(
        '--timeout',
        type=int,
        default=None,
        help='最適化タイムアウト（秒）'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/models',
        help='出力ディレクトリ（default: data/models）'
    )

    parser.add_argument(
        '--train-final',
        action='store_true',
        help='最適パラメータで最終モデルを訓練・保存'
    )

    parser.add_argument(
        '--compare-baseline',
        action='store_true',
        help='デフォルトパラメータとの比較を表示'
    )

    parser.add_argument(
        '--data-source',
        type=str,
        default='database',
        choices=['database', 'csv'],
        help='データソース（default: database）'
    )

    parser.add_argument(
        '--csv-path',
        type=str,
        default='data/processed/features_processed.csv',
        help='CSVファイルパス（data-source=csv時）'
    )

    return parser.parse_args()


def load_data_from_database(app):
    """データベースからデータを読み込む"""
    logger.info("Loading data from database")

    with app.app_context():
        from src.data.models import db

        extractor = FeatureExtractor(db.session, lookback_days=730)
        X, y = extractor.extract_features_for_training()

        if X.empty:
            raise ValueError("No training data found in database")

        logger.info(f"Loaded {len(X)} samples from database")
        return X, y


def load_data_from_csv(csv_path):
    """CSVからデータを読み込む"""
    import pandas as pd

    logger.info(f"Loading data from CSV: {csv_path}")

    X = pd.read_csv(csv_path, encoding='utf-8-sig')

    # ターゲット列の抽出
    if 'finish_position' in X.columns:
        y = X['finish_position']
        X = X.drop('finish_position', axis=1)
    else:
        # labels_regression.csvから読み込み
        labels_path = csv_path.replace('features_processed', 'labels_regression')
        if os.path.exists(labels_path):
            y = pd.read_csv(labels_path, encoding='utf-8-sig').squeeze()
        else:
            raise ValueError(f"Target column not found. Create labels file at: {labels_path}")

    logger.info(f"Loaded {len(X)} samples from CSV")
    return X, y


def preprocess_data(X, y, task):
    """データの前処理"""
    # 欠損値処理
    X = handle_missing_values(X, strategy='median')

    # 識別子列を除外
    feature_cols = [
        c for c in X.columns
        if c not in ['race_id', 'horse_id', 'race_entry_id', 'race_date']
    ]
    X_features = X[feature_cols]

    # ターゲット変換（分類タスクの場合）
    if task == 'classification':
        y = create_target_variable(y, target_type='binary_win')

    # 前処理
    preprocessor = FeaturePreprocessor()
    X_processed = preprocessor.fit_transform(X_features)

    return X_processed, y, preprocessor


def train_and_evaluate_model(
    X_train, y_train, X_test, y_test,
    model_type, task, params, label='Model'
):
    """モデルを訓練・評価する"""
    if model_type == 'xgboost':
        model = XGBoostRaceModel(task=task, **params)
    else:
        model = RandomForestRaceModel(task=task, **params)

    # 訓練
    model.train(X_train, y_train)

    # 予測
    y_pred = model.predict(X_test)

    # 評価
    if task == 'regression':
        metrics = evaluate_regression_model(y_test, y_pred, label)
    else:
        metrics = evaluate_classification_model(y_test, y_pred, label)

    return model, metrics


def main():
    """メイン処理"""
    args = parse_args()

    print("=" * 80)
    print("ハイパーパラメータ最適化")
    print("=" * 80)
    print(f"モデル: {args.model}")
    print(f"タスク: {args.task}")
    print(f"試行回数: {args.n_trials}")
    print(f"CV分割数: {args.cv_folds}")
    print("=" * 80)

    # データ読み込み
    if args.data_source == 'database':
        app = create_app()
        X, y = load_data_from_database(app)
    else:
        X, y = load_data_from_csv(args.csv_path)

    # 前処理
    X_processed, y_processed, preprocessor = preprocess_data(X, y, args.task)
    logger.info(f"Preprocessed data shape: {X_processed.shape}")

    # 訓練/テスト分割（時系列）
    split_idx = int(len(X_processed) * 0.8)
    X_train = X_processed.iloc[:split_idx]
    y_train = y_processed.iloc[:split_idx]
    X_test = X_processed.iloc[split_idx:]
    y_test = y_processed.iloc[split_idx:]

    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # 最適化対象モデル
    models_to_optimize = []
    if args.model in ['xgboost', 'both']:
        models_to_optimize.append('xgboost')
    if args.model in ['random_forest', 'both']:
        models_to_optimize.append('random_forest')

    results = {}

    for model_type in models_to_optimize:
        print(f"\n{'=' * 80}")
        print(f"{model_type.upper()} の最適化開始")
        print(f"{'=' * 80}")

        # ハイパーパラメータ最適化
        study, best_params = run_hyperparameter_search(
            X_train, y_train,
            model_type=model_type,
            task=args.task,
            n_trials=args.n_trials,
            n_cv_splits=args.cv_folds,
            timeout=args.timeout,
            show_progress=True
        )

        # 結果保存
        saved_paths = save_optimization_results(
            study, best_params, model_type, args.task, args.output_dir
        )

        results[model_type] = {
            'best_params': best_params,
            'best_score': study.best_value,
            'saved_paths': saved_paths
        }

        # デフォルトとの比較
        if args.compare_baseline:
            print(f"\n--- {model_type} ベースライン比較 ---")

            # デフォルトパラメータでの評価
            default_params = get_default_params(model_type)
            _, default_metrics = train_and_evaluate_model(
                X_train, y_train, X_test, y_test,
                model_type, args.task, default_params,
                f'{model_type}_default'
            )

            # 最適パラメータでの評価
            _, optimized_metrics = train_and_evaluate_model(
                X_train, y_train, X_test, y_test,
                model_type, args.task, best_params,
                f'{model_type}_optimized'
            )

            # 改善率計算
            if args.task == 'regression':
                default_score = default_metrics['rmse']
                optimized_score = optimized_metrics['rmse']
                improvement = (default_score - optimized_score) / default_score * 100
                print(f"  Default RMSE:   {default_score:.4f}")
                print(f"  Optimized RMSE: {optimized_score:.4f}")
                print(f"  改善率: {improvement:.2f}%")
            else:
                default_score = default_metrics['f1_score']
                optimized_score = optimized_metrics['f1_score']
                improvement = (optimized_score - default_score) / max(default_score, 0.001) * 100
                print(f"  Default F1:   {default_score:.4f}")
                print(f"  Optimized F1: {optimized_score:.4f}")
                print(f"  改善率: {improvement:.2f}%")

            results[model_type]['default_metrics'] = default_metrics
            results[model_type]['optimized_metrics'] = optimized_metrics
            results[model_type]['improvement'] = improvement

        # 最終モデル訓練・保存
        if args.train_final:
            print(f"\n--- {model_type} 最終モデル訓練 ---")

            model, _ = train_and_evaluate_model(
                X_train, y_train, X_test, y_test,
                model_type, args.task, best_params,
                f'{model_type}_final'
            )

            # モデル保存
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            model_path = os.path.join(
                args.output_dir,
                f"{model_type}_optimized_{args.task}_{timestamp}.pkl"
            )
            model.save(model_path)
            print(f"最終モデル保存: {model_path}")
            results[model_type]['model_path'] = model_path

    # サマリー
    print(f"\n{'=' * 80}")
    print("最適化結果サマリー")
    print(f"{'=' * 80}")

    for model_type, result in results.items():
        print(f"\n{model_type.upper()}:")
        print(f"  Best CV Score: {result['best_score']:.4f}")
        print(f"  Best Params: {result['best_params']}")
        if 'improvement' in result:
            print(f"  Improvement: {result['improvement']:.2f}%")

    print(f"\n{'=' * 80}")
    print("完了")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n最適化が中断されました")
        sys.exit(0)
    except Exception as e:
        logger.error(f"最適化エラー: {e}", exc_info=True)
        sys.exit(1)
