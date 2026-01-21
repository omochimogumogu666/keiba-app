"""
Model comparison utilities for evaluating multiple prediction models.

複数モデルの比較評価ユーティリティ。
モデル間のパフォーマンス比較、最適モデル選択を支援。
"""
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from src.ml.models.base_model import BaseRaceModel
from src.ml.evaluation import (
    evaluate_regression_model,
    evaluate_classification_model,
    evaluate_ranking_performance
)
from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)


class ModelComparator:
    """
    複数モデルの性能比較クラス。

    同一データセットで複数モデルを評価し、比較レポートを生成。
    """

    def __init__(self, task: str = 'regression'):
        """
        ModelComparatorを初期化。

        Args:
            task: 'regression' or 'classification'
        """
        self.task = task
        self.models: Dict[str, BaseRaceModel] = {}
        self.results: Dict[str, Dict[str, Any]] = {}

    def add_model(self, name: str, model: BaseRaceModel) -> None:
        """
        比較対象モデルを追加。

        Args:
            name: モデル識別名
            model: 学習済みモデル
        """
        if not model.is_trained:
            raise ValueError(f"Model '{name}' must be trained before comparison")
        self.models[name] = model
        logger.info(f"Added model '{name}' for comparison")

    def compare_models(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        race_ids: Optional[pd.Series] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        全モデルを同一データで評価。

        Args:
            X_test: テスト特徴量
            y_test: テストラベル
            race_ids: レースID（ランキング評価用）

        Returns:
            モデル名をキーとする評価結果辞書
        """
        if not self.models:
            raise ValueError("No models added for comparison")

        self.results = {}

        for name, model in self.models.items():
            logger.info(f"Evaluating model: {name}")

            try:
                # 予測
                y_pred = model.predict(X_test)

                # タスクに応じた評価
                if self.task == 'regression':
                    metrics = evaluate_regression_model(y_test, y_pred, name)

                    # ランキング評価
                    if race_ids is not None:
                        ranking_metrics = evaluate_ranking_performance(
                            y_test, y_pred, race_ids
                        )
                        metrics.update(ranking_metrics)

                else:
                    metrics = evaluate_classification_model(y_test, y_pred, name)

                    # 確率予測がある場合
                    if hasattr(model, 'predict_proba'):
                        try:
                            y_proba = model.predict_proba(X_test)
                            if y_proba.shape[1] == 2:
                                metrics['win_proba_available'] = True
                        except Exception:
                            pass

                # 特徴量重要度
                if hasattr(model, 'get_feature_importance'):
                    try:
                        importance = model.get_feature_importance()
                        metrics['top_features'] = importance.head(5).to_dict()
                    except Exception:
                        pass

                self.results[name] = {
                    'metrics': metrics,
                    'predictions': y_pred,
                    'model_info': model.get_model_info()
                }

            except Exception as e:
                logger.error(f"Error evaluating model '{name}': {e}")
                self.results[name] = {
                    'error': str(e),
                    'metrics': {}
                }

        return self.results

    def get_best_model_by_metric(
        self,
        metric: str,
        higher_is_better: bool = None
    ) -> Tuple[str, float]:
        """
        指定メトリクスで最良モデルを選択。

        Args:
            metric: 比較メトリクス名
            higher_is_better: Trueなら高い値が良い（Noneで自動判定）

        Returns:
            (モデル名, メトリクス値)のタプル
        """
        if not self.results:
            raise ValueError("No comparison results. Run compare_models first.")

        # 自動判定
        if higher_is_better is None:
            higher_is_better = metric in [
                'accuracy', 'f1_score', 'precision', 'recall',
                'r2_score', 'spearman_correlation', 'kendall_tau'
            ]

        best_name = None
        best_value = float('-inf') if higher_is_better else float('inf')

        for name, result in self.results.items():
            if 'error' in result:
                continue

            value = result['metrics'].get(metric)
            if value is None:
                continue

            if higher_is_better:
                if value > best_value:
                    best_value = value
                    best_name = name
            else:
                if value < best_value:
                    best_value = value
                    best_name = name

        if best_name is None:
            raise ValueError(f"Metric '{metric}' not found in any model results")

        return best_name, best_value

    def generate_comparison_report(self) -> pd.DataFrame:
        """
        比較レポートをDataFrame形式で生成。

        Returns:
            モデル別メトリクスの比較表
        """
        if not self.results:
            raise ValueError("No comparison results. Run compare_models first.")

        rows = []
        for name, result in self.results.items():
            if 'error' in result:
                row = {'model_name': name, 'error': result['error']}
            else:
                row = {'model_name': name}
                row.update(result['metrics'])
            rows.append(row)

        df = pd.DataFrame(rows)

        # メトリクスでソート
        if self.task == 'regression':
            if 'rmse' in df.columns:
                df = df.sort_values('rmse', ascending=True)
        else:
            if 'accuracy' in df.columns:
                df = df.sort_values('accuracy', ascending=False)

        return df

    def get_summary(self) -> Dict[str, Any]:
        """
        比較サマリーを取得。

        Returns:
            サマリー情報の辞書
        """
        if not self.results:
            return {'error': 'No comparison results'}

        summary = {
            'task': self.task,
            'n_models': len(self.models),
            'model_names': list(self.models.keys()),
            'comparison_timestamp': datetime.utcnow().isoformat()
        }

        # 最良モデルを各メトリクスで選択
        if self.task == 'regression':
            try:
                summary['best_by_rmse'] = self.get_best_model_by_metric('rmse', False)
                summary['best_by_mae'] = self.get_best_model_by_metric('mae', False)
                summary['best_by_r2'] = self.get_best_model_by_metric('r2_score', True)
            except ValueError:
                pass
        else:
            try:
                summary['best_by_accuracy'] = self.get_best_model_by_metric('accuracy', True)
                summary['best_by_f1'] = self.get_best_model_by_metric('f1_score', True)
            except ValueError:
                pass

        return summary


def compare_trained_models(
    model_paths: List[str],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task: str = 'regression',
    race_ids: Optional[pd.Series] = None
) -> Dict[str, Any]:
    """
    保存済みモデルを読み込んで比較する便利関数。

    Args:
        model_paths: モデルファイルパスのリスト
        X_test: テスト特徴量
        y_test: テストラベル
        task: 'regression' or 'classification'
        race_ids: レースID（ランキング評価用）

    Returns:
        比較結果と最良モデル情報
    """
    from src.ml.models.random_forest import RandomForestRaceModel
    from src.ml.models.xgboost_model import XGBoostRaceModel
    from src.ml.models import TORCH_AVAILABLE

    comparator = ModelComparator(task=task)

    for path in model_paths:
        try:
            # ファイル名からモデルタイプを推定
            if 'randomforest' in path.lower():
                model = RandomForestRaceModel(task=task)
            elif 'xgboost' in path.lower():
                model = XGBoostRaceModel(task=task)
            elif 'neuralnetwork' in path.lower() and TORCH_AVAILABLE:
                from src.ml.models.neural_network import NeuralNetworkRaceModel
                model = NeuralNetworkRaceModel(task=task)
            else:
                logger.warning(f"Unknown model type for path: {path}")
                continue

            model.load(path)
            model_name = path.split('/')[-1].replace('.pkl', '').replace('.pt', '')
            comparator.add_model(model_name, model)

        except Exception as e:
            logger.error(f"Failed to load model from {path}: {e}")

    if not comparator.models:
        return {'error': 'No models could be loaded'}

    # 比較実行
    comparator.compare_models(X_test, y_test, race_ids)

    return {
        'summary': comparator.get_summary(),
        'report': comparator.generate_comparison_report().to_dict('records'),
        'results': {
            name: {
                'metrics': result.get('metrics', {}),
                'model_info': result.get('model_info', {})
            }
            for name, result in comparator.results.items()
        }
    }
