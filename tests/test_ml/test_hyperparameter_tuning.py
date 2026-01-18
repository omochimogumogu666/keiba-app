"""
ハイパーパラメータ最適化モジュールのテスト

Optunaを使用した最適化機能のユニットテスト。
"""
import json
import os
import tempfile
import pytest
import numpy as np
import pandas as pd

from src.ml.hyperparameter_tuning import (
    XGBOOST_SEARCH_SPACE,
    RANDOM_FOREST_SEARCH_SPACE,
    create_xgboost_objective,
    create_random_forest_objective,
    run_hyperparameter_search,
    save_optimization_results,
    load_best_params,
    find_latest_params,
    get_default_params,
)


# テストデータ生成
@pytest.fixture
def sample_regression_data():
    """回帰タスク用サンプルデータ"""
    np.random.seed(42)
    n_samples = 200

    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
        'feature4': np.random.randn(n_samples),
        'feature5': np.random.randn(n_samples),
    })

    # 回帰ターゲット（着順1-18）
    y = pd.Series(
        1 + np.abs(X['feature1'] * 2 + X['feature2'] + np.random.randn(n_samples) * 2)
    ).clip(1, 18).astype(int)

    return X, y


@pytest.fixture
def sample_classification_data():
    """分類タスク用サンプルデータ"""
    np.random.seed(42)
    n_samples = 200

    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
        'feature4': np.random.randn(n_samples),
        'feature5': np.random.randn(n_samples),
    })

    # 分類ターゲット（0 or 1）
    proba = 1 / (1 + np.exp(-(X['feature1'] + X['feature2'])))
    y = pd.Series((proba > 0.5).astype(int))

    return X, y


class TestSearchSpaces:
    """探索空間定義のテスト"""

    def test_xgboost_search_space_structure(self):
        """XGBoost探索空間の構造チェック"""
        assert 'n_estimators' in XGBOOST_SEARCH_SPACE
        assert 'max_depth' in XGBOOST_SEARCH_SPACE
        assert 'learning_rate' in XGBOOST_SEARCH_SPACE

        # 型チェック
        assert XGBOOST_SEARCH_SPACE['n_estimators']['type'] == 'int'
        assert XGBOOST_SEARCH_SPACE['learning_rate']['type'] == 'float'

    def test_random_forest_search_space_structure(self):
        """RandomForest探索空間の構造チェック"""
        assert 'n_estimators' in RANDOM_FOREST_SEARCH_SPACE
        assert 'max_depth' in RANDOM_FOREST_SEARCH_SPACE
        assert 'max_features' in RANDOM_FOREST_SEARCH_SPACE

        # カテゴリカル型チェック
        assert RANDOM_FOREST_SEARCH_SPACE['max_features']['type'] == 'categorical'


class TestDefaultParams:
    """デフォルトパラメータのテスト"""

    def test_get_default_params_xgboost(self):
        """XGBoostデフォルトパラメータ取得"""
        params = get_default_params('xgboost')

        assert 'n_estimators' in params
        assert 'max_depth' in params
        assert 'learning_rate' in params
        assert params['n_estimators'] == 100
        assert params['max_depth'] == 6

    def test_get_default_params_random_forest(self):
        """RandomForestデフォルトパラメータ取得"""
        params = get_default_params('random_forest')

        assert 'n_estimators' in params
        assert 'max_depth' in params
        assert params['n_estimators'] == 100

    def test_get_default_params_invalid_model(self):
        """無効なモデルタイプでエラー"""
        with pytest.raises(ValueError):
            get_default_params('invalid_model')


@pytest.mark.slow
class TestHyperparameterSearch:
    """ハイパーパラメータ最適化のテスト（時間がかかる）"""

    def test_xgboost_optimization_regression(self, sample_regression_data):
        """XGBoost回帰タスクの最適化"""
        X, y = sample_regression_data

        study, best_params = run_hyperparameter_search(
            X, y,
            model_type='xgboost',
            task='regression',
            n_trials=3,  # テスト用に少数
            n_cv_splits=2,
            show_progress=False
        )

        assert study is not None
        assert best_params is not None
        assert 'n_estimators' in best_params
        assert study.best_value > 0  # RMSEは正

    def test_xgboost_optimization_classification(self, sample_classification_data):
        """XGBoost分類タスクの最適化"""
        X, y = sample_classification_data

        study, best_params = run_hyperparameter_search(
            X, y,
            model_type='xgboost',
            task='classification',
            n_trials=3,
            n_cv_splits=2,
            show_progress=False
        )

        assert study is not None
        assert best_params is not None
        assert study.best_value < 0  # 負のF1スコア

    def test_random_forest_optimization(self, sample_regression_data):
        """RandomForest最適化"""
        X, y = sample_regression_data

        study, best_params = run_hyperparameter_search(
            X, y,
            model_type='random_forest',
            task='regression',
            n_trials=3,
            n_cv_splits=2,
            show_progress=False
        )

        assert study is not None
        assert best_params is not None
        assert 'n_estimators' in best_params

    def test_invalid_model_type(self, sample_regression_data):
        """無効なモデルタイプでエラー"""
        X, y = sample_regression_data

        with pytest.raises(ValueError):
            run_hyperparameter_search(
                X, y,
                model_type='invalid',
                task='regression',
                n_trials=1
            )


class TestSaveAndLoad:
    """保存・読み込み機能のテスト"""

    def test_save_and_load_params(self):
        """パラメータの保存と読み込み"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # テスト用パラメータ
            params = {
                'n_estimators': 150,
                'max_depth': 8,
                'learning_rate': 0.05
            }

            # 保存
            params_path = os.path.join(tmpdir, 'test_params.json')
            with open(params_path, 'w') as f:
                json.dump(params, f)

            # 読み込み
            loaded_params = load_best_params(params_path)

            assert loaded_params == params

    def test_find_latest_params(self):
        """最新パラメータファイルの検索"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 複数のパラメータファイルを作成
            files = [
                'xgboost_regression_best_params_20260101_000000.json',
                'xgboost_regression_best_params_20260102_000000.json',
                'xgboost_regression_best_params_20260103_000000.json',
            ]

            for f in files:
                with open(os.path.join(tmpdir, f), 'w') as fp:
                    json.dump({'test': True}, fp)

            # 最新を検索
            latest = find_latest_params('xgboost', 'regression', tmpdir)

            assert latest is not None
            assert '20260103' in latest

    def test_find_latest_params_not_found(self):
        """パラメータファイルが見つからない場合"""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = find_latest_params('xgboost', 'regression', tmpdir)
            assert result is None

    @pytest.mark.slow
    def test_save_optimization_results(self, sample_regression_data):
        """最適化結果の保存"""
        X, y = sample_regression_data

        with tempfile.TemporaryDirectory() as tmpdir:
            study, best_params = run_hyperparameter_search(
                X, y,
                model_type='xgboost',
                task='regression',
                n_trials=2,
                n_cv_splits=2,
                show_progress=False
            )

            saved_paths = save_optimization_results(
                study, best_params,
                'xgboost', 'regression',
                tmpdir
            )

            # ファイル存在チェック
            assert os.path.exists(saved_paths['params_path'])
            assert os.path.exists(saved_paths['history_path'])
            assert os.path.exists(saved_paths['summary_path'])

            # パラメータ読み込みチェック
            loaded = load_best_params(saved_paths['params_path'])
            assert loaded == best_params


class TestObjectiveFunctions:
    """目的関数のテスト"""

    def test_create_xgboost_objective(self, sample_regression_data):
        """XGBoost目的関数の作成"""
        X, y = sample_regression_data

        objective = create_xgboost_objective(X, y, 'regression', n_cv_splits=2)

        assert callable(objective)

    def test_create_random_forest_objective(self, sample_regression_data):
        """RandomForest目的関数の作成"""
        X, y = sample_regression_data

        objective = create_random_forest_objective(X, y, 'regression', n_cv_splits=2)

        assert callable(objective)
