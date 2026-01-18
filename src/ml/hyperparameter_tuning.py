"""
ハイパーパラメータ最適化モジュール

Optunaを使用してXGBoostとRandomForestのハイパーパラメータを最適化する。
時系列クロスバリデーションを用いて、過学習を防ぎながら最適なパラメータを探索。
"""
import json
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, f1_score

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)


# XGBoost探索空間
XGBOOST_SEARCH_SPACE = {
    'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
    'max_depth': {'type': 'int', 'low': 3, 'high': 12},
    'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
    'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
    'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
    'min_child_weight': {'type': 'int', 'low': 1, 'high': 10},
    'gamma': {'type': 'float', 'low': 0.0, 'high': 1.0},
    'reg_alpha': {'type': 'float', 'low': 1e-8, 'high': 1.0, 'log': True},
    'reg_lambda': {'type': 'float', 'low': 1e-8, 'high': 1.0, 'log': True},
}

# RandomForest探索空間
RANDOM_FOREST_SEARCH_SPACE = {
    'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
    'max_depth': {'type': 'int', 'low': 5, 'high': 30},
    'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
    'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
    'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', None]},
}


def _suggest_param(trial: 'optuna.Trial', name: str, config: Dict[str, Any]) -> Any:
    """
    Optunaのtrialからパラメータを提案する。

    Args:
        trial: Optuna trial
        name: パラメータ名
        config: パラメータ設定

    Returns:
        提案されたパラメータ値
    """
    param_type = config['type']

    if param_type == 'int':
        return trial.suggest_int(name, config['low'], config['high'])
    elif param_type == 'float':
        log = config.get('log', False)
        return trial.suggest_float(name, config['low'], config['high'], log=log)
    elif param_type == 'categorical':
        return trial.suggest_categorical(name, config['choices'])
    else:
        raise ValueError(f"Unknown parameter type: {param_type}")


def create_xgboost_objective(
    X: pd.DataFrame,
    y: pd.Series,
    task: str = 'regression',
    n_cv_splits: int = 5,
    random_state: int = 42
) -> Callable:
    """
    XGBoost用のOptuna目的関数を作成する。

    Args:
        X: 特徴量DataFrame
        y: ターゲットSeries
        task: 'regression' or 'classification'
        n_cv_splits: クロスバリデーション分割数
        random_state: ランダムシード

    Returns:
        Optuna目的関数
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("XGBoost is not installed")

    def objective(trial: 'optuna.Trial') -> float:
        # パラメータ提案
        params = {}
        for name, config in XGBOOST_SEARCH_SPACE.items():
            params[name] = _suggest_param(trial, name, config)

        params['random_state'] = random_state
        params['verbosity'] = 0  # 出力を抑制

        # 時系列CV
        tscv = TimeSeriesSplit(n_splits=n_cv_splits)
        scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # モデル作成・学習
            if task == 'regression':
                model = xgb.XGBRegressor(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=10,
                    verbose=False
                )
                y_pred = model.predict(X_val)
                score = np.sqrt(mean_squared_error(y_val, y_pred))  # RMSE
            else:  # classification
                model = xgb.XGBClassifier(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=10,
                    verbose=False
                )
                y_pred = model.predict(X_val)
                score = -f1_score(y_val, y_pred, average='binary', zero_division=0)  # 負のF1（最大化のため）

            scores.append(score)

            # Pruning（早期停止）
            trial.report(np.mean(scores), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(scores)

    return objective


def create_random_forest_objective(
    X: pd.DataFrame,
    y: pd.Series,
    task: str = 'regression',
    n_cv_splits: int = 5,
    random_state: int = 42
) -> Callable:
    """
    RandomForest用のOptuna目的関数を作成する。

    Args:
        X: 特徴量DataFrame
        y: ターゲットSeries
        task: 'regression' or 'classification'
        n_cv_splits: クロスバリデーション分割数
        random_state: ランダムシード

    Returns:
        Optuna目的関数
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    def objective(trial: 'optuna.Trial') -> float:
        # パラメータ提案
        params = {}
        for name, config in RANDOM_FOREST_SEARCH_SPACE.items():
            params[name] = _suggest_param(trial, name, config)

        params['random_state'] = random_state
        params['n_jobs'] = -1  # 全CPUを使用

        # 時系列CV
        tscv = TimeSeriesSplit(n_splits=n_cv_splits)
        scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # モデル作成・学習
            if task == 'regression':
                model = RandomForestRegressor(**params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                score = np.sqrt(mean_squared_error(y_val, y_pred))  # RMSE
            else:  # classification
                model = RandomForestClassifier(**params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                score = -f1_score(y_val, y_pred, average='binary', zero_division=0)  # 負のF1

            scores.append(score)

            # Pruning
            trial.report(np.mean(scores), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(scores)

    return objective


def run_hyperparameter_search(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = 'xgboost',
    task: str = 'regression',
    n_trials: int = 100,
    n_cv_splits: int = 5,
    timeout: Optional[int] = None,
    random_state: int = 42,
    study_name: Optional[str] = None,
    show_progress: bool = True
) -> Tuple['optuna.Study', Dict[str, Any]]:
    """
    ハイパーパラメータ最適化を実行する。

    Args:
        X: 特徴量DataFrame
        y: ターゲットSeries
        model_type: 'xgboost' or 'random_forest'
        task: 'regression' or 'classification'
        n_trials: 試行回数
        n_cv_splits: CV分割数
        timeout: タイムアウト（秒）
        random_state: ランダムシード
        study_name: Study名（省略時は自動生成）
        show_progress: 進捗表示

    Returns:
        (Optuna Study, 最適パラメータ辞書)
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is not installed. Run: pip install optuna")

    logger.info(f"Starting hyperparameter optimization for {model_type} ({task})")
    logger.info(f"Data shape: {X.shape}, n_trials: {n_trials}, n_cv_splits: {n_cv_splits}")

    # 目的関数作成
    if model_type == 'xgboost':
        objective = create_xgboost_objective(X, y, task, n_cv_splits, random_state)
    elif model_type == 'random_forest':
        objective = create_random_forest_objective(X, y, task, n_cv_splits, random_state)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Study作成
    if study_name is None:
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        study_name = f"{model_type}_{task}_{timestamp}"

    # 最適化方向の決定（回帰=RMSE最小化、分類=負のF1最小化=F1最大化）
    direction = 'minimize'

    sampler = TPESampler(seed=random_state)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        pruner=pruner
    )

    # 最適化実行
    optuna.logging.set_verbosity(optuna.logging.WARNING if not show_progress else optuna.logging.INFO)

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=show_progress,
        gc_after_trial=True
    )

    # 結果取得
    best_params = study.best_params.copy()
    best_params['random_state'] = random_state

    logger.info(f"Optimization complete. Best score: {study.best_value:.4f}")
    logger.info(f"Best parameters: {best_params}")

    return study, best_params


def save_optimization_results(
    study: 'optuna.Study',
    best_params: Dict[str, Any],
    model_type: str,
    task: str,
    output_dir: str = 'data/models'
) -> Dict[str, str]:
    """
    最適化結果を保存する。

    Args:
        study: Optuna Study
        best_params: 最適パラメータ
        model_type: モデルタイプ
        task: タスク
        output_dir: 出力ディレクトリ

    Returns:
        保存ファイルパスの辞書
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')

    # パラメータ保存（JSON）
    params_path = os.path.join(
        output_dir,
        f"{model_type}_{task}_best_params_{timestamp}.json"
    )
    with open(params_path, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)
    logger.info(f"Best parameters saved to: {params_path}")

    # 最適化履歴保存（CSV）
    history_path = os.path.join(
        output_dir,
        f"{model_type}_{task}_optimization_history_{timestamp}.csv"
    )
    trials_df = study.trials_dataframe()
    trials_df.to_csv(history_path, index=False, encoding='utf-8-sig')
    logger.info(f"Optimization history saved to: {history_path}")

    # 最適化サマリー保存
    summary_path = os.path.join(
        output_dir,
        f"{model_type}_{task}_optimization_summary_{timestamp}.json"
    )
    summary = {
        'study_name': study.study_name,
        'model_type': model_type,
        'task': task,
        'n_trials': len(study.trials),
        'best_value': study.best_value,
        'best_params': best_params,
        'best_trial_number': study.best_trial.number,
        'optimization_time': str(datetime.utcnow()),
        'n_pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'n_complete_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Optimization summary saved to: {summary_path}")

    return {
        'params_path': params_path,
        'history_path': history_path,
        'summary_path': summary_path
    }


def load_best_params(params_path: str) -> Dict[str, Any]:
    """
    保存された最適パラメータを読み込む。

    Args:
        params_path: パラメータファイルパス

    Returns:
        パラメータ辞書
    """
    with open(params_path, 'r', encoding='utf-8') as f:
        params = json.load(f)
    logger.info(f"Loaded parameters from: {params_path}")
    return params


def find_latest_params(
    model_type: str,
    task: str,
    params_dir: str = 'data/models'
) -> Optional[str]:
    """
    最新の最適パラメータファイルを検索する。

    Args:
        model_type: モデルタイプ
        task: タスク
        params_dir: 検索ディレクトリ

    Returns:
        最新パラメータファイルパス（見つからない場合はNone）
    """
    if not os.path.exists(params_dir):
        return None

    pattern = f"{model_type}_{task}_best_params_"
    matching_files = [
        f for f in os.listdir(params_dir)
        if f.startswith(pattern) and f.endswith('.json')
    ]

    if not matching_files:
        return None

    # タイムスタンプでソート（最新を取得）
    matching_files.sort(reverse=True)
    return os.path.join(params_dir, matching_files[0])


def get_default_params(model_type: str) -> Dict[str, Any]:
    """
    モデルのデフォルトパラメータを取得する。

    Args:
        model_type: 'xgboost' or 'random_forest'

    Returns:
        デフォルトパラメータ辞書
    """
    if model_type == 'xgboost':
        return {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
    elif model_type == 'random_forest':
        return {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")
