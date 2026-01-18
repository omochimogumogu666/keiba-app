"""
予測タスクマネージャー - スレッドセーフな予測実行と進捗追跡

このモジュールは予測タスクをバックグラウンドスレッドで実行し、
SSEを通じてリアルタイムで進捗状況を配信します。
"""
import os
import threading
import uuid
from datetime import datetime, date, timedelta
from typing import Callable, Dict, Optional, Any
import pandas as pd
from flask import Flask
from src.data.models import db, Race, Prediction, RaceEntry
from src.data.database import save_race_to_db, save_race_entries_to_db
from src.scrapers.netkeiba_scraper import NetkeibaScraper
from src.ml.feature_engineering import FeatureExtractor
from src.ml.preprocessing import handle_missing_values
from src.ml.models.xgboost_model import XGBoostRaceModel
from src.ml.models.random_forest import RandomForestRaceModel
from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)


def get_latest_model_path(model_type='xgboost', models_dir='data/models'):
    """
    最新の訓練済みモデルを取得

    Args:
        model_type: モデルタイプ ('xgboost' or 'random_forest')
        models_dir: モデルディレクトリ

    Returns:
        モデルファイルのパス
    """
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"モデルディレクトリが見つかりません: {models_dir}")

    # モデルファイル一覧を取得
    model_files = [
        f for f in os.listdir(models_dir)
        if f.startswith(model_type.lower()) and f.endswith('.pkl')
    ]

    if not model_files:
        raise FileNotFoundError(f"{model_type}モデルが見つかりません")

    # 最新のファイルを取得 (タイムスタンプでソート)
    model_files.sort(reverse=True)
    return os.path.join(models_dir, model_files[0])


def load_model(model_path: str, model_type: str = 'xgboost'):
    """
    訓練済みモデルを読み込み

    Args:
        model_path: モデルファイルのパス
        model_type: モデルタイプ

    Returns:
        読み込んだモデル
    """
    if model_type == 'xgboost':
        model = XGBoostRaceModel()
    else:
        model = RandomForestRaceModel()

    model.load(model_path)
    return model


class PredictionTaskManager:
    """
    スレッドセーフな予測タスクマネージャー

    タスクのライフサイクル（開始、進捗追跡、キャンセル）を管理します。
    """

    MAX_CONCURRENT_TASKS = 1
    TASK_TTL_HOURS = 24

    def __init__(self):
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._threads: Dict[str, threading.Thread] = {}

    def start_task(
        self,
        app: Flask,
        target_date: date,
        model_type: str = 'xgboost',
        skip_scraping: bool = False,
        scraper_delay: int = 3
    ) -> tuple[bool, str, Optional[str]]:
        """
        予測タスクを開始

        Args:
            app: Flaskアプリケーションインスタンス
            target_date: 予測対象日
            model_type: 使用するモデルタイプ
            skip_scraping: Trueの場合、スクレイピングをスキップ
            scraper_delay: スクレイパーのリクエスト間隔（秒）

        Returns:
            (成功フラグ, タスクIDまたはエラーメッセージ, エラータイプ)
        """
        with self._lock:
            # 実行中タスクをチェック
            running_tasks = [
                tid for tid, task in self._tasks.items()
                if task['status'] == 'running'
            ]

            if len(running_tasks) >= self.MAX_CONCURRENT_TASKS:
                return False, "既に予測タスクが実行中です", "concurrent_limit"

            task_id = str(uuid.uuid4())

            self._tasks[task_id] = {
                'task_id': task_id,
                'status': 'running',
                'started_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'params': {
                    'target_date': target_date.isoformat(),
                    'model_type': model_type,
                    'skip_scraping': skip_scraping,
                },
                'progress': {
                    'percent_complete': 0,
                    'phase': 'initializing',
                    'phase_text': '初期化中...',
                    'races_found': 0,
                    'races_processed': 0,
                    'predictions_generated': 0,
                    'current_race': None,
                },
                'error': None,
                'result': None,
                'cancelled': False,
            }

        # ワーカースレッドを開始
        thread = threading.Thread(
            target=self._prediction_worker,
            args=(app, task_id, target_date, model_type, skip_scraping, scraper_delay),
            daemon=True
        )
        self._threads[task_id] = thread
        thread.start()

        logger.info(f"Started prediction task {task_id} for {target_date}")
        return True, task_id, None

    def _prediction_worker(
        self,
        app: Flask,
        task_id: str,
        target_date: date,
        model_type: str,
        skip_scraping: bool,
        scraper_delay: int
    ):
        """予測ワーカー関数"""
        try:
            with app.app_context():
                self._run_prediction(
                    task_id, target_date, model_type, skip_scraping, scraper_delay
                )

                with self._lock:
                    if task_id in self._tasks:
                        if self._tasks[task_id].get('cancelled'):
                            self._tasks[task_id]['status'] = 'cancelled'
                        else:
                            self._tasks[task_id]['status'] = 'completed'
                            self._tasks[task_id]['progress']['phase'] = 'completed'
                            self._tasks[task_id]['progress']['phase_text'] = '完了'
                            self._tasks[task_id]['progress']['percent_complete'] = 100
                        self._tasks[task_id]['updated_at'] = datetime.now().isoformat()

                logger.info(f"Prediction task {task_id} completed")

        except Exception as e:
            logger.exception(f"Prediction task {task_id} failed")
            with self._lock:
                if task_id in self._tasks:
                    self._tasks[task_id]['status'] = 'failed'
                    self._tasks[task_id]['error'] = str(e)
                    self._tasks[task_id]['progress']['phase'] = 'error'
                    self._tasks[task_id]['progress']['phase_text'] = f'エラー: {str(e)}'
                    self._tasks[task_id]['updated_at'] = datetime.now().isoformat()

    def _run_prediction(
        self,
        task_id: str,
        target_date: date,
        model_type: str,
        skip_scraping: bool,
        scraper_delay: int
    ):
        """予測処理を実行"""
        race_ids = []

        # Phase 1: スクレイピング（必要な場合）
        if not skip_scraping:
            self._update_progress(task_id, {
                'phase': 'scraping',
                'phase_text': 'レース情報を取得中...',
                'percent_complete': 5,
            })

            race_ids = self._scrape_races(task_id, target_date, scraper_delay)
        else:
            # 既存のレースをデータベースから取得
            self._update_progress(task_id, {
                'phase': 'loading',
                'phase_text': 'データベースからレースを読み込み中...',
                'percent_complete': 10,
            })

            races = db.session.query(Race).filter(
                Race.race_date == target_date,
                Race.status == 'upcoming'
            ).all()
            race_ids = [r.id for r in races]

            self._update_progress(task_id, {
                'races_found': len(race_ids),
            })

        if not race_ids:
            self._update_progress(task_id, {
                'phase': 'no_races',
                'phase_text': 'レースが見つかりませんでした',
                'percent_complete': 100,
            })
            return

        # キャンセルチェック
        if self._is_cancelled(task_id):
            return

        # Phase 2: モデルを読み込み
        self._update_progress(task_id, {
            'phase': 'loading_model',
            'phase_text': f'{model_type}モデルを読み込み中...',
            'percent_complete': 30,
        })

        model_path = get_latest_model_path(model_type)
        model = load_model(model_path, model_type)

        logger.info(f"Loaded model: {model.model_name} v{model.version}")

        # Phase 3: 予測を生成
        self._update_progress(task_id, {
            'phase': 'predicting',
            'phase_text': '予測を生成中...',
            'percent_complete': 40,
        })

        predictions_df = self._generate_predictions(task_id, race_ids, model)

        if predictions_df is None or predictions_df.empty:
            self._update_progress(task_id, {
                'phase': 'no_predictions',
                'phase_text': '予測を生成できませんでした',
                'percent_complete': 100,
            })
            return

        # Phase 4: 予測をデータベースに保存
        self._update_progress(task_id, {
            'phase': 'saving',
            'phase_text': '予測をデータベースに保存中...',
            'percent_complete': 90,
        })

        self._save_predictions(predictions_df, model.model_name, model.version)

        # 結果を記録
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id]['result'] = {
                    'races_processed': len(race_ids),
                    'predictions_generated': len(predictions_df),
                    'model_name': model.model_name,
                    'model_version': model.version,
                    'target_date': target_date.isoformat(),
                }

    def _scrape_races(self, task_id: str, target_date: date, delay: int) -> list:
        """レース情報をスクレイピング"""
        race_ids = []

        with NetkeibaScraper(delay=delay) as scraper:
            # レースカレンダーを取得
            races = scraper.scrape_race_calendar(target_date)

            if not races:
                logger.warning(f"{target_date}のレースが見つかりませんでした")
                return race_ids

            self._update_progress(task_id, {
                'races_found': len(races),
                'phase_text': f'{len(races)}件のレースを発見',
            })

            # 各レースの出馬表をスクレイピング
            for i, race_info in enumerate(races, 1):
                if self._is_cancelled(task_id):
                    break

                try:
                    race_id = race_info['netkeiba_race_id']
                    track = race_info.get('track', '不明')

                    self._update_progress(task_id, {
                        'current_race': f"{track} {race_info.get('race_number', '')}R",
                        'phase_text': f'出馬表を取得中 ({i}/{len(races)})',
                        'percent_complete': 5 + int(25 * i / len(races)),
                    })

                    race_card = scraper.scrape_race_card(race_id)

                    if not race_card:
                        logger.warning(f"レース {race_id} の出馬表が取得できませんでした")
                        continue

                    # データベースに保存
                    race_data = {
                        'netkeiba_race_id': race_id,
                        'race_date': target_date,
                        'track': track,
                        'race_number': race_info.get('race_number', 1),
                        'race_name': race_card.get('race_name', ''),
                        'distance': race_card.get('distance', 0),
                        'surface': race_card.get('surface', 'turf'),
                        'track_condition': race_card.get('track_condition', '良'),
                        'weather': race_card.get('weather', '晴'),
                        'race_class': race_card.get('race_class', ''),
                        'prize_money': race_card.get('prize_money', 0),
                        'status': 'upcoming'
                    }

                    db_race = save_race_to_db(race_data)

                    # 出走馬を保存
                    if 'entries' in race_card and race_card['entries']:
                        valid_entries = [
                            entry for entry in race_card['entries']
                            if all(key in entry for key in [
                                'netkeiba_horse_id', 'netkeiba_jockey_id', 'netkeiba_trainer_id'
                            ])
                        ]

                        if valid_entries:
                            save_race_entries_to_db(db_race.id, valid_entries)
                            race_ids.append(db_race.id)
                            logger.info(f"レース {race_id} を保存完了 ({len(valid_entries)}頭)")

                except Exception as e:
                    logger.error(f"レース処理中にエラー: {e}")
                    db.session.rollback()
                    continue

        return race_ids

    def _generate_predictions(
        self,
        task_id: str,
        race_ids: list,
        model
    ) -> Optional[pd.DataFrame]:
        """予測を生成"""
        all_predictions = []
        extractor = FeatureExtractor(db.session)

        for i, race_id in enumerate(race_ids, 1):
            if self._is_cancelled(task_id):
                break

            try:
                race = Race.query.get(race_id)
                track_name = race.track.name if race.track else '不明'

                self._update_progress(task_id, {
                    'races_processed': i,
                    'current_race': f"{track_name} {race.race_number}R",
                    'phase_text': f'予測を生成中 ({i}/{len(race_ids)})',
                    'percent_complete': 40 + int(50 * i / len(race_ids)),
                })

                # 特徴量を抽出
                X = extractor.extract_features_for_race(race_id)

                if X.empty:
                    logger.warning(f"レース {race_id} の特徴量が取得できませんでした")
                    continue

                # 欠損値を処理
                X_filled = handle_missing_values(X, strategy='median')

                # 数値型変換
                id_columns = [col for col in X_filled.columns if col.endswith('_id') or col == 'race_date']
                feature_columns = [col for col in X_filled.columns if col not in id_columns]

                for col in feature_columns:
                    if X_filled[col].dtype == 'object':
                        X_filled[col] = pd.to_numeric(X_filled[col], errors='coerce').fillna(0.0)

                # 予測を生成
                if model.task == 'regression':
                    predictions = model.predict(X_filled)
                    X_filled['predicted_position'] = predictions

                    max_position = X_filled['predicted_position'].max()
                    X_filled['win_probability'] = 1 - (X_filled['predicted_position'] - 1) / max_position
                    X_filled['confidence_score'] = X_filled['win_probability']
                else:
                    probas = model.predict_proba(X_filled)
                    win_proba = probas[:, 1] if probas.shape[1] == 2 else probas[:, -1]

                    X_filled['win_probability'] = win_proba
                    X_filled['confidence_score'] = win_proba
                    X_filled['predicted_position'] = X_filled['win_probability'].rank(
                        ascending=False, method='first'
                    ).astype(int)

                predictions_df = X_filled[[
                    'race_id', 'horse_id', 'predicted_position',
                    'win_probability', 'confidence_score'
                ]].copy()

                all_predictions.append(predictions_df)

                self._update_progress(task_id, {
                    'predictions_generated': sum(len(p) for p in all_predictions),
                })

            except Exception as e:
                logger.error(f"レース {race_id} の予測生成中にエラー: {e}")
                continue

        if not all_predictions:
            return None

        return pd.concat(all_predictions, ignore_index=True)

    def _save_predictions(self, predictions_df: pd.DataFrame, model_name: str, model_version: str):
        """予測をデータベースに保存"""
        for _, row in predictions_df.iterrows():
            existing = db.session.query(Prediction).filter_by(
                race_id=int(row['race_id']),
                horse_id=int(row['horse_id']),
                model_name=model_name,
                model_version=model_version
            ).first()

            if existing:
                existing.predicted_position = int(row['predicted_position'])
                existing.win_probability = float(row['win_probability'])
                existing.confidence_score = float(row['confidence_score'])
            else:
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
        logger.info(f"{len(predictions_df)}件の予測を保存完了")

    def _update_progress(self, task_id: str, data: Dict[str, Any]):
        """進捗を更新"""
        with self._lock:
            if task_id not in self._tasks:
                return

            task = self._tasks[task_id]
            task['updated_at'] = datetime.now().isoformat()

            for key, value in data.items():
                if key in task['progress']:
                    task['progress'][key] = value

    def _is_cancelled(self, task_id: str) -> bool:
        """タスクがキャンセルされたかチェック"""
        with self._lock:
            task = self._tasks.get(task_id)
            return task.get('cancelled', False) if task else True

    def get_progress(self, task_id: str) -> Optional[Dict[str, Any]]:
        """タスクの現在の進捗を取得"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                return task.copy()
            return None

    def cancel_task(self, task_id: str) -> bool:
        """実行中のタスクをキャンセル"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task['status'] == 'running':
                task['cancelled'] = True
                logger.info(f"Cancellation requested for prediction task {task_id}")
                return True
            return False

    def get_running_task(self) -> Optional[Dict[str, Any]]:
        """実行中のタスクを取得"""
        with self._lock:
            for task in self._tasks.values():
                if task['status'] == 'running':
                    return task.copy()
            return None

    def get_recent_tasks(self, limit: int = 10) -> list[Dict[str, Any]]:
        """最近のタスクを取得"""
        with self._lock:
            tasks = list(self._tasks.values())
            tasks.sort(key=lambda x: x['started_at'], reverse=True)
            return [t.copy() for t in tasks[:limit]]


# グローバルシングルトンインスタンス
prediction_task_manager = PredictionTaskManager()
