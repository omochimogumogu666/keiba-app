"""
予想精度計算ロジック

予想と結果を突き合わせて的中率・複勝圏内率・ROIを計算。
"""
from datetime import date
from typing import Optional, Dict, Any
from sqlalchemy import and_
from src.data.models import db, Race, Prediction, RaceResult, RaceEntry
from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)


class AccuracyCalculator:
    """
    予想精度を計算するクラス

    完了済みレースの予想と結果を突き合わせて、
    的中率・複勝圏内率・ROIを計算します。
    """

    BET_AMOUNT = 100.0  # 単勝1回あたりの賭け金（円）

    def calculate_metrics(
        self,
        start_date: date,
        end_date: date,
        model_name: Optional[str] = None,
        track_id: Optional[int] = None,
        race_class: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        指定期間・条件で精度指標を計算

        Args:
            start_date: 集計開始日
            end_date: 集計終了日
            model_name: モデル名フィルター（省略時は全モデル）
            track_id: 競馬場IDフィルター（省略時は全競馬場）
            race_class: レースクラスフィルター（省略時は全クラス）

        Returns:
            {
                'total_races': int,
                'total_predictions': int,
                'win_predictions': int,
                'win_hits': int,
                'win_accuracy': float,
                'top3_predictions': int,
                'top3_hits': int,
                'top3_accuracy': float,
                'total_bet': float,
                'total_return': float,
                'roi': float
            }
        """
        # 初期値
        metrics = {
            'total_races': 0,
            'total_predictions': 0,
            'win_predictions': 0,
            'win_hits': 0,
            'win_accuracy': 0.0,
            'top3_predictions': 0,
            'top3_hits': 0,
            'top3_accuracy': 0.0,
            'total_bet': 0.0,
            'total_return': 0.0,
            'roi': 0.0
        }

        # クエリ構築
        # RaceResultはrace_entry経由でrace_idとhorse_idを持つ
        query = db.session.query(
            Prediction,
            RaceResult,
            Race
        ).join(
            RaceEntry,
            and_(
                Prediction.race_id == RaceEntry.race_id,
                Prediction.horse_id == RaceEntry.horse_id
            )
        ).join(
            RaceResult,
            RaceEntry.id == RaceResult.race_entry_id
        ).join(
            Race,
            Prediction.race_id == Race.id
        ).filter(
            Race.status == 'finished',
            Race.race_date.between(start_date, end_date)
        )

        # フィルター適用
        if model_name:
            query = query.filter(Prediction.model_name == model_name)

        if track_id:
            query = query.filter(Race.track_id == track_id)

        if race_class:
            query = query.filter(Race.race_class == race_class)

        results = query.all()

        if not results:
            logger.info(f"No predictions found for period {start_date} to {end_date}")
            return metrics

        # レース数をカウント
        unique_races = set(pred.race_id for pred, _, _ in results)
        metrics['total_races'] = len(unique_races)
        metrics['total_predictions'] = len(results)

        # 各予想を処理
        for prediction, result, race in results:
            # 的中率計算
            if prediction.predicted_position == 1:
                metrics['win_predictions'] += 1
                if result.finish_position == 1:
                    metrics['win_hits'] += 1

            # 複勝圏内率計算
            if prediction.predicted_position <= 3:
                metrics['top3_predictions'] += 1
                if result.finish_position <= 3:
                    metrics['top3_hits'] += 1

            # ROI計算（1着予想馬に単勝100円購入）
            if prediction.predicted_position == 1:
                metrics['total_bet'] += self.BET_AMOUNT
                if result.finish_position == 1:
                    metrics['total_return'] += self.BET_AMOUNT * result.final_odds

        # パーセンテージ計算
        if metrics['win_predictions'] > 0:
            metrics['win_accuracy'] = round(
                metrics['win_hits'] / metrics['win_predictions'] * 100, 2
            )

        if metrics['top3_predictions'] > 0:
            metrics['top3_accuracy'] = round(
                metrics['top3_hits'] / metrics['top3_predictions'] * 100, 2
            )

        if metrics['total_bet'] > 0:
            metrics['roi'] = round(
                (metrics['total_return'] - metrics['total_bet']) / metrics['total_bet'] * 100, 2
            )

        logger.info(
            f"Calculated metrics for {metrics['total_races']} races, "
            f"{metrics['total_predictions']} predictions"
        )

        return metrics
