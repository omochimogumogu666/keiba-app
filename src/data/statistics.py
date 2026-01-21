"""
統計計算モジュール。

このモジュールは馬、騎手、調教師の成績統計を計算します。
DRY原則に基づき、重複していた統計計算ロジックを統一的に処理します。

主な機能:
- 馬の成績統計（勝率、連対率、複勝率）
- 騎手の成績統計
- 調教師の成績統計
- 収支計算（単勝、複勝）
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from sqlalchemy import func, and_, case
from sqlalchemy.orm import Session
from src.data.models import (
    db, Horse, Jockey, Trainer, RaceEntry, RaceResult,
    Race, Prediction, ModelPerformance, Track
)


@dataclass
class EntityStatistics:
    """
    エンティティ（馬、騎手、調教師）の成績統計を表すデータクラス。

    Attributes:
        total_races: 総出走回数
        total_results: 結果が記録されている回数
        wins: 1着回数
        places: 3着以内の回数（複勝）
        win_rate: 勝率（0.0-1.0）
        place_rate: 複勝率（0.0-1.0）
        total_earnings: 総獲得賞金（円）
        avg_finish_position: 平均着順
    """
    total_races: int
    total_results: int
    wins: int
    places: int
    win_rate: float
    place_rate: float
    total_earnings: Optional[int] = None
    avg_finish_position: Optional[float] = None


def calculate_horse_statistics(horse_id: int) -> EntityStatistics:
    """
    馬の成績統計を計算します。

    Args:
        horse_id: 馬のID

    Returns:
        EntityStatistics: 馬の統計情報

    Example:
        >>> stats = calculate_horse_statistics(123)
        >>> print(f"勝率: {stats.win_rate:.1%}")
        勝率: 25.0%
    """
    # 総出走回数
    total_races = RaceEntry.query.filter_by(horse_id=horse_id).count()

    # 結果が記録されている回数
    total_results = db.session.query(RaceResult).join(RaceEntry).filter(
        RaceEntry.horse_id == horse_id
    ).count()

    # 1着回数
    wins = db.session.query(RaceResult).join(RaceEntry).filter(
        RaceEntry.horse_id == horse_id,
        RaceResult.finish_position == 1
    ).count()

    # 3着以内の回数（複勝）
    places = db.session.query(RaceResult).join(RaceEntry).filter(
        RaceEntry.horse_id == horse_id,
        RaceResult.finish_position <= 3
    ).count()

    # 勝率と複勝率の計算
    win_rate = wins / total_results if total_results > 0 else 0.0
    place_rate = places / total_results if total_results > 0 else 0.0

    # 平均着順の計算
    avg_position_result = db.session.query(
        func.avg(RaceResult.finish_position)
    ).join(RaceEntry).filter(
        RaceEntry.horse_id == horse_id,
        RaceResult.finish_position.isnot(None)
    ).scalar()
    avg_finish_position = float(avg_position_result) if avg_position_result else None

    return EntityStatistics(
        total_races=total_races,
        total_results=total_results,
        wins=wins,
        places=places,
        win_rate=win_rate,
        place_rate=place_rate,
        avg_finish_position=avg_finish_position
    )


def calculate_jockey_statistics(jockey_id: int) -> EntityStatistics:
    """
    騎手の成績統計を計算します。

    Args:
        jockey_id: 騎手のID

    Returns:
        EntityStatistics: 騎手の統計情報
    """
    # 総騎乗回数
    total_races = RaceEntry.query.filter_by(jockey_id=jockey_id).count()

    # 結果が記録されている回数
    total_results = db.session.query(RaceResult).join(RaceEntry).filter(
        RaceEntry.jockey_id == jockey_id
    ).count()

    # 1着回数
    wins = db.session.query(RaceResult).join(RaceEntry).filter(
        RaceEntry.jockey_id == jockey_id,
        RaceResult.finish_position == 1
    ).count()

    # 3着以内の回数（複勝）
    places = db.session.query(RaceResult).join(RaceEntry).filter(
        RaceEntry.jockey_id == jockey_id,
        RaceResult.finish_position <= 3
    ).count()

    # 勝率と複勝率の計算
    win_rate = wins / total_results if total_results > 0 else 0.0
    place_rate = places / total_results if total_results > 0 else 0.0

    # 平均着順の計算
    avg_position_result = db.session.query(
        func.avg(RaceResult.finish_position)
    ).join(RaceEntry).filter(
        RaceEntry.jockey_id == jockey_id,
        RaceResult.finish_position.isnot(None)
    ).scalar()
    avg_finish_position = float(avg_position_result) if avg_position_result else None

    return EntityStatistics(
        total_races=total_races,
        total_results=total_results,
        wins=wins,
        places=places,
        win_rate=win_rate,
        place_rate=place_rate,
        avg_finish_position=avg_finish_position
    )


def calculate_trainer_statistics(trainer_id: int) -> EntityStatistics:
    """
    調教師の成績統計を計算します。

    Args:
        trainer_id: 調教師のID

    Returns:
        EntityStatistics: 調教師の統計情報
    """
    # 調教師が管理する馬のリストを取得
    horses_query = db.session.query(RaceEntry).join(Horse).filter(
        Horse.trainer_id == trainer_id
    )

    # 総出走回数
    total_races = horses_query.count()

    # 結果が記録されている回数
    total_results = db.session.query(RaceResult).join(RaceEntry).join(Horse).filter(
        Horse.trainer_id == trainer_id
    ).count()

    # 1着回数
    wins = db.session.query(RaceResult).join(RaceEntry).join(Horse).filter(
        Horse.trainer_id == trainer_id,
        RaceResult.finish_position == 1
    ).count()

    # 3着以内の回数（複勝）
    places = db.session.query(RaceResult).join(RaceEntry).join(Horse).filter(
        Horse.trainer_id == trainer_id,
        RaceResult.finish_position <= 3
    ).count()

    # 勝率と複勝率の計算
    win_rate = wins / total_results if total_results > 0 else 0.0
    place_rate = places / total_results if total_results > 0 else 0.0

    # 平均着順の計算
    avg_position_result = db.session.query(
        func.avg(RaceResult.finish_position)
    ).join(RaceEntry).join(Horse).filter(
        Horse.trainer_id == trainer_id,
        RaceResult.finish_position.isnot(None)
    ).scalar()
    avg_finish_position = float(avg_position_result) if avg_position_result else None

    return EntityStatistics(
        total_races=total_races,
        total_results=total_results,
        wins=wins,
        places=places,
        win_rate=win_rate,
        place_rate=place_rate,
        avg_finish_position=avg_finish_position
    )


def statistics_to_dict(stats: EntityStatistics) -> Dict[str, Any]:
    """
    EntityStatisticsオブジェクトを辞書形式に変換します。

    Args:
        stats: 統計情報

    Returns:
        Dict[str, Any]: 辞書形式の統計情報
    """
    return {
        'total_races': stats.total_races,
        'total_results': stats.total_results,
        'wins': stats.wins,
        'places': stats.places,
        'win_rate': round(stats.win_rate, 3),
        'place_rate': round(stats.place_rate, 3),
        'avg_finish_position': round(stats.avg_finish_position, 2) if stats.avg_finish_position else None
    }


# =============================================================================
# 予測精度・ROI計算関数
# =============================================================================

def calculate_prediction_accuracy(
    model_name: Optional[str] = None,
    days: int = 30,
    session: Optional[Session] = None
) -> Dict[str, Any]:
    """
    予測精度を計算する。

    Args:
        model_name: モデル名（Noneで全モデル）
        days: 過去何日分を対象とするか
        session: DBセッション（Noneでdb.sessionを使用）

    Returns:
        精度メトリクスの辞書
    """
    if session is None:
        session = db.session

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)

    # 基本クエリ: 完了したレースの予測と結果を結合
    query = session.query(
        Prediction.predicted_position,
        RaceResult.finish_position,
        RaceResult.final_odds,
        Prediction.win_probability,
        Prediction.confidence_score,
        Prediction.model_name
    ).join(
        Race, Prediction.race_id == Race.id
    ).join(
        RaceEntry, and_(
            RaceEntry.race_id == Prediction.race_id,
            RaceEntry.horse_id == Prediction.horse_id
        )
    ).join(
        RaceResult, RaceResult.race_entry_id == RaceEntry.id
    ).filter(
        Race.status == 'completed',
        Race.race_date >= start_date,
        Race.race_date <= end_date
    )

    if model_name:
        query = query.filter(Prediction.model_name == model_name)

    results = query.all()

    if not results:
        return {
            'total_predictions': 0,
            'win_accuracy': 0.0,
            'top3_accuracy': 0.0,
            'roi': 0.0,
            'avg_position_error': 0.0
        }

    # 統計計算
    total = len(results)
    win_correct = sum(1 for r in results if r.predicted_position == 1 and r.finish_position == 1)
    top3_correct = sum(1 for r in results if r.predicted_position <= 3 and r.finish_position <= 3)

    # 着順誤差
    position_errors = [
        abs(r.predicted_position - r.finish_position)
        for r in results
        if r.predicted_position and r.finish_position
    ]
    avg_error = sum(position_errors) / len(position_errors) if position_errors else 0

    # ROI計算（1着予測で単勝購入した場合）
    top1_predictions = [r for r in results if r.predicted_position == 1]
    if top1_predictions:
        investment = len(top1_predictions) * 100  # 100円ずつ購入
        returns = sum(
            r.final_odds * 100
            for r in top1_predictions
            if r.finish_position == 1 and r.final_odds
        )
        roi = (returns / investment * 100) if investment > 0 else 0
    else:
        roi = 0

    return {
        'total_predictions': total,
        'win_accuracy': win_correct / total if total > 0 else 0,
        'top3_accuracy': top3_correct / total if total > 0 else 0,
        'roi': roi,
        'avg_position_error': avg_error,
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat()
    }


def calculate_roi_by_model(
    days: int = 30,
    session: Optional[Session] = None
) -> Dict[str, Dict[str, float]]:
    """
    モデル別のROIを計算する。

    Args:
        days: 過去何日分を対象とするか
        session: DBセッション

    Returns:
        モデル名をキーとするROIメトリクス辞書
    """
    if session is None:
        session = db.session

    # モデル別に集計
    model_names = session.query(
        Prediction.model_name
    ).distinct().all()

    results = {}
    for (model_name,) in model_names:
        if model_name:
            results[model_name] = calculate_prediction_accuracy(
                model_name=model_name,
                days=days,
                session=session
            )

    return results


def calculate_daily_performance(
    model_name: Optional[str] = None,
    days: int = 30,
    session: Optional[Session] = None
) -> List[Dict[str, Any]]:
    """
    日別のパフォーマンス推移を計算する。

    Args:
        model_name: モデル名（Noneで全モデル）
        days: 過去何日分を対象とするか
        session: DBセッション

    Returns:
        日別パフォーマンスのリスト
    """
    if session is None:
        session = db.session

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)

    # 日付別の集計クエリ
    query = session.query(
        Race.race_date,
        func.count(Prediction.id).label('total_predictions'),
        func.sum(
            case(
                (and_(Prediction.predicted_position == 1, RaceResult.finish_position == 1), 1),
                else_=0
            )
        ).label('win_correct'),
        func.sum(
            case(
                (and_(Prediction.predicted_position <= 3, RaceResult.finish_position <= 3), 1),
                else_=0
            )
        ).label('top3_correct')
    ).join(
        Race, Prediction.race_id == Race.id
    ).join(
        RaceEntry, and_(
            RaceEntry.race_id == Prediction.race_id,
            RaceEntry.horse_id == Prediction.horse_id
        )
    ).join(
        RaceResult, RaceResult.race_entry_id == RaceEntry.id
    ).filter(
        Race.status == 'completed',
        Race.race_date >= start_date,
        Race.race_date <= end_date
    )

    if model_name:
        query = query.filter(Prediction.model_name == model_name)

    query = query.group_by(Race.race_date).order_by(Race.race_date)

    results = []
    for row in query.all():
        total = row.total_predictions or 0
        win_correct = row.win_correct or 0
        top3_correct = row.top3_correct or 0

        results.append({
            'date': row.race_date.isoformat(),
            'total_predictions': total,
            'win_accuracy': win_correct / total if total > 0 else 0,
            'top3_accuracy': top3_correct / total if total > 0 else 0
        })

    return results


def calculate_track_accuracy(
    model_name: Optional[str] = None,
    days: int = 90,
    session: Optional[Session] = None
) -> Dict[str, Dict[str, float]]:
    """
    競馬場別の予測精度を計算する。

    Args:
        model_name: モデル名（Noneで全モデル）
        days: 過去何日分を対象とするか
        session: DBセッション

    Returns:
        競馬場名をキーとする精度メトリクス辞書
    """
    if session is None:
        session = db.session

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)

    # 競馬場別の集計クエリ
    query = session.query(
        Track.name.label('track_name'),
        func.count(Prediction.id).label('total_predictions'),
        func.sum(
            case(
                (and_(Prediction.predicted_position == 1, RaceResult.finish_position == 1), 1),
                else_=0
            )
        ).label('win_correct'),
        func.sum(
            case(
                (and_(Prediction.predicted_position <= 3, RaceResult.finish_position <= 3), 1),
                else_=0
            )
        ).label('top3_correct')
    ).join(
        Race, Prediction.race_id == Race.id
    ).join(
        Track, Race.track_id == Track.id
    ).join(
        RaceEntry, and_(
            RaceEntry.race_id == Prediction.race_id,
            RaceEntry.horse_id == Prediction.horse_id
        )
    ).join(
        RaceResult, RaceResult.race_entry_id == RaceEntry.id
    ).filter(
        Race.status == 'completed',
        Race.race_date >= start_date,
        Race.race_date <= end_date
    )

    if model_name:
        query = query.filter(Prediction.model_name == model_name)

    query = query.group_by(Track.name)

    results = {}
    for row in query.all():
        total = row.total_predictions or 0
        win_correct = row.win_correct or 0
        top3_correct = row.top3_correct or 0

        results[row.track_name] = {
            'total_predictions': total,
            'win_accuracy': win_correct / total if total > 0 else 0,
            'top3_accuracy': top3_correct / total if total > 0 else 0
        }

    return results


def get_model_comparison_summary(
    days: int = 30,
    session: Optional[Session] = None
) -> List[Dict[str, Any]]:
    """
    モデル比較サマリーを取得する。

    Args:
        days: 過去何日分を対象とするか
        session: DBセッション

    Returns:
        モデル別の比較サマリーリスト
    """
    model_stats = calculate_roi_by_model(days=days, session=session)

    results = []
    for model_name, stats in model_stats.items():
        results.append({
            'model_name': model_name,
            'total_predictions': stats['total_predictions'],
            'win_accuracy': stats['win_accuracy'],
            'top3_accuracy': stats['top3_accuracy'],
            'roi': stats['roi'],
            'avg_position_error': stats['avg_position_error']
        })

    # ROI順でソート
    results.sort(key=lambda x: x['roi'], reverse=True)

    return results


def update_model_performance(
    model_name: str,
    model_version: str,
    session: Optional[Session] = None
) -> int:
    """
    ModelPerformanceテーブルを更新する。

    既存の予測と結果を比較し、パフォーマンス記録を作成/更新。

    Args:
        model_name: モデル名
        model_version: モデルバージョン
        session: DBセッション

    Returns:
        更新したレコード数
    """
    if session is None:
        session = db.session

    # 完了したレースの予測で、まだModelPerformanceに記録されていないもの
    query = session.query(
        Prediction.race_id,
        Prediction.horse_id,
        Prediction.predicted_position,
        Prediction.win_probability,
        Prediction.confidence_score,
        Race.race_date,
        RaceResult.finish_position,
        RaceResult.final_odds
    ).join(
        Race, Prediction.race_id == Race.id
    ).join(
        RaceEntry, and_(
            RaceEntry.race_id == Prediction.race_id,
            RaceEntry.horse_id == Prediction.horse_id
        )
    ).join(
        RaceResult, RaceResult.race_entry_id == RaceEntry.id
    ).filter(
        Race.status == 'completed',
        Prediction.model_name == model_name
    )

    count = 0
    for row in query.all():
        # 既存チェック
        existing = session.query(ModelPerformance).filter(
            ModelPerformance.model_name == model_name,
            ModelPerformance.model_version == model_version,
            ModelPerformance.race_id == row.race_id,
            ModelPerformance.horse_id == row.horse_id
        ).first()

        if existing:
            continue

        # 新規作成
        is_correct = (row.predicted_position == 1 and row.finish_position == 1)
        is_top3_correct = (row.predicted_position <= 3 and row.finish_position <= 3)
        position_error = abs(row.predicted_position - row.finish_position) if row.predicted_position and row.finish_position else None

        # ROI貢献度（1着予測で単勝100円購入した場合）
        if row.predicted_position == 1:
            if row.finish_position == 1 and row.final_odds:
                roi_contribution = row.final_odds * 100 - 100
            else:
                roi_contribution = -100
        else:
            roi_contribution = 0

        perf = ModelPerformance(
            model_name=model_name,
            model_version=model_version,
            race_id=row.race_id,
            horse_id=row.horse_id,
            race_date=row.race_date,
            predicted_position=row.predicted_position,
            actual_position=row.finish_position,
            win_probability=row.win_probability,
            confidence_score=row.confidence_score,
            actual_odds=row.final_odds,
            is_correct=is_correct,
            is_top3_correct=is_top3_correct,
            position_error=position_error,
            roi_contribution=roi_contribution
        )
        session.add(perf)
        count += 1

    if count > 0:
        session.commit()

    return count
