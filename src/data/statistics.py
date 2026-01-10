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
from typing import Optional, Dict, Any
from sqlalchemy import func
from src.data.models import db, Horse, Jockey, Trainer, RaceEntry, RaceResult


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
