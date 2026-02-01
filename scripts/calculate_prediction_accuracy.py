"""
予想精度集計スクリプト

完了済みレースの予想精度を計算し、PredictionAccuracyテーブルに保存。
日別・週別・月別の集計に対応。

Usage:
    # 昨日分を集計
    python scripts/calculate_prediction_accuracy.py --yesterday

    # 期間指定で集計
    python scripts/calculate_prediction_accuracy.py --start 2026-01-01 --end 2026-01-31

    # モデル指定
    python scripts/calculate_prediction_accuracy.py --start 2026-01-01 --end 2026-01-31 --model xgboost

    # 強制再集計（既存データを上書き）
    python scripts/calculate_prediction_accuracy.py --start 2026-01-01 --end 2026-01-31 --force
"""
import argparse
from datetime import date, timedelta, datetime
from typing import Optional, List
from src.web.app import create_app
from src.data.models import db, PredictionAccuracy, Track
from src.analysis.accuracy_calculator import AccuracyCalculator
from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)


def get_week_start(target_date: date) -> date:
    """その週の月曜日を返す"""
    return target_date - timedelta(days=target_date.weekday())


def get_month_start(target_date: date) -> date:
    """その月の1日を返す"""
    return target_date.replace(day=1)


def aggregate_daily(
    target_date: date,
    model_name: Optional[str] = None,
    force: bool = False
) -> None:
    """
    日別集計を実行

    Args:
        target_date: 集計対象日
        model_name: モデル名（省略時は全モデル）
        force: 既存データを上書きするか
    """
    calc = AccuracyCalculator()

    # モデル別に集計
    models_to_process = [model_name] if model_name else ['xgboost', 'random_forest']

    for model in models_to_process:
        # 既存レコードをチェック
        existing = PredictionAccuracy.query.filter_by(
            aggregation_type='daily',
            aggregation_date=target_date,
            model_name=model,
            track_id=None,
            race_class=None
        ).first()

        if existing and not force:
            logger.info(f"Daily record already exists for {target_date} {model}, skipping")
            continue

        # 精度計算
        metrics = calc.calculate_metrics(
            start_date=target_date,
            end_date=target_date,
            model_name=model
        )

        if metrics['total_predictions'] == 0:
            logger.info(f"No predictions for {target_date} {model}")
            continue

        # 保存または更新
        if existing:
            # 更新
            existing.total_races = metrics['total_races']
            existing.total_predictions = metrics['total_predictions']
            existing.win_predictions = metrics['win_predictions']
            existing.win_hits = metrics['win_hits']
            existing.win_accuracy = metrics['win_accuracy']
            existing.top3_predictions = metrics['top3_predictions']
            existing.top3_hits = metrics['top3_hits']
            existing.top3_accuracy = metrics['top3_accuracy']
            existing.total_bet_amount = metrics['total_bet']
            existing.total_return_amount = metrics['total_return']
            existing.roi = metrics['roi']
            existing.updated_at = datetime.utcnow()

            logger.info(f"Updated daily record for {target_date} {model}")
        else:
            # 新規作成
            accuracy = PredictionAccuracy(
                aggregation_type='daily',
                aggregation_date=target_date,
                model_name=model,
                total_races=metrics['total_races'],
                total_predictions=metrics['total_predictions'],
                win_predictions=metrics['win_predictions'],
                win_hits=metrics['win_hits'],
                win_accuracy=metrics['win_accuracy'],
                top3_predictions=metrics['top3_predictions'],
                top3_hits=metrics['top3_hits'],
                top3_accuracy=metrics['top3_accuracy'],
                total_bet_amount=metrics['total_bet'],
                total_return_amount=metrics['total_return'],
                roi=metrics['roi']
            )
            db.session.add(accuracy)

            logger.info(f"Created daily record for {target_date} {model}")

        db.session.commit()


def aggregate_weekly(
    target_date: date,
    model_name: Optional[str] = None,
    force: bool = False
) -> None:
    """
    週別集計を実行（その週の月曜日が基準日）

    Args:
        target_date: 週に含まれる日付
        model_name: モデル名
        force: 既存データを上書きするか
    """
    calc = AccuracyCalculator()
    week_start = get_week_start(target_date)
    week_end = week_start + timedelta(days=6)

    models_to_process = [model_name] if model_name else ['xgboost', 'random_forest']

    for model in models_to_process:
        existing = PredictionAccuracy.query.filter_by(
            aggregation_type='weekly',
            aggregation_date=week_start,
            model_name=model,
            track_id=None,
            race_class=None
        ).first()

        if existing and not force:
            logger.info(f"Weekly record already exists for {week_start} {model}, skipping")
            continue

        metrics = calc.calculate_metrics(
            start_date=week_start,
            end_date=week_end,
            model_name=model
        )

        if metrics['total_predictions'] == 0:
            logger.info(f"No predictions for week {week_start} {model}")
            continue

        if existing:
            # 更新
            existing.total_races = metrics['total_races']
            existing.total_predictions = metrics['total_predictions']
            existing.win_predictions = metrics['win_predictions']
            existing.win_hits = metrics['win_hits']
            existing.win_accuracy = metrics['win_accuracy']
            existing.top3_predictions = metrics['top3_predictions']
            existing.top3_hits = metrics['top3_hits']
            existing.top3_accuracy = metrics['top3_accuracy']
            existing.total_bet_amount = metrics['total_bet']
            existing.total_return_amount = metrics['total_return']
            existing.roi = metrics['roi']
            existing.updated_at = datetime.utcnow()

            logger.info(f"Updated weekly record for {week_start} {model}")
        else:
            accuracy = PredictionAccuracy(
                aggregation_type='weekly',
                aggregation_date=week_start,
                model_name=model,
                total_races=metrics['total_races'],
                total_predictions=metrics['total_predictions'],
                win_predictions=metrics['win_predictions'],
                win_hits=metrics['win_hits'],
                win_accuracy=metrics['win_accuracy'],
                top3_predictions=metrics['top3_predictions'],
                top3_hits=metrics['top3_hits'],
                top3_accuracy=metrics['top3_accuracy'],
                total_bet_amount=metrics['total_bet'],
                total_return_amount=metrics['total_return'],
                roi=metrics['roi']
            )
            db.session.add(accuracy)

            logger.info(f"Created weekly record for {week_start} {model}")

        db.session.commit()


def aggregate_monthly(
    target_date: date,
    model_name: Optional[str] = None,
    force: bool = False
) -> None:
    """
    月別集計を実行（その月の1日が基準日）

    Args:
        target_date: 月に含まれる日付
        model_name: モデル名
        force: 既存データを上書きするか
    """
    calc = AccuracyCalculator()
    month_start = get_month_start(target_date)

    # 月末を計算
    if month_start.month == 12:
        month_end = month_start.replace(year=month_start.year + 1, month=1, day=1) - timedelta(days=1)
    else:
        month_end = month_start.replace(month=month_start.month + 1, day=1) - timedelta(days=1)

    models_to_process = [model_name] if model_name else ['xgboost', 'random_forest']

    for model in models_to_process:
        existing = PredictionAccuracy.query.filter_by(
            aggregation_type='monthly',
            aggregation_date=month_start,
            model_name=model,
            track_id=None,
            race_class=None
        ).first()

        if existing and not force:
            logger.info(f"Monthly record already exists for {month_start} {model}, skipping")
            continue

        metrics = calc.calculate_metrics(
            start_date=month_start,
            end_date=month_end,
            model_name=model
        )

        if metrics['total_predictions'] == 0:
            logger.info(f"No predictions for month {month_start} {model}")
            continue

        if existing:
            # 更新
            existing.total_races = metrics['total_races']
            existing.total_predictions = metrics['total_predictions']
            existing.win_predictions = metrics['win_predictions']
            existing.win_hits = metrics['win_hits']
            existing.win_accuracy = metrics['win_accuracy']
            existing.top3_predictions = metrics['top3_predictions']
            existing.top3_hits = metrics['top3_hits']
            existing.top3_accuracy = metrics['top3_accuracy']
            existing.total_bet_amount = metrics['total_bet']
            existing.total_return_amount = metrics['total_return']
            existing.roi = metrics['roi']
            existing.updated_at = datetime.utcnow()

            logger.info(f"Updated monthly record for {month_start} {model}")
        else:
            accuracy = PredictionAccuracy(
                aggregation_type='monthly',
                aggregation_date=month_start,
                model_name=model,
                total_races=metrics['total_races'],
                total_predictions=metrics['total_predictions'],
                win_predictions=metrics['win_predictions'],
                win_hits=metrics['win_hits'],
                win_accuracy=metrics['win_accuracy'],
                top3_predictions=metrics['top3_predictions'],
                top3_hits=metrics['top3_hits'],
                top3_accuracy=metrics['top3_accuracy'],
                total_bet_amount=metrics['total_bet'],
                total_return_amount=metrics['total_return'],
                roi=metrics['roi']
            )
            db.session.add(accuracy)

            logger.info(f"Created monthly record for {month_start} {model}")

        db.session.commit()


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description='予想精度集計スクリプト')

    # 日付オプション
    parser.add_argument('--yesterday', action='store_true', help='昨日分を集計')
    parser.add_argument('--start', type=str, help='集計開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='集計終了日 (YYYY-MM-DD)')

    # フィルターオプション
    parser.add_argument('--model', type=str, help='モデル名フィルター (xgboost/random_forest)')

    # 集計タイプ
    parser.add_argument(
        '--aggregation',
        type=str,
        nargs='+',
        default=['daily', 'weekly', 'monthly'],
        choices=['daily', 'weekly', 'monthly'],
        help='集計タイプ（デフォルト: すべて）'
    )

    # その他
    parser.add_argument('--force', action='store_true', help='既存データを上書き')

    args = parser.parse_args()

    # 日付範囲の決定
    if args.yesterday:
        start_date = date.today() - timedelta(days=1)
        end_date = start_date
    elif args.start and args.end:
        start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
        end_date = datetime.strptime(args.end, '%Y-%m-%d').date()
    else:
        parser.error('--yesterday または --start と --end を指定してください')

    # Flaskアプリケーション作成
    app = create_app('development')

    with app.app_context():
        logger.info(f"Starting aggregation for {start_date} to {end_date}")

        # 日別集計
        if 'daily' in args.aggregation:
            current = start_date
            while current <= end_date:
                logger.info(f"Processing daily aggregation for {current}")
                aggregate_daily(current, args.model, args.force)
                current += timedelta(days=1)

        # 週別集計
        if 'weekly' in args.aggregation:
            current = start_date
            processed_weeks = set()
            while current <= end_date:
                week_start = get_week_start(current)
                if week_start not in processed_weeks:
                    logger.info(f"Processing weekly aggregation for week starting {week_start}")
                    aggregate_weekly(current, args.model, args.force)
                    processed_weeks.add(week_start)
                current += timedelta(days=1)

        # 月別集計
        if 'monthly' in args.aggregation:
            current = start_date
            processed_months = set()
            while current <= end_date:
                month_start = get_month_start(current)
                if month_start not in processed_months:
                    logger.info(f"Processing monthly aggregation for {month_start.strftime('%Y-%m')}")
                    aggregate_monthly(current, args.model, args.force)
                    processed_months.add(month_start)
                current += timedelta(days=1)

        logger.info("Aggregation completed successfully")


if __name__ == '__main__':
    main()
