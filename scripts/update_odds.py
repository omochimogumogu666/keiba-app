"""
レース開始15分前のオッズ自動更新スクリプト

このスクリプトは以下の機能を提供します:
1. 今日のレースから15分以内に発走するレースを検出
2. 最新オッズをスクレイピング
3. データベースに保存
4. ログ記録

使用方法:
    python scripts/update_odds.py --race-id 202601040511
    python scripts/update_odds.py --all  # 今日の全レース
"""
import os
import sys
import argparse
from datetime import datetime, time as dtime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.web.app import create_app
from src.data.models import db, Race, RaceEntry
from src.scrapers.netkeiba_scraper import NetkeibaScraper
from src.utils.logger import get_app_logger
from config.settings import get_config

logger = get_app_logger(__name__)


def update_race_odds(race_id: str, scraper: NetkeibaScraper, app) -> bool:
    """
    指定されたレースのオッズを更新

    Args:
        race_id: NetkeibaレースID
        scraper: Netkeibaスクレイパーインスタンス
        app: Flaskアプリケーションインスタンス

    Returns:
        成功時True、失敗時False
    """
    try:
        with app.app_context():
            # レースを取得
            race = Race.query.filter_by(netkeiba_race_id=race_id).first()

            if not race:
                logger.warning(f"Race not found: {race_id}")
                return False

            # レース情報をログ出力
            race_time_str = race.post_time.strftime('%H:%M') if race.post_time else '不明'
            logger.info(f"Updating odds for race: {race.race_name} ({race_time_str})")

            # 最新オッズを取得
            odds_data = scraper.scrape_latest_odds(race_id)

            if not odds_data:
                logger.warning(f"No odds data found for race {race_id}")
                return False

            # データベースを更新
            update_count = 0
            current_time = datetime.utcnow()

            for horse_number, latest_odds in odds_data.items():
                # 馬番に対応するエントリーを取得
                entry = RaceEntry.query.filter_by(
                    race_id=race.id,
                    horse_number=horse_number
                ).first()

                if entry:
                    entry.latest_odds = latest_odds
                    entry.odds_updated_at = current_time
                    update_count += 1
                    logger.debug(f"Updated odds for horse #{horse_number}: {latest_odds}")

            db.session.commit()
            logger.info(f"Successfully updated odds for {update_count} horses in race {race_id}")
            return True

    except Exception as e:
        logger.error(f"Error updating odds for race {race_id}: {e}", exc_info=True)
        return False


def get_races_needing_odds_update(app, minutes_before: int = 15) -> list:
    """
    オッズ更新が必要なレースを取得

    Args:
        app: Flaskアプリケーションインスタンス
        minutes_before: レース開始何分前まで対象にするか

    Returns:
        レースオブジェクトのリスト
    """
    with app.app_context():
        # 今日の日付
        today = datetime.now().date()

        # 現在時刻
        now = datetime.now()
        current_time = now.time()

        # 今日のレースで、まだ完了していないものを取得
        upcoming_races = Race.query.filter(
            Race.race_date == today,
            Race.status == 'upcoming',
            Race.post_time.isnot(None)
        ).all()

        races_to_update = []

        for race in upcoming_races:
            # レース開始時刻をdatetimeに変換
            race_datetime = datetime.combine(today, race.post_time)

            # 現在時刻との差分を計算
            time_until_race = race_datetime - now

            # 15分以内に開始するレースを対象
            if timedelta(0) <= time_until_race <= timedelta(minutes=minutes_before):
                races_to_update.append(race)
                logger.info(
                    f"Race needs update: {race.race_name} "
                    f"(starts in {time_until_race.total_seconds() / 60:.1f} minutes)"
                )

        return races_to_update


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description='レース開始15分前のオッズ自動更新'
    )

    parser.add_argument(
        '--race-id',
        type=str,
        help='更新対象のレースID (指定しない場合は自動検出)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='今日の全レースのオッズを更新'
    )

    parser.add_argument(
        '--minutes-before',
        type=int,
        default=15,
        help='レース開始何分前まで対象にするか (デフォルト: 15)'
    )

    parser.add_argument(
        '--scraping-delay',
        type=int,
        default=3,
        help='スクレイピング時のリクエスト間隔(秒) (デフォルト: 3)'
    )

    parser.add_argument(
        '--env',
        type=str,
        choices=['development', 'production', 'testing'],
        default='development',
        help='実行環境 (デフォルト: development)'
    )

    return parser.parse_args()


def main():
    """メイン処理"""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("オッズ更新スクリプト開始")
    logger.info("=" * 80)

    # Flaskアプリケーションを作成
    app = create_app(args.env)

    try:
        with NetkeibaScraper(delay=args.scraping_delay) as scraper:
            if args.race_id:
                # 特定のレースのオッズを更新
                logger.info(f"Updating odds for race: {args.race_id}")
                success = update_race_odds(args.race_id, scraper, app)

                if success:
                    logger.info("Odds update completed successfully")
                else:
                    logger.error("Odds update failed")
                    sys.exit(1)

            elif args.all:
                # 今日の全レースのオッズを更新
                logger.info("Updating odds for all races today")

                with app.app_context():
                    today = datetime.now().date()
                    races = Race.query.filter(
                        Race.race_date == today,
                        Race.status == 'upcoming'
                    ).all()

                    if not races:
                        logger.info("No races found for today")
                        return

                    logger.info(f"Found {len(races)} races for today")

                    success_count = 0
                    for race in races:
                        if update_race_odds(race.netkeiba_race_id, scraper, app):
                            success_count += 1

                    logger.info(f"Updated odds for {success_count}/{len(races)} races")

            else:
                # 15分以内に開始するレースを自動検出して更新
                logger.info(f"Auto-detecting races starting within {args.minutes_before} minutes")

                races_to_update = get_races_needing_odds_update(app, args.minutes_before)

                if not races_to_update:
                    logger.info("No races need odds update at this time")
                    return

                logger.info(f"Found {len(races_to_update)} races needing update")

                success_count = 0
                for race in races_to_update:
                    if update_race_odds(race.netkeiba_race_id, scraper, app):
                        success_count += 1

                logger.info(f"Updated odds for {success_count}/{len(races_to_update)} races")

    except Exception as e:
        logger.error(f"Error in odds update script: {e}", exc_info=True)
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("オッズ更新スクリプト終了")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
