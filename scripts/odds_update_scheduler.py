"""
オッズ自動更新スケジューラー

このスクリプトは以下の機能を提供します:
1. 定期的に今日のレースをチェック
2. レース開始15分前に最新オッズを自動取得
3. データベースに保存
4. ログ記録

使用方法:
    python scripts/odds_update_scheduler.py
    python scripts/odds_update_scheduler.py --check-interval 5  # 5分ごとにチェック
"""
import os
import sys
import time
import argparse
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.web.app import create_app
from src.data.models import db, Race, RaceEntry
from src.scrapers.netkeiba_scraper import NetkeibaScraper
from src.utils.logger import get_app_logger
from config.settings import get_config

logger = get_app_logger(__name__)


def update_race_odds(race, scraper: NetkeibaScraper, app) -> bool:
    """
    指定されたレースのオッズを更新

    Args:
        race: Raceオブジェクト
        scraper: Netkeibaスクレイパーインスタンス
        app: Flaskアプリケーションインスタンス

    Returns:
        成功時True、失敗時False
    """
    try:
        with app.app_context():
            race_time_str = race.post_time.strftime('%H:%M') if race.post_time else '不明'
            logger.info(
                f"Updating odds for race: {race.race_name} "
                f"(R{race.race_number}, {race_time_str})"
            )

            # 最新オッズを取得
            odds_data = scraper.scrape_latest_odds(race.netkeiba_race_id)

            if not odds_data:
                logger.warning(f"No odds data found for race {race.netkeiba_race_id}")
                return False

            # データベースを更新
            update_count = 0
            current_time = datetime.utcnow()

            for horse_number, latest_odds in odds_data.items():
                entry = RaceEntry.query.filter_by(
                    race_id=race.id,
                    horse_number=horse_number
                ).first()

                if entry:
                    entry.latest_odds = latest_odds
                    entry.odds_updated_at = current_time
                    update_count += 1

            db.session.commit()
            logger.info(
                f"Successfully updated odds for {update_count} horses "
                f"in race {race.netkeiba_race_id}"
            )
            return True

    except Exception as e:
        logger.error(f"Error updating odds for race {race.netkeiba_race_id}: {e}", exc_info=True)
        db.session.rollback()
        return False


def check_and_update_odds(app, scraper: NetkeibaScraper, minutes_before: int = 15) -> int:
    """
    オッズ更新が必要なレースをチェックして更新

    Args:
        app: Flaskアプリケーションインスタンス
        scraper: Netkeibaスクレイパーインスタンス
        minutes_before: レース開始何分前まで対象にするか

    Returns:
        更新したレース数
    """
    with app.app_context():
        today = datetime.now().date()
        now = datetime.now()

        # 今日のレースで、まだ完了していないものを取得
        upcoming_races = Race.query.filter(
            Race.race_date == today,
            Race.status == 'upcoming',
            Race.post_time.isnot(None)
        ).all()

        if not upcoming_races:
            logger.debug("No upcoming races found for today")
            return 0

        updated_count = 0

        for race in upcoming_races:
            # レース開始時刻をdatetimeに変換
            race_datetime = datetime.combine(today, race.post_time)

            # 現在時刻との差分を計算
            time_until_race = race_datetime - now

            # 15分以内に開始するレースを対象
            # また、オッズが既に更新済みかチェック
            if timedelta(0) <= time_until_race <= timedelta(minutes=minutes_before):
                # 最後にオッズ更新されてから5分以上経過している場合のみ更新
                entries = RaceEntry.query.filter_by(race_id=race.id).first()

                should_update = True
                if entries and entries.odds_updated_at:
                    time_since_update = now - entries.odds_updated_at.replace(tzinfo=None)
                    if time_since_update < timedelta(minutes=5):
                        logger.debug(
                            f"Skipping race {race.race_name}: "
                            f"updated {time_since_update.total_seconds() / 60:.1f} minutes ago"
                        )
                        should_update = False

                if should_update:
                    logger.info(
                        f"Race starting in {time_until_race.total_seconds() / 60:.1f} minutes: "
                        f"{race.race_name}"
                    )
                    if update_race_odds(race, scraper, app):
                        updated_count += 1

        return updated_count


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description='オッズ自動更新スケジューラー'
    )

    parser.add_argument(
        '--check-interval',
        type=int,
        default=3,
        help='チェック間隔（分） (デフォルト: 3)'
    )

    parser.add_argument(
        '--minutes-before',
        type=int,
        default=15,
        help='レース開始何分前から更新対象にするか (デフォルト: 15)'
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

    parser.add_argument(
        '--once',
        action='store_true',
        help='1回だけ実行して終了（テスト用）'
    )

    return parser.parse_args()


def main():
    """メイン処理"""
    args = parse_args()

    print("=" * 80)
    print("オッズ自動更新スケジューラー")
    print("=" * 80)
    print(f"チェック間隔: {args.check_interval}分")
    print(f"更新タイミング: レース開始{args.minutes_before}分前")
    print(f"実行環境: {args.env}")
    print("=" * 80)

    logger.info("オッズ自動更新スケジューラー起動")

    # Flaskアプリケーションを作成
    app = create_app(args.env)

    try:
        with NetkeibaScraper(delay=args.scraping_delay) as scraper:
            if args.once:
                # 1回だけ実行
                logger.info("Running once (test mode)")
                updated = check_and_update_odds(app, scraper, args.minutes_before)
                logger.info(f"Updated {updated} races")
                return

            # 無限ループで定期実行
            logger.info("Starting continuous monitoring (Ctrl+C to stop)")
            print("\nMonitoring races... (Press Ctrl+C to stop)")

            while True:
                try:
                    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    logger.info(f"Checking for races at {current_time}")

                    updated = check_and_update_odds(app, scraper, args.minutes_before)

                    if updated > 0:
                        logger.info(f"Updated odds for {updated} races")
                        print(f"[{current_time}] Updated {updated} races")
                    else:
                        logger.debug("No races needed update")

                    # 次のチェックまで待機
                    wait_seconds = args.check_interval * 60
                    next_check = datetime.now() + timedelta(seconds=wait_seconds)
                    logger.debug(f"Next check at {next_check.strftime('%H:%M:%S')}")

                    time.sleep(wait_seconds)

                except KeyboardInterrupt:
                    logger.info("Received stop signal")
                    raise

                except Exception as e:
                    logger.error(f"Error in check cycle: {e}", exc_info=True)
                    # エラーが発生しても1分後に再試行
                    logger.info("Retrying in 1 minute...")
                    time.sleep(60)

    except KeyboardInterrupt:
        print("\nStopping scheduler...")
        logger.info("Scheduler stopped by user")

    except Exception as e:
        logger.error(f"Fatal error in scheduler: {e}", exc_info=True)
        sys.exit(1)

    logger.info("オッズ自動更新スケジューラー終了")


if __name__ == '__main__':
    main()
