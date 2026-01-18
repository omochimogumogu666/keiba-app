"""
払い戻しデータを修正するスクリプト

既存の完了したレースの払い戻しデータを再スクレイピングして更新します。
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.web.app import create_app
from src.data.models import db, Race, Payout
from src.scrapers.netkeiba_scraper import NetkeibaScraper
from src.data.database import save_payouts_to_db
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fix_payout_data():
    """既存の完了したレースの払い戻しデータを再取得して更新"""

    app = create_app('development')

    with app.app_context():
        # 完了したレースで払い戻しデータがあるものを取得
        races = Race.query.filter_by(status='completed').all()

        logger.info(f"Found {len(races)} completed races")

        updated_count = 0
        skipped_count = 0
        error_count = 0

        with NetkeibaScraper(delay=3) as scraper:
            for i, race in enumerate(races, 1):
                try:
                    logger.info(f"[{i}/{len(races)}] Processing race {race.netkeiba_race_id} - {race.race_name}")

                    # 既存の払い戻しデータがあるか確認
                    existing_payouts = Payout.query.filter_by(race_id=race.id).all()

                    if existing_payouts:
                        logger.info(f"  Found {len(existing_payouts)} existing payouts")

                        # レース結果を再スクレイピング
                        result = scraper.scrape_race_result(race.netkeiba_race_id)

                        if result and 'payouts' in result and result['payouts']:
                            # 既存の払い戻しデータを削除
                            for payout in existing_payouts:
                                db.session.delete(payout)
                            db.session.commit()
                            logger.info(f"  Deleted {len(existing_payouts)} old payouts")

                            # 新しい払い戻しデータを保存
                            new_payouts = save_payouts_to_db(race.id, result['payouts'])
                            logger.info(f"  ✓ Updated with {len(new_payouts)} new payouts")
                            updated_count += 1
                        else:
                            logger.warning(f"  ⚠ No payout data found in scraping result")
                            skipped_count += 1
                    else:
                        logger.info(f"  No existing payouts, skipping")
                        skipped_count += 1

                    # レート制限のためのディレイ
                    if i < len(races):
                        time.sleep(3)

                except Exception as e:
                    logger.error(f"  ✗ Error processing race {race.netkeiba_race_id}: {e}")
                    error_count += 1
                    continue

        logger.info("\n" + "="*60)
        logger.info("Summary:")
        logger.info(f"  Total races: {len(races)}")
        logger.info(f"  Updated: {updated_count}")
        logger.info(f"  Skipped: {skipped_count}")
        logger.info(f"  Errors: {error_count}")
        logger.info("="*60)


if __name__ == '__main__':
    fix_payout_data()
