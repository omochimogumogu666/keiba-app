"""
レース予想の定期実行スケジューラー

このスクリプトは以下の機能を提供します:
1. 定期的に今後のレースをスクレイピング
2. 予想を自動生成
3. ログ記録
4. エラーハンドリング

使用方法:
    python scripts/scheduler.py --schedule daily --time 08:00
    python scripts/scheduler.py --schedule weekly --day friday --time 20:00
    python scripts/scheduler.py --once  # 即座に1回実行
"""
import os
import sys
import time
import schedule
import argparse
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.logger import get_app_logger
import subprocess

logger = get_app_logger(__name__)


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description='レース予想の定期実行スケジューラー'
    )

    parser.add_argument(
        '--schedule',
        type=str,
        choices=['daily', 'weekly', 'once'],
        default='daily',
        help='スケジュール頻度 (デフォルト: daily)'
    )

    parser.add_argument(
        '--time',
        type=str,
        default='08:00',
        help='実行時刻 HH:MM形式 (デフォルト: 08:00)'
    )

    parser.add_argument(
        '--day',
        type=str,
        choices=['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'],
        help='週次実行の曜日 (weekly選択時に必須)'
    )

    parser.add_argument(
        '--days-ahead',
        type=int,
        default=1,
        help='何日後までのレースを処理するか (デフォルト: 1)'
    )

    parser.add_argument(
        '--model-type',
        type=str,
        default='xgboost',
        choices=['xgboost', 'random_forest'],
        help='使用するモデルタイプ (デフォルト: xgboost)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/predictions',
        help='予想結果のCSV出力先ディレクトリ (デフォルト: data/predictions)'
    )

    parser.add_argument(
        '--scraping-delay',
        type=int,
        default=3,
        help='スクレイピング時のリクエスト間隔(秒) (デフォルト: 3)'
    )

    return parser.parse_args()


def run_prediction_job(args):
    """
    予想ジョブを実行

    Args:
        args: コマンドライン引数
    """
    logger.info("=" * 80)
    logger.info("予想ジョブを開始します")
    logger.info("=" * 80)

    start_time = datetime.now()

    try:
        # 出力ディレクトリを作成
        os.makedirs(args.output_dir, exist_ok=True)

        # CSVファイル名 (タイムスタンプ付き)
        timestamp = start_time.strftime('%Y%m%d_%H%M%S')
        output_csv = os.path.join(
            args.output_dir,
            f"predictions_{timestamp}.csv"
        )

        # predict_upcoming_races.pyを実行
        script_path = os.path.join(
            os.path.dirname(__file__),
            'predict_upcoming_races.py'
        )

        cmd = [
            sys.executable,
            script_path,
            '--days-ahead', str(args.days_ahead),
            '--model-type', args.model_type,
            '--output-csv', output_csv,
            '--scraping-delay', str(args.scraping_delay)
        ]

        logger.info(f"コマンド実行: {' '.join(cmd)}")

        # サブプロセスとして実行
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        # 標準出力とエラー出力をログに記録
        if result.stdout:
            for line in result.stdout.splitlines():
                logger.info(f"[stdout] {line}")

        if result.stderr:
            for line in result.stderr.splitlines():
                logger.warning(f"[stderr] {line}")

        # リターンコードをチェック
        if result.returncode == 0:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info("=" * 80)
            logger.info(f"予想ジョブが正常に完了しました (実行時間: {duration:.1f}秒)")
            logger.info(f"結果: {output_csv}")
            logger.info("=" * 80)
        else:
            logger.error(f"予想ジョブがエラーコード {result.returncode} で終了しました")

    except Exception as e:
        logger.error(f"予想ジョブ中にエラーが発生しました: {e}", exc_info=True)


def setup_schedule(args):
    """
    スケジュールを設定

    Args:
        args: コマンドライン引数
    """
    if args.schedule == 'once':
        # 即座に1回実行
        logger.info("即座に予想ジョブを実行します")
        run_prediction_job(args)
        return

    # 定期実行の設定
    if args.schedule == 'daily':
        schedule.every().day.at(args.time).do(run_prediction_job, args)
        logger.info(f"毎日 {args.time} に予想ジョブを実行します")

    elif args.schedule == 'weekly':
        if not args.day:
            raise ValueError("週次スケジュールには --day オプションが必要です")

        day_func = getattr(schedule.every(), args.day)
        day_func.at(args.time).do(run_prediction_job, args)
        logger.info(f"毎週 {args.day} の {args.time} に予想ジョブを実行します")

    # スケジューラーを実行
    logger.info("スケジューラーを起動しました (Ctrl+Cで停止)")

    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # 1分ごとにチェック
    except KeyboardInterrupt:
        logger.info("スケジューラーを停止しました")


def main():
    """メイン処理"""
    args = parse_args()

    print("=" * 80)
    print("レース予想スケジューラー")
    print("=" * 80)
    print(f"スケジュール: {args.schedule}")

    if args.schedule == 'daily':
        print(f"実行時刻: 毎日 {args.time}")
    elif args.schedule == 'weekly':
        print(f"実行時刻: 毎週 {args.day} {args.time}")
    elif args.schedule == 'once':
        print("実行モード: 即座に1回実行")

    print(f"対象期間: 今日から{args.days_ahead}日後まで")
    print(f"モデルタイプ: {args.model_type}")
    print(f"出力先: {args.output_dir}")
    print("=" * 80)

    setup_schedule(args)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"スケジューラーでエラーが発生しました: {e}", exc_info=True)
        sys.exit(1)
