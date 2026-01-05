"""
今後のレースを自動的にスクレイピングして予想を生成するスクリプト

このスクリプトは以下の処理を自動化します:
1. 指定日のレースカレンダーをスクレイピング
2. 各レースの出馬表(shutuba)をスクレイピング
3. データベースに保存
4. 訓練済みモデルで予想を生成
5. 予想をデータベースとCSVに保存
"""
import os
import sys
from datetime import datetime, date, timedelta
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.web.app import create_app
from src.scrapers.netkeiba_scraper import NetkeibaScraper
from src.data.database import save_race_to_db, save_race_entries_to_db
from src.ml.feature_engineering import FeatureExtractor
from src.ml.preprocessing import FeaturePreprocessor, handle_missing_values
from src.ml.models.xgboost_model import XGBoostRaceModel
from src.ml.models.random_forest import RandomForestRaceModel
from src.data.models import db, Prediction, Race
from src.utils.logger import get_app_logger
from config.settings import get_config

logger = get_app_logger(__name__)


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description='今後のレースをスクレイピングして予想を生成'
    )

    parser.add_argument(
        '--date',
        type=str,
        help='レース日付 (YYYY-MM-DD形式、デフォルト: 今日)'
    )

    parser.add_argument(
        '--days-ahead',
        type=int,
        default=0,
        help='今日から何日後まで処理するか (デフォルト: 0=今日のみ)'
    )

    parser.add_argument(
        '--model-path',
        type=str,
        help='使用する訓練済みモデルのパス (指定しない場合は最新のモデルを使用)'
    )

    parser.add_argument(
        '--model-type',
        type=str,
        default='xgboost',
        choices=['xgboost', 'random_forest'],
        help='モデルタイプ (デフォルト: xgboost)'
    )

    parser.add_argument(
        '--output-csv',
        type=str,
        help='予想結果のCSV出力先 (オプション)'
    )

    parser.add_argument(
        '--skip-scraping',
        action='store_true',
        help='スクレイピングをスキップ (既にDBにデータがある場合)'
    )

    parser.add_argument(
        '--scraping-delay',
        type=int,
        default=3,
        help='スクレイピング時のリクエスト間隔(秒) (デフォルト: 3)'
    )

    return parser.parse_args()


def get_latest_model_path(model_type='xgboost', models_dir='data/models'):
    """
    最新の訓練済みモデルを取得

    Args:
        model_type: モデルタイプ
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
    latest_model = os.path.join(models_dir, model_files[0])

    logger.info(f"最新モデルを使用: {latest_model}")
    return latest_model


def scrape_upcoming_races(target_date, delay=3):
    """
    指定日のレース情報をスクレイピング

    Args:
        target_date: レース日付 (date型)
        delay: リクエスト間隔(秒)

    Returns:
        スクレイピングしたレースIDのリスト
    """
    logger.info(f"レースカレンダーをスクレイピング: {target_date}")

    race_ids = []

    with NetkeibaScraper(delay=delay) as scraper:
        # レースカレンダーを取得
        races = scraper.scrape_race_calendar(target_date)

        if not races:
            logger.warning(f"{target_date}のレースが見つかりませんでした")
            return race_ids

        logger.info(f"{len(races)}件のレースを検出")

        # 各レースの出馬表をスクレイピング
        for i, race_info in enumerate(races, 1):
            try:
                race_id = race_info['jra_race_id']
                logger.info(f"[{i}/{len(races)}] レース {race_id} の出馬表を取得中...")

                # 出馬表をスクレイピング
                race_card = scraper.scrape_race_card(race_id)

                if not race_card:
                    logger.warning(f"レース {race_id} の出馬表が取得できませんでした")
                    continue

                # データベースに保存
                race_data = {
                    'jra_race_id': race_id,
                    'race_date': target_date,
                    'track_name': race_info.get('track_name', '不明'),
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

                # レースを保存
                db_race = save_race_to_db(race_data)

                # 出走馬を保存
                if 'entries' in race_card and race_card['entries']:
                    save_race_entries_to_db(db_race.id, race_card['entries'])
                    logger.info(f"レース {race_id} を保存完了 ({len(race_card['entries'])}頭)")
                    race_ids.append(db_race.id)
                else:
                    logger.warning(f"レース {race_id} に出走馬データがありません")

            except Exception as e:
                logger.error(f"レース {race_id} の処理中にエラー: {e}", exc_info=True)
                continue

    return race_ids


def load_model(model_path, model_type='xgboost'):
    """
    訓練済みモデルを読み込み

    Args:
        model_path: モデルファイルのパス
        model_type: モデルタイプ

    Returns:
        読み込んだモデル
    """
    logger.info(f"モデルを読み込み中: {model_path}")

    if model_type == 'xgboost':
        model = XGBoostRaceModel()
    else:
        model = RandomForestRaceModel()

    model.load(model_path)
    logger.info(f"モデル読み込み完了: {model.model_name} v{model.version}")

    return model


def generate_predictions(race_ids, model):
    """
    レースの予想を生成

    Args:
        race_ids: レースIDのリスト
        model: 訓練済みモデル

    Returns:
        予想結果のDataFrame
    """
    logger.info(f"{len(race_ids)}件のレースの予想を生成中...")

    all_predictions = []
    extractor = FeatureExtractor(db.session)

    for i, race_id in enumerate(race_ids, 1):
        try:
            logger.info(f"[{i}/{len(race_ids)}] レース {race_id} の予想を生成中...")

            # 特徴量を抽出
            X = extractor.extract_features_for_race(race_id)

            if X.empty:
                logger.warning(f"レース {race_id} の特徴量が取得できませんでした")
                continue

            # 欠損値を処理
            X_filled = handle_missing_values(X, strategy='median')

            # 予想を生成
            if model.task == 'regression':
                # 着順を予測
                predictions = model.predict(X_filled)
                X_filled['predicted_position'] = predictions

                # 勝率を計算 (予想着順が低いほど高い)
                max_position = X_filled['predicted_position'].max()
                X_filled['win_probability'] = 1 - (X_filled['predicted_position'] - 1) / max_position
                X_filled['confidence_score'] = X_filled['win_probability']
            else:
                # 分類: 勝率を予測
                probas = model.predict_proba(X_filled)

                if probas.shape[1] == 2:
                    win_proba = probas[:, 1]
                else:
                    win_proba = probas[:, -1]

                X_filled['win_probability'] = win_proba
                X_filled['confidence_score'] = win_proba

                # 勝率に基づいて予想着順を計算
                X_filled['predicted_position'] = X_filled['win_probability'].rank(
                    ascending=False, method='first'
                ).astype(int)

            predictions_df = X_filled[[
                'race_id', 'horse_id', 'predicted_position',
                'win_probability', 'confidence_score'
            ]].copy()

            all_predictions.append(predictions_df)

            # トップ3を表示
            top_3 = predictions_df.nsmallest(3, 'predicted_position')
            print(f"\nレース {race_id} - トップ3予想:")
            for _, row in top_3.iterrows():
                print(f"  {int(row['predicted_position'])}位: 馬ID {int(row['horse_id'])} "
                      f"(勝率: {row['win_probability']:.1%}, "
                      f"信頼度: {row['confidence_score']:.1%})")

        except Exception as e:
            logger.error(f"レース {race_id} の予想生成中にエラー: {e}", exc_info=True)
            continue

    if not all_predictions:
        logger.warning("予想が生成できませんでした")
        return None

    import pandas as pd
    combined = pd.concat(all_predictions, ignore_index=True)
    logger.info(f"{len(combined)}件の予想を生成完了")

    return combined


def save_predictions(predictions_df, model_name, model_version):
    """
    予想をデータベースに保存

    Args:
        predictions_df: 予想結果のDataFrame
        model_name: モデル名
        model_version: モデルバージョン
    """
    logger.info(f"{len(predictions_df)}件の予想をデータベースに保存中...")

    for _, row in predictions_df.iterrows():
        # 既存の予想をチェック
        existing = db.session.query(Prediction).filter_by(
            race_id=int(row['race_id']),
            horse_id=int(row['horse_id']),
            model_name=model_name,
            model_version=model_version
        ).first()

        if existing:
            # 既存の予想を更新
            existing.predicted_position = int(row['predicted_position'])
            existing.win_probability = float(row['win_probability'])
            existing.confidence_score = float(row['confidence_score'])
        else:
            # 新規予想を作成
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
    logger.info("予想の保存が完了しました")


def main():
    """メイン処理"""
    args = parse_args()

    print("=" * 80)
    print("今後のレース予想システム")
    print("=" * 80)

    # 対象日付を決定
    if args.date:
        target_date = datetime.strptime(args.date, '%Y-%m-%d').date()
    else:
        target_date = date.today()

    # 処理する日付のリスト
    dates_to_process = [
        target_date + timedelta(days=i)
        for i in range(args.days_ahead + 1)
    ]

    print(f"処理対象日: {', '.join(str(d) for d in dates_to_process)}")
    print("=" * 80)

    # Flaskアプリを作成
    app = create_app()

    all_race_ids = []

    # スクレイピング
    if not args.skip_scraping:
        with app.app_context():
            for target_date in dates_to_process:
                print(f"\n{target_date}のレースをスクレイピング中...")
                race_ids = scrape_upcoming_races(
                    target_date,
                    delay=args.scraping_delay
                )
                all_race_ids.extend(race_ids)
                print(f"{target_date}: {len(race_ids)}件のレースを取得")
    else:
        # データベースから既存のレースを取得
        with app.app_context():
            for target_date in dates_to_process:
                races = db.session.query(Race).filter(
                    Race.race_date == target_date,
                    Race.status == 'upcoming'
                ).all()
                race_ids = [r.id for r in races]
                all_race_ids.extend(race_ids)
                print(f"{target_date}: {len(race_ids)}件のレースを発見")

    if not all_race_ids:
        print("\n予想するレースが見つかりませんでした")
        return

    print(f"\n合計 {len(all_race_ids)}件のレースを処理します")
    print("=" * 80)

    # モデルを読み込み
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = get_latest_model_path(args.model_type)

    model = load_model(model_path, args.model_type)

    print(f"\nモデル情報:")
    print(f"  名前: {model.model_name}")
    print(f"  バージョン: {model.version}")
    print(f"  タスク: {model.task}")
    print(f"  特徴量数: {len(model.feature_columns)}")
    print("=" * 80)

    # 予想を生成
    with app.app_context():
        predictions_df = generate_predictions(all_race_ids, model)

        if predictions_df is None or predictions_df.empty:
            print("\n予想が生成できませんでした")
            return

        # データベースに保存
        save_predictions(predictions_df, model.model_name, model.version)
        print(f"\n{len(predictions_df)}件の予想をデータベースに保存しました")

        # CSVに保存 (オプション)
        if args.output_csv:
            predictions_df.to_csv(
                args.output_csv,
                index=False,
                encoding='utf-8-sig'
            )
            print(f"予想をCSVに保存しました: {args.output_csv}")

    print("\n" + "=" * 80)
    print("処理が完了しました!")
    print("=" * 80)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}", exc_info=True)
        sys.exit(1)
