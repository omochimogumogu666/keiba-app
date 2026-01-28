#!/usr/bin/env python3
"""
既存の予測データの重複順位を修正するスクリプト

バグにより生成された重複順位を持つ予測データを、
win_probabilityに基づいて正しい順位に再計算します。
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.web.app import create_app
from src.data.models import db, Prediction, Race
import pandas as pd
from collections import defaultdict


def fix_duplicate_predictions():
    """重複した予測順位を修正"""
    app = create_app('development')

    with app.app_context():
        # 予測データが存在するすべてのレースを取得
        races_with_predictions = db.session.query(Race.id).join(Prediction).distinct().all()
        race_ids = [r[0] for r in races_with_predictions]

        print(f"Found {len(race_ids)} races with predictions")

        fixed_races = 0
        fixed_predictions = 0

        for race_id in race_ids:
            # レースの全予測を取得
            predictions = Prediction.query.filter_by(race_id=race_id).all()

            if not predictions:
                continue

            # 重複チェック
            positions = [p.predicted_position for p in predictions if p.predicted_position]
            if len(positions) == len(set(positions)):
                # 重複なし
                continue

            print(f"\nRace {race_id}: Found {len(predictions)} predictions with duplicates")
            print(f"  Original positions: {sorted(positions)}")

            # win_probabilityに基づいて再計算
            pred_data = []
            for p in predictions:
                if p.win_probability is not None:
                    pred_data.append({
                        'id': p.id,
                        'horse_id': p.horse_id,
                        'win_probability': p.win_probability,
                        'old_position': p.predicted_position
                    })

            if not pred_data:
                print(f"  Skipping: No win_probability data")
                continue

            # DataFrameで順位を再計算
            df = pd.DataFrame(pred_data)
            df['new_position'] = df['win_probability'].rank(ascending=False, method='first').astype(int)

            # データベースを更新
            for _, row in df.iterrows():
                prediction = Prediction.query.get(row['id'])
                if prediction:
                    old_pos = prediction.predicted_position
                    new_pos = int(row['new_position'])

                    if old_pos != new_pos:
                        prediction.predicted_position = new_pos
                        fixed_predictions += 1

            db.session.commit()

            new_positions = sorted([int(row['new_position']) for _, row in df.iterrows()])
            print(f"  Fixed positions: {new_positions}")
            print(f"  Unique positions: {len(set(new_positions))}/{len(new_positions)}")

            fixed_races += 1

        print(f"\n{'='*60}")
        print(f"Summary:")
        print(f"  Races processed: {len(race_ids)}")
        print(f"  Races fixed: {fixed_races}")
        print(f"  Predictions updated: {fixed_predictions}")
        print(f"{'='*60}")


if __name__ == '__main__':
    fix_duplicate_predictions()
