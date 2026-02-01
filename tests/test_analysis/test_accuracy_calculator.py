"""
精度計算ロジックのテスト
"""
import pytest
from datetime import date
from src.analysis.accuracy_calculator import AccuracyCalculator
from src.data.models import db, Race, Track, Horse, Jockey, Trainer, RaceEntry, RaceResult, Prediction
from src.web.app import create_app


@pytest.fixture
def app():
    """テスト用Flaskアプリ"""
    app = create_app('testing')
    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()


@pytest.fixture
def sample_data(app):
    """テスト用データ"""
    with app.app_context():
        # Track
        track = Track(name='東京', location='東京都')
        db.session.add(track)

        # Race
        race = Race(
            netkeiba_race_id='202601050101',
            track_id=1,
            race_date=date(2026, 1, 5),
            race_number=1,
            race_name='新春賞',
            distance=1600,
            surface='turf',
            race_class='3勝',
            status='finished'
        )
        db.session.add(race)

        # Horses
        horses = []
        for i in range(1, 6):
            horse = Horse(
                netkeiba_horse_id=f'2020100{i}',
                name=f'テスト馬{i}'
            )
            horses.append(horse)
            db.session.add(horse)

        # Jockey & Trainer (最小限)
        jockey = Jockey(netkeiba_jockey_id='00001', name='テスト騎手')
        trainer = Trainer(netkeiba_trainer_id='00001', name='テスト調教師')
        db.session.add_all([jockey, trainer])

        db.session.commit()

        # RaceEntries
        entries = []
        for i, horse in enumerate(horses, 1):
            entry = RaceEntry(
                race_id=race.id,
                horse_id=horse.id,
                jockey_id=jockey.id,
                post_position=i,
                horse_number=i
            )
            entries.append(entry)
            db.session.add(entry)

        db.session.commit()

        # Results (1着: 馬3, 2着: 馬1, 3着: 馬5)
        results_data = [
            (entries[2].id, 1, 5.2),   # 馬3が1着
            (entries[0].id, 2, 3.1),   # 馬1が2着
            (entries[4].id, 3, 12.5),  # 馬5が3着
            (entries[1].id, 4, 8.0),   # 馬2が4着
            (entries[3].id, 5, 20.0),  # 馬4が5着
        ]

        for entry_id, position, odds in results_data:
            result = RaceResult(
                race_entry_id=entry_id,
                finish_position=position,
                final_odds=odds
            )
            db.session.add(result)

        # Predictions (モデル予想: 1位=馬1, 2位=馬3, 3位=馬2, 4位=馬5, 5位=馬4)
        predictions_data = [
            (horses[0].id, 1, 0.30),  # 馬1を1位予想（実際は2着）
            (horses[2].id, 2, 0.25),  # 馬3を2位予想（実際は1着）
            (horses[1].id, 3, 0.20),  # 馬2を3位予想（実際は4着）
            (horses[4].id, 4, 0.15),  # 馬5を4位予想（実際は3着）
            (horses[3].id, 5, 0.10),  # 馬4を5位予想（実際は5着）
        ]

        for horse_id, pred_pos, prob in predictions_data:
            pred = Prediction(
                race_id=race.id,
                horse_id=horse_id,
                predicted_position=pred_pos,
                win_probability=prob,
                confidence_score=prob,
                model_name='xgboost',
                model_version='1.0'
            )
            db.session.add(pred)

        db.session.commit()

        return {
            'race': race,
            'horses': horses,
            'track': track
        }


def test_calculate_win_accuracy(app, sample_data):
    """的中率が正しく計算されるかテスト"""
    with app.app_context():
        calc = AccuracyCalculator()

        metrics = calc.calculate_metrics(
            start_date=date(2026, 1, 5),
            end_date=date(2026, 1, 5),
            model_name='xgboost'
        )

        # 1着予想: 馬1 → 実際は2着 → 外れ
        # win_accuracy = 0 / 1 * 100 = 0%
        assert metrics['win_predictions'] == 1
        assert metrics['win_hits'] == 0
        assert metrics['win_accuracy'] == 0.0


def test_calculate_top3_accuracy(app, sample_data):
    """複勝圏内率が正しく計算されるかテスト"""
    with app.app_context():
        calc = AccuracyCalculator()

        metrics = calc.calculate_metrics(
            start_date=date(2026, 1, 5),
            end_date=date(2026, 1, 5),
            model_name='xgboost'
        )

        # 予想上位3頭: 馬1(2着), 馬3(1着), 馬2(4着)
        # 3着以内: 馬1, 馬3 = 2頭
        # top3_accuracy = 2 / 3 * 100 = 66.67%
        assert metrics['top3_predictions'] == 3
        assert metrics['top3_hits'] == 2
        assert abs(metrics['top3_accuracy'] - 66.67) < 0.1


def test_calculate_roi(app, sample_data):
    """ROIが正しく計算されるかテスト"""
    with app.app_context():
        calc = AccuracyCalculator()

        metrics = calc.calculate_metrics(
            start_date=date(2026, 1, 5),
            end_date=date(2026, 1, 5),
            model_name='xgboost'
        )

        # 1着予想: 馬1に100円 → 実際は2着 → 0円払戻
        # ROI = (0 - 100) / 100 * 100 = -100%
        assert metrics['total_bet'] == 100.0
        assert metrics['total_return'] == 0.0
        assert metrics['roi'] == -100.0


def test_zero_predictions_safety(app):
    """予想が0件でもエラーにならないかテスト"""
    with app.app_context():
        calc = AccuracyCalculator()

        metrics = calc.calculate_metrics(
            start_date=date(2026, 1, 1),
            end_date=date(2026, 1, 1),
            model_name='nonexistent'
        )

        assert metrics['win_accuracy'] == 0.0
        assert metrics['top3_accuracy'] == 0.0
        assert metrics['roi'] == 0.0
