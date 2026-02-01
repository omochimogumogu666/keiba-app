# tests/test_data/test_prediction_accuracy_model.py
"""
予想精度モデルのテスト
"""
import pytest
from datetime import date
from src.data.models import db, PredictionAccuracy, Track
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
def client(app):
    """テストクライアント"""
    return app.test_client()


def test_create_prediction_accuracy(app):
    """PredictionAccuracyレコードが作成できるかテスト"""
    with app.app_context():
        # テストデータ作成
        accuracy = PredictionAccuracy(
            aggregation_type='daily',
            aggregation_date=date(2026, 1, 15),
            model_name='xgboost',
            total_races=10,
            total_predictions=160,
            win_predictions=10,
            win_hits=3,
            win_accuracy=30.0,
            top3_predictions=30,
            top3_hits=18,
            top3_accuracy=60.0,
            total_bet_amount=1000.0,
            total_return_amount=850.0,
            roi=-15.0
        )

        db.session.add(accuracy)
        db.session.commit()

        # 取得して検証
        saved = PredictionAccuracy.query.first()
        assert saved is not None
        assert saved.aggregation_type == 'daily'
        assert saved.model_name == 'xgboost'
        assert saved.win_accuracy == 30.0
        assert saved.roi == -15.0


def test_prediction_accuracy_unique_constraint(app):
    """同一条件のレコードは重複作成できないことをテスト"""
    with app.app_context():
        # 競馬場を作成（NULL値を避けるため）
        track = Track(name='東京', location='東京都')
        db.session.add(track)
        db.session.commit()

        # 1つ目
        acc1 = PredictionAccuracy(
            aggregation_type='daily',
            aggregation_date=date(2026, 1, 15),
            model_name='xgboost',
            track_id=track.id,
            race_class='G1',
            win_accuracy=30.0
        )
        db.session.add(acc1)
        db.session.commit()

        # 2つ目（同じ条件）
        acc2 = PredictionAccuracy(
            aggregation_type='daily',
            aggregation_date=date(2026, 1, 15),
            model_name='xgboost',
            track_id=track.id,
            race_class='G1',
            win_accuracy=35.0
        )
        db.session.add(acc2)

        # IntegrityErrorが発生するはず
        from sqlalchemy.exc import IntegrityError
        with pytest.raises(IntegrityError):
            db.session.commit()


def test_prediction_accuracy_track_relationship(app):
    """競馬場とのリレーションが機能するかテスト"""
    with app.app_context():
        # 競馬場を作成
        track = Track(name='東京', location='東京都')
        db.session.add(track)
        db.session.commit()

        # 競馬場別の精度データ
        accuracy = PredictionAccuracy(
            aggregation_type='daily',
            aggregation_date=date(2026, 1, 15),
            model_name='xgboost',
            track_id=track.id,
            win_accuracy=28.5
        )
        db.session.add(accuracy)
        db.session.commit()

        # リレーションを確認
        saved = PredictionAccuracy.query.first()
        assert saved.track is not None
        assert saved.track.name == '東京'
