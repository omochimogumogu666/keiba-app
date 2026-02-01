"""
精度分析APIのテスト
"""
import pytest
from datetime import date
from src.web.app import create_app
from src.data.models import db, PredictionAccuracy


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


@pytest.fixture
def sample_accuracy_data(app):
    """テスト用精度データ"""
    with app.app_context():
        # 日別データ
        for i in range(1, 11):
            accuracy = PredictionAccuracy(
                aggregation_type='daily',
                aggregation_date=date(2026, 1, i),
                model_name='xgboost',
                total_races=10,
                total_predictions=160,
                win_predictions=10,
                win_hits=2 + i % 3,
                win_accuracy=20.0 + i * 2,
                top3_predictions=30,
                top3_hits=18,
                top3_accuracy=60.0,
                total_bet_amount=1000.0,
                total_return_amount=800.0 + i * 10,
                roi=-20.0 + i
            )
            db.session.add(accuracy)

        db.session.commit()


def test_get_accuracy_summary(client, sample_accuracy_data):
    """サマリーAPIが正しいデータを返すかテスト"""
    response = client.get('/analysis/api/accuracy/summary?start_date=2026-01-01&end_date=2026-01-10')

    assert response.status_code == 200
    data = response.get_json()

    assert 'period' in data
    assert data['period']['start'] == '2026-01-01'
    assert data['period']['end'] == '2026-01-10'

    assert 'overall' in data
    assert 'total_races' in data['overall']
    assert 'win_accuracy' in data['overall']
    assert 'top3_accuracy' in data['overall']
    assert 'roi' in data['overall']


def test_get_accuracy_timeseries(client, sample_accuracy_data):
    """時系列APIがChart.js形式でデータを返すかテスト"""
    response = client.get('/analysis/api/accuracy/timeseries?start_date=2026-01-01&end_date=2026-01-10&aggregation=daily')

    assert response.status_code == 200
    data = response.get_json()

    assert 'labels' in data
    assert len(data['labels']) == 10

    assert 'datasets' in data
    assert len(data['datasets']) >= 2  # 的中率とROI

    # データセット構造確認
    for dataset in data['datasets']:
        assert 'label' in dataset
        assert 'data' in dataset
        assert len(dataset['data']) == 10


def test_get_model_comparison(client, sample_accuracy_data):
    """モデル比較APIが正しいデータを返すかテスト"""
    # Random Forestデータも追加
    with client.application.app_context():
        accuracy = PredictionAccuracy(
            aggregation_type='daily',
            aggregation_date=date(2026, 1, 5),
            model_name='random_forest',
            win_accuracy=18.0,
            top3_accuracy=55.0,
            roi=-25.0
        )
        db.session.add(accuracy)
        db.session.commit()

    response = client.get('/analysis/api/accuracy/models?start_date=2026-01-01&end_date=2026-01-10')

    assert response.status_code == 200
    data = response.get_json()

    assert 'models' in data
    assert len(data['models']) >= 1


def test_missing_parameters(client):
    """必須パラメータが不足している場合のエラーハンドリング"""
    response = client.get('/analysis/api/accuracy/summary')

    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data
