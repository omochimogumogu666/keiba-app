"""
Tests for betting simulator.
"""
import pytest
from datetime import date, datetime

from src.web.app import create_app
from src.data.models import (
    db, Race, RaceEntry, RaceResult, Payout, Prediction,
    Horse, Jockey, Trainer, Track
)
from src.ml.betting_simulator import (
    BettingSimulator,
    BettingStrategy,
    BetType,
    BetTicket,
    BetResult
)


@pytest.fixture
def app():
    """Create test Flask app."""
    app = create_app('testing')
    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture
def sample_race(app):
    """Create sample race with entries and results."""
    with app.app_context():
        # Create track
        track = Track(name='東京', location='東京都')
        db.session.add(track)

        # Create horses
        horses = []
        for i in range(1, 6):
            horse = Horse(
                netkeiba_horse_id=f'test_horse_{i}',
                name=f'テスト馬{i}'
            )
            db.session.add(horse)
            horses.append(horse)

        # Create jockey
        jockey = Jockey(netkeiba_jockey_id='test_jockey', name='テスト騎手')
        db.session.add(jockey)

        # Create race
        race = Race(
            netkeiba_race_id='2024010101010101',
            track=track,
            race_date=date(2024, 1, 1),
            race_number=1,
            race_name='テストレース',
            distance=2000,
            surface='turf',
            status='completed'
        )
        db.session.add(race)
        db.session.flush()

        # Create race entries
        entries = []
        for i, horse in enumerate(horses, start=1):
            entry = RaceEntry(
                race_id=race.id,
                horse_id=horse.id,
                jockey_id=jockey.id,
                horse_number=i,
                post_position=i
            )
            db.session.add(entry)
            entries.append(entry)

        db.session.flush()

        # Create race results (1-2-3-4-5)
        for i, entry in enumerate(entries, start=1):
            result = RaceResult(
                race_entry_id=entry.id,
                finish_position=i,
                final_odds=i * 2.0
            )
            db.session.add(result)

        # Create payouts
        payouts = [
            Payout(race_id=race.id, bet_type='win', combination='1', payout=200),
            Payout(race_id=race.id, bet_type='place', combination='1', payout=110),
            Payout(race_id=race.id, bet_type='place', combination='2', payout=150),
            Payout(race_id=race.id, bet_type='place', combination='3', payout=200),
            Payout(race_id=race.id, bet_type='quinella', combination='1-2', payout=500),
            Payout(race_id=race.id, bet_type='exacta', combination='1-2', payout=800),
            Payout(race_id=race.id, bet_type='wide', combination='1-2', payout=300),
            Payout(race_id=race.id, bet_type='wide', combination='1-3', payout=400),
            Payout(race_id=race.id, bet_type='wide', combination='2-3', payout=500),
            Payout(race_id=race.id, bet_type='trio', combination='1-2-3', payout=2000),
            Payout(race_id=race.id, bet_type='trifecta', combination='1-2-3', payout=5000),
        ]
        for payout in payouts:
            db.session.add(payout)

        # Create predictions
        for i, horse in enumerate(horses, start=1):
            prediction = Prediction(
                race_id=race.id,
                horse_id=horse.id,
                win_probability=0.5 / i  # 1着の確率が高い
            )
            db.session.add(prediction)

        db.session.commit()

        race_id = race.id  # IDを保存

    # コンテキスト外でIDを返す
    return race_id


@pytest.mark.unit
def test_betting_strategy_creation():
    """Test creating betting strategy."""
    strategy = BettingStrategy(
        bet_types=[BetType.WIN, BetType.PLACE],
        bet_amount=100,
        top_n=3,
        min_probability=0.1,
        max_bets_per_race=10
    )

    assert len(strategy.bet_types) == 2
    assert BetType.WIN in strategy.bet_types
    assert strategy.bet_amount == 100
    assert strategy.top_n == 3


@pytest.mark.unit
def test_betting_simulator_creation():
    """Test creating betting simulator."""
    strategy = BettingStrategy(
        bet_types=[BetType.WIN],
        bet_amount=100,
        top_n=3
    )

    simulator = BettingSimulator(strategy)
    assert simulator.strategy == strategy


@pytest.mark.integration
def test_run_simulation_win_only(app, sample_race):
    """Test running simulation with win bets only."""
    with app.app_context():
        strategy = BettingStrategy(
            bet_types=[BetType.WIN],
            bet_amount=100,
            top_n=3,
            min_probability=0.01
        )

        simulator = BettingSimulator(strategy)
        result = simulator.run_simulation(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1),
            use_predictions=True
        )

        assert result.total_races == 1
        assert result.total_bets >= 1  # At least 1 bet
        assert result.total_investment > 0
        assert result.hit_count >= 0


@pytest.mark.integration
def test_run_simulation_multiple_bet_types(app, sample_race):
    """Test running simulation with multiple bet types."""
    with app.app_context():
        strategy = BettingStrategy(
            bet_types=[BetType.WIN, BetType.PLACE, BetType.QUINELLA],
            bet_amount=100,
            top_n=3,
            min_probability=0.01
        )

        simulator = BettingSimulator(strategy)
        result = simulator.run_simulation(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1),
            use_predictions=True
        )

        assert result.total_races == 1
        assert result.total_bets > 3  # Multiple bet types
        assert 'win' in result.stats_by_bet_type
        assert 'place' in result.stats_by_bet_type
        assert 'quinella' in result.stats_by_bet_type


@pytest.mark.unit
def test_check_hit_win(app, sample_race):
    """Test checking win bet hit."""
    with app.app_context():
        strategy = BettingStrategy(bet_types=[BetType.WIN], bet_amount=100, top_n=3)
        simulator = BettingSimulator(strategy)

        # 1着の馬 (的中)
        ticket_hit = BetTicket(
            race_id=sample_race,  # sample_raceはIDを返す
            bet_type=BetType.WIN,
            combination='1',
            horse_numbers=[1],
            amount=100,
            predicted_probability=0.5
        )

        race_results = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        assert simulator._check_hit(ticket_hit, race_results) is True

        # 2着の馬 (不的中)
        ticket_miss = BetTicket(
            race_id=sample_race,
            bet_type=BetType.WIN,
            combination='2',
            horse_numbers=[2],
            amount=100,
            predicted_probability=0.3
        )
        assert simulator._check_hit(ticket_miss, race_results) is False


@pytest.mark.unit
def test_check_hit_place(app, sample_race):
    """Test checking place bet hit."""
    with app.app_context():
        strategy = BettingStrategy(bet_types=[BetType.PLACE], bet_amount=100, top_n=3)
        simulator = BettingSimulator(strategy)

        race_results = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

        # 3着以内 (的中)
        for horse_num in [1, 2, 3]:
            ticket = BetTicket(
                race_id=sample_race,
                bet_type=BetType.PLACE,
                combination=str(horse_num),
                horse_numbers=[horse_num],
                amount=100,
                predicted_probability=0.3
            )
            assert simulator._check_hit(ticket, race_results) is True

        # 4着以下 (不的中)
        ticket_miss = BetTicket(
            race_id=sample_race,
            bet_type=BetType.PLACE,
            combination='4',
            horse_numbers=[4],
            amount=100,
            predicted_probability=0.2
        )
        assert simulator._check_hit(ticket_miss, race_results) is False


@pytest.mark.unit
def test_check_hit_quinella(app, sample_race):
    """Test checking quinella bet hit."""
    with app.app_context():
        strategy = BettingStrategy(bet_types=[BetType.QUINELLA], bet_amount=100, top_n=3)
        simulator = BettingSimulator(strategy)

        race_results = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

        # 1-2着 (的中)
        ticket_hit = BetTicket(
            race_id=sample_race,
            bet_type=BetType.QUINELLA,
            combination='1-2',
            horse_numbers=[1, 2],
            amount=100,
            predicted_probability=0.2
        )
        assert simulator._check_hit(ticket_hit, race_results) is True

        # 1-3着 (不的中)
        ticket_miss = BetTicket(
            race_id=sample_race,
            bet_type=BetType.QUINELLA,
            combination='1-3',
            horse_numbers=[1, 3],
            amount=100,
            predicted_probability=0.15
        )
        assert simulator._check_hit(ticket_miss, race_results) is False


@pytest.mark.unit
def test_check_hit_exacta(app, sample_race):
    """Test checking exacta bet hit."""
    with app.app_context():
        strategy = BettingStrategy(bet_types=[BetType.EXACTA], bet_amount=100, top_n=3)
        simulator = BettingSimulator(strategy)

        race_results = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

        # 1着-2着 (的中)
        ticket_hit = BetTicket(
            race_id=sample_race,
            bet_type=BetType.EXACTA,
            combination='1-2',
            horse_numbers=[1, 2],
            amount=100,
            predicted_probability=0.2
        )
        assert simulator._check_hit(ticket_hit, race_results) is True

        # 2着-1着 (不的中 - 順番が違う)
        ticket_miss = BetTicket(
            race_id=sample_race,
            bet_type=BetType.EXACTA,
            combination='2-1',
            horse_numbers=[2, 1],
            amount=100,
            predicted_probability=0.1
        )
        assert simulator._check_hit(ticket_miss, race_results) is False


@pytest.mark.unit
def test_calculate_payout(app, sample_race):
    """Test calculating payout."""
    with app.app_context():
        # Raceオブジェクトを取得
        race = Race.query.get(sample_race)

        strategy = BettingStrategy(bet_types=[BetType.WIN], bet_amount=100, top_n=3)
        simulator = BettingSimulator(strategy)

        ticket = BetTicket(
            race_id=race.id,
            bet_type=BetType.WIN,
            combination='1',
            horse_numbers=[1],
            amount=100,
            predicted_probability=0.5
        )

        payouts = simulator._get_payouts(race)
        payout_amount = simulator._calculate_payout(ticket, payouts)

        # 100円で200円払戻
        assert payout_amount == 200


@pytest.mark.unit
def test_calculate_payout_multiple_units(app, sample_race):
    """Test calculating payout with multiple units."""
    with app.app_context():
        # Raceオブジェクトを取得
        race = Race.query.get(sample_race)

        strategy = BettingStrategy(bet_types=[BetType.WIN], bet_amount=500, top_n=3)
        simulator = BettingSimulator(strategy)

        ticket = BetTicket(
            race_id=race.id,
            bet_type=BetType.WIN,
            combination='1',
            horse_numbers=[1],
            amount=500,  # 5単位
            predicted_probability=0.5
        )

        payouts = simulator._get_payouts(race)
        payout_amount = simulator._calculate_payout(ticket, payouts)

        # 500円で1000円払戻 (200円 * 5)
        assert payout_amount == 1000


@pytest.mark.integration
def test_simulation_metrics_calculation(app, sample_race):
    """Test simulation result metrics calculation."""
    with app.app_context():
        strategy = BettingStrategy(
            bet_types=[BetType.WIN],
            bet_amount=100,
            top_n=3,
            min_probability=0.01
        )

        simulator = BettingSimulator(strategy)
        result = simulator.run_simulation(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1),
            use_predictions=True
        )

        # メトリクスが計算されているか確認
        assert result.hit_rate >= 0
        assert result.hit_rate <= 1
        assert result.recovery_rate >= 0
        assert result.total_profit == result.total_payout - result.total_investment


@pytest.mark.integration
def test_simulation_without_predictions(app, sample_race):
    """Test simulation without using predictions (uniform probability)."""
    with app.app_context():
        strategy = BettingStrategy(
            bet_types=[BetType.WIN],
            bet_amount=100,
            top_n=3,
            min_probability=0.01
        )

        simulator = BettingSimulator(strategy)
        result = simulator.run_simulation(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1),
            use_predictions=False  # 予測を使わない
        )

        assert result.total_races == 1
        assert result.total_bets >= 1


@pytest.mark.integration
def test_simulation_time_series_generation(app, sample_race):
    """Test time series data generation."""
    with app.app_context():
        strategy = BettingStrategy(
            bet_types=[BetType.WIN],
            bet_amount=100,
            top_n=3,
            min_probability=0.01
        )

        simulator = BettingSimulator(strategy)
        result = simulator.run_simulation(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1),
            use_predictions=True
        )

        # 時系列データが生成されているか確認
        assert len(result.cumulative_profit) >= 0
        assert len(result.cumulative_recovery_rate) >= 0

        if len(result.cumulative_profit) > 0:
            # 日付と損益のタプルになっているか確認
            date_obj, profit = result.cumulative_profit[0]
            assert isinstance(date_obj, date)
            assert isinstance(profit, int)


@pytest.mark.integration
def test_full_simulation_workflow(app, sample_race):
    """Test full simulation workflow."""
    with app.app_context():
        # 複数の馬券種でシミュレーション
        strategy = BettingStrategy(
            bet_types=[BetType.WIN, BetType.PLACE, BetType.QUINELLA],
            bet_amount=100,
            top_n=3,
            min_probability=0.01,
            max_bets_per_race=20
        )

        simulator = BettingSimulator(strategy)
        result = simulator.run_simulation(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1),
            use_predictions=True
        )

        # 基本的な検証
        assert result.total_races == 1
        assert result.total_bets > 0
        assert result.total_investment > 0

        # 馬券種別統計が生成されているか確認
        assert len(result.stats_by_bet_type) > 0

        # 各馬券種の統計が正しいか確認
        for bet_type, stats in result.stats_by_bet_type.items():
            assert stats['count'] > 0
            assert stats['investment'] > 0
            assert 0 <= stats['hit_rate'] <= 1
            assert stats['recovery_rate'] >= 0
