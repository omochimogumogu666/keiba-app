"""
Simulation routes for betting simulations.
"""
from datetime import datetime, date
from flask import Blueprint, request, jsonify, render_template
from sqlalchemy import desc

from src.data.models import db, SimulationRun, SimulationBet, Race
from src.ml.betting_simulator import (
    BettingSimulator,
    BettingStrategy,
    BetType,
    SimulationResult
)
from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)

simulation_bp = Blueprint('simulation', __name__, url_prefix='/simulation')


@simulation_bp.route('/')
def index():
    """シミュレーション画面"""
    return render_template('simulation/index.html')


@simulation_bp.route('/history')
def history():
    """シミュレーション履歴一覧"""
    runs = SimulationRun.query.order_by(desc(SimulationRun.created_at)).limit(50).all()
    return render_template('simulation/history.html', runs=runs)


@simulation_bp.route('/run/<int:run_id>')
def view_run(run_id):
    """シミュレーション詳細"""
    run = SimulationRun.query.get_or_404(run_id)
    bets = SimulationBet.query.filter_by(simulation_run_id=run_id).order_by(
        SimulationBet.race_id, SimulationBet.created_at
    ).all()
    return render_template('simulation/detail.html', run=run, bets=bets)


# API Endpoints
@simulation_bp.route('/api/run', methods=['POST'])
def api_run_simulation():
    """
    シミュレーションを実行

    Request JSON:
    {
        "name": "2024年シミュレーション",
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "strategy": {
            "bet_types": ["win", "place", "quinella"],
            "bet_amount": 100,
            "top_n": 3,
            "min_probability": 0.1,
            "max_bets_per_race": 10
        },
        "use_predictions": true,
        "save_to_db": true
    }

    Response JSON:
    {
        "success": true,
        "simulation_run_id": 1,
        "result": {
            "total_races": 100,
            "total_bets": 500,
            "total_investment": 50000,
            "total_payout": 45000,
            "total_profit": -5000,
            "hit_count": 50,
            "hit_rate": 0.1,
            "recovery_rate": 90.0,
            "stats_by_bet_type": {...}
        }
    }
    """
    try:
        data = request.json

        # パラメータ検証
        if not data:
            return jsonify({'error': 'Request body is required'}), 400

        # 日付パース
        try:
            start_date = datetime.strptime(data['start_date'], '%Y-%m-%d').date()
            end_date = datetime.strptime(data['end_date'], '%Y-%m-%d').date()
        except (KeyError, ValueError) as e:
            return jsonify({'error': f'Invalid date format: {str(e)}'}), 400

        if start_date > end_date:
            return jsonify({'error': 'start_date must be before end_date'}), 400

        # 戦略設定
        strategy_data = data.get('strategy', {})

        try:
            bet_types = [BetType(bt) for bt in strategy_data.get('bet_types', ['win', 'place'])]
        except ValueError as e:
            return jsonify({'error': f'Invalid bet_type: {str(e)}'}), 400

        strategy = BettingStrategy(
            bet_types=bet_types,
            bet_amount=strategy_data.get('bet_amount', 100),
            top_n=strategy_data.get('top_n', 3),
            min_probability=strategy_data.get('min_probability', 0.1),
            max_bets_per_race=strategy_data.get('max_bets_per_race', 10)
        )

        # シミュレーション実行
        simulator = BettingSimulator(strategy)
        use_predictions = data.get('use_predictions', True)

        logger.info(f"Running simulation from {start_date} to {end_date}")
        result = simulator.run_simulation(start_date, end_date, use_predictions)

        # DBに保存
        simulation_run = None
        if data.get('save_to_db', True):
            simulation_run = _save_simulation_to_db(
                name=data.get('name', f'Simulation {datetime.now().strftime("%Y-%m-%d %H:%M")}'),
                result=result,
                strategy=strategy
            )
            db.session.commit()
            logger.info(f"Simulation saved with ID: {simulation_run.id}")

        # レスポンス
        response_data = {
            'success': True,
            'result': _serialize_result(result)
        }

        if simulation_run:
            response_data['simulation_run_id'] = simulation_run.id

        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"Simulation error: {str(e)}", exc_info=True)
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@simulation_bp.route('/api/runs', methods=['GET'])
def api_list_runs():
    """
    シミュレーション実行履歴を取得

    Query params:
        limit: int (default: 50)
        offset: int (default: 0)

    Response JSON:
    {
        "runs": [
            {
                "id": 1,
                "name": "2024年シミュレーション",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "total_races": 100,
                "total_bets": 500,
                "hit_rate": 0.1,
                "recovery_rate": 90.0,
                "total_profit": -5000,
                "created_at": "2024-01-01T00:00:00"
            }
        ],
        "total": 10
    }
    """
    try:
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)

        query = SimulationRun.query.order_by(desc(SimulationRun.created_at))
        total = query.count()
        runs = query.limit(limit).offset(offset).all()

        return jsonify({
            'runs': [_serialize_simulation_run(run) for run in runs],
            'total': total
        }), 200

    except Exception as e:
        logger.error(f"Error listing simulation runs: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@simulation_bp.route('/api/runs/<int:run_id>', methods=['GET'])
def api_get_run(run_id):
    """
    シミュレーション詳細を取得

    Response JSON:
    {
        "run": {...},
        "bets": [...],
        "time_series": {
            "cumulative_profit": [[date, profit], ...],
            "cumulative_recovery_rate": [[date, rate], ...]
        }
    }
    """
    try:
        run = SimulationRun.query.get_or_404(run_id)
        bets = SimulationBet.query.filter_by(simulation_run_id=run_id).order_by(
            SimulationBet.race_id
        ).all()

        # 時系列データを生成
        time_series = _generate_time_series(bets)

        return jsonify({
            'run': _serialize_simulation_run(run, include_strategy=True),
            'bets': [_serialize_simulation_bet(bet) for bet in bets],
            'time_series': time_series
        }), 200

    except Exception as e:
        logger.error(f"Error getting simulation run: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@simulation_bp.route('/api/runs/<int:run_id>', methods=['DELETE'])
def api_delete_run(run_id):
    """シミュレーション削除"""
    try:
        run = SimulationRun.query.get_or_404(run_id)
        db.session.delete(run)
        db.session.commit()

        logger.info(f"Deleted simulation run {run_id}")
        return jsonify({'success': True}), 200

    except Exception as e:
        logger.error(f"Error deleting simulation run: {str(e)}", exc_info=True)
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@simulation_bp.route('/api/stats', methods=['GET'])
def api_get_stats():
    """
    全体統計を取得

    Response JSON:
    {
        "total_simulations": 10,
        "avg_recovery_rate": 95.5,
        "avg_hit_rate": 0.12,
        "best_recovery_rate": 120.0,
        "worst_recovery_rate": 70.0
    }
    """
    try:
        from sqlalchemy import func

        stats = db.session.query(
            func.count(SimulationRun.id).label('total'),
            func.avg(SimulationRun.recovery_rate).label('avg_recovery'),
            func.avg(SimulationRun.hit_rate).label('avg_hit_rate'),
            func.max(SimulationRun.recovery_rate).label('best_recovery'),
            func.min(SimulationRun.recovery_rate).label('worst_recovery')
        ).first()

        return jsonify({
            'total_simulations': stats.total or 0,
            'avg_recovery_rate': float(stats.avg_recovery or 0),
            'avg_hit_rate': float(stats.avg_hit_rate or 0),
            'best_recovery_rate': float(stats.best_recovery or 0),
            'worst_recovery_rate': float(stats.worst_recovery or 0)
        }), 200

    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# Helper functions
def _save_simulation_to_db(name: str, result: SimulationResult, strategy: BettingStrategy) -> SimulationRun:
    """シミュレーション結果をDBに保存"""
    # 戦略設定をJSONに変換
    strategy_config = {
        'bet_types': [bt.value for bt in strategy.bet_types],
        'bet_amount': strategy.bet_amount,
        'top_n': strategy.top_n,
        'min_probability': strategy.min_probability,
        'max_bets_per_race': strategy.max_bets_per_race
    }

    # SimulationRunを作成
    simulation_run = SimulationRun(
        name=name,
        start_date=result.start_date,
        end_date=result.end_date,
        strategy_config=strategy_config,
        total_races=result.total_races,
        total_bets=result.total_bets,
        total_investment=result.total_investment,
        total_payout=result.total_payout,
        total_profit=result.total_profit,
        hit_count=result.hit_count,
        hit_rate=result.hit_rate,
        recovery_rate=result.recovery_rate,
        stats_by_bet_type=result.stats_by_bet_type
    )
    db.session.add(simulation_run)
    db.session.flush()  # IDを取得

    # 各購入馬券を保存
    for bet_result in result.bet_results:
        simulation_bet = SimulationBet(
            simulation_run_id=simulation_run.id,
            race_id=bet_result.ticket.race_id,
            bet_type=bet_result.ticket.bet_type.value,
            combination=bet_result.ticket.combination,
            horse_numbers=bet_result.ticket.horse_numbers,
            amount=bet_result.ticket.amount,
            predicted_probability=bet_result.ticket.predicted_probability,
            is_hit=bet_result.is_hit,
            payout_amount=bet_result.payout_amount,
            profit=bet_result.profit
        )
        db.session.add(simulation_bet)

    return simulation_run


def _serialize_result(result: SimulationResult) -> dict:
    """SimulationResultをJSONシリアライズ"""
    return {
        'start_date': result.start_date.isoformat(),
        'end_date': result.end_date.isoformat(),
        'total_races': result.total_races,
        'total_bets': result.total_bets,
        'total_investment': result.total_investment,
        'total_payout': result.total_payout,
        'total_profit': result.total_profit,
        'hit_count': result.hit_count,
        'hit_rate': result.hit_rate,
        'recovery_rate': result.recovery_rate,
        'stats_by_bet_type': result.stats_by_bet_type,
        'cumulative_profit': [[d.isoformat(), p] for d, p in result.cumulative_profit],
        'cumulative_recovery_rate': [[d.isoformat(), r] for d, r in result.cumulative_recovery_rate]
    }


def _serialize_simulation_run(run: SimulationRun, include_strategy: bool = False) -> dict:
    """SimulationRunをJSONシリアライズ"""
    data = {
        'id': run.id,
        'name': run.name,
        'start_date': run.start_date.isoformat(),
        'end_date': run.end_date.isoformat(),
        'total_races': run.total_races,
        'total_bets': run.total_bets,
        'total_investment': run.total_investment,
        'total_payout': run.total_payout,
        'total_profit': run.total_profit,
        'hit_count': run.hit_count,
        'hit_rate': run.hit_rate,
        'recovery_rate': run.recovery_rate,
        'stats_by_bet_type': run.stats_by_bet_type,
        'created_at': run.created_at.isoformat()
    }

    if include_strategy:
        data['strategy_config'] = run.strategy_config

    return data


def _serialize_simulation_bet(bet: SimulationBet) -> dict:
    """SimulationBetをJSONシリアライズ"""
    # レース情報を取得
    race = Race.query.get(bet.race_id)
    race_info = None
    if race:
        race_info = {
            'date': race.race_date.isoformat(),
            'track': race.track.name if race.track else None,
            'race_number': race.race_number,
            'race_name': race.race_name
        }

    return {
        'id': bet.id,
        'race_id': bet.race_id,
        'race_info': race_info,
        'bet_type': bet.bet_type,
        'combination': bet.combination,
        'horse_numbers': bet.horse_numbers,
        'amount': bet.amount,
        'predicted_probability': bet.predicted_probability,
        'is_hit': bet.is_hit,
        'payout_amount': bet.payout_amount,
        'profit': bet.profit
    }


def _generate_time_series(bets: list) -> dict:
    """購入馬券リストから時系列データを生成"""
    # レースIDから日付を取得
    race_dates = {}
    for bet in bets:
        if bet.race_id not in race_dates:
            race = Race.query.get(bet.race_id)
            if race:
                race_dates[bet.race_id] = race.race_date

    # 日付ごとに集計
    daily_data = {}
    for bet in bets:
        race_date = race_dates.get(bet.race_id)
        if not race_date:
            continue

        if race_date not in daily_data:
            daily_data[race_date] = {
                'investment': 0,
                'payout': 0
            }

        daily_data[race_date]['investment'] += bet.amount
        daily_data[race_date]['payout'] += bet.payout_amount

    # 累積データを生成
    sorted_dates = sorted(daily_data.keys())
    cumulative_profit = []
    cumulative_recovery_rate = []

    cumulative_investment = 0
    cumulative_payout = 0

    for race_date in sorted_dates:
        data = daily_data[race_date]
        cumulative_investment += data['investment']
        cumulative_payout += data['payout']
        profit = cumulative_payout - cumulative_investment

        cumulative_profit.append([race_date.isoformat(), profit])

        if cumulative_investment > 0:
            recovery_rate = (cumulative_payout / cumulative_investment) * 100
            cumulative_recovery_rate.append([race_date.isoformat(), recovery_rate])

    return {
        'cumulative_profit': cumulative_profit,
        'cumulative_recovery_rate': cumulative_recovery_rate
    }
