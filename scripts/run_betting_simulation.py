"""
CLI script to run betting simulations.

Usage:
    python scripts/run_betting_simulation.py --start-date 2024-01-01 --end-date 2024-12-31
    python scripts/run_betting_simulation.py --start-date 2024-01-01 --end-date 2024-12-31 --bet-types win place --top-n 5
    python scripts/run_betting_simulation.py --start-date 2024-01-01 --end-date 2024-12-31 --no-predictions
"""
import argparse
from datetime import datetime, date
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.web.app import create_app
from src.data.models import db, SimulationRun, SimulationBet
from src.ml.betting_simulator import (
    BettingSimulator,
    BettingStrategy,
    BetType,
    SimulationResult
)
from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run betting simulation')

    # 必須パラメータ
    parser.add_argument(
        '--start-date',
        type=str,
        required=True,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        required=True,
        help='End date (YYYY-MM-DD)'
    )

    # オプショナルパラメータ
    parser.add_argument(
        '--name',
        type=str,
        help='Simulation name'
    )
    parser.add_argument(
        '--bet-types',
        nargs='+',
        choices=['win', 'place', 'quinella', 'exacta', 'wide', 'trio', 'trifecta'],
        default=['win', 'place'],
        help='Bet types to simulate (default: win place)'
    )
    parser.add_argument(
        '--bet-amount',
        type=int,
        default=100,
        help='Bet amount per ticket in yen (default: 100)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=3,
        help='Number of top probability bets to place (default: 3)'
    )
    parser.add_argument(
        '--min-probability',
        type=float,
        default=0.05,
        help='Minimum probability threshold (default: 0.05)'
    )
    parser.add_argument(
        '--max-bets-per-race',
        type=int,
        default=10,
        help='Maximum number of bets per race (default: 10)'
    )
    parser.add_argument(
        '--no-predictions',
        action='store_true',
        help='Use uniform probability instead of predictions'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to database'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='development',
        choices=['development', 'production', 'testing'],
        help='Configuration to use (default: development)'
    )

    return parser.parse_args()


def save_simulation_to_db(name: str, result: SimulationResult, strategy: BettingStrategy) -> SimulationRun:
    """Save simulation result to database."""
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
    db.session.flush()

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


def print_result(result: SimulationResult):
    """Print simulation result to console."""
    print("\n" + "=" * 80)
    print("シミュレーション結果")
    print("=" * 80)

    print(f"\n期間: {result.start_date} 〜 {result.end_date}")
    print(f"対象レース数: {result.total_races}")
    print(f"購入点数: {result.total_bets}")

    print("\n【成績サマリー】")
    print(f"  総投資額: {result.total_investment:,}円")
    print(f"  総払戻額: {result.total_payout:,}円")
    print(f"  総損益: {result.total_profit:+,}円")
    print(f"  的中数: {result.hit_count} / {result.total_bets}")
    print(f"  的中率: {result.hit_rate * 100:.1f}%")
    print(f"  回収率: {result.recovery_rate:.1f}%")

    if result.stats_by_bet_type:
        print("\n【馬券種別集計】")
        bet_type_names = {
            'win': '単勝',
            'place': '複勝',
            'quinella': '馬連',
            'exacta': '馬単',
            'wide': 'ワイド',
            'trio': '3連複',
            'trifecta': '3連単'
        }

        header = f"{'馬券種':<10} {'購入数':>8} {'的中数':>8} {'的中率':>8} {'投資額':>12} {'払戻額':>12} {'損益':>12} {'回収率':>8}"
        print(header)
        print("-" * len(header))

        for bet_type, stats in result.stats_by_bet_type.items():
            bet_type_name = bet_type_names.get(bet_type, bet_type)
            print(
                f"{bet_type_name:<10} "
                f"{stats['count']:>8} "
                f"{stats['hit_count']:>8} "
                f"{stats['hit_rate'] * 100:>7.1f}% "
                f"{stats['investment']:>11,}円 "
                f"{stats['payout']:>11,}円 "
                f"{stats['profit']:>+11,}円 "
                f"{stats['recovery_rate']:>7.1f}%"
            )

    print("\n" + "=" * 80 + "\n")


def main():
    """Main function."""
    args = parse_args()

    # Parse dates
    try:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        sys.exit(1)

    if start_date > end_date:
        logger.error("start_date must be before end_date")
        sys.exit(1)

    # Create Flask app
    app = create_app(args.config)

    with app.app_context():
        # Create betting strategy
        bet_types = [BetType(bt) for bt in args.bet_types]
        strategy = BettingStrategy(
            bet_types=bet_types,
            bet_amount=args.bet_amount,
            top_n=args.top_n,
            min_probability=args.min_probability,
            max_bets_per_race=args.max_bets_per_race
        )

        # Run simulation
        simulator = BettingSimulator(strategy)
        use_predictions = not args.no_predictions

        logger.info(f"Running simulation from {start_date} to {end_date}")
        logger.info(f"Bet types: {', '.join(bt.value for bt in bet_types)}")
        logger.info(f"Using predictions: {use_predictions}")

        result = simulator.run_simulation(start_date, end_date, use_predictions)

        # Print result
        print_result(result)

        # Save to database
        if not args.no_save:
            simulation_name = args.name or f"Simulation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            simulation_run = save_simulation_to_db(simulation_name, result, strategy)
            db.session.commit()

            logger.info(f"Simulation saved with ID: {simulation_run.id}")
            print(f"シミュレーション結果を保存しました (ID: {simulation_run.id})")
            print(f"詳細: http://localhost:5000/simulation/run/{simulation_run.id}")
        else:
            logger.info("Simulation not saved (--no-save flag)")


if __name__ == '__main__':
    main()
