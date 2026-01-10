"""
Betting simulation module for horse racing.

このモジュールは競馬の馬券シミュレーションを実行します。
- 過去のレース結果と予想を使用した戦略シミュレーション
- 7券種対応（単勝、複勝、馬連、馬単、ワイド、3連複、3連単）
- 的中率・回収率・期待値の計算
- 時系列の損益推移分析

Simulates betting strategies using historical race results and predictions.
"""
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

from sqlalchemy import and_
from sqlalchemy.orm import joinedload

from src.data.models import db, Race, RaceEntry, RaceResult, Payout, Prediction
from src.ml.constants import (
    DEFAULT_BET_AMOUNT,
    DEFAULT_TOP_N_HORSES,
    MIN_BET_PROBABILITY,
    MAX_BETS_PER_RACE,
    PLACE_PROBABILITY_MULTIPLIER,
    PLACE_PROBABILITY_MAX,
    COMBINATION_PROBABILITY_FACTOR,
    TRIO_PROBABILITY_FACTOR
)
from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)


class BetType(Enum):
    """馬券種別"""
    WIN = "win"  # 単勝
    PLACE = "place"  # 複勝
    QUINELLA = "quinella"  # 馬連
    EXACTA = "exacta"  # 馬単
    WIDE = "wide"  # ワイド
    TRIO = "trio"  # 3連複
    TRIFECTA = "trifecta"  # 3連単


@dataclass
class BettingStrategy:
    """馬券購入戦略設定"""
    bet_types: List[BetType]  # 対象馬券種
    bet_amount: int = DEFAULT_BET_AMOUNT  # 1点あたりの購入金額（円）
    top_n: int = DEFAULT_TOP_N_HORSES  # 予想確率上位N点を購入
    min_probability: float = MIN_BET_PROBABILITY  # 最小購入確率閾値
    max_bets_per_race: int = MAX_BETS_PER_RACE  # 1レースあたりの最大購入点数


@dataclass
class BetTicket:
    """購入馬券"""
    race_id: int
    bet_type: BetType
    combination: str  # 例: "1", "1-2", "1-2-3"
    horse_numbers: List[int]
    amount: int  # 購入金額
    predicted_probability: float
    odds: Optional[float] = None
    expected_value: Optional[float] = None  # 期待値 = probability * odds * amount


@dataclass
class BetResult:
    """馬券結果"""
    ticket: BetTicket
    is_hit: bool  # 的中したか
    payout_amount: int = 0  # 払戻金額
    profit: int = 0  # 損益 = payout - amount


@dataclass
class SimulationResult:
    """シミュレーション結果"""
    start_date: date
    end_date: date
    strategy: BettingStrategy
    total_races: int = 0
    total_bets: int = 0
    total_investment: int = 0  # 総投資額
    total_payout: int = 0  # 総払戻金
    total_profit: int = 0  # 総損益
    hit_count: int = 0  # 的中数
    hit_rate: float = 0.0  # 的中率
    recovery_rate: float = 0.0  # 回収率
    bet_results: List[BetResult] = field(default_factory=list)

    # 馬券種別ごとの集計
    stats_by_bet_type: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # 時系列データ
    cumulative_profit: List[Tuple[date, int]] = field(default_factory=list)
    cumulative_recovery_rate: List[Tuple[date, float]] = field(default_factory=list)

    def calculate_metrics(self):
        """メトリクスを計算"""
        if self.total_bets > 0:
            self.hit_rate = self.hit_count / self.total_bets

        if self.total_investment > 0:
            self.recovery_rate = (self.total_payout / self.total_investment) * 100

        self.total_profit = self.total_payout - self.total_investment

        # 馬券種別ごとの集計
        self._calculate_stats_by_bet_type()

        # 時系列データの生成
        self._generate_time_series()

    def _calculate_stats_by_bet_type(self):
        """馬券種別ごとの統計を計算"""
        for bet_type in BetType:
            results_of_type = [r for r in self.bet_results if r.ticket.bet_type == bet_type]
            if not results_of_type:
                continue

            total = len(results_of_type)
            hits = sum(1 for r in results_of_type if r.is_hit)
            investment = sum(r.ticket.amount for r in results_of_type)
            payout = sum(r.payout_amount for r in results_of_type)

            self.stats_by_bet_type[bet_type.value] = {
                'count': total,
                'hit_count': hits,
                'hit_rate': hits / total if total > 0 else 0,
                'investment': investment,
                'payout': payout,
                'profit': payout - investment,
                'recovery_rate': (payout / investment * 100) if investment > 0 else 0
            }

    def _generate_time_series(self):
        """時系列データを生成"""
        if not self.bet_results:
            return

        # 日付ごとにグループ化
        date_results = {}
        for result in self.bet_results:
            # race_idからレース日付を取得する必要があるため、
            # この関数は後でBettingSimulator内で実装
            pass


class BettingSimulator:
    """馬券シミュレーター"""

    def __init__(self, strategy: BettingStrategy):
        """
        Args:
            strategy: 馬券購入戦略
        """
        self.strategy = strategy
        self.logger = get_app_logger(__name__)

    def run_simulation(
        self,
        start_date: date,
        end_date: date,
        use_predictions: bool = True
    ) -> SimulationResult:
        """
        指定期間でシミュレーションを実行

        Args:
            start_date: 開始日
            end_date: 終了日
            use_predictions: 予測データを使用するか（Falseの場合は完全ランダム）

        Returns:
            SimulationResult: シミュレーション結果
        """
        self.logger.info(f"Starting simulation from {start_date} to {end_date}")

        result = SimulationResult(
            start_date=start_date,
            end_date=end_date,
            strategy=self.strategy
        )

        # 対象レースを取得
        races = self._get_races(start_date, end_date)
        result.total_races = len(races)

        self.logger.info(f"Found {len(races)} completed races in the period")

        # 各レースでシミュレーション
        for race in races:
            race_bets = self._generate_bets_for_race(race, use_predictions)
            race_results = self._evaluate_bets(race, race_bets)

            result.bet_results.extend(race_results)
            result.total_bets += len(race_results)
            result.total_investment += sum(r.ticket.amount for r in race_results)
            result.total_payout += sum(r.payout_amount for r in race_results)
            result.hit_count += sum(1 for r in race_results if r.is_hit)

        # メトリクス計算
        result.calculate_metrics()
        self._generate_time_series_for_result(result)

        self.logger.info(
            f"Simulation completed: {result.total_bets} bets, "
            f"hit rate: {result.hit_rate:.2%}, recovery rate: {result.recovery_rate:.1f}%"
        )

        return result

    def _get_races(self, start_date: date, end_date: date) -> List[Race]:
        """
        指定期間の完了済みレースを取得

        Args:
            start_date: 開始日
            end_date: 終了日

        Returns:
            List[Race]: レース一覧
        """
        races = Race.query.filter(
            and_(
                Race.race_date >= start_date,
                Race.race_date <= end_date,
                Race.status == 'completed'
            )
        ).options(
            joinedload(Race.race_entries).joinedload(RaceEntry.result),
            joinedload(Race.predictions)
        ).order_by(Race.race_date, Race.race_number).all()

        return races

    def _generate_bets_for_race(
        self,
        race: Race,
        use_predictions: bool
    ) -> List[BetTicket]:
        """
        1レース分の購入馬券を生成

        Args:
            race: レース
            use_predictions: 予測データを使用するか

        Returns:
            List[BetTicket]: 購入馬券リスト
        """
        tickets = []

        # 出走馬の予測確率を取得
        predictions = self._get_predictions_for_race(race, use_predictions)

        if not predictions:
            self.logger.warning(f"No predictions for race {race.id}, skipping")
            return tickets

        # 各馬券種で馬券を生成
        for bet_type in self.strategy.bet_types:
            type_tickets = self._generate_bets_by_type(
                race, bet_type, predictions
            )
            tickets.extend(type_tickets)

        # 最大購入点数でフィルタ
        if len(tickets) > self.strategy.max_bets_per_race:
            # 予測確率が高い順にソート
            tickets.sort(key=lambda t: t.predicted_probability, reverse=True)
            tickets = tickets[:self.strategy.max_bets_per_race]

        return tickets

    def _get_predictions_for_race(
        self,
        race: Race,
        use_predictions: bool
    ) -> Dict[int, float]:
        """
        レースの予測確率を取得

        Args:
            race: レース
            use_predictions: DBの予測データを使用するか

        Returns:
            Dict[int, float]: {馬番: 勝率}
        """
        predictions = {}

        if use_predictions and race.predictions:
            # DBから予測データを取得
            for pred in race.predictions:
                # 馬番を取得
                entry = next(
                    (e for e in race.race_entries if e.horse_id == pred.horse_id),
                    None
                )
                if entry and pred.win_probability:
                    predictions[entry.horse_number] = pred.win_probability

        if not predictions:
            # 予測データがない場合は均等確率
            num_horses = len(race.race_entries)
            if num_horses > 0:
                uniform_prob = 1.0 / num_horses
                for entry in race.race_entries:
                    predictions[entry.horse_number] = uniform_prob

        return predictions

    def _generate_bets_by_type(
        self,
        race: Race,
        bet_type: BetType,
        predictions: Dict[int, float]
    ) -> List[BetTicket]:
        """
        特定馬券種の購入馬券を生成

        Args:
            race: レース
            bet_type: 馬券種
            predictions: 予測確率

        Returns:
            List[BetTicket]: 購入馬券リスト
        """
        tickets = []

        if bet_type == BetType.WIN:
            tickets = self._generate_win_bets(race, predictions)
        elif bet_type == BetType.PLACE:
            tickets = self._generate_place_bets(race, predictions)
        elif bet_type == BetType.QUINELLA:
            tickets = self._generate_quinella_bets(race, predictions)
        elif bet_type == BetType.EXACTA:
            tickets = self._generate_exacta_bets(race, predictions)
        elif bet_type == BetType.WIDE:
            tickets = self._generate_wide_bets(race, predictions)
        elif bet_type == BetType.TRIO:
            tickets = self._generate_trio_bets(race, predictions)
        elif bet_type == BetType.TRIFECTA:
            tickets = self._generate_trifecta_bets(race, predictions)

        return tickets

    def _generate_win_bets(
        self,
        race: Race,
        predictions: Dict[int, float]
    ) -> List[BetTicket]:
        """単勝馬券を生成"""
        tickets = []

        # 予測確率上位N頭
        sorted_horses = sorted(
            predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.strategy.top_n]

        for horse_num, prob in sorted_horses:
            if prob < self.strategy.min_probability:
                continue

            tickets.append(BetTicket(
                race_id=race.id,
                bet_type=BetType.WIN,
                combination=str(horse_num),
                horse_numbers=[horse_num],
                amount=self.strategy.bet_amount,
                predicted_probability=prob
            ))

        return tickets

    def _generate_place_bets(
        self,
        race: Race,
        predictions: Dict[int, float]
    ) -> List[BetTicket]:
        """複勝馬券を生成"""
        tickets = []

        # 複勝は上位3頭に入る確率を考慮（簡易的に勝率の3倍）
        sorted_horses = sorted(
            predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.strategy.top_n]

        for horse_num, prob in sorted_horses:
            # 複勝確率は単勝確率より高いと仮定
            place_prob = min(prob * PLACE_PROBABILITY_MULTIPLIER, PLACE_PROBABILITY_MAX)
            if place_prob < self.strategy.min_probability:
                continue

            tickets.append(BetTicket(
                race_id=race.id,
                bet_type=BetType.PLACE,
                combination=str(horse_num),
                horse_numbers=[horse_num],
                amount=self.strategy.bet_amount,
                predicted_probability=place_prob
            ))

        return tickets

    def _generate_quinella_bets(
        self,
        race: Race,
        predictions: Dict[int, float]
    ) -> List[BetTicket]:
        """馬連馬券を生成"""
        tickets = []

        # 上位N頭の組み合わせ
        sorted_horses = sorted(
            predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.strategy.top_n]

        for i, (horse1, prob1) in enumerate(sorted_horses):
            for horse2, prob2 in sorted_horses[i+1:]:
                # 2頭が1-2着に入る確率（簡易計算）
                combo_prob = prob1 * prob2
                if combo_prob < self.strategy.min_probability:
                    continue

                combination = f"{min(horse1, horse2)}-{max(horse1, horse2)}"
                tickets.append(BetTicket(
                    race_id=race.id,
                    bet_type=BetType.QUINELLA,
                    combination=combination,
                    horse_numbers=sorted([horse1, horse2]),
                    amount=self.strategy.bet_amount,
                    predicted_probability=combo_prob
                ))

        return tickets

    def _generate_exacta_bets(
        self,
        race: Race,
        predictions: Dict[int, float]
    ) -> List[BetTicket]:
        """馬単馬券を生成"""
        tickets = []

        sorted_horses = sorted(
            predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.strategy.top_n]

        for i, (horse1, prob1) in enumerate(sorted_horses):
            for horse2, prob2 in sorted_horses:
                if horse1 == horse2:
                    continue

                # 1着horse1, 2着horse2の確率（簡易計算）
                combo_prob = prob1 * prob2 * COMBINATION_PROBABILITY_FACTOR
                if combo_prob < self.strategy.min_probability:
                    continue

                combination = f"{horse1}-{horse2}"
                tickets.append(BetTicket(
                    race_id=race.id,
                    bet_type=BetType.EXACTA,
                    combination=combination,
                    horse_numbers=[horse1, horse2],
                    amount=self.strategy.bet_amount,
                    predicted_probability=combo_prob
                ))

        return tickets

    def _generate_wide_bets(
        self,
        race: Race,
        predictions: Dict[int, float]
    ) -> List[BetTicket]:
        """ワイド馬券を生成"""
        tickets = []

        sorted_horses = sorted(
            predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.strategy.top_n]

        for i, (horse1, prob1) in enumerate(sorted_horses):
            for horse2, prob2 in sorted_horses[i+1:]:
                # 2頭が3着以内に入る確率（簡易計算）
                combo_prob = (prob1 * 3) * (prob2 * 3)
                if combo_prob < self.strategy.min_probability:
                    continue

                combination = f"{min(horse1, horse2)}-{max(horse1, horse2)}"
                tickets.append(BetTicket(
                    race_id=race.id,
                    bet_type=BetType.WIDE,
                    combination=combination,
                    horse_numbers=sorted([horse1, horse2]),
                    amount=self.strategy.bet_amount,
                    predicted_probability=combo_prob
                ))

        return tickets

    def _generate_trio_bets(
        self,
        race: Race,
        predictions: Dict[int, float]
    ) -> List[BetTicket]:
        """3連複馬券を生成"""
        tickets = []

        sorted_horses = sorted(
            predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.strategy.top_n]

        for i, (horse1, prob1) in enumerate(sorted_horses):
            for j, (horse2, prob2) in enumerate(sorted_horses[i+1:], start=i+1):
                for horse3, prob3 in sorted_horses[j+1:]:
                    # 3頭が3着以内に入る確率（簡易計算）
                    combo_prob = prob1 * prob2 * prob3
                    if combo_prob < self.strategy.min_probability:
                        continue

                    horses = sorted([horse1, horse2, horse3])
                    combination = f"{horses[0]}-{horses[1]}-{horses[2]}"
                    tickets.append(BetTicket(
                        race_id=race.id,
                        bet_type=BetType.TRIO,
                        combination=combination,
                        horse_numbers=horses,
                        amount=self.strategy.bet_amount,
                        predicted_probability=combo_prob
                    ))

        return tickets

    def _generate_trifecta_bets(
        self,
        race: Race,
        predictions: Dict[int, float]
    ) -> List[BetTicket]:
        """3連単馬券を生成"""
        tickets = []

        sorted_horses = sorted(
            predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.strategy.top_n]

        for i, (horse1, prob1) in enumerate(sorted_horses):
            for j, (horse2, prob2) in enumerate(sorted_horses):
                if horse1 == horse2:
                    continue
                for horse3, prob3 in sorted_horses:
                    if horse3 == horse1 or horse3 == horse2:
                        continue

                    # 1着horse1, 2着horse2, 3着horse3の確率（簡易計算）
                    combo_prob = prob1 * prob2 * prob3 * TRIO_PROBABILITY_FACTOR
                    if combo_prob < self.strategy.min_probability:
                        continue

                    combination = f"{horse1}-{horse2}-{horse3}"
                    tickets.append(BetTicket(
                        race_id=race.id,
                        bet_type=BetType.TRIFECTA,
                        combination=combination,
                        horse_numbers=[horse1, horse2, horse3],
                        amount=self.strategy.bet_amount,
                        predicted_probability=combo_prob
                    ))

        return tickets

    def _evaluate_bets(
        self,
        race: Race,
        tickets: List[BetTicket]
    ) -> List[BetResult]:
        """
        馬券の結果を判定

        Args:
            race: レース
            tickets: 購入馬券リスト

        Returns:
            List[BetResult]: 馬券結果リスト
        """
        results = []

        # レース結果を取得
        race_results = self._get_race_results(race)
        if not race_results:
            self.logger.warning(f"No results for race {race.id}")
            return results

        # 払戻金を取得
        payouts = self._get_payouts(race)

        for ticket in tickets:
            is_hit = self._check_hit(ticket, race_results)
            payout_amount = 0

            if is_hit:
                payout_amount = self._calculate_payout(ticket, payouts)

            results.append(BetResult(
                ticket=ticket,
                is_hit=is_hit,
                payout_amount=payout_amount,
                profit=payout_amount - ticket.amount
            ))

        return results

    def _get_race_results(self, race: Race) -> Dict[int, int]:
        """
        レース結果を取得

        Args:
            race: レース

        Returns:
            Dict[int, int]: {馬番: 着順}
        """
        results = {}

        for entry in race.race_entries:
            if entry.result and entry.result.finish_position:
                results[entry.horse_number] = entry.result.finish_position

        return results

    def _get_payouts(self, race: Race) -> Dict[str, Dict[str, int]]:
        """
        払戻金を取得

        Args:
            race: レース

        Returns:
            Dict[str, Dict[str, int]]: {馬券種: {組み合わせ: 払戻金}}
        """
        payouts = {}

        for payout in race.payouts:
            if payout.bet_type not in payouts:
                payouts[payout.bet_type] = {}
            payouts[payout.bet_type][payout.combination] = payout.payout

        return payouts

    def _check_hit(
        self,
        ticket: BetTicket,
        race_results: Dict[int, int]
    ) -> bool:
        """
        馬券が的中したかチェック

        Args:
            ticket: 購入馬券
            race_results: {馬番: 着順}

        Returns:
            bool: 的中したか
        """
        bet_type = ticket.bet_type
        horses = ticket.horse_numbers

        # 着順を取得
        positions = [race_results.get(h) for h in horses]
        if None in positions:
            return False

        if bet_type == BetType.WIN:
            # 単勝: 1着
            return positions[0] == 1

        elif bet_type == BetType.PLACE:
            # 複勝: 3着以内
            return positions[0] <= 3

        elif bet_type == BetType.QUINELLA:
            # 馬連: 2頭が1-2着（順不同）
            return set(positions) <= {1, 2}

        elif bet_type == BetType.EXACTA:
            # 馬単: 1着horses[0], 2着horses[1]
            return positions[0] == 1 and positions[1] == 2

        elif bet_type == BetType.WIDE:
            # ワイド: 2頭が3着以内
            return all(p <= 3 for p in positions)

        elif bet_type == BetType.TRIO:
            # 3連複: 3頭が1-2-3着（順不同）
            return set(positions) == {1, 2, 3}

        elif bet_type == BetType.TRIFECTA:
            # 3連単: 1着horses[0], 2着horses[1], 3着horses[2]
            return (positions[0] == 1 and
                    positions[1] == 2 and
                    positions[2] == 3)

        return False

    def _calculate_payout(
        self,
        ticket: BetTicket,
        payouts: Dict[str, Dict[str, int]]
    ) -> int:
        """
        払戻金を計算

        Args:
            ticket: 購入馬券
            payouts: {馬券種: {組み合わせ: 払戻金}}

        Returns:
            int: 払戻金（円）
        """
        bet_type_str = ticket.bet_type.value

        if bet_type_str not in payouts:
            return 0

        bet_payouts = payouts[bet_type_str]
        combination = ticket.combination

        if combination in bet_payouts:
            # 100円あたりの払戻金を購入金額に応じて計算
            payout_per_100 = bet_payouts[combination]
            return int((ticket.amount / 100) * payout_per_100)

        return 0

    def _generate_time_series_for_result(self, result: SimulationResult):
        """
        時系列データを生成してSimulationResultに追加

        Args:
            result: シミュレーション結果
        """
        if not result.bet_results:
            return

        # レースIDから日付を取得するマップを作成
        race_dates = {}
        for bet_result in result.bet_results:
            race_id = bet_result.ticket.race_id
            if race_id not in race_dates:
                race = Race.query.get(race_id)
                if race:
                    race_dates[race_id] = race.race_date

        # 日付ごとに集計
        daily_data = {}
        for bet_result in result.bet_results:
            race_date = race_dates.get(bet_result.ticket.race_id)
            if not race_date:
                continue

            if race_date not in daily_data:
                daily_data[race_date] = {
                    'investment': 0,
                    'payout': 0
                }

            daily_data[race_date]['investment'] += bet_result.ticket.amount
            daily_data[race_date]['payout'] += bet_result.payout_amount

        # 時系列データを生成（累積）
        sorted_dates = sorted(daily_data.keys())
        cumulative_profit = 0
        cumulative_investment = 0
        cumulative_payout = 0

        for race_date in sorted_dates:
            data = daily_data[race_date]
            cumulative_investment += data['investment']
            cumulative_payout += data['payout']
            cumulative_profit = cumulative_payout - cumulative_investment

            result.cumulative_profit.append((race_date, cumulative_profit))

            if cumulative_investment > 0:
                recovery_rate = (cumulative_payout / cumulative_investment) * 100
                result.cumulative_recovery_rate.append((race_date, recovery_rate))


def create_default_strategy() -> BettingStrategy:
    """デフォルトの購入戦略を作成"""
    return BettingStrategy(
        bet_types=[BetType.WIN, BetType.PLACE],
        bet_amount=100,
        top_n=3,
        min_probability=0.05,
        max_bets_per_race=10
    )
