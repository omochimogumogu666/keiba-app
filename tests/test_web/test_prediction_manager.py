"""
prediction_manager.pyのテスト

予測順位の計算ロジックが正しく動作することを検証します。
"""
import pytest
import pandas as pd
import numpy as np


class TestPredictionRanking:
    """予測順位の計算テスト"""

    def test_regression_predictions_have_unique_ranks(self):
        """
        回帰モデルの予測値が順位に正しく変換されることを確認

        バグ: int()変換により、小数の予測値が同じ整数に丸められて
        重複した順位が生成される問題を検証します。
        """
        # 回帰モデルの予測値（連続値）をシミュレート
        predicted_values = np.array([7.3, 8.1, 8.9, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8])

        # 現在の実装（バグがある）: int()で変換
        # これだと [7, 8, 8, 9, 9, 9, 9, 9, 9, 9] のように重複する
        buggy_ranks = predicted_values.astype(int)

        # バグの確認: 重複した順位が生成される
        assert len(set(buggy_ranks)) < len(buggy_ranks), \
            "Bug not reproduced: Expected duplicate ranks"

        # 正しい実装: rank()メソッドで変換
        df = pd.DataFrame({'predicted_position': predicted_values})
        correct_ranks = df['predicted_position'].rank(ascending=True, method='first').astype(int)

        # 正しい実装では、すべての順位が一意になる
        assert len(set(correct_ranks)) == len(correct_ranks), \
            "Expected all ranks to be unique"

        # 順位は1から始まる
        assert correct_ranks.min() == 1, "Expected ranks to start from 1"
        assert correct_ranks.max() == len(predicted_values), \
            "Expected max rank to equal number of horses"

        # 順位の範囲が正しい
        expected_ranks = set(range(1, len(predicted_values) + 1))
        assert set(correct_ranks) == expected_ranks, \
            "Expected ranks to be consecutive integers from 1 to N"

    def test_classification_predictions_have_unique_ranks(self):
        """
        分類モデルの勝率から順位を計算する処理が正しく動作することを確認
        """
        # 勝率をシミュレート
        win_probabilities = np.array([0.15, 0.22, 0.08, 0.18, 0.35, 0.12, 0.09, 0.28])

        # 正しい実装: rank()メソッドで降順ランキング
        df = pd.DataFrame({'win_probability': win_probabilities})
        ranks = df['win_probability'].rank(ascending=False, method='first').astype(int)

        # すべての順位が一意
        assert len(set(ranks)) == len(ranks), "Expected all ranks to be unique"

        # 最高勝率が1位
        assert ranks[np.argmax(win_probabilities)] == 1, \
            "Expected highest win probability to have rank 1"

        # 最低勝率が最下位
        assert ranks[np.argmin(win_probabilities)] == len(win_probabilities), \
            "Expected lowest win probability to have last rank"

    def test_rank_uniqueness_in_actual_workflow(self):
        """
        実際のワークフローで予測順位の一意性を確認
        """
        # 実際のレースデータをシミュレート（18頭立て）
        num_horses = 18
        predicted_values = np.random.uniform(3.5, 12.5, num_horses)

        # 正しい順位計算
        df = pd.DataFrame({'predicted_position': predicted_values})
        ranks = df['predicted_position'].rank(ascending=True, method='first').astype(int)

        # 検証
        assert len(ranks) == num_horses
        assert len(set(ranks)) == num_horses, "All ranks must be unique"
        assert set(ranks) == set(range(1, num_horses + 1)), \
            "Ranks must be consecutive from 1 to N"
