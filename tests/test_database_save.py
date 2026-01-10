"""
データベース保存関数のテスト

save_race_to_db関数とsave_race_entries_to_db関数が正しく動作することを確認する
"""
import pytest
from datetime import date
from src.data.database import save_race_to_db, save_race_entries_to_db
from src.data.models import db, Race, RaceEntry, Track


@pytest.mark.integration
class TestSaveRaceToDb:
    """save_race_to_db関数のテスト"""

    def test_save_race_with_netkeiba_race_id(self, test_db):
        """netkeiba_race_idを使用してレースを保存"""
        race_data = {
            'netkeiba_race_id': '202506050712',
            'track': '中山',
            'race_date': date(2025, 12, 28),
            'race_number': 12,
            'race_name': 'ホープフルステークス',
            'distance': 2000,
            'surface': 'turf',
            'track_condition': '良',
            'weather': '晴',
            'race_class': 'G1',
            'prize_money': 100000000,
            'status': 'upcoming'
        }

        # レースを保存
        race = save_race_to_db(race_data)

        # 検証
        assert race is not None
        assert race.id is not None
        assert race.netkeiba_race_id == '202506050712'
        assert race.race_name == 'ホープフルステークス'
        assert race.distance == 2000
        assert race.surface == 'turf'
        assert race.status == 'upcoming'

        # トラックが自動作成されているか確認
        track = Track.query.filter_by(name='中山').first()
        assert track is not None
        assert race.track_id == track.id

    def test_save_race_missing_required_field(self, test_db):
        """必須フィールドが欠けている場合はエラー"""
        race_data = {
            'track': '中山',
            'race_date': date(2025, 12, 28),
            # netkeiba_race_idが欠けている
        }

        with pytest.raises(KeyError) as exc_info:
            save_race_to_db(race_data)

        assert 'netkeiba_race_id' in str(exc_info.value)

    def test_update_existing_race(self, test_db):
        """既存のレースを更新"""
        race_data = {
            'netkeiba_race_id': '202506050712',
            'track': '中山',
            'race_date': date(2025, 12, 28),
            'race_name': '初期名',
            'distance': 2000,
        }

        # 初回保存
        race1 = save_race_to_db(race_data)
        race1_id = race1.id

        # 同じnetkeiba_race_idで更新
        race_data['race_name'] = '更新後の名前'
        race_data['distance'] = 2400
        race2 = save_race_to_db(race_data)

        # 検証
        assert race2.id == race1_id  # IDは同じ
        assert race2.race_name == '更新後の名前'
        assert race2.distance == 2400

        # データベース内のレース数を確認
        race_count = Race.query.filter_by(netkeiba_race_id='202506050712').count()
        assert race_count == 1


@pytest.mark.integration
class TestSaveRaceEntriesToDb:
    """save_race_entries_to_db関数のテスト"""

    def test_save_race_entries(self, test_db):
        """レースエントリーを保存"""
        # まずレースを作成
        race_data = {
            'netkeiba_race_id': '202506050712',
            'track': '中山',
            'race_date': date(2025, 12, 28),
        }
        race = save_race_to_db(race_data)

        # エントリーデータ
        entries = [
            {
                'netkeiba_horse_id': '2021105438',
                'horse_name': 'テストホース1',
                'netkeiba_jockey_id': '01093',
                'jockey_name': 'テスト騎手1',
                'netkeiba_trainer_id': '01062',
                'trainer_name': 'テスト調教師1',
                'post_position': 1,
                'horse_number': 1,
                'weight': 57.0,
                'horse_weight': 480,
                'horse_weight_change': -2,
                'morning_odds': 3.5
            },
            {
                'netkeiba_horse_id': '2021105439',
                'horse_name': 'テストホース2',
                'netkeiba_jockey_id': '01094',
                'jockey_name': 'テスト騎手2',
                'netkeiba_trainer_id': '01063',
                'trainer_name': 'テスト調教師2',
                'post_position': 2,
                'horse_number': 2,
                'weight': 56.0,
                'horse_weight': 470,
                'horse_weight_change': 0,
                'morning_odds': 5.2
            }
        ]

        # エントリーを保存
        saved_entries = save_race_entries_to_db(race.id, entries)

        # 検証
        assert len(saved_entries) == 2
        assert saved_entries[0].race_id == race.id
        assert saved_entries[0].post_position == 1
        assert saved_entries[0].horse_number == 1
        assert saved_entries[0].weight == 57.0

        # データベースを直接確認
        db_entries = RaceEntry.query.filter_by(race_id=race.id).all()
        assert len(db_entries) == 2

    def test_update_existing_entries(self, test_db):
        """既存のエントリーを更新"""
        # レース作成
        race_data = {
            'netkeiba_race_id': '202506050712',
            'track': '中山',
            'race_date': date(2025, 12, 28),
        }
        race = save_race_to_db(race_data)

        # 初回エントリー
        entries1 = [
            {
                'netkeiba_horse_id': '2021105438',
                'horse_name': 'テストホース1',
                'netkeiba_jockey_id': '01093',
                'jockey_name': 'テスト騎手1',
                'netkeiba_trainer_id': '01062',
                'trainer_name': 'テスト調教師1',
                'post_position': 1,
                'horse_number': 1,
                'weight': 57.0,
                'morning_odds': 3.5
            }
        ]
        save_race_entries_to_db(race.id, entries1)

        # 同じ馬で更新（オッズ変更）
        entries2 = [
            {
                'netkeiba_horse_id': '2021105438',
                'horse_name': 'テストホース1',
                'netkeiba_jockey_id': '01093',
                'jockey_name': 'テスト騎手1',
                'netkeiba_trainer_id': '01062',
                'trainer_name': 'テスト調教師1',
                'post_position': 1,
                'horse_number': 1,
                'weight': 57.0,
                'morning_odds': 2.8  # オッズが変更
            }
        ]
        save_race_entries_to_db(race.id, entries2)

        # 検証
        entries = RaceEntry.query.filter_by(race_id=race.id).all()
        assert len(entries) == 1  # 重複せず1件のまま
        assert entries[0].morning_odds == 2.8  # 更新されている
