"""
Tests for database models.
"""
import pytest
from datetime import date, datetime
from src.data.models import (
    db, Horse, Jockey, Trainer, Track, Race, RaceEntry, RaceResult, Prediction
)


class TestTrack:
    """Tests for Track model."""

    def test_create_track(self, test_db):
        """Test creating a track."""
        track = Track(
            name='中山',
            location='千葉県船橋市',
            surface_types=['turf', 'dirt']
        )
        test_db.session.add(track)
        test_db.session.commit()

        assert track.id is not None
        assert track.name == '中山'
        assert track.location == '千葉県船橋市'
        assert 'turf' in track.surface_types
        assert 'dirt' in track.surface_types

    def test_track_repr(self, sample_track):
        """Test track string representation."""
        assert repr(sample_track) == '<Track 東京>'

    def test_track_unique_name(self, test_db, sample_track):
        """Test track name uniqueness."""
        duplicate_track = Track(
            name='東京',
            location='別の場所'
        )
        test_db.session.add(duplicate_track)

        with pytest.raises(Exception):  # IntegrityError
            test_db.session.commit()


class TestJockey:
    """Tests for Jockey model."""

    def test_create_jockey(self, test_db):
        """Test creating a jockey."""
        jockey = Jockey(
            netkeiba_jockey_id='J99999',
            name='新人騎手',
            weight=50.0
        )
        test_db.session.add(jockey)
        test_db.session.commit()

        assert jockey.id is not None
        assert jockey.name == '新人騎手'
        assert jockey.weight == 50.0

    def test_jockey_unique_id(self, test_db, sample_jockey):
        """Test jockey JRA ID uniqueness."""
        duplicate_jockey = Jockey(
            netkeiba_jockey_id='J12345',  # Same as sample_jockey
            name='別の騎手'
        )
        test_db.session.add(duplicate_jockey)

        with pytest.raises(Exception):
            test_db.session.commit()


class TestTrainer:
    """Tests for Trainer model."""

    def test_create_trainer(self, test_db):
        """Test creating a trainer."""
        trainer = Trainer(
            netkeiba_trainer_id='T99999',
            name='新人調教師',
            stable='栗東'
        )
        test_db.session.add(trainer)
        test_db.session.commit()

        assert trainer.id is not None
        assert trainer.name == '新人調教師'
        assert trainer.stable == '栗東'


class TestHorse:
    """Tests for Horse model."""

    def test_create_horse(self, test_db, sample_trainer):
        """Test creating a horse."""
        horse = Horse(
            netkeiba_horse_id='H99999',
            name='テスト馬',
            birth_date=date(2021, 4, 1),
            sex='牝',
            trainer_id=sample_trainer.id
        )
        test_db.session.add(horse)
        test_db.session.commit()

        assert horse.id is not None
        assert horse.name == 'テスト馬'
        assert horse.sex == '牝'
        assert horse.trainer.name == 'テスト調教師'

    def test_horse_with_pedigree(self, test_db, sample_trainer):
        """Test horse with sire and dam."""
        # Create sire and dam
        sire = Horse(
            netkeiba_horse_id='H_SIRE',
            name='父馬',
            birth_date=date(2015, 3, 1),
            sex='牡',
            trainer_id=sample_trainer.id
        )
        dam = Horse(
            netkeiba_horse_id='H_DAM',
            name='母馬',
            birth_date=date(2016, 4, 1),
            sex='牝',
            trainer_id=sample_trainer.id
        )
        test_db.session.add_all([sire, dam])
        test_db.session.commit()

        # Create offspring
        offspring = Horse(
            netkeiba_horse_id='H_OFFSPRING',
            name='子馬',
            birth_date=date(2021, 5, 1),
            sex='牡',
            sire_id=sire.id,
            dam_id=dam.id,
            trainer_id=sample_trainer.id
        )
        test_db.session.add(offspring)
        test_db.session.commit()

        assert offspring.sire.name == '父馬'
        assert offspring.dam.name == '母馬'
        assert offspring in sire.offspring_as_sire

    def test_horse_repr(self, sample_horse):
        """Test horse string representation."""
        assert repr(sample_horse) == '<Horse テストホース>'


class TestRace:
    """Tests for Race model."""

    def test_create_race(self, test_db, sample_track):
        """Test creating a race."""
        race = Race(
            netkeiba_race_id='2024020201',
            track_id=sample_track.id,
            race_date=date(2024, 2, 2),
            race_number=5,
            race_name='新馬戦',
            distance=1800,
            surface='turf',
            track_condition='稍重',
            weather='曇',
            race_class='新馬',
            prize_money=5000000,
            status='upcoming'
        )
        test_db.session.add(race)
        test_db.session.commit()

        assert race.id is not None
        assert race.race_name == '新馬戦'
        assert race.distance == 1800
        assert race.track.name == '東京'

    def test_race_with_entries(self, test_db, sample_race, sample_horse, sample_jockey):
        """Test race with entries."""
        entry = RaceEntry(
            race_id=sample_race.id,
            horse_id=sample_horse.id,
            jockey_id=sample_jockey.id,
            post_position=2,
            horse_number=2,
            weight=56.0
        )
        test_db.session.add(entry)
        test_db.session.commit()

        assert len(sample_race.race_entries) == 1
        assert sample_race.race_entries[0].horse.name == 'テストホース'


class TestRaceEntry:
    """Tests for RaceEntry model."""

    def test_create_race_entry(self, test_db, sample_race, sample_horse, sample_jockey):
        """Test creating a race entry."""
        entry = RaceEntry(
            race_id=sample_race.id,
            horse_id=sample_horse.id,
            jockey_id=sample_jockey.id,
            post_position=3,
            horse_number=3,
            weight=58.0,
            horse_weight=480,
            horse_weight_change=5,
            morning_odds=5.5
        )
        test_db.session.add(entry)
        test_db.session.commit()

        assert entry.id is not None
        assert entry.post_position == 3
        assert entry.weight == 58.0
        assert entry.race.race_name == 'テストレース'
        assert entry.horse.name == 'テストホース'
        assert entry.jockey.name == 'テスト騎手'


class TestRaceResult:
    """Tests for RaceResult model."""

    def test_create_race_result(self, test_db, sample_race_entry):
        """Test creating a race result."""
        result = RaceResult(
            race_entry_id=sample_race_entry.id,
            finish_position=2,
            finish_time=97.8,
            margin='ハナ',
            final_odds=4.5,
            popularity=3,
            running_positions=[5, 4, 3, 2]
        )
        test_db.session.add(result)
        test_db.session.commit()

        assert result.id is not None
        assert result.finish_position == 2
        assert result.finish_time == 97.8
        assert result.running_positions == [5, 4, 3, 2]
        assert result.race_entry.horse.name == 'テストホース'


class TestPrediction:
    """Tests for Prediction model."""

    def test_create_prediction(self, test_db, sample_race, sample_horse):
        """Test creating a prediction."""
        prediction = Prediction(
            race_id=sample_race.id,
            horse_id=sample_horse.id,
            predicted_position=2,
            win_probability=0.28,
            confidence_score=0.68,
            model_version='2.0',
            model_name='XGBoost'
        )
        test_db.session.add(prediction)
        test_db.session.commit()

        assert prediction.id is not None
        assert prediction.predicted_position == 2
        assert prediction.win_probability == 0.28
        assert prediction.confidence_score == 0.68
        assert prediction.model_name == 'XGBoost'
        assert prediction.race.race_name == 'テストレース'
        assert prediction.horse.name == 'テストホース'

    def test_multiple_predictions_for_race(self, test_db, sample_race):
        """Test multiple predictions for one race."""
        horses = []
        for i in range(3):
            horse = Horse(
                netkeiba_horse_id=f'H_PRED_{i}',
                name=f'予想馬{i}',
                birth_date=date(2020, 3, 1),
                sex='牡'
            )
            horses.append(horse)
            test_db.session.add(horse)
        test_db.session.commit()

        predictions = []
        for i, horse in enumerate(horses, 1):
            pred = Prediction(
                race_id=sample_race.id,
                horse_id=horse.id,
                predicted_position=i,
                win_probability=0.4 - (i * 0.1),
                confidence_score=0.8,
                model_name='RandomForest'
            )
            predictions.append(pred)
            test_db.session.add(pred)
        test_db.session.commit()

        assert len(sample_race.predictions) == 3
        assert sample_race.predictions[0].predicted_position == 1


class TestRelationships:
    """Tests for model relationships."""

    def test_horse_race_entries(self, test_db, sample_horse, sample_race, sample_jockey):
        """Test horse can have multiple race entries."""
        # Create multiple races
        races = []
        for i in range(3):
            race = Race(
                netkeiba_race_id=f'202403030{i}',
                track_id=sample_race.track_id,
                race_date=date(2024, 3, 3),
                race_number=i + 1,
                distance=1600,
                surface='turf',
                status='upcoming'
            )
            races.append(race)
            test_db.session.add(race)
        test_db.session.commit()

        # Create entries for each race
        for race in races:
            entry = RaceEntry(
                race_id=race.id,
                horse_id=sample_horse.id,
                jockey_id=sample_jockey.id,
                post_position=1,
                horse_number=1,
                weight=57.0
            )
            test_db.session.add(entry)
        test_db.session.commit()

        assert len(sample_horse.race_entries) == 3

    def test_cascade_delete(self, test_db, sample_race, sample_horse, sample_jockey):
        """Test cascade delete of race entries when race is deleted."""
        entry = RaceEntry(
            race_id=sample_race.id,
            horse_id=sample_horse.id,
            jockey_id=sample_jockey.id,
            post_position=1,
            horse_number=1,
            weight=57.0
        )
        test_db.session.add(entry)
        test_db.session.commit()

        entry_id = entry.id

        # Delete race
        test_db.session.delete(sample_race)
        test_db.session.commit()

        # Entry should be deleted
        deleted_entry = RaceEntry.query.get(entry_id)
        assert deleted_entry is None
