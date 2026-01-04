"""
SQLAlchemy ORM models for JRA horse racing data.
"""
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Horse(db.Model):
    """Horse model."""
    __tablename__ = 'horses'

    id = db.Column(db.Integer, primary_key=True)
    netkeiba_horse_id = db.Column(db.String(20), unique=True, index=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    birth_date = db.Column(db.Date)
    sex = db.Column(db.String(10))  # 牡, 牝, セン
    sire_id = db.Column(db.Integer, db.ForeignKey('horses.id'))
    dam_id = db.Column(db.Integer, db.ForeignKey('horses.id'))
    trainer_id = db.Column(db.Integer, db.ForeignKey('trainers.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    sire = db.relationship('Horse', foreign_keys=[sire_id], remote_side=[id], backref='offspring_as_sire')
    dam = db.relationship('Horse', foreign_keys=[dam_id], remote_side=[id], backref='offspring_as_dam')
    trainer = db.relationship('Trainer', back_populates='horses')
    race_entries = db.relationship('RaceEntry', back_populates='horse', cascade='all, delete-orphan')

    def __repr__(self):
        return f'<Horse {self.name}>'


class Jockey(db.Model):
    """Jockey model."""
    __tablename__ = 'jockeys'

    id = db.Column(db.Integer, primary_key=True)
    netkeiba_jockey_id = db.Column(db.String(20), unique=True, index=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    weight = db.Column(db.Float)  # kg
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    race_entries = db.relationship('RaceEntry', back_populates='jockey')

    def __repr__(self):
        return f'<Jockey {self.name}>'


class Trainer(db.Model):
    """Trainer model."""
    __tablename__ = 'trainers'

    id = db.Column(db.Integer, primary_key=True)
    netkeiba_trainer_id = db.Column(db.String(20), unique=True, index=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    stable = db.Column(db.String(100))  # 所属厩舎
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    horses = db.relationship('Horse', back_populates='trainer')

    def __repr__(self):
        return f'<Trainer {self.name}>'


class Track(db.Model):
    """Race track model."""
    __tablename__ = 'tracks'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False, unique=True)  # 東京, 中山, etc.
    location = db.Column(db.String(100))
    surface_types = db.Column(db.JSON)  # ['turf', 'dirt']
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    races = db.relationship('Race', back_populates='track')

    def __repr__(self):
        return f'<Track {self.name}>'


class Race(db.Model):
    """Race model."""
    __tablename__ = 'races'

    id = db.Column(db.Integer, primary_key=True)
    netkeiba_race_id = db.Column(db.String(20), unique=True, index=True, nullable=False)
    track_id = db.Column(db.Integer, db.ForeignKey('tracks.id'), nullable=False)
    race_date = db.Column(db.Date, nullable=False, index=True)
    race_number = db.Column(db.Integer, nullable=False)  # 1R, 2R, etc.
    race_name = db.Column(db.String(200))  # レース名
    distance = db.Column(db.Integer)  # meters - nullable because calendar doesn't have this info
    surface = db.Column(db.String(20), default='turf')  # turf, dirt - default to turf if unknown
    track_condition = db.Column(db.String(20))  # 良, 稍重, 重, 不良
    weather = db.Column(db.String(20))  # 晴, 曇, 雨, etc.
    race_class = db.Column(db.String(50))  # G1, G2, G3, OP, etc.
    prize_money = db.Column(db.Integer)  # 円
    status = db.Column(db.String(20), default='upcoming')  # upcoming, completed
    kaisai_code = db.Column(db.String(10))  # netkeiba開催コード
    meeting_number = db.Column(db.Integer)  # 回次
    day_number = db.Column(db.Integer)  # 日目
    course_type = db.Column(db.String(20))  # 右, 左
    track_variant = db.Column(db.String(10))  # A, B, C, D
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    track = db.relationship('Track', back_populates='races')
    race_entries = db.relationship('RaceEntry', back_populates='race', cascade='all, delete-orphan')
    predictions = db.relationship('Prediction', back_populates='race', cascade='all, delete-orphan')

    def __repr__(self):
        return f'<Race {self.netkeiba_race_id} {self.race_name}>'


class RaceEntry(db.Model):
    """Race entry model (horse participation in a race)."""
    __tablename__ = 'race_entries'

    id = db.Column(db.Integer, primary_key=True)
    race_id = db.Column(db.Integer, db.ForeignKey('races.id'), nullable=False, index=True)
    horse_id = db.Column(db.Integer, db.ForeignKey('horses.id'), nullable=False, index=True)
    jockey_id = db.Column(db.Integer, db.ForeignKey('jockeys.id'), nullable=False)
    post_position = db.Column(db.Integer)  # 枠番
    horse_number = db.Column(db.Integer)  # 馬番
    weight = db.Column(db.Float)  # 斤量 (kg)
    horse_weight = db.Column(db.Integer)  # 馬体重 (kg)
    horse_weight_change = db.Column(db.Integer)  # 馬体重増減
    morning_odds = db.Column(db.Float)  # 朝オッズ
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    race = db.relationship('Race', back_populates='race_entries')
    horse = db.relationship('Horse', back_populates='race_entries')
    jockey = db.relationship('Jockey', back_populates='race_entries')
    result = db.relationship('RaceResult', back_populates='race_entry', uselist=False, cascade='all, delete-orphan')

    def __repr__(self):
        return f'<RaceEntry race={self.race_id} horse={self.horse_id}>'


class RaceResult(db.Model):
    """Race result model."""
    __tablename__ = 'race_results'

    id = db.Column(db.Integer, primary_key=True)
    race_entry_id = db.Column(db.Integer, db.ForeignKey('race_entries.id'), nullable=False, unique=True, index=True)
    finish_position = db.Column(db.Integer)  # 着順
    finish_time = db.Column(db.Float)  # 秒
    margin = db.Column(db.String(50))  # 着差
    final_odds = db.Column(db.Float)  # 確定オッズ
    popularity = db.Column(db.Integer)  # 人気
    running_positions = db.Column(db.JSON)  # コーナー通過順位 [1, 1, 1, 1]
    comment = db.Column(db.Text)  # レースコメント
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    race_entry = db.relationship('RaceEntry', back_populates='result')

    def __repr__(self):
        return f'<RaceResult entry={self.race_entry_id} position={self.finish_position}>'


class Payout(db.Model):
    """Payout model for race bet returns."""
    __tablename__ = 'payouts'

    id = db.Column(db.Integer, primary_key=True)
    race_id = db.Column(db.Integer, db.ForeignKey('races.id'), nullable=False, index=True)
    bet_type = db.Column(db.String(20), nullable=False)  # win, place, quinella, exacta, wide, trio, trifecta
    combination = db.Column(db.String(50), nullable=False)  # '1' or '1-2' or '1-2-3'
    payout = db.Column(db.Integer, nullable=False)  # 円
    popularity = db.Column(db.Integer)  # 人気
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    race = db.relationship('Race', backref='payouts')

    def __repr__(self):
        return f'<Payout race={self.race_id} {self.bet_type} {self.combination}>'


class Prediction(db.Model):
    """Prediction model."""
    __tablename__ = 'predictions'

    id = db.Column(db.Integer, primary_key=True)
    race_id = db.Column(db.Integer, db.ForeignKey('races.id'), nullable=False, index=True)
    horse_id = db.Column(db.Integer, db.ForeignKey('horses.id'), nullable=False, index=True)
    predicted_position = db.Column(db.Integer)  # 予想着順
    win_probability = db.Column(db.Float)  # 勝率
    confidence_score = db.Column(db.Float)  # 信頼度スコア (0-1)
    model_version = db.Column(db.String(50))  # モデルバージョン
    model_name = db.Column(db.String(50))  # モデル名 (RandomForest, XGBoost, etc.)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    race = db.relationship('Race', back_populates='predictions')
    horse = db.relationship('Horse')

    def __repr__(self):
        return f'<Prediction race={self.race_id} horse={self.horse_id} pos={self.predicted_position}>'
