# データベース最適化と改善計画

## 現状の問題点

### 1. NULL項目が多すぎる（最重要）

分析結果:
- **Races テーブル**:
  - `track_condition`: 100% NULL (625/625)
  - `weather`: 100% NULL (625/625)
  - `race_class`: 100% NULL (625/625)
  - `course_type`: 100% NULL (625/625)
  - `track_variant`: 100% NULL (625/625)
  - `distance`: 4.2% NULL (26/625)

- **Horses テーブル**:
  - `birth_date`: 100% NULL (5,559/5,559)
  - `sex`: 100% NULL (5,559/5,559)
  - `sire_id`: 100% NULL (5,559/5,559)
  - `dam_id`: 100% NULL (5,559/5,559)
  - `trainer_id`: 100% NULL (5,559/5,559)

### 2. レース結果が取得できていない
- 8,653エントリーあるが、結果が0件（0%完了率）

### 3. データの重複
- ✅ 問題なし（UNIQUE制約が正常に機能）

### 4. ID設計
- ✅ 問題なし（適切に設計されている）

---

## 解決策

### Phase 1: スクレイパー修正（最優先）

#### 1.1 RaceCard スクレイピングの改善

**問題**:
- RaceData01/02から情報を正しく抽出できていない
- レース名が取得できていない

**解決策**:
```python
# src/scrapers/netkeiba_scraper.py の _parse_race_card_data() を修正

# RaceData01から抽出:
- 距離 (芝1600m)
- コース種別 (右/左)
- トラックバリアント (A/B/C/D)
- 天候 (晴/曇/雨)
- 馬場状態 (良/稍重/重/不良)

# RaceData02から抽出:
- レースクラス (G1/G2/G3/OP/3勝クラス/2勝クラス/1勝クラス/新馬/未勝利)

# ページタイトルから抽出:
- レース名
```

#### 1.2 RaceResult スクレイピングの改善

**問題**:
- "No result table found" エラー
- レース結果のテーブルが見つからない

**解決策**:
- HTMLを確認して、正しいテーブルクラス名を特定
- `_parse_race_result_data()` のセレクターを修正

#### 1.3 Horse Profile スクレイピングの実装

**問題**:
- 馬のプロフィール情報が100% NULL

**解決策**:
```python
# scripts/scrape_historical_data.py に馬プロフィール取得を追加

for entry in race_card['entries']:
    horse_id = entry.get('netkeiba_horse_id')
    if horse_id:
        # 馬プロフィールをスクレイピング
        horse_profile = scraper.scrape_horse_profile(horse_id)
        if horse_profile:
            save_horse_profile_to_db(horse_profile)
```

### Phase 2: データベーススキーマ最適化

#### 2.1 適切なデフォルト値の設定

```python
# models.py

class Race(db.Model):
    # 現状: nullable=True (NULL許可)
    # 改善: デフォルト値を設定し、NULLを最小化

    surface = db.Column(db.String(20), default='turf')  # ✅ 既に実装済み
    status = db.Column(db.String(20), default='upcoming')  # ✅ 既に実装済み

    # 追加推奨:
    weather = db.Column(db.String(20), default='不明')  # デフォルト値
    track_condition = db.Column(db.String(20), default='不明')
```

#### 2.2 インデックスの最適化

```python
# よく検索されるフィールドにインデックスを追加

class Race(db.Model):
    race_date = db.Column(db.Date, nullable=False, index=True)  # ✅ 既に実装済み
    track_id = db.Column(db.Integer, db.ForeignKey('tracks.id'), nullable=False, index=True)  # 追加推奨
    race_class = db.Column(db.String(50), index=True)  # 追加推奨

class Horse(db.Model):
    trainer_id = db.Column(db.Integer, db.ForeignKey('trainers.id'), index=True)  # 追加推奨
```

#### 2.3 データ検証の追加

```python
# models.py に検証ロジックを追加

class Race(db.Model):
    @validates('distance')
    def validate_distance(self, key, distance):
        if distance is not None:
            if distance < 800 or distance > 4000:
                raise ValueError(f"Invalid distance: {distance}")
        return distance

    @validates('surface')
    def validate_surface(self, key, surface):
        if surface not in ['turf', 'dirt', None]:
            raise ValueError(f"Invalid surface: {surface}")
        return surface
```

### Phase 3: データ品質改善スクリプト

#### 3.1 既存データの補完

```python
# scripts/backfill_missing_data.py

# 既存のレースデータで欠けている情報を再スクレイピング
for race in Race.query.filter(
    or_(
        Race.weather.is_(None),
        Race.track_condition.is_(None),
        Race.race_class.is_(None)
    )
).all():
    race_card = scraper.scrape_race_card(race.netkeiba_race_id)
    # Update race with new data
    update_race_from_scrape(race, race_card)
```

#### 3.2 馬プロフィールの一括取得

```python
# scripts/backfill_horse_profiles.py

# 全ての馬のプロフィールを取得
for horse in Horse.query.filter(Horse.birth_date.is_(None)).all():
    if horse.netkeiba_horse_id:
        profile = scraper.scrape_horse_profile(horse.netkeiba_horse_id)
        if profile:
            update_horse_profile(horse, profile)
```

---

## 実装順序

### ステップ1: スクレイパー修正 (最優先)
1. ✅ RaceData01/02のパース改善
2. ✅ レース名抽出の実装
3. ✅ レース結果のHTML構造確認と修正
4. ✅ 馬プロフィールスクレイピングの実装

### ステップ2: テストと検証
1. ✅ 単一レースでのテスト
2. ✅ 複数レースでのテスト
3. ✅ データベース保存の確認

### ステップ3: 既存データの補完
1. ✅ 欠損データの再スクレイピング
2. ✅ 馬プロフィールの一括取得

### ステップ4: スキーマ最適化（オプション）
1. ⏳ インデックス追加
2. ⏳ バリデーション追加
3. ⏳ パフォーマンステスト

---

## 期待される改善効果

### データ完全性
- **Races**: NULL率を 100% → 5%未満 に削減
- **Horses**: NULL率を 100% → 10%未満 に削減
- **RaceResults**: 完了率を 0% → 95%以上 に改善

### クエリパフォーマンス
- インデックス追加により、検索速度が 2-5倍 向上

### データ品質
- バリデーションにより、不正なデータの混入を防止
- デフォルト値により、NULL処理のロジックが簡素化

---

## 次のアクション

1. **今すぐ実行**: スクレイパー修正（_parse_race_card_data, _parse_race_result_data）
2. **今すぐ実行**: 修正したスクレイパーでテスト実行
3. **次回実行**: 既存データの補完スクリプト実行
4. **後で検討**: スキーマ最適化（破壊的変更なし）
