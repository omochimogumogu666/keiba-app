# バグフィックス: predict_upcoming_races.pyのフィールド名修正

**日付**: 2026-01-10
**報告者**: tori
**修正者**: Claude Sonnet 4.5

## 問題の概要

`python scripts/predict_upcoming_races.py --date 2025-12-28` を実行すると以下のエラーが発生していました:

```
2026-01-10 14:27:33 - ERROR - レース 202506050706 の処理中にエラー: 'Missing required key in race_data: netkeiba_race_id'
Traceback (most recent call last):
  File "/app/scripts/predict_upcoming_races.py", line 174, in scrape_upcoming_races
    db_race = save_race_to_db(race_data)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^
```

## 根本原因

[scripts/predict_upcoming_races.py](../scripts/predict_upcoming_races.py)の158-171行目で、`race_data`辞書を作成する際に、データベース関数が期待するフィールド名と異なるキーを使用していました:

### 問題のあったコード
```python
race_data = {
    'jra_race_id': race_id,          # ❌ 間違い
    'track_name': race_info.get('track', '不明'),  # ❌ 間違い
    # ... その他のフィールド
}
```

### データベース関数の期待
[src/data/database.py:268-291](../src/data/database.py#L268-L291)の`save_race_to_db()`関数は以下のフィールドを必須としています:
- `netkeiba_race_id` (レースID)
- `track` (競馬場名、文字列)
- `race_date` (レース日付)

## 修正内容

[scripts/predict_upcoming_races.py:158-171](../scripts/predict_upcoming_races.py#L158-L171)を以下のように修正:

```python
race_data = {
    'netkeiba_race_id': race_id,     # ✅ 修正
    'track': race_info.get('track', '不明'),  # ✅ 修正
    'race_date': target_date,
    'race_number': race_info.get('race_number', 1),
    'race_name': race_card.get('race_name', ''),
    'distance': race_card.get('distance', 0),
    'surface': race_card.get('surface', 'turf'),
    'track_condition': race_card.get('track_condition', '良'),
    'weather': race_card.get('weather', '晴'),
    'race_class': race_card.get('race_class', ''),
    'prize_money': race_card.get('prize_money', 0),
    'status': 'upcoming'
}
```

## 変更されたファイル

- [scripts/predict_upcoming_races.py](../scripts/predict_upcoming_races.py) (L159, L161)

## テスト結果

### 1. 新規統合テスト
[tests/test_database_save.py](../tests/test_database_save.py)を作成し、以下のテストを追加:

- `test_save_race_with_netkeiba_race_id`: `netkeiba_race_id`を使用したレース保存
- `test_save_race_missing_required_field`: 必須フィールドが欠けている場合のエラー検証
- `test_update_existing_race`: 既存レースの更新
- `test_save_race_entries`: レースエントリーの保存
- `test_update_existing_entries`: 既存エントリーの更新

**結果**: 5 passed ✅

### 2. 既存スクレイパーテスト
```bash
pytest tests/test_scrapers/ -v -k "not slow"
```
**結果**: 17 passed, 1 skipped ✅

### 3. 統合テスト
```bash
docker-compose exec -T web python -c "..." # save_race_to_dbの直接テスト
```
**結果**: ✅ レース保存成功

### 4. 実際のスクリプト実行
```bash
docker-compose exec -T web python scripts/predict_upcoming_races.py --date 2025-12-28
```
**結果**:
- ✅ 48件のレースを正常にスクレイピング
- ✅ データベースに保存完了
- ℹ️ モデルファイルが見つからないエラーは別の問題(モデル未学習)

## 影響範囲

- ✅ 今後のレース予想スクリプトが正常に動作するようになりました
- ✅ データベース保存機能の整合性が確保されました
- ✅ 既存のテストには影響なし

## 今後の対応

1. モデル学習: XGBoostモデルを学習して`data/models/`に配置する必要があります
2. ドキュメント更新: [CLAUDE.md](../CLAUDE.md)の「Critical Patterns」セクションに、正しいフィールド名を明記することを推奨

## 参考情報

### データベース保存関数の必須フィールド

**save_race_to_db()**:
- `netkeiba_race_id` (str): レースID
- `track` (str): 競馬場名
- `race_date` (date): レース日付

**save_race_entries_to_db()**:
- `race_id` (int): DB内部のレースID (netkeiba_race_idではない)
- `entries` (list): エントリーデータのリスト
  - `netkeiba_horse_id` (str)
  - `netkeiba_jockey_id` (str)
  - `netkeiba_trainer_id` (str)
  - その他のフィールド

### スクレイパー出力フォーマット

NetkeibaScraper.scrape_race_calendar()が返す辞書:
- `netkeiba_race_id`: レースID
- `track`: 競馬場名
- `race_number`: レース番号
- その他のメタデータ

## まとめ

フィールド名の不一致によるバグを修正し、すべてのテストが成功することを確認しました。今後は`save_race_to_db()`関数の必須フィールド(`netkeiba_race_id`, `track`, `race_date`)を使用することで、同様の問題を防ぐことができます。
