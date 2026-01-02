# REST API Documentation

JRA競馬予想アプリケーションのRESTful APIエンドポイント一覧

## Base URL

```
http://localhost:5000/api
```

## 共通仕様

### ページネーション

すべてのリストエンドポイントでページネーションをサポートしています。

**パラメータ:**
- `page`: ページ番号 (デフォルト: 1)
- `per_page`: 1ページあたりのアイテム数 (デフォルト: 20, 最大: 100)

**レスポンス形式:**
```json
{
  "data": [...],
  "meta": {
    "page": 1,
    "per_page": 20,
    "total_items": 100,
    "total_pages": 5,
    "has_next": true,
    "has_prev": false
  }
}
```

### エラーレスポンス

エラー発生時のレスポンス形式:
```json
{
  "error": "エラーメッセージ"
}
```

---

## レース (Races)

### GET /api/races

レース一覧を取得します。

**パラメータ:**
- `date` (optional): 日付フィルター (YYYY-MM-DD)
- `track_id` (optional): 競馬場IDフィルター
- `status` (optional): レースステータス (`upcoming`, `in_progress`, `completed`, `cancelled`)
- `sort` (optional): ソート項目 (`date`, `race_number`)
- `order` (optional): ソート順 (`asc`, `desc`)
- `page`, `per_page`: ページネーション

**レスポンス例:**
```json
{
  "races": [
    {
      "id": 1,
      "jra_race_id": "2026010101011",
      "race_name": "○○ステークス",
      "race_number": 11,
      "race_date": "2026-01-01",
      "track_id": 1,
      "track_name": "東京",
      "distance": 1600,
      "surface": "芝",
      "weather": "晴",
      "track_condition": "良",
      "status": "completed",
      "prize_money": 100000000,
      "race_class": "G1"
    }
  ],
  "meta": {...}
}
```

### GET /api/races/{id}

特定のレースの詳細情報を取得します。

**パラメータ:**
- `include_entries` (optional): 出走馬情報を含める (`true`/`false`)
- `include_results` (optional): レース結果を含める (`true`/`false`)
- `include_predictions` (optional): 予想情報を含める (`true`/`false`)

**レスポンス例:**
```json
{
  "id": 1,
  "jra_race_id": "2026010101011",
  "race_name": "○○ステークス",
  "race_number": 11,
  "race_date": "2026-01-01",
  "track": {
    "id": 1,
    "name": "東京",
    "location": "東京都"
  },
  "distance": 1600,
  "surface": "芝",
  "weather": "晴",
  "track_condition": "良",
  "status": "completed",
  "prize_money": 100000000,
  "race_class": "G1"
}
```

`include_entries=true`を指定した場合、`entries`配列が含まれます:
```json
{
  "entries": [
    {
      "id": 1,
      "horse_number": 1,
      "post_position": 1,
      "horse": {
        "id": 1,
        "name": "サンプルホース",
        "jra_horse_id": "2020000001"
      },
      "jockey": {
        "id": 1,
        "name": "○○騎手",
        "jra_jockey_id": "01234"
      },
      "trainer": {
        "id": 1,
        "name": "△△調教師",
        "jra_trainer_id": "01234"
      },
      "weight": 54.0,
      "horse_weight": 480,
      "morning_odds": 2.5
    }
  ]
}
```

---

## 馬 (Horses)

### GET /api/horses

馬の一覧を取得します。

**パラメータ:**
- `search` (optional): 馬名検索
- `sort` (optional): ソート項目 (`name`, `race_count`)
- `order` (optional): ソート順 (`asc`, `desc`)
- `page`, `per_page`: ページネーション

**レスポンス例:**
```json
{
  "horses": [
    {
      "id": 1,
      "jra_horse_id": "2020000001",
      "name": "サンプルホース",
      "birth_date": "2020-03-15",
      "sex": "牡"
    }
  ],
  "meta": {...}
}
```

### GET /api/horses/{id}

特定の馬の詳細情報を取得します。

**パラメータ:**
- `include_stats` (optional): 統計情報を含める (`true`/`false`)
- `include_races` (optional): 最近のレース情報を含める (`true`/`false`)
- `race_limit` (optional): 含めるレース数 (デフォルト: 10)

**レスポンス例 (include_stats=true):**
```json
{
  "id": 1,
  "jra_horse_id": "2020000001",
  "name": "サンプルホース",
  "birth_date": "2020-03-15",
  "sex": "牡",
  "trainer_id": 1,
  "statistics": {
    "total_races": 15,
    "total_results": 15,
    "wins": 5,
    "places": 10,
    "win_rate": 0.333,
    "place_rate": 0.667
  }
}
```

---

## 騎手 (Jockeys)

### GET /api/jockeys

騎手の一覧を取得します。

**パラメータ:**
- `search` (optional): 騎手名検索
- `sort` (optional): ソート項目 (`name`, `race_count`)
- `order` (optional): ソート順 (`asc`, `desc`)
- `page`, `per_page`: ページネーション

**レスポンス例:**
```json
{
  "jockeys": [
    {
      "id": 1,
      "jra_jockey_id": "01234",
      "name": "○○騎手"
    }
  ],
  "meta": {...}
}
```

### GET /api/jockeys/{id}

特定の騎手の詳細情報を取得します。

**パラメータ:**
- `include_stats` (optional): 統計情報を含める (`true`/`false`)
- `include_races` (optional): 最近のレース情報を含める (`true`/`false`)
- `race_limit` (optional): 含めるレース数 (デフォルト: 10)

**レスポンス例 (include_stats=true):**
```json
{
  "id": 1,
  "jra_jockey_id": "01234",
  "name": "○○騎手",
  "statistics": {
    "total_races": 500,
    "total_results": 480,
    "wins": 100,
    "places": 250,
    "win_rate": 0.208,
    "place_rate": 0.521
  }
}
```

---

## 調教師 (Trainers)

### GET /api/trainers

調教師の一覧を取得します。

**パラメータ:**
- `search` (optional): 調教師名検索
- `sort` (optional): ソート項目 (`name`, `race_count`)
- `order` (optional): ソート順 (`asc`, `desc`)
- `page`, `per_page`: ページネーション

**レスポンス例:**
```json
{
  "trainers": [
    {
      "id": 1,
      "jra_trainer_id": "01234",
      "name": "△△調教師",
      "stable": "美浦"
    }
  ],
  "meta": {...}
}
```

### GET /api/trainers/{id}

特定の調教師の詳細情報を取得します。

**パラメータ:**
- `include_stats` (optional): 統計情報を含める (`true`/`false`)
- `include_races` (optional): 最近のレース情報を含める (`true`/`false`)
- `race_limit` (optional): 含めるレース数 (デフォルト: 10)

**レスポンス例 (include_stats=true):**
```json
{
  "id": 1,
  "jra_trainer_id": "01234",
  "name": "△△調教師",
  "stable": "美浦",
  "statistics": {
    "total_races": 300,
    "total_results": 290,
    "wins": 60,
    "places": 150,
    "win_rate": 0.207,
    "place_rate": 0.517
  }
}
```

---

## 予想 (Predictions)

### GET /api/predictions

予想情報の一覧を取得します。

**パラメータ:**
- `race_id` (optional): レースIDフィルター
- `model_name` (optional): モデル名フィルター
- `page`, `per_page`: ページネーション

**レスポンス例:**
```json
{
  "predictions": [
    {
      "id": 1,
      "race_id": 1,
      "race_name": "○○ステークス",
      "horse_id": 1,
      "horse_name": "サンプルホース",
      "predicted_position": 1,
      "win_probability": 0.45,
      "confidence_score": 0.82,
      "model_name": "xgboost_v1",
      "created_at": "2026-01-01T12:00:00"
    }
  ],
  "meta": {...}
}
```

### GET /api/predictions/race/{race_id}

特定のレースの予想情報を取得します。

**レスポンス例:**
```json
{
  "race": {
    "id": 1,
    "race_name": "○○ステークス",
    "race_date": "2026-01-01",
    "track_name": "東京"
  },
  "predictions": [
    {
      "horse_id": 1,
      "horse_name": "サンプルホース",
      "horse_number": 1,
      "jockey_name": "○○騎手",
      "predicted_position": 1,
      "win_probability": 0.45,
      "confidence_score": 0.82,
      "model_name": "xgboost_v1",
      "morning_odds": 2.5
    }
  ]
}
```

---

## 競馬場 (Tracks)

### GET /api/tracks

競馬場の一覧を取得します。

**レスポンス例:**
```json
{
  "tracks": [
    {
      "id": 1,
      "name": "東京",
      "location": "東京都",
      "surface_types": ["turf", "dirt"]
    }
  ],
  "meta": {
    "total_items": 10
  }
}
```

### GET /api/tracks/{id}

特定の競馬場の詳細情報を取得します。

**パラメータ:**
- `include_races` (optional): 最近のレース情報を含める (`true`/`false`)
- `race_limit` (optional): 含めるレース数 (デフォルト: 10)

**レスポンス例 (include_races=true):**
```json
{
  "id": 1,
  "name": "東京",
  "location": "東京都",
  "surface_types": ["turf", "dirt"],
  "recent_races": [
    {
      "id": 1,
      "race_name": "○○ステークス",
      "race_date": "2026-01-01",
      "race_number": 11,
      "distance": 1600,
      "surface": "芝"
    }
  ]
}
```

---

## 使用例

### cURL

```bash
# レース一覧を取得
curl http://localhost:5000/api/races

# 特定日のレースを取得
curl "http://localhost:5000/api/races?date=2026-01-01"

# レース詳細と出走馬情報を取得
curl "http://localhost:5000/api/races/1?include_entries=true"

# 馬の詳細と統計を取得
curl "http://localhost:5000/api/horses/1?include_stats=true"

# 特定レースの予想を取得
curl http://localhost:5000/api/predictions/race/1
```

### Python (requests)

```python
import requests

# レース一覧を取得
response = requests.get('http://localhost:5000/api/races')
races = response.json()

# 出走馬情報を含むレース詳細を取得
response = requests.get(
    'http://localhost:5000/api/races/1',
    params={'include_entries': 'true', 'include_results': 'true'}
)
race_detail = response.json()

# 馬の検索
response = requests.get(
    'http://localhost:5000/api/horses',
    params={'search': 'サンプル', 'page': 1, 'per_page': 20}
)
horses = response.json()

# レースの予想を取得
response = requests.get('http://localhost:5000/api/predictions/race/1')
predictions = response.json()
```

### JavaScript (fetch)

```javascript
// レース一覧を取得
fetch('http://localhost:5000/api/races')
  .then(response => response.json())
  .then(data => console.log(data));

// レース詳細と予想を取得
fetch('http://localhost:5000/api/races/1?include_predictions=true')
  .then(response => response.json())
  .then(data => console.log(data));

// 馬の統計を取得
fetch('http://localhost:5000/api/horses/1?include_stats=true&include_races=true')
  .then(response => response.json())
  .then(data => console.log(data));
```

---

## 注意事項

- すべてのエンドポイントはGETメソッドのみサポートしています（読み取り専用API）
- データの作成・更新・削除はスクレイピングスクリプト経由で行います
- 大量のリクエストを送信する場合は、適切なレート制限を設けてください
- 本APIで取得したデータはJRAのデータに基づいており、商用利用には制限があります
