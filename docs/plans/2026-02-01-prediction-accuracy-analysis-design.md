# 予想精度分析機能 設計書

**作成日**: 2026-02-01
**ステータス**: 承認済み

## 概要

予測モデルのパフォーマンスを定量的に評価するための分析機能。的中率、複勝圏内率、ROI（投資回収率）を時系列・モデル別に可視化し、モデル改善の指針を提供する。

## 要件

### 分析指標

1. **的中率（Win Accuracy）**
   - 1着予想が実際に1着になった割合
   - 計算式: `的中数 / 総予想数 × 100`

2. **複勝圏内率（Top-3 Accuracy）**
   - 予想上位3頭が3着以内に入った割合
   - 計算式: `圏内的中数 / (総予想数 × 3) × 100`

3. **回収率（ROI: Return on Investment）**
   - 予想に基づいて馬券を購入した場合の投資回収率
   - 計算式: `(総払戻額 - 総投資額) / 総投資額 × 100`
   - 前提: 1着予想馬に単勝100円を購入

### 分析軸

1. **時系列分析**
   - 日別・週別・月別の精度推移
   - モデルの成長・劣化を追跡

2. **モデル別比較**
   - XGBoost vs Random Forest
   - 各モデルの得意不得意を特定

## データモデル

### 新規テーブル: `prediction_accuracy`

```sql
CREATE TABLE prediction_accuracy (
    id INTEGER PRIMARY KEY,

    -- 集計軸
    aggregation_type VARCHAR(20) NOT NULL,  -- 'daily', 'weekly', 'monthly'
    aggregation_date DATE NOT NULL,         -- 集計期間の開始日
    model_name VARCHAR(50) NOT NULL,
    track_id INTEGER,                       -- NULL = 全競馬場
    race_class VARCHAR(50),                 -- NULL = 全クラス

    -- 基本統計
    total_races INTEGER DEFAULT 0,
    total_predictions INTEGER DEFAULT 0,

    -- 的中率
    win_predictions INTEGER DEFAULT 0,
    win_hits INTEGER DEFAULT 0,
    win_accuracy FLOAT,

    -- 複勝圏内率
    top3_predictions INTEGER DEFAULT 0,
    top3_hits INTEGER DEFAULT 0,
    top3_accuracy FLOAT,

    -- ROI
    total_bet_amount FLOAT DEFAULT 0.0,
    total_return_amount FLOAT DEFAULT 0.0,
    roi FLOAT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP,

    UNIQUE(aggregation_type, aggregation_date, model_name, track_id, race_class)
);

CREATE INDEX idx_accuracy_date ON prediction_accuracy(aggregation_date);
CREATE INDEX idx_accuracy_model ON prediction_accuracy(model_name);
```

### SQLAlchemy ORM モデル

```python
class PredictionAccuracy(db.Model):
    __tablename__ = 'prediction_accuracy'

    id = db.Column(db.Integer, primary_key=True)
    aggregation_type = db.Column(db.String(20), nullable=False)
    aggregation_date = db.Column(db.Date, nullable=False)
    model_name = db.Column(db.String(50), nullable=False)
    track_id = db.Column(db.Integer, db.ForeignKey('tracks.id'))
    race_class = db.Column(db.String(50))

    total_races = db.Column(db.Integer, default=0)
    total_predictions = db.Column(db.Integer, default=0)

    win_predictions = db.Column(db.Integer, default=0)
    win_hits = db.Column(db.Integer, default=0)
    win_accuracy = db.Column(db.Float)

    top3_predictions = db.Column(db.Integer, default=0)
    top3_hits = db.Column(db.Integer, default=0)
    top3_accuracy = db.Column(db.Float)

    total_bet_amount = db.Column(db.Float, default=0.0)
    total_return_amount = db.Column(db.Float, default=0.0)
    roi = db.Column(db.Float)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, onupdate=datetime.utcnow)

    track = db.relationship('Track', backref='accuracy_stats')
```

## アーキテクチャ

### 1. バッチ集計スクリプト

**ファイル**: `scripts/calculate_prediction_accuracy.py`

```python
def aggregate_accuracy(
    start_date: date,
    end_date: date,
    aggregation_types: list = ['daily', 'weekly', 'monthly'],
    models: list = None
) -> None:
    """
    予想精度を集計してDBに保存

    処理フロー:
    1. 完了済みレース（status='finished'）を抽出
    2. 予想と結果をJOINで一括取得（N+1回避）
    3. 各集計軸で指標を計算
    4. PredictionAccuracyテーブルに保存（既存データは更新）
    """
```

**最適化されたクエリ**:

```python
query = db.session.query(
    Prediction,
    RaceResult,
    Race,
    Track
).join(
    RaceResult,
    (Prediction.race_id == RaceResult.race_id) &
    (Prediction.horse_id == RaceResult.horse_id)
).join(
    Race, Prediction.race_id == Race.id
).join(
    Track, Race.track_id == Track.id
).filter(
    Race.status == 'finished',
    Race.race_date.between(start_date, end_date)
).all()
```

### 2. 精度計算ロジック

**ファイル**: `src/analysis/accuracy_calculator.py`

```python
def calculate_metrics(predictions_with_results: list) -> dict:
    """
    予想と結果のペアから精度指標を計算

    Args:
        predictions_with_results: [(Prediction, RaceResult, Race), ...]

    Returns:
        {
            'total_races': int,
            'total_predictions': int,
            'win_accuracy': float,
            'top3_accuracy': float,
            'roi': float,
            'by_model': {...},
            'by_track': {...}
        }
    """
    metrics = {
        'win_predictions': 0,
        'win_hits': 0,
        'top3_predictions': 0,
        'top3_hits': 0,
        'total_bet': 0.0,
        'total_return': 0.0
    }

    for pred, result, race in predictions_with_results:
        # 的中率
        if pred.predicted_position == 1:
            metrics['win_predictions'] += 1
            if result.final_position == 1:
                metrics['win_hits'] += 1

        # 複勝圏内率
        if pred.predicted_position <= 3:
            metrics['top3_predictions'] += 1
            if result.final_position <= 3:
                metrics['top3_hits'] += 1

        # ROI（1着予想馬に単勝100円）
        if pred.predicted_position == 1:
            bet = 100
            metrics['total_bet'] += bet
            if result.final_position == 1:
                metrics['total_return'] += bet * result.win_odds

    # 最終計算
    metrics['win_accuracy'] = (
        metrics['win_hits'] / metrics['win_predictions'] * 100
        if metrics['win_predictions'] > 0 else 0.0
    )
    metrics['top3_accuracy'] = (
        metrics['top3_hits'] / metrics['top3_predictions'] * 100
        if metrics['top3_predictions'] > 0 else 0.0
    )
    metrics['roi'] = (
        (metrics['total_return'] - metrics['total_bet']) / metrics['total_bet'] * 100
        if metrics['total_bet'] > 0 else 0.0
    )

    return metrics
```

### 3. Web UI

**ルート**: `src/web/routes/analysis.py`

```python
@analysis_bp.route('/accuracy')
def accuracy_dashboard():
    """精度分析ダッシュボード画面"""
    return render_template('analysis/accuracy.html')

@analysis_bp.route('/api/accuracy/summary')
def get_accuracy_summary():
    """
    集計サマリーAPIエンドポイント

    Query Parameters:
        - start_date (required): YYYY-MM-DD
        - end_date (required): YYYY-MM-DD
        - model_name (optional): モデル名フィルター
        - track_id (optional): 競馬場IDフィルター

    Response:
        {
            "period": {"start": "2026-01-01", "end": "2026-01-31"},
            "overall": {
                "total_races": 100,
                "win_accuracy": 25.5,
                "top3_accuracy": 62.3,
                "roi": -12.8
            },
            "by_model": [
                {
                    "model_name": "xgboost",
                    "win_accuracy": 26.2,
                    "top3_accuracy": 64.1,
                    "roi": -10.5
                },
                ...
            ]
        }
    """

@analysis_bp.route('/api/accuracy/timeseries')
def get_accuracy_timeseries():
    """
    時系列データAPIエンドポイント（Chart.js用）

    Query Parameters:
        - start_date, end_date, model_name
        - aggregation: 'daily' | 'weekly' | 'monthly'

    Response:
        {
            "labels": ["2026-01-01", "2026-01-08", ...],
            "datasets": [
                {
                    "label": "的中率",
                    "data": [25.0, 30.5, 22.3, ...]
                },
                {
                    "label": "ROI",
                    "data": [-15.2, -8.3, -20.1, ...]
                }
            ]
        }
    """
```

**テンプレート**: `src/web/templates/analysis/accuracy.html`

```html
<!-- サマリーカード -->
<div class="row">
    <div class="col-md-4">
        <div class="card">
            <div class="card-body">
                <h5>的中率</h5>
                <h2 id="win-accuracy">--</h2>
            </div>
        </div>
    </div>
    <div class="col-md-4">
        <div class="card">
            <div class="card-body">
                <h5>複勝圏内率</h5>
                <h2 id="top3-accuracy">--</h2>
            </div>
        </div>
    </div>
    <div class="col-md-4">
        <div class="card">
            <div class="card-body">
                <h5>ROI</h5>
                <h2 id="roi">--</h2>
            </div>
        </div>
    </div>
</div>

<!-- 時系列チャート -->
<canvas id="accuracyChart"></canvas>

<!-- モデル比較テーブル -->
<table class="table">
    <thead>
        <tr>
            <th>モデル</th>
            <th>的中率</th>
            <th>複勝圏内率</th>
            <th>ROI</th>
        </tr>
    </thead>
    <tbody id="model-comparison">
    </tbody>
</table>

<script>
// APIからデータ取得してグラフ描画
fetch('/api/accuracy/timeseries?start_date=2026-01-01&end_date=2026-01-31')
    .then(res => res.json())
    .then(data => {
        new Chart(document.getElementById('accuracyChart'), {
            type: 'line',
            data: data,
            options: { responsive: true }
        });
    });
</script>
```

## エラーハンドリング

### 1. データ不整合

```python
# 予想はあるが結果がないケース
if result is None:
    logger.warning(f"No result for prediction {pred.id} (race {pred.race_id})")
    continue  # スキップ
```

### 2. ゼロ除算

```python
accuracy = (hits / total * 100) if total > 0 else 0.0
roi = ((ret - bet) / bet * 100) if bet > 0 else 0.0
```

### 3. 再集計時の重複回避

```python
existing = PredictionAccuracy.query.filter_by(
    aggregation_type='daily',
    aggregation_date=date,
    model_name=model
).first()

if existing:
    existing.win_accuracy = new_value
    existing.updated_at = datetime.utcnow()
else:
    db.session.add(PredictionAccuracy(...))
```

## テスト戦略

### ユニットテスト

```python
# tests/test_analysis/test_accuracy_calculator.py

def test_win_accuracy_calculation():
    """的中率が正しく計算されるかテスト"""
    # 10予想中2的中 = 20%

def test_roi_calculation():
    """ROIが正しく計算されるかテスト"""
    # 1000円投資、700円払戻 = -30%

def test_zero_division_safety():
    """予想0件でもエラーにならないかテスト"""
```

### 統合テスト

```python
# tests/test_web/test_accuracy_api.py

def test_accuracy_api_response_format(client):
    """APIレスポンス形式のテスト"""

def test_accuracy_api_filters(client):
    """フィルターパラメータが機能するかテスト"""
```

## 実装順序

1. **Phase 1: データ層**
   - PredictionAccuracyモデル追加
   - マイグレーション実行

2. **Phase 2: 計算ロジック**
   - accuracy_calculator.py実装
   - ユニットテスト作成

3. **Phase 3: バッチスクリプト**
   - calculate_prediction_accuracy.py実装
   - テストデータで動作確認

4. **Phase 4: API**
   - /api/accuracy/* エンドポイント実装
   - APIテスト作成

5. **Phase 5: UI**
   - accuracy.htmlテンプレート作成
   - Chart.js統合
   - 手動テスト

## 運用

### 定期実行

```bash
# cron設定例（毎日23:00に前日分を集計）
0 23 * * * cd /path/to/keiba-app && python scripts/calculate_prediction_accuracy.py --yesterday
```

### 過去データの再集計

```bash
# 2026年1月全体を再集計
python scripts/calculate_prediction_accuracy.py --start 2026-01-01 --end 2026-01-31 --force
```

## パフォーマンス考慮

- **想定データ量**: 1日100レース × 16頭 × 365日 = 584,000予想/年
- **集計時間**: JOINクエリで一括取得により、数秒以内で完了見込み
- **キャッシュ**: 集計結果はDBに保存され、APIは事前計算済みデータを返すため高速

## 将来拡張

- 競馬場別・クラス別の詳細分析
- 信頼度スコア別の精度分析
- オッズ帯別のROI分析
- アンサンブルモデルの追加
