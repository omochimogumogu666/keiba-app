# 予想一覧画面の改善設計

**作成日**: 2026-01-29
**目的**: 予想一覧画面のレース名表示を目立たせ、ソート機能を追加してユーザビリティを向上

## 要件

### 機能要件
1. レース名を視覚的に目立たせる
2. 日付、競馬場名、レース番号でソート可能にする
3. サーバー側（SQL）でソートを実行

### 非機能要件
- ページ遷移時のソート条件を保持
- 既存の予想データ表示機能に影響を与えない
- レスポンシブデザインを維持

## UI/UX改善

### レース名の強調表示

**現在の表示**:
```
東京 1R - 3歳未勝利
```

**変更後**:
```
東京 1R
3歳未勝利  ← 大きく目立つフォント
```

**実装方針**:
- レース名を独立した行に配置
- `<h4>`タグと`fw-bold`クラスで大きく太字表示
- 競馬場名とレース番号は`<small>`タグでサブタイトルとして上部配置
- レース名がない場合は`race_class`（例：2勝クラス）をフォールバック表示

## ソート機能

### URL設計

クエリパラメータでソート条件を指定：

```
/predictions/                    # デフォルト（日付降順）
/predictions/?sort=date_asc      # 日付昇順
/predictions/?sort=date_desc     # 日付降順
/predictions/?sort=track         # 競馬場名
/predictions/?sort=race_number   # レース番号
```

### バックエンド実装

**ファイル**: `src/web/routes/predictions.py`

**変更内容**:

```python
@predictions_bp.route('/')
def predictions_index():
    """Predictions index page with sorting."""
    # クエリパラメータからソート条件を取得（デフォルト: 日付降順）
    sort_by = request.args.get('sort', 'date_desc')

    # ベースクエリ
    query = db.session.query(Race).join(Prediction).distinct()

    # ソート条件を適用
    if sort_by == 'date_asc':
        query = query.order_by(Race.race_date.asc(), Race.race_number.asc())
    elif sort_by == 'date_desc':
        query = query.order_by(Race.race_date.desc(), Race.race_number.asc())
    elif sort_by == 'track':
        query = query.join(Race.track).order_by(Track.name, Race.race_date.desc())
    elif sort_by == 'race_number':
        query = query.order_by(Race.race_number.asc(), Race.race_date.desc())
    else:
        # 不正なソート条件の場合はデフォルトにフォールバック
        query = query.order_by(Race.race_date.desc(), Race.race_number.asc())

    races_with_predictions = query.all()

    return render_template(
        'predictions/index.html',
        races=races_with_predictions,
        current_sort=sort_by  # テンプレートに現在のソート条件を渡す
    )
```

**注意点**:
- `sort=track`の場合、`Track`テーブルとJOINが必要
- セカンダリソート条件を設定して一意な並び順を保証
- 不正なソート条件はデフォルトにフォールバック

### フロントエンド実装

**ファイル**: `src/web/templates/predictions/index.html`

**ソートUI**:

予想一覧の上部にボタングループを配置：

```html
<div class="d-flex justify-content-between align-items-center mb-3">
    <div>
        <label class="me-2">並び替え:</label>
        <div class="btn-group" role="group" aria-label="Sort options">
            <a href="?sort=date_desc"
               class="btn btn-sm btn-outline-primary {% if current_sort == 'date_desc' %}active{% endif %}">
                <i class="bi bi-calendar-event"></i> 日付（新しい順）
            </a>
            <a href="?sort=date_asc"
               class="btn btn-sm btn-outline-primary {% if current_sort == 'date_asc' %}active{% endif %}">
                <i class="bi bi-calendar-event"></i> 日付（古い順）
            </a>
            <a href="?sort=track"
               class="btn btn-sm btn-outline-primary {% if current_sort == 'track' %}active{% endif %}">
                <i class="bi bi-geo-alt"></i> 競馬場
            </a>
            <a href="?sort=race_number"
               class="btn btn-sm btn-outline-primary {% if current_sort == 'race_number' %}active{% endif %}">
                <i class="bi bi-123"></i> レース番号
            </a>
        </div>
    </div>
</div>
```

**レース表示の改善**:

```html
<a href="{{ url_for('predictions.race_predictions', race_id=race.id) }}"
   class="list-group-item list-group-item-action">
    <div class="d-flex w-100 justify-content-between align-items-start">
        <div class="flex-grow-1">
            <!-- 競馬場とレース番号（小さめのサブタイトル） -->
            <small class="text-muted d-block mb-1">
                <i class="bi bi-geo-alt-fill"></i>
                {{ race.track.name if race.track else '競馬場' }}
                {{ race.race_number }}R
            </small>

            <!-- レース名（大きく目立つ） -->
            <h4 class="mb-2 fw-bold text-dark">
                {% if race.race_name %}
                    {{ race.race_name }}
                {% else %}
                    {{ race.race_class if race.race_class else 'レース名未設定' }}
                {% endif %}
            </h4>

            <!-- バッジ情報 -->
            <p class="mb-1">
                <span class="badge bg-secondary">{{ race.surface }}</span>
                <span class="badge bg-info">{{ race.distance }}m</span>
                {% if race.race_class %}
                    <span class="badge bg-warning text-dark">{{ race.race_class }}</span>
                {% endif %}
                <span class="badge bg-success">予想あり</span>
            </p>
        </div>

        <!-- 日付（右寄せ） -->
        <small class="text-muted ms-3">
            <i class="bi bi-calendar3"></i>
            {{ race.race_date.strftime('%Y/%m/%d') if race.race_date else 'TBD' }}
        </small>
    </div>
</a>
```

## 実装ファイル

### 変更ファイル
1. `src/web/routes/predictions.py` - ソートロジック追加
2. `src/web/templates/predictions/index.html` - UI改善

### テストファイル
- `tests/test_web/test_predictions_routes.py` - ソート機能のテストケース追加

## 実装ステップ

1. バックエンドのソート機能実装
   - `predictions_index()`関数を修正
   - クエリパラメータのバリデーション
   - 各ソート条件のSQL実装

2. フロントエンドUI実装
   - ソートボタングループの追加
   - レース表示レイアウトの改善
   - アクティブ状態の表示

3. テスト実装
   - 各ソート条件のテストケース
   - エッジケース（不正なソート条件など）

4. 動作確認
   - 各ソート条件が正しく動作するか
   - レース名の表示が目立っているか
   - レスポンシブデザインが崩れていないか

## 懸念事項とリスク

### パフォーマンス
- `sort=track`の場合、追加のJOINが発生
- 予想データが大量にある場合、クエリが遅くなる可能性
- **対策**: 必要に応じてインデックスを追加

### データの欠損
- `track`が`None`の場合、ソートが正しく動作しない可能性
- **対策**: NULLを最後に配置するか、'不明'として扱う

### ブラウザの戻るボタン
- ソート条件がクエリパラメータなので、ブラウザの戻るボタンで正しく戻れる
- 特別な対応は不要

## 今後の拡張案

- ソート条件を複数組み合わせる（例：競馬場 → 日付 → レース番号）
- フィルタ機能の追加（競馬場、クラス、距離など）
- ページネーションの追加（データが多い場合）
- お気に入り機能（特定のレースをピン留め）
