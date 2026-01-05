# モデル定期再訓練ガイド

このドキュメントでは、JRA競馬予想アプリケーションのモデル定期再訓練機能について説明します。

## 概要

モデルの定期再訓練機能は、最新のレースデータを使用して機械学習モデルを自動的に再訓練し、予想精度を維持・向上させます。

### 主な機能

1. **自動データ取得**: データベースから最新のレースデータを自動的に抽出
2. **モデルバージョニング**: 訓練済みモデルを自動的にバージョン管理
3. **性能比較**: 新モデルと前回モデルのパフォーマンスを自動比較
4. **自動デプロイ**: 性能が改善した場合のみ新モデルをアクティブ化
5. **モデル履歴管理**: 最新N個のモデルバージョンを保持
6. **通知機能**: 再訓練完了時にEmail/Slack通知（オプション）

## クイックスタート

### 1. 今すぐ再訓練を実行

```bash
python scripts/model_retraining_scheduler.py --mode once
```

このコマンドは、設定ファイルに基づいてすべてのモデルを再訓練します。

### 2. 定期再訓練のスケジュール設定

```bash
# 毎週月曜2時に自動再訓練（推奨）
python scripts/model_retraining_scheduler.py --mode schedule --interval weekly --time 02:00

# 毎日2時に自動再訓練
python scripts/model_retraining_scheduler.py --mode schedule --interval daily --time 02:00

# 毎月1日2時に自動再訓練
python scripts/model_retraining_scheduler.py --mode schedule --interval monthly --time 02:00
```

## 設定ファイル

再訓練の設定は `config/retraining_config.json` で管理します。

### デフォルト設定

```json
{
  "models_to_train": ["random_forest", "xgboost"],
  "task": "regression",
  "training_window_days": 365,
  "validation_split": 0.15,
  "test_split": 0.15,
  "min_samples": 1000,
  "performance_threshold": 0.05,
  "keep_last_n_models": 5,
  "models_dir": "data/models",
  "notification": {
    "enabled": false,
    "email_recipients": [],
    "smtp_server": "",
    "smtp_port": 587,
    "smtp_user": "",
    "smtp_password": ""
  }
}
```

### 設定パラメータの説明

| パラメータ | 説明 | デフォルト値 |
|-----------|------|-------------|
| `models_to_train` | 訓練するモデルのリスト | `["random_forest", "xgboost"]` |
| `task` | タスクタイプ（`regression`または`classification`） | `"regression"` |
| `training_window_days` | 訓練データの期間（日数） | `365` |
| `validation_split` | 検証データの割合 | `0.15` (15%) |
| `test_split` | テストデータの割合 | `0.15` (15%) |
| `min_samples` | 訓練に必要な最小サンプル数 | `1000` |
| `performance_threshold` | デプロイに必要な最小改善率 | `0.05` (5%) |
| `keep_last_n_models` | 保持するモデルバージョン数 | `5` |
| `models_dir` | モデル保存先ディレクトリ | `"data/models"` |

## モデルバージョニング

### モデルレジストリ

再訓練されたすべてのモデルは `data/models/model_registry.json` に記録されます。

**レジストリの内容:**
```json
{
  "models": [
    {
      "model_type": "random_forest",
      "task": "regression",
      "timestamp": "20260104_120000",
      "filename": "random_forest_regression_20260104_120000.pkl",
      "path": "data/models/random_forest_regression_20260104_120000.pkl",
      "train_samples": 5000,
      "test_samples": 1200,
      "training_time_seconds": 45.2,
      "train_metrics": {
        "rmse": 2.15,
        "mae": 1.68,
        "r2_score": 0.52
      },
      "test_metrics": {
        "rmse": 2.23,
        "mae": 1.75,
        "r2_score": 0.48
      },
      "comparison": {
        "has_previous": true,
        "previous_model": "random_forest_regression_20251228_120000.pkl",
        "previous_rmse": 2.35,
        "new_rmse": 2.23,
        "improvement": 0.051,
        "should_deploy": true
      },
      "is_active": true,
      "created_at": "2026-01-04T12:00:00"
    }
  ]
}
```

### モデルのデプロイ判定

新しいモデルは以下の条件を満たす場合にのみアクティブ化されます:

1. **初回モデル**: 前回のモデルが存在しない場合
2. **性能改善**: 前回のモデルと比較して、設定された閾値以上の改善が見られる場合

**改善率の計算:**
```
improvement = (previous_rmse - new_rmse) / previous_rmse
```

デフォルトでは、`performance_threshold = 0.05`（5%）以上の改善が必要です。

### モデルのクリーンアップ

古いモデルは自動的に削除されます:
- 各モデルタイプで最新N個のバージョンを保持（デフォルト: 5）
- アクティブ/非アクティブに関わらず、タイムスタンプ順で保持
- 削除されたモデルはレジストリからも削除されます

## 再訓練ワークフロー

### 1. データ抽出

```python
# データベースから最新データを抽出
start_date = now - training_window_days
X, y = extract_features_for_training(min_date=start_date, max_date=now)
```

### 2. データ前処理

```python
# 欠損値処理
X = handle_missing_values(X, strategy='median')

# 時系列分割
train_df, test_df, val_df = split_by_date(
    X,
    train_end_date,
    val_end_date
)

# スケーリング
preprocessor = FeaturePreprocessor()
X_train_scaled = preprocessor.fit_transform(X_train)
X_val_scaled = preprocessor.transform(X_val)
X_test_scaled = preprocessor.transform(X_test)
```

### 3. モデル訓練

```python
# RandomForestとXGBoostを訓練
for model_type in ['random_forest', 'xgboost']:
    model = ModelClass(task='regression')
    model.train(X_train_scaled, y_train, X_val_scaled, y_val)

    # テストセットで評価
    y_pred = model.predict(X_test_scaled)
    metrics = evaluate_regression_model(y_test, y_pred, model_type)
```

### 4. 性能比較とデプロイ判定

```python
# 前回モデルと比較
comparison = compare_with_previous(new_model_info)

if comparison['improvement'] >= performance_threshold:
    # 新モデルをアクティブ化
    update_registry(new_model_info, is_active=True)
else:
    # 前モデルを維持
    update_registry(new_model_info, is_active=False)
```

### 5. 通知送信（オプション）

```python
if notification_enabled:
    send_notification(config, results, status='success')
```

## 通知機能

### Email通知の設定

`config/retraining_config.json` で設定:

```json
{
  "notification": {
    "enabled": true,
    "email_recipients": ["your@email.com"],
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "smtp_user": "your@email.com",
    "smtp_password": "your-app-password"
  }
}
```

**Gmail設定例:**
1. Gmailのアプリパスワードを生成（2段階認証が必要）
2. `smtp_server`: `smtp.gmail.com`
3. `smtp_port`: `587`
4. `smtp_user`: Gmailアドレス
5. `smtp_password`: アプリパスワード

### Slack通知の設定

```json
{
  "notification": {
    "enabled": true,
    "slack_webhook_url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
  }
}
```

**Slack Webhook URLの取得:**
1. Slack Appを作成
2. Incoming Webhooksを有効化
3. Webhook URLをコピー

## 運用のベストプラクティス

### 推奨スケジュール

- **週次再訓練**: 毎週月曜日の深夜に実行（推奨）
  - 週末のレース結果を反映
  - データが十分に蓄積
  - 計算リソースの競合を回避

```bash
python scripts/model_retraining_scheduler.py --mode schedule --interval weekly --time 02:00
```

- **月次再訓練**: データ量が少ない場合
- **日次再訓練**: データ量が豊富で頻繁な更新が必要な場合

### データ量の確認

再訓練前にデータベースのレコード数を確認:

```bash
# データベース内のレース数を確認
python -c "
from src.web.app import create_app
from src.data.models import db, Race

app = create_app()
with app.app_context():
    count = db.session.query(Race).count()
    print(f'Total races: {count}')
"
```

最低でも1000レース以上のデータが推奨されます。

### パフォーマンスモニタリング

モデルレジストリを定期的に確認し、以下を監視:

1. **改善トレンド**: 新モデルが継続的に改善しているか
2. **訓練時間**: 訓練時間が異常に長くなっていないか
3. **サンプル数**: 十分なデータで訓練されているか

```bash
# レジストリの確認
cat data/models/model_registry.json | python -m json.tool
```

### トラブルシューティング

**問題: 再訓練が失敗する**
- ログファイルを確認: `logs/app.log`
- データベース接続を確認
- 十分なディスク空き容量があるか確認

**問題: 新モデルが常にデプロイされない**
- `performance_threshold`を調整（例: `0.02`に下げる）
- 前回モデルの性能を確認
- 訓練データの質を確認

**問題: 訓練時間が長すぎる**
- `training_window_days`を短くする（例: 180日）
- モデルのハイパーパラメータを調整
- サーバースペックをアップグレード

## コマンドラインオプション

```bash
python scripts/model_retraining_scheduler.py --help
```

### 主なオプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--mode` | 実行モード（`once`または`schedule`） | `once` |
| `--interval` | スケジュール間隔（`daily`, `weekly`, `monthly`） | `weekly` |
| `--time` | 実行時刻（HH:MM形式） | `02:00` |
| `--config` | 設定ファイルパス | `config/retraining_config.json` |

## サンプル実行例

### 開発環境でテスト実行

```bash
# 設定ファイルをコピー
cp config/retraining_config.json config/retraining_config_test.json

# 設定を調整（訓練期間を短く、通知を無効化）
# {
#   "training_window_days": 90,
#   "min_samples": 100,
#   "notification": { "enabled": false }
# }

# テスト実行
python scripts/model_retraining_scheduler.py --mode once --config config/retraining_config_test.json
```

### 本番環境でスケジュール実行

```bash
# スケジューラーをバックグラウンドで起動（Linux/Mac）
nohup python scripts/model_retraining_scheduler.py --mode schedule --interval weekly --time 02:00 > logs/retraining.log 2>&1 &

# Windows（PowerShellでバックグラウンド実行）
Start-Process python -ArgumentList "scripts/model_retraining_scheduler.py --mode schedule --interval weekly --time 02:00" -WindowStyle Hidden
```

### systemdサービス化（Linux）

```ini
# /etc/systemd/system/keiba-retraining.service
[Unit]
Description=Keiba Model Retraining Scheduler
After=network.target

[Service]
Type=simple
User=keiba
WorkingDirectory=/path/to/keiba-app
Environment="PATH=/path/to/keiba-app/venv/bin"
ExecStart=/path/to/keiba-app/venv/bin/python scripts/model_retraining_scheduler.py --mode schedule --interval weekly --time 02:00
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# サービス有効化
sudo systemctl enable keiba-retraining
sudo systemctl start keiba-retraining

# ステータス確認
sudo systemctl status keiba-retraining
```

## まとめ

モデルの定期再訓練機能により、以下が実現されます:

✅ **自動化**: 人手を介さずにモデルを最新状態に保つ
✅ **品質管理**: 性能が改善した場合のみデプロイ
✅ **履歴管理**: すべてのモデルバージョンを追跡
✅ **通知**: 再訓練完了時に自動通知
✅ **柔軟性**: 設定ファイルで簡単にカスタマイズ

これにより、予想精度を継続的に維持・向上させることができます。
