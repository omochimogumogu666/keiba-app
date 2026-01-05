# Docker環境セットアップガイド

このガイドでは、Docker環境でのkeiba-appのセットアップとデータベースマイグレーションの実施方法を説明します。

## 前提条件

- Docker Desktop (Windows) または Docker Engine (Linux/Mac) がインストールされていること
- Docker Compose が利用可能であること

## クイックスタート

### 1. Docker環境の起動

```bash
# Docker Composeでサービスを起動
docker compose up -d

# ログを確認
docker compose logs -f
```

これにより以下のサービスが起動します:
- `db`: PostgreSQL 15データベース
- `web`: Flaskアプリケーション

### 2. データベースマイグレーションの実施

#### 方法A: 自動マイグレーション（推奨）

```bash
# コンテナ内でマイグレーションスクリプトを実行
docker compose exec web python scripts/init_docker_db.py
```

このスクリプトは以下を自動的に実行します:
- データベース接続の待機
- マイグレーションの実行
- サンプル競馬場データの追加

#### 方法B: 手動マイグレーション

```bash
# Flask-Migrateコマンドを使用
docker compose exec web flask db upgrade

# サンプルデータを個別に追加する場合
docker compose exec web python scripts/init_db.py
```

### 3. アプリケーションへのアクセス

ブラウザで以下のURLにアクセス:
```
http://localhost:5000
```

## データベース管理

### マイグレーションファイルの作成

モデルを変更した場合、新しいマイグレーションファイルを作成:

```bash
# コンテナ内でマイグレーションを生成
docker compose exec web flask db migrate -m "Add new column to horses table"
```

### マイグレーションの適用

```bash
docker compose exec web flask db upgrade
```

### マイグレーションのロールバック

```bash
# 1つ前のバージョンに戻す
docker compose exec web flask db downgrade

# 特定のリビジョンに戻す
docker compose exec web flask db downgrade <revision_id>
```

### マイグレーション履歴の確認

```bash
docker compose exec web flask db history
```

## データベース直接アクセス

PostgreSQLに直接接続する場合:

```bash
# psqlを使用して接続
docker compose exec db psql -U keiba_user -d keiba_db

# またはホストから接続
psql -h localhost -p 5432 -U keiba_user -d keiba_db
```

デフォルトのパスワード: `keiba_password`

## トラブルシューティング

### データベース接続エラー

```bash
# データベースコンテナのログを確認
docker compose logs db

# データベースが起動しているか確認
docker compose ps
```

### マイグレーションエラー

```bash
# マイグレーションの状態を確認
docker compose exec web flask db current

# マイグレーションをリセット（開発環境のみ）
docker compose down -v  # ボリュームも削除
docker compose up -d
docker compose exec web python scripts/init_docker_db.py
```

### コンテナの完全リセット

```bash
# すべてのコンテナとボリュームを削除
docker compose down -v

# イメージを再ビルド
docker compose build --no-cache

# 再起動
docker compose up -d
```

## 環境設定

### 環境変数のカスタマイズ

`docker-compose.yml`を編集して環境変数を変更できます:

```yaml
services:
  db:
    environment:
      POSTGRES_USER: your_user
      POSTGRES_PASSWORD: your_password
      POSTGRES_DB: your_database

  web:
    environment:
      - DATABASE_URL=postgresql://your_user:your_password@db:5432/your_database
```

### 本番環境への移行

本番環境では、環境変数を`.env`ファイルで管理:

```bash
# .env.production を作成
cat > .env.production << EOF
FLASK_ENV=production
DATABASE_URL=postgresql://user:password@db:5432/keiba_db
SECRET_KEY=your-secret-key-here
SCRAPING_DELAY=3
EOF

# docker-compose.ymlで読み込み
docker compose --env-file .env.production up -d
```

## 開発ワークフロー

### コードの変更を反映

ボリュームマウントにより、ローカルの変更は自動的にコンテナに反映されます:

```bash
# アプリケーションを再起動（必要な場合）
docker compose restart web
```

### テストの実行

```bash
# コンテナ内でpytestを実行
docker compose exec web pytest

# カバレッジレポート付き
docker compose exec web pytest --cov=src --cov-report=html
```

### シェルアクセス

```bash
# Webコンテナのシェルに入る
docker compose exec web bash

# データベースコンテナのシェルに入る
docker compose exec db bash
```

## クリーンアップ

```bash
# コンテナを停止
docker compose down

# ボリュームも含めて完全削除
docker compose down -v

# イメージも削除
docker compose down --rmi all -v
```

## 参考資料

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Flask-Migrate Documentation](https://flask-migrate.readthedocs.io/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
