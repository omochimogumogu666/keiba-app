# JRA競馬予想アプリケーション

JRAの競馬データをスクレイピングし、機械学習モデルで予想を行うWebアプリケーション。

## 概要

このアプリケーションは以下の機能を提供します:
- JRAウェブサイトからのレースデータの自動収集
- 機械学習モデルによる勝馬予想
- Webインターフェースでの予想結果表示
- REST APIによるデータアクセス

## 技術スタック

- **バックエンド**: Python 3.9+, Flask
- **機械学習**: scikit-learn, XGBoost
- **データ処理**: pandas, numpy
- **スクレイピング**: BeautifulSoup4, Requests
- **データベース**: PostgreSQL (本番), SQLite (開発)
- **フロントエンド**: Jinja2, Bootstrap 5

## セットアップ

### 1. 仮想環境の作成

```bash
python -m venv venv
```

### 2. 仮想環境のアクティベート

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### 3. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 4. 環境変数の設定

`.env.example`を`.env`にコピーして、必要な環境変数を設定します。

```bash
cp .env.example .env
```

### 5. データベースの初期化

```bash
python scripts/init_db.py
```

## 使用方法

### データのスクレイピング

```bash
python scripts/scrape_data.py
```

### モデルの訓練

```bash
python scripts/train_model.py
```

### Webアプリケーションの起動

```bash
python -m src.web.app
```

ブラウザで `http://localhost:5000` にアクセス。

## プロジェクト構造

```
keiba-app/
├── config/              # 設定ファイル
├── src/
│   ├── scrapers/       # スクレイピングモジュール
│   ├── data/           # データモデルと前処理
│   ├── ml/             # 機械学習モデル
│   ├── web/            # Webアプリケーション
│   └── utils/          # ユーティリティ
├── scripts/            # 実行スクリプト
├── data/               # データファイル
├── tests/              # テスト
└── notebooks/          # Jupyter notebooks
```

## 注意事項

### 法的配慮
- JRAウェブサイトの利用規約を遵守してください
- スクレイピングは適切な間隔で実行してください（2-5秒の遅延）
- データの商用利用については事前にJRAに確認してください

### 免責事項
- このアプリケーションの予想結果は参考情報です
- 予想の正確性を保証するものではありません
- 実際の馬券購入は自己責任で行ってください

## ライセンス

MIT License

## 開発者向け

### テストの実行

```bash
pytest
```

### コードフォーマット

```bash
black .
```

### リント

```bash
flake8
```

## ロードマップ

- [x] プロジェクト構造の作成
- [ ] データスクレイパーの実装
- [ ] データベースモデルの実装
- [ ] 特徴量エンジニアリング
- [ ] 機械学習モデルの訓練
- [ ] Webインターフェースの実装
- [ ] APIエンドポイントの実装
- [ ] テストの実装
- [ ] デプロイ
