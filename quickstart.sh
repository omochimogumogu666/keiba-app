#!/bin/bash
# JRA競馬予想アプリ - クイックスタートスクリプト (Mac/Linux)

set -e

echo "================================================"
echo "JRA競馬予想アプリ - クイックスタート"
echo "================================================"
echo "このスクリプトは初期セットアップを自動化します"
echo "================================================"
echo ""

# カラー定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================
# [1/6] 必要なツールのチェック
# ============================================
echo "[1/6] 必要なツールをチェック中..."

# Dockerのチェック
if ! command -v docker &> /dev/null; then
    echo -e "${RED}[ERROR] Dockerがインストールされていません${NC}"
    echo "   Docker Desktopをインストールしてください: https://www.docker.com/products/docker-desktop"
    exit 1
fi
echo -e "${GREEN}[OK] Docker検出${NC}"

# Docker Composeのチェック
if ! docker compose version &> /dev/null; then
    echo -e "${RED}[ERROR] Docker Composeが利用できません${NC}"
    echo "   Docker Desktopを最新版に更新してください"
    exit 1
fi
echo -e "${GREEN}[OK] Docker Compose検出${NC}"

# Pythonのチェック（SECRET_KEY生成用）
PYTHON_AVAILABLE=0
if command -v python3 &> /dev/null; then
    echo -e "${GREEN}[OK] Python3検出${NC}"
    PYTHON_AVAILABLE=1
elif command -v python &> /dev/null; then
    echo -e "${GREEN}[OK] Python検出${NC}"
    PYTHON_AVAILABLE=1
else
    echo -e "${YELLOW}[WARN] Pythonが未検出 - SECRET_KEYの自動生成をスキップします${NC}"
    echo "   手動で.envファイルのSECRET_KEYを変更してください"
fi

echo ""

# ============================================
# [2/6] .envファイルのセットアップ
# ============================================
echo "[2/6] 環境変数ファイルをセットアップ中..."

if [ -f .env ]; then
    echo -e "${GREEN}[OK] .envファイルは既に存在します${NC}"
    echo "   既存の設定をそのまま使用します"
else
    if [ ! -f .env.personal.example ]; then
        if [ ! -f .env.example ]; then
            echo -e "${RED}[ERROR] .env.exampleファイルが見つかりません${NC}"
            echo "   リポジトリが正しくクローンされているか確認してください"
            exit 1
        fi
        cp .env.example .env
    else
        cp .env.personal.example .env
    fi

    if [ $PYTHON_AVAILABLE -eq 1 ]; then
        echo "[INFO] SECRET_KEYを自動生成しています..."
        if command -v python3 &> /dev/null; then
            SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
        else
            SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
        fi

        # macOS と Linux の sed の違いに対応
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s/your-secret-key-here-change-this/${SECRET_KEY}/" .env
        else
            sed -i "s/your-secret-key-here-change-this/${SECRET_KEY}/" .env
        fi
        echo -e "${GREEN}[OK] SECRET_KEYを生成しました${NC}"
    else
        echo -e "${YELLOW}[WARN] SECRET_KEYは未設定です${NC}"
        echo "   .envファイルを開いてSECRET_KEYを手動で変更してください"
        echo "   生成コマンド: python3 -c \"import secrets; print(secrets.token_hex(32))\""
    fi

    echo -e "${GREEN}[OK] .envファイルを作成しました${NC}"
fi

echo ""

# ============================================
# [3/6] 必要なディレクトリの作成
# ============================================
echo "[3/6] 必要なディレクトリを作成中..."

mkdir -p data/raw data/processed data/models logs

echo -e "${GREEN}[OK] ディレクトリを作成しました${NC}"

echo ""

# ============================================
# [4/6] Dockerコンテナのビルドと起動
# ============================================
echo "[4/6] Dockerコンテナを起動中..."
echo "   これには数分かかる場合があります..."

docker compose up -d --build

if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR] Dockerコンテナの起動に失敗しました${NC}"
    echo "   エラーログを確認してください: docker compose logs"
    exit 1
fi

echo -e "${GREEN}[OK] Dockerコンテナを起動しました${NC}"

echo ""

# ============================================
# [5/6] データベース初期化を待機
# ============================================
echo "[5/6] データベースの初期化を待機中..."
echo "   データベースが起動するまで15秒待機します..."

sleep 15

# データベース初期化スクリプトの実行
if [ -f scripts/init_db.py ]; then
    echo "[INFO] データベース初期化スクリプトを実行中..."
    docker compose exec -T web python scripts/init_db.py
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}[WARN] データベース初期化に失敗しました${NC}"
        echo "   手動で実行してください: docker compose exec web python scripts/init_db.py"
    else
        echo -e "${GREEN}[OK] データベースを初期化しました${NC}"
    fi
else
    echo "[INFO] データベースは自動的に初期化されます"
fi

echo ""

# ============================================
# [6/6] サービスの状態確認
# ============================================
echo "[6/6] サービスの状態を確認中..."

docker compose ps

echo ""

# ============================================
# 完了メッセージ
# ============================================
echo ""
echo "================================================"
echo -e "${GREEN}[完了] 起動成功!${NC}"
echo "================================================"
echo ""
echo "[アプリケーションURL]"
echo "   http://localhost:5001"
echo ""
echo "[管理画面]"
echo "   http://localhost:5001/admin/"
echo ""
echo "[便利なコマンド]"
echo "   ログ確認: docker compose logs -f web"
echo "   停止:     docker compose down"
echo "   再起動:   docker compose restart"
echo ""
echo "[ドキュメント]"
echo "   QUICKSTART.md を参照してください"
echo ""
echo "================================================"

# ブラウザを開く
echo ""
read -p "ブラウザを開きますか? (y/N): " OPEN_BROWSER

if [[ "$OPEN_BROWSER" =~ ^[Yy]$ ]]; then
    echo "ブラウザを開いています..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open http://localhost:5001
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        xdg-open http://localhost:5001 2>/dev/null || echo "http://localhost:5001 を手動で開いてください"
    fi
else
    echo "http://localhost:5001 を手動で開いてください"
fi

echo ""
echo "セットアップが完了しました!"
