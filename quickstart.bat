@echo off
chcp 932 > nul
setlocal enabledelayedexpansion

echo ================================================
echo JRA馬馬予想アプリ - クイックスタート
echo ================================================
echo このスクリプトは初回セットアップを自動化します
echo ================================================
echo.

REM ============================================
REM [1/6] 必要なツールのチェック
REM ============================================
echo [1/6] 必要なツールをチェック中...

REM Dockerのチェック
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Dockerがインストールされていません
    echo    Docker Desktopをインストールしてください: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)
echo [OK] Docker検出

REM Docker Composeのチェック
docker compose version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker Composeが利用できません
    echo    Docker Desktopを最新版に更新してください
    pause
    exit /b 1
)
echo [OK] Docker Compose検出

REM Pythonのチェック(SECRET_KEY生成用)
python --version >nul 2>&1
if errorlevel 1 (
    echo [WARN] Python未検出 - SECRET_KEYの自動生成をスキップします
    echo    手動で.envファイルのSECRET_KEYを変更してください
    set PYTHON_AVAILABLE=0
) else (
    echo [OK] Python検出
    set PYTHON_AVAILABLE=1
)

echo.

REM ============================================
REM [2/6] .envファイルのセットアップ
REM ============================================
echo [2/6] 環境変数ファイルをセットアップ中...

if exist .env (
    echo [OK] .envファイルは既に存在します
    echo    既存の設定をそのまま使用します
) else (
    if not exist .env.personal.example (
        echo [ERROR] .env.personal.exampleファイルが見つかりません
        echo    リポジトリが正しくクローンされているか確認してください
        pause
        exit /b 1
    )

    echo [INFO] .envファイルを作成しています...
    copy .env.personal.example .env > nul

    if !PYTHON_AVAILABLE!==1 (
        echo [INFO] SECRET_KEYを自動生成しています...
        for /f "delims=" %%i in ('python -c "import secrets; print(secrets.token_hex(32))"') do set SECRET_KEY=%%i

        REM PowerShellを使ってSECRET_KEYを置換
        powershell -Command "(Get-Content .env) -replace 'your-secret-key-here-change-this', '!SECRET_KEY!' | Set-Content .env"
        echo [OK] SECRET_KEYを生成しました
    ) else (
        echo [WARN] SECRET_KEYが未設定です
        echo    .envファイルを開いてSECRET_KEYを手動で変更してください
        echo    生成コマンド: python -c "import secrets; print(secrets.token_hex(32))"
    )

    echo [OK] .envファイルを作成しました
)

echo.

REM ============================================
REM [3/6] 必要なディレクトリの作成
REM ============================================
echo [3/6] 必要なディレクトリを作成中...

if not exist data mkdir data
if not exist data\raw mkdir data\raw
if not exist data\processed mkdir data\processed
if not exist data\models mkdir data\models
if not exist logs mkdir logs

echo [OK] ディレクトリを作成しました

echo.

REM ============================================
REM [4/6] Dockerコンテナのビルドと起動
REM ============================================
echo [4/6] Dockerコンテナを起動中...
echo    これには数分かかる場合があります...

docker compose up -d --build
if errorlevel 1 (
    echo [ERROR] Dockerコンテナの起動に失敗しました
    echo    エラーログを確認してください: docker compose logs
    pause
    exit /b 1
)

echo [OK] Dockerコンテナを起動しました

echo.

REM ============================================
REM [5/6] データベース初期化を待機
REM ============================================
echo [5/6] データベースの初期化を待機中...
echo    データベースが起動するまで15秒待機します...

timeout /t 15 /nobreak > nul

REM データベース初期化スクリプトの実行
if exist scripts\init_db.py (
    echo [INFO] データベース初期化スクリプトを実行中...
    docker compose exec -T web python scripts/init_db.py
    if errorlevel 1 (
        echo [WARN] データベース初期化に失敗しました
        echo    手動で実行してください: docker compose exec web python scripts/init_db.py
    ) else (
        echo [OK] データベースを初期化しました
    )
) else (
    echo [INFO] データベースは自動的に初期化されます
)

echo.

REM ============================================
REM [6/6] サービスの状態確認
REM ============================================
echo [6/6] サービスの状態を確認中...

docker compose ps

echo.

REM ============================================
REM 起動完了メッセージ
REM ============================================
echo.
echo ================================================
echo [完了] 起動完了！
echo ================================================
echo.
echo [アプリケーションURL]
echo    http://localhost:5000
echo.
echo [管理画面]
echo    http://localhost:5000/admin/
echo.
echo [便利なコマンド]
echo    ログ確認: docker compose logs -f web
echo    停止:     docker compose down
echo    再起動:   docker compose restart
echo.
echo [ドキュメント]
echo    README_PERSONAL.md を参照してください
echo.
echo ================================================

REM ブラウザを開く
echo ブラウザを開きますか？ (Y/N)
set /p OPEN_BROWSER=選択:

if /i "%OPEN_BROWSER%"=="Y" (
    echo ブラウザを開いています...
    start http://localhost:5000
) else (
    echo http://localhost:5000 を手動で開いてください
)

echo.
echo セットアップが完了しました！
pause
