"""
Run script for the JRA Horse Racing Prediction application.
Optimized for personal use - localhost only for security.
"""
import os
from config.logging_config import setup_logging
from src.web.app import create_app

# Setup logging
setup_logging(log_level='INFO')

# Create Flask app
app = create_app()

if __name__ == '__main__':
    # Get environment variables
    # Force localhost binding for security (personal use)
    host = os.getenv('FLASK_HOST', '127.0.0.1')
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_ENV', 'development') == 'development'

    # Security warning if host is not localhost
    if host not in ['127.0.0.1', 'localhost']:
        print("\n" + "=" * 80)
        print("⚠️  WARNING: Running on non-localhost address!")
        print(f"   Host: {host}")
        print("   This application is designed for personal use only.")
        print("   For security, it's recommended to use 127.0.0.1 or localhost.")
        print("=" * 80 + "\n")

    print(f"""
    ╔════════════════════════════════════════════════════╗
    ║   JRA競馬予想アプリケーション（個人使用版）       ║
    ╠════════════════════════════════════════════════════╣
    ║   Server: http://{host}:{port:<25} ║
    ║   Environment: {'Development' if debug else 'Production':<35} ║
    ║   Security: Localhost Only (Personal Use)          ║
    ╚════════════════════════════════════════════════════╝

    📍 アクセスURL: http://{host}:{port}
    🔒 セキュリティ: localhost専用バインド
    🛠️  管理画面: http://{host}:{port}/admin/
    📖 ドキュメント: README_PERSONAL.md参照

    停止: Ctrl+C
    """)

    app.run(host=host, port=port, debug=debug)
